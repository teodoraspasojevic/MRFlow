"""Score a trained pathology classifier on a held-out split of its own manifest.

    python -m evaluation.pathology.evaluate --ckpt <output_dir>/best.pt --split test

Thresholds and the failure prior are the checkpoint's (chosen on val, from train), never refit
here. Writes `<out>/predictions.csv` (one row per split case, prepare failures included),
`<out>/metrics.json` (every `summarize` metric, study-bootstrap CIs, per-contrast and per-plane
macro AUROC) and `<out>/report.md`. `--out` defaults to `<ckpt dir>/eval-<split>`.
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation.modality import file_sha256
from evaluation.modality.data import ModalityDataset, write_csv
from evaluation.pathology import LABELS, NUM_LABELS, load_classifier, predict_proba, read_labels
from evaluation.pathology.metrics import bootstrap_ci, markdown, scored_labels, summarize

ID_FIELDS = ("case_id", "patient_uid", "study_uid", "series_id", "modality", "plane", "archive",
             "member")


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation of the pathology classifier.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval-{args.split}")
    os.makedirs(out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_classifier(args.ckpt, device)
    config = meta["config"]
    labels = read_labels(config["labels_root"])
    dataset = ModalityDataset(config["data_root"], args.split)
    inputs = np.stack([dataset.load(i) for i in range(len(dataset))])
    ok_probs = dict(zip((r["case_id"] for r in dataset.rows_ok),
                        predict_proba(model, inputs, args.batch_size, device)))

    rows = dataset.rows  # prepare failures included, as NaN predictions
    y = np.stack([labels[r["study_uid"]] for r in rows]).astype(np.float64)
    probs = np.stack([ok_probs.get(r["case_id"], np.full(NUM_LABELS, np.nan)) for r in rows])
    groups = [r["study_uid"] for r in rows]
    thresholds, prior = np.array(meta["thresholds"]), np.array(meta["prior"])
    scored = scored_labels(y, config["min_positives"])

    summary = summarize(y, probs, thresholds, prior, scored=scored)
    ci = bootstrap_ci(y, probs, groups, thresholds, prior, scored, n_boot=args.n_boot,
                      seed=config["seed"])
    breakdown = {}
    for field in ("modality", "plane"):
        for value in sorted({r[field] for r in rows}):
            idx = np.array([r[field] == value for r in rows])
            sub = scored_labels(y[idx], config["min_positives"])
            breakdown[f"{field}={value}"] = {
                "n": int(idx.sum()), "n_scored_labels": len(sub),
                "macro_auroc": summarize(y[idx], probs[idx], thresholds, prior,
                                         scored=sub)["macro_auroc"]}

    records = []
    for r, target, p in zip(rows, y, probs):
        record = {k: r[k] for k in ID_FIELDS}
        record.update(status=r["status"], error=r["error"])
        record.update({f"target_{n}": int(t) for n, t in zip(LABELS, target)})
        record.update({f"p_{n}": float(v) for n, v in zip(LABELS, p)})
        record.update({f"pred_{n}": int(v >= thr) if not np.isnan(v) else ""
                       for n, v, thr in zip(LABELS, p, thresholds)})
        records.append(record)
    write_csv(os.path.join(out, "predictions.csv"), records, list(records[0]))

    result = {"split": args.split, "ckpt": args.ckpt, "ckpt_sha256": file_sha256(args.ckpt),
              "selected_epoch": meta["epoch"], "labels": list(LABELS),
              "thresholds": meta["thresholds"], "prior": meta["prior"], "summary": summary,
              "bootstrap_ci_95": ci, "breakdown": breakdown}
    with open(os.path.join(out, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    report = markdown(summary, ci, f"Pathology classifier, held-out `{args.split}` split")
    report += "\n| subset | n | scored labels | macro AUROC |\n|---|---|---|---|\n" + "".join(
        f"| {k} | {v['n']} | {v['n_scored_labels']} | {v['macro_auroc']:.4f} |\n"
        for k, v in breakdown.items())
    with open(os.path.join(out, "report.md"), "w") as handle:
        handle.write(report)
    print(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
