"""Score a trained classifier on a held-out split of its own manifest.

    python -m evaluation.modality.evaluate --ckpt <output_dir>/best.pt --split test

Writes `<out>/predictions.csv` (one row per split case, prepare failures included as failed
predictions), `<out>/metrics.json` (every `summarize` metric, bootstrap CIs, a per-plane
breakdown) and `<out>/report.md`. `--out` defaults to `<ckpt dir>/eval-<split>`.
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation.modality import CLASSES, file_sha256, load_classifier, predict_proba
from evaluation.modality.data import ModalityDataset, write_csv
from evaluation.modality.metrics import (arrays, bootstrap_ci, case_record, markdown,
                                         summarize)

ID_FIELDS = ("case_id", "patient_uid", "study_uid", "series_id", "plane", "archive", "member")


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation of the modality classifier.")
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
    dataset = ModalityDataset(meta["config"]["data_root"], args.split)
    inputs = np.stack([dataset.load(i) for i in range(len(dataset))])
    probs = dict(zip((r["case_id"] for r in dataset.rows_ok),
                     predict_proba(model, inputs, args.batch_size, device)))

    records = [{**{k: r[k] for k in ID_FIELDS},
                **case_record(r["modality"], probs.get(r["case_id"]), r["error"])}
               for r in dataset.rows]
    write_csv(os.path.join(out, "predictions.csv"), records, list(records[0]))

    summary = summarize(*arrays(records))
    ci = bootstrap_ci(*arrays(records), n_boot=args.n_boot, seed=meta["config"]["seed"])
    by_plane = {plane: summarize(*arrays([r for r in records if r["plane"] == plane]))
                for plane in sorted({r["plane"] for r in records})}
    result = {"split": args.split, "ckpt": args.ckpt, "ckpt_sha256": file_sha256(args.ckpt),
              "selected_epoch": meta["epoch"], "classes": list(CLASSES),
              "summary": summary, "bootstrap_ci_95": ci,
              "by_plane": {p: {k: s[k] for k in ("n", "accuracy", "balanced_accuracy",
                                                 "macro_f1")} for p, s in by_plane.items()}}
    with open(os.path.join(out, "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    report = markdown(summary, ci, f"Modality classifier, held-out `{args.split}` split")
    report += "\n| plane | n | accuracy | balanced accuracy |\n|---|---|---|---|\n" + "".join(
        f"| {p} | {s['n']} | {s['accuracy']:.4f} | {s['balanced_accuracy']:.4f} |\n"
        for p, s in by_plane.items())
    with open(os.path.join(out, "report.md"), "w") as handle:
        handle.write(report)
    print(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
