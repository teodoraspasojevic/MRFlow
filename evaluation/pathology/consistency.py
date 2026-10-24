"""Pathology consistency: does a generated volume show the findings of the report it came from?

    python -m evaluation.pathology.consistency --run <eval out dir> \
        --ckpt <classifier>/best.pt --cases baselines/cases-test-n100.json

The trained multi-label classifier scores, for every in-scope case of an evaluation run, the
generated volume and its ground truth against the **same** target: the study's 11 pathology
labels. The generator was conditioned on that study's report, and the labels were extracted from
the report, so the ground truth is the empirical ceiling -- how much of the report's pathology a
classifier can read off one real series of that study.

Inputs come through the modality metric's own path (`evaluation.modality.consistency._inputs`:
`canonicalize_gt` / `canonicalize_generated`, then `classifier_input`), and the run is checked by
its `load_run`. Every expected case stays in every denominator: a missing or unreadable volume
scores the train prior (see `metrics.py`) and is counted.

    gap          GT - generated, same metric, same cases, same scored labels
    normalized   macro AUROC: (gen - 0.5) / (GT - 0.5), defined when GT > 0.5;
                 macro AUPRC: (gen - p) / (GT - p) with p the mean prevalence of the scored labels
                 (AUPRC's chance level), defined when GT > p
    paired CI    percentile bootstrap of GT - generated, resampling **studies** jointly for both
                 sides -- a study contributes several cases and one label vector
    correlation  per scored label, Pearson r between the generated and the GT probability over
                 cases where both exist, then the mean: does a generation carry its real
                 counterpart's pathology signal

Writes `<save>/pathology_consistency.json`, `cases.csv` and `report.md`; `--save` defaults to
`<run>/pathology_consistency`. With `--pathology_ckpt`, `evaluation/main.py` adds the
`pathology_*` keys to `metrics.json`.
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from evaluation.modality import classifier_input, file_sha256
from evaluation.pathology import LABELS, NUM_LABELS, labels_sha256, load_classifier, \
    predict_proba, read_labels
from evaluation.pathology.metrics import (BOOTSTRAP_KEYS, bootstrap_ci, markdown,
                                          paired_bootstrap, scored_labels, summarize)

SIDES = ("gen", "gt")
BOOTSTRAP_SEED = 0
MODALITIES = ("T1w", "T2w", "SWI", "FLAIR")


def score_keys():
    keys = []
    for metric in ("macro_auroc", "macro_auprc"):
        keys += [f"pathology_gen_{metric}", f"pathology_gt_{metric}", f"pathology_{metric}_gap",
                 f"pathology_{metric}_gap_ci_low", f"pathology_{metric}_gap_ci_high",
                 f"pathology_{metric}_normalized"]
    keys += ["pathology_gen_macro_f1", "pathology_gt_macro_f1", "pathology_macro_f1_gap",
             "pathology_gen_gt_prob_correlation"]
    return tuple(keys) + tuple(f"pathology_{side}_auroc_{name}"
                               for name in LABELS for side in SIDES)


def count_keys():
    return ("n_pathology_cases", "n_pathology_studies", "n_pathology_scored_labels",
            "n_pathology_unlabelled") + tuple(f"n_pathology_{side}_{what}" for side in SIDES
                                              for what in ("evaluated", "missing", "failed"))


def metric_keys():
    return score_keys() + count_keys()


class PathologyConsistency:
    """Per-case label probabilities for both sides of every pair it is handed."""

    def __init__(self, ckpt, device="auto", batch_size=2):
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.meta = load_classifier(ckpt, device)
        root = self.meta["config"]["labels_root"]
        if labels_sha256(root) != self.meta["labels_sha256"]:
            raise ValueError(f"{root} is not the label table {ckpt} was trained against")
        self.labels = read_labels(root)
        self.ckpt_sha256 = file_sha256(ckpt)
        self.batch_size = batch_size
        self.records = []
        self.n_unlabelled = 0

    def add(self, case_id, bucket, study_uid, gt=None, gen=None, gt_error="", gen_error="",
            gt_status="missing", gen_status="missing"):
        """`gt` / `gen` are canonicalized `[0, 1]` volumes, or None with an error saying why."""
        self.add_inputs(case_id, bucket, study_uid,
                        None if gt is None else classifier_input(gt),
                        None if gen is None else classifier_input(gen),
                        gt_error, gen_error, gt_status, gen_status)

    def add_inputs(self, case_id, bucket, study_uid, gt_input=None, gen_input=None, gt_error="",
                   gen_error="", gt_status="missing", gen_status="missing"):
        """As `add`, from `classifier_input` arrays. A study without labels is counted, not scored."""
        if study_uid not in self.labels:
            self.n_unlabelled += 1
            return
        given = {s: x for s, x in (("gt", gt_input), ("gen", gen_input)) if x is not None}
        probs = dict(zip(given, predict_proba(self.model, np.stack(list(given.values())),
                                              self.batch_size))) if given else {}
        errors = {"gt": (gt_error, gt_status), "gen": (gen_error, gen_status)}
        record = {"case_id": case_id, "bucket": bucket, "study_uid": study_uid,
                  "target": self.labels[study_uid].tolist()}
        for side in SIDES:
            p = probs.get(side)
            record[side] = {"status": "ok" if p is not None else errors[side][1],
                            "error": "" if p is not None else errors[side][0],
                            "probs": p.tolist() if p is not None else [float("nan")] * NUM_LABELS}
        self.records.append(record)

    def state(self):
        return {"ckpt_sha256": self.ckpt_sha256, "records": list(self.records),
                "n_unlabelled": self.n_unlabelled, "thresholds": self.meta["thresholds"],
                "prior": self.meta["prior"],
                "min_positives": self.meta["config"]["min_positives"]}


def merge_states(states):
    if any(st["ckpt_sha256"] != states[0]["ckpt_sha256"] for st in states):
        raise ValueError("shards were scored with different pathology classifiers")
    return {**states[0], "records": [r for st in states for r in st["records"]],
            "n_unlabelled": sum(st["n_unlabelled"] for st in states)}


def _correlation(a, b, scored):
    ok = ~np.isnan(a).any(axis=1) & ~np.isnan(b).any(axis=1)
    rs = [np.corrcoef(a[ok, j], b[ok, j])[0, 1] for j, n in enumerate(LABELS)
          if n in scored and ok.sum() > 2 and a[ok, j].std() > 0 and b[ok, j].std() > 0]
    return float(np.mean(rs)) if rs else float("nan")


def summarize_pairs(state, n_boot=2000, seed=BOOTSTRAP_SEED):
    """Both sides, their gap, normalized scores, study-paired CIs, per-label and per-contrast."""
    records = state["records"]
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id among pathology-consistency records")
    y = np.array([r["target"] for r in records], dtype=np.float64).reshape(-1, NUM_LABELS)
    probs = {side: np.array([r[side]["probs"] for r in records],
                            dtype=np.float64).reshape(-1, NUM_LABELS) for side in SIDES}
    groups = [r["study_uid"] for r in records]
    thresholds, prior = np.array(state["thresholds"]), np.array(state["prior"])
    scored = scored_labels(y, state["min_positives"])

    sides = {s: summarize(y, probs[s], thresholds, prior, scored=scored) for s in SIDES}
    ci = {s: bootstrap_ci(y, probs[s], groups, thresholds, prior, scored, n_boot, seed)
          for s in SIDES}
    paired = paired_bootstrap(y, probs["gt"], probs["gen"], groups, thresholds, prior, scored,
                              n_boot, seed)
    gen, gt = sides["gen"], sides["gt"]
    chance_ap = float(np.mean([y[:, LABELS.index(n)].mean() for n in scored])) if scored else 0.0

    per_label = {n: {"scored": n in scored, "n_pos": gt["per_label"][n]["n_pos"],
                     **{f"{s}_{k}": sides[s]["per_label"][n][k] for s in SIDES
                        for k in ("auroc", "auprc", "f1")}} for n in LABELS}
    for row in per_label.values():
        row["auroc_gap"] = row["gt_auroc"] - row["gen_auroc"]

    by_modality = {}
    for m in MODALITIES:
        rows = np.array([r["bucket"].split("__")[0] == m for r in records])
        if rows.any():
            sub = scored_labels(y[rows], state["min_positives"])
            by_modality[m] = {"n": int(rows.sum()), "scored_labels": sub, **{
                f"{s}_macro_auroc": summarize(y[rows], probs[s][rows], thresholds, prior,
                                              scored=sub)["macro_auroc"] for s in SIDES}}

    return {
        "ckpt_sha256": state["ckpt_sha256"], "n_cases": len(records),
        "n_studies": len(set(groups)), "n_unlabelled": state["n_unlabelled"],
        "scored_labels": scored,
        "generated": gen, "ground_truth": gt,
        "bootstrap_ci_95": {"generated": ci["gen"], "ground_truth": ci["gt"]},
        "gap_gt_minus_generated": {k: gt[k] - gen[k] for k in BOOTSTRAP_KEYS},
        "gap_paired_bootstrap_ci_95": paired,
        "normalized": {
            "macro_auroc": ((gen["macro_auroc"] - 0.5) / (gt["macro_auroc"] - 0.5)
                            if gt["macro_auroc"] > 0.5 else float("nan")),
            "macro_auprc": ((gen["macro_auprc"] - chance_ap) / (gt["macro_auprc"] - chance_ap)
                            if gt["macro_auprc"] > chance_ap else float("nan")),
            "macro_auprc_chance": chance_ap,
        },
        "gen_gt_prob_correlation": _correlation(probs["gen"], probs["gt"], scored),
        "per_label": per_label,
        "by_modality": by_modality,
        "status_counts": {name: {v: sum(r[side]["status"] == v for r in records)
                                 for v in ("ok", "missing", "failed")}
                          for side, name in (("gen", "generated"), ("gt", "ground_truth"))},
    }


def scalar_metrics(summary):
    """The `pathology_*` keys `EvalAccumulator` reports; all nan (counts 0) when the metric is off."""
    if summary is None:
        return {k: (0 if k.startswith("n_") else float("nan")) for k in metric_keys()}
    out = {"n_pathology_cases": summary["n_cases"], "n_pathology_studies": summary["n_studies"],
           "n_pathology_scored_labels": len(summary["scored_labels"]),
           "n_pathology_unlabelled": summary["n_unlabelled"],
           "pathology_gen_gt_prob_correlation": summary["gen_gt_prob_correlation"]}
    for side, name in (("gen", "generated"), ("gt", "ground_truth")):
        s = summary[name]
        for k in ("macro_auroc", "macro_auprc", "macro_f1"):
            out[f"pathology_{side}_{k}"] = s[k]
        for label in LABELS:
            out[f"pathology_{side}_auroc_{label}"] = s["per_label"][label]["auroc"]
        out[f"n_pathology_{side}_evaluated"] = s["n_evaluated"]
        out[f"n_pathology_{side}_missing"] = summary["status_counts"][name]["missing"]
        out[f"n_pathology_{side}_failed"] = summary["status_counts"][name]["failed"]
    for k in ("macro_auroc", "macro_auprc"):
        out[f"pathology_{k}_gap"] = summary["gap_gt_minus_generated"][k]
        out[f"pathology_{k}_gap_ci_low"], out[f"pathology_{k}_gap_ci_high"] = (
            summary["gap_paired_bootstrap_ci_95"][k])
        out[f"pathology_{k}_normalized"] = summary["normalized"][k]
    out["pathology_macro_f1_gap"] = summary["gap_gt_minus_generated"]["macro_f1"]
    return {k: out[k] for k in metric_keys()}


def report(summary, title):
    gen, gt = summary["generated"], summary["ground_truth"]
    ci, paired = summary["bootstrap_ci_95"], summary["gap_paired_bootstrap_ci_95"]
    gap, norm = summary["gap_gt_minus_generated"], summary["normalized"]

    def cell(value, interval):
        return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"

    lines = [f"## {title}", "",
             f"{summary['n_cases']} cases from {summary['n_studies']} studies "
             f"({summary['n_unlabelled']} unlabelled, not scored). Status counts: "
             f"{json.dumps(summary['status_counts'])}. Macro means over "
             f"{len(summary['scored_labels'])} scored labels: {', '.join(summary['scored_labels'])}.",
             "", "| metric | generated | ground truth (ceiling) | gap GT - gen [paired 95% CI] |",
             "|---|---|---|---|"]
    for k in BOOTSTRAP_KEYS:
        lines.append(f"| {k} | {cell(gen[k], ci['generated'][k])} | "
                     f"{cell(gt[k], ci['ground_truth'][k])} | {cell(gap[k], paired[k])} |")
    lines += ["", f"Normalized macro AUROC ((gen - 0.5) / (GT - 0.5)): {norm['macro_auroc']:.4f}. "
              f"Normalized macro AUPRC ((gen - p) / (GT - p), p = {norm['macro_auprc_chance']:.4f}):"
              f" {norm['macro_auprc']:.4f}. Mean per-label correlation between generated and GT "
              f"probabilities: {summary['gen_gt_prob_correlation']:.4f}.", "",
              "| label | scored | pos | gen AUROC | GT AUROC | gap | gen AUPRC | GT AUPRC |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in summary["per_label"].items():
        lines.append(f"| {name} | {'yes' if r['scored'] else 'no'} | {r['n_pos']} | "
                     f"{r['gen_auroc']:.4f} | {r['gt_auroc']:.4f} | {r['auroc_gap']:.4f} | "
                     f"{r['gen_auprc']:.4f} | {r['gt_auprc']:.4f} |")
    lines += ["", "| contrast | n | gen macro AUROC | GT macro AUROC |", "|---|---|---|---|"]
    for m, r in summary["by_modality"].items():
        lines.append(f"| {m} | {r['n']} | {r['gen_macro_auroc']:.4f} | {r['gt_macro_auroc']:.4f} |")
    lines += ["", markdown(gen, ci["generated"], "Generated vs report labels"),
              markdown(gt, ci["ground_truth"], "Ground truth vs report labels")]
    return "\n".join(lines)


def main():
    from evaluation.challenge_metrics import ALLOWED_MODALITIES
    from evaluation.modality.consistency import _inputs, load_run
    from evaluation.modality.data import write_csv

    parser = argparse.ArgumentParser(description="Pathology consistency of an evaluation run.")
    parser.add_argument("--run", required=True, help="An evaluation --out dir.")
    parser.add_argument("--ckpt", required=True, help="The pathology classifier's best.pt.")
    parser.add_argument("--cases", default=None, help="The frozen population the run must match.")
    parser.add_argument("--save", default=None, help="Default: <run>/pathology_consistency.")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count()))
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()
    save = args.save or os.path.join(args.run, "pathology_consistency")
    os.makedirs(save, exist_ok=True)

    cases = load_run(args.run, args.cases)
    scored = [c for c in cases if c["modality"].lower() in ALLOWED_MODALITIES]
    print(f"[pathology] {len(cases)} cases, {len(scored)} in scope")

    metric = PathologyConsistency(args.ckpt)
    with Pool(args.workers) as pool:
        for case, (case_id, out) in zip(scored, pool.imap(_inputs, [(args.run, c) for c in scored],
                                                         chunksize=2)):
            assert case_id == case["case_id"]
            metric.add_inputs(case_id, case["bucket"], case["study_uid"], out["gt"], out["gen"],
                              out["gt_error"], out["gen_error"], out["gt_status"],
                              out["gen_status"])

    summary = summarize_pairs(metric.state(), n_boot=args.n_boot)
    by_case = {c["case_id"]: c for c in cases}
    rows = []
    for r in metric.records:
        row = {"case_id": r["case_id"], "bucket": r["bucket"], "study_uid": r["study_uid"],
               "series_member": by_case[r["case_id"]]["member"],
               "generated_file": by_case[r["case_id"]].get("generated") or "",
               **{f"target_{n}": v for n, v in zip(LABELS, r["target"])}}
        for side in SIDES:
            row[f"{side}_status"], row[f"{side}_error"] = r[side]["status"], r[side]["error"]
            row.update({f"{side}_p_{n}": v for n, v in zip(LABELS, r[side]["probs"])})
        rows.append(row)
    write_csv(os.path.join(save, "cases.csv"), rows, list(rows[0]))
    result = {"run": args.run, "ckpt": args.ckpt, "cases": args.cases,
              "n_manifest_cases": len(cases), "n_out_of_scope": len(cases) - len(scored),
              "labels": list(LABELS), "thresholds": metric.meta["thresholds"],
              "metrics": scalar_metrics(summary), "summary": summary}
    with open(os.path.join(save, "pathology_consistency.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    text = report(summary, f"Pathology consistency: {args.run}")
    with open(os.path.join(save, "report.md"), "w") as handle:
        handle.write(text)
    print(text)
    print(f"-> {save}")


if __name__ == "__main__":
    main()
