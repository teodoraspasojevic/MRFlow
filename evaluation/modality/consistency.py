"""Modality consistency: is a volume the contrast it is supposed to be?

    python -m evaluation.modality.consistency --run <eval out dir> \
        --ckpt <classifier>/best.pt --cases baselines/cases-test-n100.json

For every in-scope case of an evaluation run, the trained classifier is asked for the contrast of

    generated     the cached rollout, expected to be the modality it was **conditioned** on
    ground truth  the MR-RATE series, expected to be its **dataset** label

Both through the evaluation's own input path -- `canonicalize_generated` / `canonicalize_gt`, the
arrays `EvalAccumulator.add` builds -- and then only the classifier's own `classifier_input`. In
MR-RATE the two expected labels are the same field (the rollout is conditioned on the series'
modality), so the ground truth is the empirical ceiling for the generated score.

Every expected case stays in every denominator: a missing generation or an unreadable volume is a
wrong answer in the strict metrics and is counted (`n_modality_*_missing` / `_failed`).

    gap          GT - generated, same metric, same cases
    normalized   accuracy: generated / GT (defined when GT > 0);
                 balanced accuracy: (generated - 1/4) / (GT - 1/4), chance-corrected, defined when
                 GT > 1/4
    paired CI    percentile bootstrap of GT - generated, resampling cases jointly for both sides

The CLI writes `<save>/modality_consistency.json`, `cases.csv` (one row per case, both sides) and
`report.md`; `--save` defaults to `<run>/modality_consistency`. With `--modality_ckpt`,
`evaluation/main.py` computes the same thing inside `EvalAccumulator` and adds the `modality_*`
keys to `metrics.json`.
"""

import argparse
import json
import os
from glob import glob
from multiprocessing import Pool

import numpy as np

from evaluation.modality import (CLASSES, NUM_CLASSES, classifier_input, file_sha256,
                                 load_classifier, modality_index, predict_proba)
from evaluation.modality.metrics import (BOOTSTRAP_KEYS, arrays, bootstrap_ci, case_record,
                                         markdown, paired_bootstrap, summarize)

SIDES = ("gen", "gt")
BOOTSTRAP_SEED = 0


def score_keys():
    keys = []
    for metric in ("accuracy", "balanced_accuracy"):
        keys += [f"modality_gen_{metric}", f"modality_gt_{metric}", f"modality_{metric}_gap",
                 f"modality_{metric}_gap_ci_low", f"modality_{metric}_gap_ci_high",
                 f"modality_{metric}_normalized"]
    for metric in ("macro_f1", "mean_p_expected"):
        keys += [f"modality_gen_{metric}", f"modality_gt_{metric}"]
    return tuple(keys) + tuple(f"modality_{side}_recall_{name}"
                               for name in CLASSES for side in SIDES)


def count_keys():
    return ("n_modality_cases",) + tuple(f"n_modality_{side}_{what}" for side in SIDES
                                         for what in ("evaluated", "missing", "failed"))


def metric_keys():
    return score_keys() + count_keys()


class ModalityConsistency:
    """Per-case predictions for both sides of every pair it is handed."""

    def __init__(self, ckpt, device="auto", batch_size=2):
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.meta = load_classifier(ckpt, device)
        self.ckpt, self.ckpt_sha256 = ckpt, file_sha256(ckpt)
        self.batch_size = batch_size
        self.records = []

    def add(self, case_id, bucket, modality, gt=None, gen=None, gt_error="", gen_error="",
            gt_status="missing", gen_status="missing"):
        """`gt` / `gen` are canonicalized `[0, 1]` volumes, or None with an error saying why."""
        inputs = {}
        for side, volume in (("gt", gt), ("gen", gen)):
            inputs[side] = None if volume is None else classifier_input(volume)
        self.add_inputs(case_id, bucket, modality, inputs["gt"], inputs["gen"],
                        gt_error, gen_error, gt_status, gen_status)

    def add_inputs(self, case_id, bucket, modality, gt_input=None, gen_input=None, gt_error="",
                   gen_error="", gt_status="missing", gen_status="missing"):
        """As `add`, from `classifier_input` arrays already computed."""
        expected = CLASSES[modality_index(modality)]
        given = {s: x for s, x in (("gt", gt_input), ("gen", gen_input)) if x is not None}
        probs = dict(zip(given, predict_proba(self.model, np.stack(list(given.values())),
                                              self.batch_size))) if given else {}
        errors = {"gt": (gt_error, gt_status), "gen": (gen_error, gen_status)}
        self.records.append({"case_id": case_id, "bucket": bucket, **{
            side: case_record(expected, probs.get(side), *errors[side]) for side in SIDES}})

    def state(self):
        return {"ckpt_sha256": self.ckpt_sha256, "records": list(self.records)}


def merge_states(states):
    if any(st["ckpt_sha256"] != states[0]["ckpt_sha256"] for st in states):
        raise ValueError("shards were scored with different modality classifiers")
    return {"ckpt_sha256": states[0]["ckpt_sha256"],
            "records": [r for st in states for r in st["records"]]}


def summarize_pairs(state, n_boot=2000, seed=BOOTSTRAP_SEED):
    """Both sides, their gap, the normalized score, paired CIs and a per-class breakdown."""
    records = state["records"]
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id among modality-consistency records")
    y, pred = {}, {}
    for side in SIDES:
        y[side], *pred[side] = arrays([r[side] for r in records])
    assert np.array_equal(y["gen"], y["gt"]), "generated and ground truth expect different labels"
    y = y["gt"]

    sides = {side: summarize(y, *pred[side]) for side in SIDES}
    ci = {side: bootstrap_ci(y, *pred[side], n_boot=n_boot, seed=seed) for side in SIDES}
    paired = paired_bootstrap(y, *pred["gt"], *pred["gen"], n_boot=n_boot, seed=seed)
    gen, gt = sides["gen"], sides["gt"]
    chance = 1.0 / NUM_CLASSES

    per_class = {}
    for c, name in enumerate(CLASSES):
        rows = y == c
        per_class[name] = {"n": int(rows.sum()), **{
            f"{side}_{k}": sides[side]["per_class"][name][k] for side in SIDES
            for k in ("recall", "precision", "f1")}}
        for side in SIDES:
            p = pred[side][1][rows, c]
            per_class[name][f"{side}_mean_p_expected"] = (
                float(np.nan_to_num(p).sum() / rows.sum()) if rows.any() else float("nan"))
        per_class[name]["recall_gap"] = per_class[name]["gt_recall"] - per_class[name]["gen_recall"]

    both_ok = (pred["gen"][0] >= 0) & (pred["gt"][0] >= 0)
    return {
        "ckpt_sha256": state["ckpt_sha256"], "n_cases": len(records),
        "generated": gen, "ground_truth": gt,
        "bootstrap_ci_95": {"generated": ci["gen"], "ground_truth": ci["gt"]},
        "gap_gt_minus_generated": {k: gt[k] - gen[k] for k in BOOTSTRAP_KEYS},
        "gap_paired_bootstrap_ci_95": paired,
        "normalized": {
            "accuracy": gen["accuracy"] / gt["accuracy"] if gt["accuracy"] > 0 else float("nan"),
            "balanced_accuracy_chance_corrected": (
                (gen["balanced_accuracy"] - chance) / (gt["balanced_accuracy"] - chance)
                if gt["balanced_accuracy"] > chance else float("nan")),
        },
        "prediction_agreement_gen_vs_gt": (
            float(np.mean(pred["gen"][0][both_ok] == pred["gt"][0][both_ok]))
            if both_ok.any() else float("nan")),
        "per_class": per_class,
        "status_counts": {name: {v: sum(r[side]["status"] == v for r in records)
                                 for v in ("ok", "missing", "failed")}
                          for side, name in (("gen", "generated"), ("gt", "ground_truth"))},
    }


def scalar_metrics(summary):
    """The `modality_*` keys `EvalAccumulator` reports; all nan when the metric was off."""
    if summary is None:
        return {k: (0 if k.startswith("n_") else float("nan")) for k in metric_keys()}
    out = {"n_modality_cases": summary["n_cases"]}
    names = {"gen": "generated", "gt": "ground_truth"}
    for side, name in names.items():
        s = summary[name]
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "mean_p_expected"):
            out[f"modality_{side}_{k}"] = s[k]
        for c in CLASSES:
            out[f"modality_{side}_recall_{c}"] = s["per_class"][c]["recall"]
        out[f"n_modality_{side}_evaluated"] = s["n_evaluated"]
        out[f"n_modality_{side}_missing"] = summary["status_counts"][name]["missing"]
        out[f"n_modality_{side}_failed"] = summary["status_counts"][name]["failed"]
    for k in ("accuracy", "balanced_accuracy"):
        out[f"modality_{k}_gap"] = summary["gap_gt_minus_generated"][k]
        out[f"modality_{k}_gap_ci_low"], out[f"modality_{k}_gap_ci_high"] = (
            summary["gap_paired_bootstrap_ci_95"][k])
    out["modality_accuracy_normalized"] = summary["normalized"]["accuracy"]
    out["modality_balanced_accuracy_normalized"] = (
        summary["normalized"]["balanced_accuracy_chance_corrected"])
    return {k: out[k] for k in metric_keys()}


### Scoring an existing evaluation run ###


def load_run(run, cases_path=None):
    """Every case of a run's manifests, checked against the frozen population when given.

    Raises on a duplicated case id, a case set that is not the frozen one, a case whose identity
    fields disagree with it, a cached file that is not named after its case, or a
    `conditioning_uid` that does not hash to the recorded study/modality/plane -- the last being
    what says the recorded modality is the one the rollout was conditioned on.
    """
    from evaluation.main import conditioning_uid

    manifests = sorted(glob(os.path.join(run, "shard-*.json")))
    if not manifests:
        raise SystemExit(f"no shard-*.json in {run}")
    cases = [c for path in manifests for c in json.load(open(path))["cases"]]
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{run}: duplicate case_id in the manifests")

    if cases_path:
        with open(cases_path) as handle:
            frozen = {c["case_id"]: c for c in json.load(handle)["cases"]}
        if set(ids) != set(frozen):
            raise ValueError(f"{run}: {len(set(ids) - set(frozen))} cases not in {cases_path}, "
                             f"{len(set(frozen) - set(ids))} of its cases missing")
        for c in cases:
            for key in ("modality", "plane", "study_uid", "archive", "member"):
                if c[key] != frozen[c["case_id"]][key]:
                    raise ValueError(f"{run}: case {c['case_id']} has {key}={c[key]!r}, the "
                                     f"frozen population says {frozen[c['case_id']][key]!r}")
    for c in cases:
        if c.get("generated") and c["case_id"] not in c["generated"]:
            raise ValueError(f"{run}: case {c['case_id']} points at {c['generated']}")
        if c.get("conditioning_uid") and c["conditioning_uid"] != conditioning_uid(
                c["study_uid"], c["modality"], c["plane"]):
            raise ValueError(f"{run}: case {c['case_id']} conditioning_uid does not match its "
                             f"recorded modality/plane")
    return cases


def _inputs(job):
    """One case -> both sides' classifier inputs, through the evaluation's own loading path
    (`score_cached` reads a pair exactly like this). Errors are returned, never raised."""
    from echosyn.common.mrrate import load_native_volume, read_member
    from evaluation.paper_metrics import canonicalize_generated, canonicalize_gt

    run, case = job
    out = {"gt": None, "gen": None, "gt_error": "", "gen_error": "",
           "gt_status": "failed", "gen_status": "failed"}
    try:
        real, spacing = load_native_volume(read_member(case["archive"], case["member"]),
                                           case["plane"])
        out["gt"] = classifier_input(canonicalize_gt(real, spacing, case["plane"]))
    except Exception as e:
        out["gt_error"] = f"{type(e).__name__}: {e}"

    path = os.path.join(run, "generated", case.get("generated") or "")
    if case["status"] != "generated":
        out["gen_error"], out["gen_status"] = f"manifest status {case['status']}", "missing"
    elif not os.path.isfile(path):
        out["gen_error"], out["gen_status"] = f"{path} does not exist", "missing"
    else:
        try:
            produced = np.load(path).astype(np.float32)
            out["gen"] = classifier_input(canonicalize_generated(produced))
        except Exception as e:
            out["gen_error"] = f"{type(e).__name__}: {e}"
    return case["case_id"], out


def report(summary, title):
    gen, gt = summary["generated"], summary["ground_truth"]
    ci, paired = summary["bootstrap_ci_95"], summary["gap_paired_bootstrap_ci_95"]
    gap = summary["gap_gt_minus_generated"]

    def cell(value, interval):
        return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"

    lines = [f"## {title}", "", f"{summary['n_cases']} cases. Status counts: "
             f"{json.dumps(summary['status_counts'])}. Strict metrics count every missing or "
             f"failed case as wrong.", "",
             "| metric | generated | ground truth (ceiling) | gap GT - gen [paired 95% CI] |",
             "|---|---|---|---|"]
    for k in BOOTSTRAP_KEYS:
        lines.append(f"| {k} | {cell(gen[k], ci['generated'][k])} | "
                     f"{cell(gt[k], ci['ground_truth'][k])} | {cell(gap[k], paired[k])} |")
    norm = summary["normalized"]
    lines += ["", f"Normalized accuracy (gen / GT): {norm['accuracy']:.4f}. Chance-corrected "
              f"normalized balanced accuracy ((gen - 1/4) / (GT - 1/4)): "
              f"{norm['balanced_accuracy_chance_corrected']:.4f}. Prediction agreement "
              f"between the two sides (both evaluated): "
              f"{summary['prediction_agreement_gen_vs_gt']:.4f}.", "",
              "| class | n | gen recall | GT recall | recall gap | gen mean p | GT mean p |",
              "|---|---|---|---|---|---|---|"]
    for name, row in summary["per_class"].items():
        lines.append(f"| {name} | {row['n']} | {row['gen_recall']:.4f} | {row['gt_recall']:.4f} "
                     f"| {row['recall_gap']:.4f} | {row['gen_mean_p_expected']:.4f} | "
                     f"{row['gt_mean_p_expected']:.4f} |")
    lines += ["", markdown(gen, ci["generated"], "Generated vs conditioning modality"),
              markdown(gt, ci["ground_truth"], "Ground truth vs dataset label")]
    return "\n".join(lines)


def main():
    from evaluation.challenge_metrics import ALLOWED_MODALITIES
    from evaluation.modality.data import write_csv

    parser = argparse.ArgumentParser(description="Modality consistency of an evaluation run.")
    parser.add_argument("--run", required=True, help="An evaluation --out dir: shard-*.json + "
                                                     "generated/*.npy.")
    parser.add_argument("--ckpt", required=True, help="The classifier's best.pt.")
    parser.add_argument("--cases", default=None,
                        help="The frozen population the run must match, e.g. "
                             "baselines/cases-test-n100.json.")
    parser.add_argument("--save", default=None, help="Default: <run>/modality_consistency.")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count()))
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()
    save = args.save or os.path.join(args.run, "modality_consistency")
    os.makedirs(save, exist_ok=True)

    cases = load_run(args.run, args.cases)
    scored = [c for c in cases if c["modality"].lower() in ALLOWED_MODALITIES]
    print(f"[modality] {len(cases)} cases, {len(scored)} in scope, "
          f"{len(cases) - len(scored)} out-of-scope modality")

    metric = ModalityConsistency(args.ckpt)
    with Pool(args.workers) as pool:
        for case, (case_id, out) in zip(scored, pool.imap(_inputs, [(args.run, c) for c in scored],
                                                         chunksize=2)):
            assert case_id == case["case_id"]
            metric.add_inputs(case_id, case["bucket"], case["modality"], out["gt"], out["gen"],
                              out["gt_error"], out["gen_error"], out["gt_status"],
                              out["gen_status"])

    summary = summarize_pairs(metric.state(), n_boot=args.n_boot)
    by_case = {c["case_id"]: c for c in cases}
    rows = [{"case_id": r["case_id"], "bucket": r["bucket"],
             "study_uid": by_case[r["case_id"]]["study_uid"],
             "series_member": by_case[r["case_id"]]["member"],
             "generated_file": by_case[r["case_id"]].get("generated") or "",
             **{f"{side}_{k}": v for side in SIDES for k, v in r[side].items()}}
            for r in metric.records]
    write_csv(os.path.join(save, "cases.csv"), rows, list(rows[0]))
    result = {"run": args.run, "ckpt": args.ckpt, "cases": args.cases,
              "n_manifest_cases": len(cases), "n_out_of_scope": len(cases) - len(scored),
              "metrics": scalar_metrics(summary), "summary": summary}
    with open(os.path.join(save, "modality_consistency.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    text = report(summary, f"Modality consistency: {args.run}")
    with open(os.path.join(save, "report.md"), "w") as handle:
        handle.write(text)
    print(text)
    print(f"-> {save}")


if __name__ == "__main__":
    main()
