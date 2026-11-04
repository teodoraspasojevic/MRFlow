"""How much does the report drive alignment? Full-report against null-report generations, paired.

Two runs of one model over the same studies -- one generated from the real reports, one with every
report replaced by the model's null (`main.py --null_report`) -- are scored on HLIP against the
*real* reports. Only series both runs generated are kept, so the two sides hold identical studies
and scans, and the real series of those cases give the ceiling.

    python evaluation/hlip_null_report.py --full <run> --null <run> \
        --study_subset baselines/cases-test-null100.json \
        --gt_bounds <workspace>/data/mrflow_flux_trim-v1 --trim_generated --score_collapsed \
        --out <null run>/hlip_null_report.json

The headline is the drop in study-to-report AUROC, full minus null, with a 95% CI from 2,000
bootstrap resamples of studies; each study's AUROC is the fraction of wrong reports scored below its
own (ties half), as in `hlip_metrics.retrieval_metrics`. A drop near zero means the report is not
what makes the generated studies align.
"""

import argparse
import json

import numpy as np
import torch

from evaluation import hlip_metrics as H
from evaluation.main import load_trim_version
from evaluation.rescore_hlip import HELMA_RAW_ROOT, load_states, run_cases, skip_reason


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", required=True, help="the run generated from the real reports")
    parser.add_argument("--null", required=True, help="the same model, --null_report")
    parser.add_argument("--study_subset", required=True)
    parser.add_argument("--gt_bounds", default=None)
    parser.add_argument("--trim_generated", action="store_true")
    parser.add_argument("--score_collapsed", action="store_true")
    parser.add_argument("--raw_root", default=HELMA_RAW_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True, help="the JSON to write")
    return parser.parse_args()


def study_auroc(similarity):
    """Per study (column), the fraction of wrong reports scored below its own, ties half."""
    s = similarity.T
    n = s.shape[0]
    matched = np.diag(s)
    greater = (s > matched[:, None]).sum(axis=1)
    equal = (s == matched[:, None]).sum(axis=1) - 1
    return (n - 1 - greater - equal / 2) / (n - 1)


def main():
    args = parse_args()
    bounds, policy = load_trim_version(args.gt_bounds) if args.gt_bounds else (None, None)
    runs = {"full": args.full, "null": args.null}
    cases = {}
    for side, out in runs.items():
        cases[side] = {c["case_id"]: c for c in run_cases(out, args.study_subset)
                       if skip_reason(c, bounds, args.score_collapsed) is None}
    shared = sorted(set(cases["full"]) & set(cases["null"]))
    print(f"[null] full {len(cases['full'])}, null {len(cases['null'])}, both {len(shared)} series")

    studies = {}
    for side, out in runs.items():
        states = load_states(out, [cases[side][c] for c in shared], args.raw_root, bounds, policy,
                             args.trim_generated, args.workers)
        studies[side] = H.merge_states(states)["studies"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = H.load_hlip(device)
    uids = sorted(u for u, s in studies["full"].items() if s["sentences"])
    reports, _ = H._encode_reports(model, tokenizer, device,
                                   [studies["full"][u]["sentences"] for u in uids])
    similarity = {
        "full": [H._encode_study(model, device, studies["full"][u]["scans"], "fake") for u in uids],
        "null": [H._encode_study(model, device, studies["null"][u]["scans"], "fake") for u in uids],
        "real": [H._encode_study(model, device, studies["full"][u]["scans"], "real") for u in uids],
    }
    similarity = {k: reports @ np.stack(v).T for k, v in similarity.items()}

    result = {"full_run": args.full, "null_run": args.null, "study_subset": args.study_subset,
              "gt_bounds": args.gt_bounds, "trim_generated": args.trim_generated,
              "score_collapsed": args.score_collapsed, "n_studies": len(uids),
              "n_scans": sum(len(studies["full"][u]["scans"]) for u in uids),
              "chance_r5": 5 / len(uids)}
    for side, s in similarity.items():
        result[side] = H.retrieval_metrics(s, "hlip")

    drop = study_auroc(similarity["full"]) - study_auroc(similarity["null"])
    rng = np.random.default_rng(0)
    boot = [drop[rng.integers(0, len(drop), len(drop))].mean() for _ in range(2000)]
    result["auroc_drop"] = float(drop.mean())
    result["auroc_drop_ci95"] = [float(v) for v in np.percentile(boot, [2.5, 97.5])]

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    for side in similarity:
        r = result[side]
        print(f"{side:5s} auroc {r['hlip_study_to_report_auroc']:.3f}  "
              f"r5 {r['hlip_study_to_report_r5']:.3f}  cos {r['hlip_cosine_mean']:.3f}")
    print(f"drop {result['auroc_drop']:.3f} {result['auroc_drop_ci95']} -> {args.out}")


if __name__ == "__main__":
    main()
