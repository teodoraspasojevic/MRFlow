"""Rescore a finished eval or baseline run with the HLIP report-study metrics only.

`main.py --combine` recomputes every metric. When only HLIP has changed, this reads the same
manifests and cached volumes, applies `score_cached`'s filters and trims, and writes the HLIP keys
to their own file beside the run's metrics; nothing else in the run is touched.

    python evaluation/rescore_hlip.py --out <run> \
        --gt_bounds <workspace>/data/mrflow_flux_trim-v1 --trim_generated

writes `<run>/hlip_study_mrflow_flux_trim-v1_trimgen.json` (`hlip_study.json` without bounds).
Generated volumes are read from `<run>/generated/` if it exists, otherwise from `generated.tar`
in place. `--raw_root` replaces the MR-RATE root the manifests record, for a machine that keeps
the archives elsewhere.
"""

import argparse
import io
import json
import os
import tarfile
from glob import glob
from multiprocessing import Pool

import numpy as np
import torch

from echosyn.common.mrrate import load_native_volume, read_member, read_report
from evaluation import EvalAccumulator, hlip_metrics
from evaluation.main import load_trim_version, trim_generated, trim_native
from evaluation.paper_metrics import canonicalize_generated, canonicalize_gt

HELMA_RAW_ROOT = "/hnvme/workspace/y100dc19-MR-Rate-raw"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="the run: shard-*.json + generated volumes")
    parser.add_argument("--gt_bounds", default=None, help="a trimmed version, as for main.py")
    parser.add_argument("--trim_generated", action="store_true")
    parser.add_argument("--score_collapsed", action="store_true")
    parser.add_argument("--raw_root", default=HELMA_RAW_ROOT,
                        help="where this machine keeps the MR-RATE archives")
    parser.add_argument("--workers", type=int, default=8, help="processes reading volumes")
    parser.add_argument("--limit", type=int, default=None, help="first N cases only (smoke runs)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_name(args):
    if not args.gt_bounds:
        return "hlip_study.json"
    return (f"hlip_study_{os.path.basename(os.path.normpath(args.gt_bounds))}"
            f"{'_trimgen' if args.trim_generated else ''}.json")


def skip_reason(case, bounds, score_collapsed):
    """`score_cached`'s filter, then `EvalAccumulator.add`'s: None when the case reaches HLIP."""
    if bounds is not None and bounds.get(case["case_id"], {}).get("included") != "True":
        return "excluded_by_bounds"
    if case["status"] != "generated" or (case.get("collapsed") and not score_collapsed):
        return "not_generated_or_collapsed"
    if not EvalAccumulator.is_scored(case["modality"]):
        return "out_of_scope_modality"
    return None


_JOB = {}


def _init(job):
    _JOB.update(job)
    torch.set_num_threads(1)


def _generated(name):
    if _JOB["tar"] is None:
        return np.load(os.path.join(_JOB["out"], "generated", name))
    offset, size = _JOB["tar"][f"generated/{name}"]
    with open(os.path.join(_JOB["out"], "generated.tar"), "rb") as handle:
        handle.seek(offset)
        return np.load(io.BytesIO(handle.read(size)))


def _load(case):
    """One case through the exact path `score_cached` takes, into a one-case HLIP state."""
    archive = case["archive"].replace(HELMA_RAW_ROOT, _JOB["raw_root"], 1)
    produced = _generated(case["generated"]).astype(np.float32)
    real, spacing = load_native_volume(read_member(archive, case["member"]), case["plane"])
    if _JOB["bounds"] is not None:
        real = trim_native(real, spacing, case["plane"], _JOB["bounds"][case["case_id"]])
    if _JOB["trim_generated"]:
        produced = trim_generated(produced, _JOB["policy"])
    acc = hlip_metrics.HlipAccumulator()
    acc.add(case["case_id"], case["study_uid"], read_report(archive, case["study_uid"]),
            canonicalize_generated(produced), canonicalize_gt(real, spacing, case["plane"]))
    return acc.state()


def main():
    args = parse_args()
    path = os.path.join(args.out, output_name(args))
    if os.path.exists(path) and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to redo it")

    cases = [c for p in sorted(glob(os.path.join(args.out, "shard-*.json")))
             for c in json.load(open(p))["cases"]][:args.limit]
    if not cases:
        raise SystemExit(f"no shard-*.json in {args.out}")
    bounds, policy = load_trim_version(args.gt_bounds) if args.gt_bounds else (None, None)
    skipped = {}
    keep = []
    for case in cases:
        reason = skip_reason(case, bounds, args.score_collapsed)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            keep.append(case)
    print(f"[hlip] {len(cases)} cases, {len(keep)} reach HLIP, skipped {skipped}")

    tar = None
    if not os.path.isdir(os.path.join(args.out, "generated")):
        with tarfile.open(os.path.join(args.out, "generated.tar")) as handle:
            tar = {m.name: (m.offset_data, m.size) for m in handle.getmembers() if m.isfile()}
    job = {"out": args.out, "tar": tar, "raw_root": args.raw_root, "bounds": bounds,
           "policy": policy, "trim_generated": args.trim_generated}
    with Pool(args.workers, initializer=_init, initargs=(job,)) as pool:
        states = pool.map(_load, keep, chunksize=4)

    metrics = hlip_metrics.summarize(hlip_metrics.merge_states(states))
    result = {"metrics": metrics, "extractors": {"hlip": hlip_metrics.signature()},
              "gt_bounds": args.gt_bounds, "trim_generated": args.trim_generated,
              "n_cases": len(cases), "skipped": skipped}
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    for key, value in metrics.items():
        print(f"{key:40s} {value}")
    print(f"[hlip] wrote {path}")


if __name__ == "__main__":
    main()
