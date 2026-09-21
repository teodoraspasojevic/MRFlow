"""Example comparison videos for a baseline run, written where `log_wandb` already looks.

    python -m baselines.common.examples --out $WS/baselines/runs/<tag> --n 10

MRFlow writes these during its rollout pass (`evaluation/main.py:414-419`) and `log_wandb` globs
`<out>/examples/*.mp4` at `--combine` time. A baseline never runs that pass -- it arrives with
`generated/*.npy` already cached -- so its W&B runs come out with the metrics table and no videos.
This fills that gap from the cached volumes instead.

**Nothing about the visualization is reimplemented.** The frames come from
`evaluation.comparison_frames`, the file from `echosyn.common.save_as_mp4`, the bucket choice from
`evaluation.main.pick_example_buckets`, and the ground truth is re-read from the MR-RATE archives
with `load_native_volume` -- the same four calls, in the same order, on the same pair of arrays that
`score_cached` hands the metrics. So a baseline's video is the same artifact MRFlow's is, and
`log_wandb` needs no change to find it: run this before `--combine` and the videos appear in the run.

One example per bucket by default rather than MRFlow's two per shard: a baseline scores in a single
pass, where MRFlow's 32-task array covers every bucket several times over.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import torch

from echosyn.common import save_as_mp4
from echosyn.common.mrrate import load_native_volume, read_member
from evaluation import comparison_frames
from evaluation.main import pick_example_buckets


def write_examples(out, n=None, fps=32):
    """One `<out>/examples/<bucket>-<case_id>.mp4` per selected bucket. Returns the paths."""
    manifests = sorted(glob(os.path.join(out, "shard-*.json")))
    if not manifests:
        raise SystemExit(f"no shard-*.json in {out} -- ingest the run first")
    cases = [c for path in manifests for c in json.load(open(path))["cases"]
             if c.get("status") == "generated"]
    if not cases:
        raise SystemExit(f"no generated cases in {out}")

    # `pick_example_buckets` wants the series-style records the generation pass has.
    series = [{"modality": c["modality"], "plane": c["plane"]} for c in cases]
    n_buckets = len({f"{e['modality']}__{e['plane']}" for e in series})
    wanted = pick_example_buckets(series, shard=0, n=n or n_buckets)

    directory = os.path.join(out, "examples")
    os.makedirs(directory, exist_ok=True)
    written = []
    for case in cases:
        if case["bucket"] not in wanted:
            continue
        wanted.discard(case["bucket"])          # the first case of that bucket, then done
        produced = np.load(os.path.join(out, "generated", case["generated"])).astype(np.float32)
        real, spacing = load_native_volume(read_member(case["archive"], case["member"]),
                                           case["plane"])
        path = os.path.join(directory, f"{case['bucket']}-{case['case_id']}.mp4")
        save_as_mp4(torch.from_numpy(comparison_frames(real, produced, spacing, case["plane"])),
                    path, fps=fps)
        written.append(path)
        print(f"  {case['bucket']:<18s} {case['case_id']}  -> {os.path.basename(path)}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Write example comparison videos for a baseline "
                                                 "run, where log_wandb already looks for them.")
    parser.add_argument("--out", required=True, help="The run's results dir (holds shard-*.json).")
    parser.add_argument("--n", type=int, default=None,
                        help="How many buckets to film. Default: all of them.")
    parser.add_argument("--fps", type=int, default=32,
                        help="MRFlow's globals.target_fps; MR slices are not a time axis, so this "
                             "is a playback speed and nothing else.")
    args = parser.parse_args()
    written = write_examples(args.out, args.n, args.fps)
    print(f"{len(written)} example videos -> {args.out}/examples")


if __name__ == "__main__":
    main()
