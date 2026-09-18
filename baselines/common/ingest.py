"""A baseline's NIfTI output -> the cache `evaluation/main.py --combine` scores.

    python -m baselines.common.ingest --cases baselines/cases-test-n100.json \
        --nifti $WS/baselines/runs/text2ct/nifti --out $WS/baselines/runs/text2ct \
        --label text2ct@887caa9

Writes `<out>/generated/<bucket>-<case_id>.npy` and `<out>/shard-NNNN.json`, which is the entire
contract `score_cached` reads -- it needs no config, no checkpoint and no model, only those two
things plus the raw archives the manifest points at. So once this has run:

    sbatch slurms/mrflow_eval_helma.sh <any mrflow config> none --split test --combine --out <out>

scores the baseline with the identical extractors, geometry and ground truth MRFlow is scored with.

**NIfTI is the interchange format, and that is a deliberate choice.** Every one of the three
baselines already writes it, and its affine carries spacing and orientation -- so this reads a
baseline's output with `read_canonical`, the *same* function the ground-truth path uses, instead of
each adapter restating what its model's axes mean. `read_canonical`/`plane_order` are already
shared between `preprocess_volume` and the evaluation for exactly this reason; a baseline is one
more consumer rather than a fourth copy. It also means the adapter side needs no MRFlow import at
all: it writes NIfTI in its own venv and stops there.

The intermediate NIfTIs are disposable -- `generated/*.npy` is what the evaluation reads from here
on, and it is ~21 MB per case. Delete the NIfTI directory once a run's manifest is written.
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from baselines.common.canonicalize import canonicalize_external
from baselines.common.cases import load_cases
from echosyn.common.mrrate import read_canonical
from evaluation import EvalAccumulator
from evaluation.paper_metrics import normalize01


def find_volume(nifti_dir, case_id):
    """The baseline's file for a case, or None. `<case_id>.nii.gz` is the contract; `.nii` is
    accepted because some upstream savers write uncompressed by default."""
    for suffix in (".nii.gz", ".nii"):
        path = os.path.join(nifti_dir, case_id + suffix)
        if os.path.exists(path):
            return path
    return None


def ingest(cases, nifti_dir, out, posterior_shift_mm=0.0):
    """Canonicalize every case's NIfTI into the `.npy` cache. Returns the manifest rows."""
    generated_dir = os.path.join(out, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    rows = []

    for case in tqdm(cases, disable=None):
        row = {k: case[k] for k in ("case_id", "bucket", "modality", "plane", "study_uid",
                                    "conditioning_uid", "archive", "member")}
        rows.append(row)

        # Out of scope before anything is read: `add_missing` counts these as excluded rather than
        # missing, and no baseline is penalized for a modality the evaluation never scores.
        if not EvalAccumulator.is_scored(case["modality"]):
            row["status"] = "excluded"
            continue

        path = find_volume(nifti_dir, case["case_id"])
        if path is None:
            # The same penalty the platform applies, and the same one MRFlow takes for a collapsed
            # rollout: no volume, no contribution to any distribution, counted in n_missing_outputs.
            row["status"] = "missing"
            continue

        with open(path, "rb") as handle:
            volume, spacing = read_canonical(handle.read())
        volume = canonicalize_external(volume, spacing, case["plane"], posterior_shift_mm)

        # **Windowed here because the cache is fp16 and raw intensities do not fit in it.**
        # Measured on six MR-RATE test volumes: three exceed float16's 65504 ceiling (worst
        # 78,083), and an `inf` in the cache would then poison `normalize01` -- and with it every
        # metric -- at scoring time. MRFlow's own `cache_generated` writes fp16 safely only because
        # `decode_latent` already emits ~[0, 1].
        #
        # This costs nothing and changes no number: `normalize01` is what `canonicalize_generated`
        # applies to every cached volume anyway, and it is idempotent -- after the first window at
        # least 0.5% of voxels sit at exactly 0 and 0.5% at exactly 1, so the second call's
        # percentiles are 0 and 1 and it is the identity. Measured across those six volumes,
        # applying it twice with the fp16 cast in between differs by 2.44e-4, which is the fp16
        # quantization step at 1.0 and not an algorithmic difference; MRFlow's cache carries the
        # identical rounding.
        volume = normalize01(volume)

        name = f"{case['bucket']}-{case['case_id']}.npy"
        np.save(os.path.join(generated_dir, name), volume.astype(np.float16))
        row["status"] = "generated"
        row["generated"] = name

    return rows


def main():
    parser = argparse.ArgumentParser(description="Ingest a baseline's NIfTI output into the "
                                                 "evaluation cache.")
    parser.add_argument("--cases", required=True, help="The frozen population (baselines/cases-*.json).")
    parser.add_argument("--nifti", required=True, help="Directory of <case_id>.nii.gz.")
    parser.add_argument("--out", required=True, help="Results dir; --combine is pointed here.")
    parser.add_argument("--label", required=True,
                        help="What produced these volumes, e.g. 'nvidia_r2v armA cfg7 @ad5dca1'. "
                             "Recorded in the manifest so a results dir says what it holds.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--posterior_shift_mm", type=float, default=0.0,
                        help="Leave at 0 unless this baseline's framing demands otherwise -- see "
                             "canonicalize_external's docstring before changing it.")
    args = parser.parse_args()

    cases = load_cases(args.cases)[args.shard::args.num_shards]
    os.makedirs(args.out, exist_ok=True)
    rows = ingest(cases, args.nifti, args.out, args.posterior_shift_mm)

    manifest = os.path.join(args.out, f"shard-{args.shard:04d}.json")
    with open(manifest, "w") as handle:
        json.dump({"shard": args.shard, "num_shards": args.num_shards,
                   "generator": args.label, "cases": rows}, handle, indent=2)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"{manifest}: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if counts.get("missing"):
        print(f"WARNING: {counts['missing']} cases have no volume -- they score as missing outputs")


if __name__ == "__main__":
    main()
