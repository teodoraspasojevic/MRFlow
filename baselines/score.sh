#!/bin/bash -l
#
# Ingest a baseline's NIfTI output and score it. Identical for every baseline -- that is the point
# of the contract in baselines/README.md.
#
#   baselines/score.sh nvidia_r2v_armA_cfg7 "nvidia_r2v armA cfg7 @ad5dca1"
#
# Two steps, and the second is a SLURM job because it needs a GPU:
#
#   1. ingest   <run>/nifti/*.nii.gz -> <run>/generated/*.npy + shard-0000.json   (login node, CPU)
#   2. combine  every metric over those volumes -> <run>/metrics.json             (sbatch, 1 GPU)
#
# The combine pass reads the ground truth fresh from the MR-RATE archives and canonicalizes it in
# memory. Nothing about a baseline's run touches the reference, which is what makes every row of
# the table share one.
#
# `ingest.py` writes no shard-*.pt, so --combine always takes `score_cached`: one sequential pass
# over all 1,010 cases on one GPU. Budget hours, not minutes -- unlike an MRFlow --combine, whose
# array tasks scored as they generated.

set -euo pipefail

TAG=${1:?usage: score.sh <run-tag> "<label>" [extra main.py flags]}
LABEL=${2:?usage: score.sh <run-tag> "<label>" [extra main.py flags]}
shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/venv
OUT=$WS/baselines/runs/$TAG
CASES=baselines/cases-test-n100.json
CONFIG=lvfm/configs/mrflow_STDiT-L2_16f8.yaml

[[ -d "$OUT/nifti" ]] || { echo "no $OUT/nifti -- run the generation array first" >&2; exit 1; }
echo "=== 1/2  ingest $(ls "$OUT/nifti" | grep -c '\.nii' || true) volumes"
PYTHONPATH=$REPO $VENV/bin/python -m baselines.common.ingest \
    --cases "$CASES" --nifti "$OUT/nifti" --out "$OUT" --label "$LABEL"

echo "=== 2/2  submit scoring"
# --ckpt is required positionally by the eval script and is meaningless for a baseline; --label is
# what names the run and lands in metrics.json.
sbatch --job-name="score_$TAG" --time=04:30:00 \
    slurms/mrflow_eval_helma.sh "$CONFIG" none \
    --split test --combine --out "$OUT" --cases "$CASES" --label "$LABEL" "$@"

echo
echo "When it lands: $OUT/metrics.json"
echo "The NIfTIs are disposable once shard-0000.json exists:  rm -rf $OUT/nifti"
