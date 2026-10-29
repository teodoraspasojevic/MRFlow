#!/bin/bash -l
#
# Rescore an existing eval or baseline run with --gt_bounds trim-v1 --trim_generated, writing
# metrics_mrflow_flux_trim-v1_trimgen.json beside the existing metrics.json.
#
# Usage (one job per run):
#
#   sbatch slurms/rescore_trimgen.sh \
#       <out_dir> <config> --split <split> [--label <label>] [--cases <cases_json>]
#
# <out_dir>  must contain generated.tar and shard-*.json (packed by pack_generated.py).
# All args after <config> are passed through to evaluation/main.py.
#
# Examples — see the submit block at the bottom of this script header for the table rows.
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --job-name=rescore_trimgen
#SBATCH --output=logs/rescore_%j.out
#SBATCH --error=logs/rescore_%j.err

set -euo pipefail

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

OUT=${1:?usage: rescore_trimgen.sh <out_dir> <config> [extra main.py flags]}
CONFIG=${2:?usage: rescore_trimgen.sh <out_dir> <config> [extra main.py flags]}
shift 2

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/venv
GT_BOUNDS=$WS/data/mrflow_flux_trim-v1

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1"

TARGET=$OUT/metrics_mrflow_flux_trim-v1_trimgen.json
if [ -f "$TARGET" ]; then
    echo "already done: $TARGET"
    exit 0
fi

# A checkout without the trim rescoring (3863cdb) would unpack the tar, then fail on the flags.
if ! grep -q -- '"--trim_generated"' evaluation/main.py; then
    echo "evaluation/main.py here has no --trim_generated; submit from a checkout with 3863cdb" >&2
    exit 1
fi

echo "=== unpack $OUT/generated.tar"
tar -xf "$OUT/generated.tar" -C "$OUT"

echo "=== rescore (trimgen)"
$VENV/bin/python evaluation/main.py \
    --config "$CONFIG" --ckpt none --combine \
    --out "$OUT" \
    --gt_bounds "$GT_BOUNDS" \
    --trim_generated \
    --no_wandb \
    "$@"

echo "=== repack"
rm -rf "$OUT/generated"

echo "done -> $TARGET"
