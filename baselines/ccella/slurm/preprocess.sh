#!/bin/bash -l
#SBATCH --job-name=ccella_prep
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/hnvme/workspace/y100dc19-mrflow-final/baselines/ccella_runs/logs/%x-%A_%a.out
#
# MR-RATE -> the CCELLA zip cache. One array task per shard.
#
#   NUM_SHARDS=4  sbatch --array=0-3  baselines/ccella/slurm/preprocess.sh <config> val
#   NUM_SHARDS=64 sbatch --array=0-63 baselines/ccella/slurm/preprocess.sh <config> train
#
# To redo a subset, keep NUM_SHARDS identical and list only the shard ids:
#   NUM_SHARDS=64 sbatch --array=0-11,15-17 baselines/ccella/slurm/preprocess.sh <config> train
#
# Let task 0 finish before training: it writes cache_meta.json, and train.py refuses a cache
# without one. Anything after the split is passed through to prepare_data.py, so --limit and
# --overwrite work as usual.
#
# This account has h200 but no h100 allocation; `-p h100` is rejected at submit and
# `-p preempt --gres=gpu:h100:1` pends forever on AssocGrpGRES. Always request h200.

set -euo pipefail

CONFIG="${1:?usage: preprocess.sh <config.yaml> <split> [extra args...]}"
SPLIT="${2:?usage: preprocess.sh <config.yaml> <split> [extra args...]}"
shift 2

# sbatch runs a spooled copy of this script, so BASH_SOURCE does not point into the repo.
# SLURM_SUBMIT_DIR is where `sbatch` was invoked, which the documented usage says is the repo root.
REPO="${MRFLOW_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
if [ ! -d "$REPO/baselines/ccella" ]; then
    echo "cannot locate the MRFlow repo (tried '$REPO'). Submit from the repo root, or set MRFLOW_REPO." >&2
    exit 2
fi
source /hnvme/workspace/y100dc19-mrflow-final/venv/bin/activate
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# NUM_SHARDS is a CONSTANT of the cache, never SLURM_ARRAY_TASK_COUNT. `shard_slice` partitions the
# study list into exactly this many pieces, so a recovery array that re-runs a subset of shards must
# use the SAME value -- deriving it from the array's own size silently re-partitions the dataset and
# produces overlapping shards. (That happened once: a 34-element recovery array rewrote shard 12 as a
# 1/34 slice, duplicating 13,621 series against shards 22 and 23.)
python -m baselines.ccella.prepare_data \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --num_shards "${NUM_SHARDS:-64}" \
    "$@"
