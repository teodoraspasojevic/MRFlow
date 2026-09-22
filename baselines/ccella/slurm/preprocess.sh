#!/bin/bash -l
#SBATCH --job-name=ccella_prep
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#
# MR-RATE -> the CCELLA zip cache. One array task per shard.
#
#   sbatch --array=0-3  baselines/ccella/slurm/preprocess.sh <config> val
#   sbatch --array=0-63 baselines/ccella/slurm/preprocess.sh <config> train
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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source /hnvme/workspace/y100dc19-mrflow-final/venv/bin/activate
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -m baselines.ccella.prepare_data \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    "$@"
