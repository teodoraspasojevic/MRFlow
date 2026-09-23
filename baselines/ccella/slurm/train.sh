#!/bin/bash -l
#SBATCH --job-name=ccella_train
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/hnvme/workspace/y100dc19-mrflow-final/baselines/ccella_runs/logs/%x-%j.out
#
#   sbatch baselines/ccella/slurm/train.sh baselines/ccella/configs/mrrate_ccella.yaml
#   sbatch baselines/ccella/slurm/train.sh <config> --resume          # continue the latest
#
# Everything after the config is passed to train.py.

set -euo pipefail

CONFIG="${1:?usage: train.sh <config.yaml> [extra args...]}"
shift

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
export OMP_NUM_THREADS=8
# A node with no outbound route otherwise hangs 90 s inside wandb.init; set offline and
# `wandb sync` the run directory afterwards.
export WANDB_MODE="${WANDB_MODE:-online}"

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
torchrun --standalone --nproc_per_node "$GPUS" \
    -m baselines.ccella.train --config "$CONFIG" "$@"
