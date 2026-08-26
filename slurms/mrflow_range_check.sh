#!/bin/bash -l
#
# One-off: which input range should MR volumes be normalized to, [-1, 1] or [0, 1]?
# Reconstruction quality (VAE round-trip) and CTFlow prior transfer (zero-shot flow loss).
#
#   sbatch slurms/mrflow_range_check.sh
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=range_check
#SBATCH --output=logs/range_check_%j.out
#SBATCH --error=logs/range_check_%j.err

unset SLURM_EXPORT_ENV
# sbatch spools the script to /var/tmp, so $0 is not in the repo -- use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VENV=/hnvme/workspace/y100dc19-mrflow/venv
CONFIG=lvfm/configs/mrflow_STDiT-L2_16f8.yaml

echo "=== VAE reconstruction ==="
$VENV/bin/python tools/vae_range_check.py --config $CONFIG --n 20

echo "=== CTFlow prior transfer ==="
$VENV/bin/python tools/ctflow_transfer_check.py --config $CONFIG --n 20
