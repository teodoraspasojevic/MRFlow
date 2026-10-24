#!/bin/bash -l
#
# Cache the pathology classifier's inputs: one zip + status CSV per array task. CPU only.
# Build the split manifest first (login node, ~1 min):
#
#   python -m evaluation.pathology.data splits
#   sbatch --array=0-31 evaluation/pathology/slurm/prepare.sh
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=pathology_prep
#SBATCH --output=logs/pathology_prep_%A_%a.out

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
export PYTHONPATH=$PWD:$PYTHONPATH OMP_NUM_THREADS=1

/hnvme/workspace/y100dc19-mrflow-final/venv/bin/python -m evaluation.pathology.data prepare \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --workers "$SLURM_CPUS_PER_TASK" "$@"
