#!/bin/bash -l
#
# Cache the modality classifier's inputs: one zip + status CSV per array task. CPU only.
# Build the split manifest first (login node, ~1 min):
#
#   python -m evaluation.modality.data splits
#   sbatch --array=0-7 evaluation/modality/slurm/prepare.sh
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=modality_prep
#SBATCH --output=logs/modality_prep_%A_%a.out

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
export PYTHONPATH=$PWD:$PYTHONPATH OMP_NUM_THREADS=1

/hnvme/workspace/y100dc19-mrflow-final/venv/bin/python -m evaluation.modality.data prepare \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --workers "$SLURM_CPUS_PER_TASK" "$@"
