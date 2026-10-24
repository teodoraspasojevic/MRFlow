#!/bin/bash -l
#
# Modality consistency of one evaluation run: its cached generations against their conditioning
# modality, and the matching ground truths against their labels. CPU only.
#
#   sbatch evaluation/modality/slurm/consistency.sh <eval out dir> [--save DIR]
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=modality_consistency
#SBATCH --output=logs/modality_consistency_%j.out

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
export PYTHONPATH=$PWD:$PYTHONPATH OMP_NUM_THREADS=1

RUN=$1
shift
CKPT=${MODALITY_CKPT:-/hnvme/workspace/y100dc19-mrflow-final/modality_classifier/runs/modality_net3d/best.pt}

/hnvme/workspace/y100dc19-mrflow-final/venv/bin/python -m evaluation.modality.consistency \
    --run "$RUN" --ckpt "$CKPT" --cases baselines/cases-test-n100.json \
    --workers "$SLURM_CPUS_PER_TASK" "$@"
