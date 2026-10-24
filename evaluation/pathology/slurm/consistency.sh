#!/bin/bash -l
#
# Pathology consistency of one evaluation run: its cached generations and the matching ground
# truths, both against the study's report-derived pathology labels. CPU only.
#
#   sbatch evaluation/pathology/slurm/consistency.sh <eval out dir> [--save DIR]
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=pathology_consistency
#SBATCH --output=logs/pathology_consistency_%j.out

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
export PYTHONPATH=$PWD:$PYTHONPATH OMP_NUM_THREADS=1

RUN=$1
shift
CKPT=${PATHOLOGY_CKPT:-/hnvme/workspace/y100dc19-mrflow-final/pathology_classifier/runs/pathology_net3d/best.pt}

/hnvme/workspace/y100dc19-mrflow-final/venv/bin/python -m evaluation.pathology.consistency \
    --run "$RUN" --ckpt "$CKPT" --cases baselines/cases-test-n100.json \
    --workers "$SLURM_CPUS_PER_TASK" "$@"
