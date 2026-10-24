#!/bin/bash -l
#
# Train the modality classifier, then score it on the held-out test split. Resumes from
# <output_dir>/last.pt, so a preempted job is simply resubmitted.
#
#   sbatch evaluation/modality/slurm/train.sh [dotlist overrides, e.g. train.epochs=2]
#
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=modality_train
#SBATCH --output=logs/modality_train_%j.out

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
export PYTHONPATH=$PWD:$PYTHONPATH
PY=/hnvme/workspace/y100dc19-mrflow-final/venv/bin/python
CONFIG=evaluation/modality/config.yaml

$PY -m evaluation.modality.train --config $CONFIG "$@" || exit 1
OUT=$($PY -c "from omegaconf import OmegaConf as O; import sys; \
c = O.merge(O.load('$CONFIG'), O.from_dotlist(sys.argv[1:])); print(c.output_dir)" "$@")
$PY -m evaluation.modality.evaluate --ckpt "$OUT/best.pt" --split val
$PY -m evaluation.modality.evaluate --ckpt "$OUT/best.pt" --split test
