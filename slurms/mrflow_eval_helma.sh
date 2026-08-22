#!/bin/bash -l
#
# Roll out a checkpoint over an MR-RATE split and score it with the official VLM3D metrics.
# One GPU per array task; each task writes its own shard-NNNN.pt, then one --combine pass pools
# them. See evaluation/README.md.
#
#   # ~1000 val cases, CT checkpoint with no fine-tuning at all (the zero-shot baseline row)
#   sbatch --array=0-31 slurms/mrflow_eval_helma.sh \
#       lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
#       /hnvme/workspace/y100dc19-mrflow/models/ctflow/checkpoint-680000/denoiser_ema \
#       --split val --limit 32 --out /hnvme/workspace/y100dc19-mrflow/eval/ctflow_zeroshot
#   sbatch slurms/mrflow_eval_helma.sh <same config> <same ckpt> \
#       --split val --combine --out /hnvme/workspace/y100dc19-mrflow/eval/ctflow_zeroshot
#
# --limit is PER SHARD, so cases = array size * limit. --shard/--num_shards are derived from the
# array here; everything after the ckpt is passed through to evaluation/main.py.
#
# Pass --out explicitly when the checkpoint is not from a finished experiment: the default is
# <config output_dir>/eval/..., which for an un-run config would create the experiment directory.
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=mrflow_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

unset SLURM_EXPORT_ENV

# sbatch spools the script to /var/tmp, so $0 is not in the repo -- use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

CONFIG=$1
CKPT=$2
shift 2

VENV=/hnvme/workspace/y100dc19-mrflow/venv

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# Compute nodes have no direct outbound route; wandb.init hangs 90 s and dies without the proxy.
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1"

# A --combine pass is submitted without an array, so default to the single-shard case.
SHARD=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

echo "shard $SHARD/$NUM_SHARDS | ckpt $CKPT"

$VENV/bin/python evaluation/main.py \
    --config "$CONFIG" --ckpt "$CKPT" \
    --shard "$SHARD" --num_shards "$NUM_SHARDS" "$@"
