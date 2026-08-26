#!/bin/bash -l
#
# Encode MR-RATE into latent blocks. Array job: each task takes an interleaved slice of the series
# list and writes its own manifest CSV, so tasks never coordinate and a failed task is re-runnable
# on its own (already-encoded samples are skipped unless --overwrite).
#
#   sbatch --array=0-63 slurms/mrflow_preprocess_helma.sh train --zip
#   sbatch --array=0-3  slurms/mrflow_preprocess_helma.sh val   --zip
#
# Task 0 also writes the two boundary latents, so let it finish before training starts.
#
# Anything after the split is passed through to the script. `--zip` bundles each shard into one
# file, which this account needs: 244k loose .pt files is well past its 102k inode quota.
#
# When the array finishes, check it before training -- a dead task is otherwise silent:
#   python tools/mrflow_verify.py --config <config> --split train --num_shards 64
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err

unset SLURM_EXPORT_ENV

SPLIT=${1:-train}
shift 2>/dev/null  # remaining args (e.g. --zip) go straight to the script
# Sharding is series[SHARD::NUM_SHARDS], so re-running a subset must keep the ORIGINAL divisor:
# `--array=3,7` would otherwise set it to 2 and silently re-shard the whole split. Override it:
#   sbatch --export=ALL,MRFLOW_NUM_SHARDS=16 --array=0,1,8-11 ... test --zip
NUM_SHARDS=${MRFLOW_NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

# sbatch spools the script to /var/tmp, so $0 is not in the repo -- use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH
# No PYTHONNOUSERSITE: this venv layers on ~/.local, where diffusers/transformers/tqdm live.
# The volume path is NIfTI decode + trilinear resample, which is CPU-bound and would otherwise
# oversubscribe the node.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VENV=/hnvme/workspace/y100dc19-mrflow/venv
CONFIG=lvfm/configs/mrflow_STDiT-L2_16f8.yaml

echo "[$(date)] shard $SHARD/$NUM_SHARDS split=$SPLIT"
$VENV/bin/python lvfm/preprocess_mrrate.py \
    --config $CONFIG --split "$SPLIT" \
    --shard "$SHARD" --num_shards "$NUM_SHARDS" "$@"
echo "[$(date)] done"
