#!/bin/bash -l
#
# Train CCELLA on MR-RATE. 4 GPUs per node, any number of nodes.
#
#   sbatch baselines/ccella/slurm/train.sh baselines/ccella/configs/mrrate_ccella.yaml
#   sbatch baselines/ccella/slurm/train.sh <config> --resume         # continue the latest
#   sbatch --nodes=4 baselines/ccella/slurm/train.sh <config>        # more nodes, no other change
#
# **h200 nodes have 4 GPUs, not 8** (`gpu:h200:4`, 128 CPUs), so the default here is 2 nodes: 8
# GPUs at `micro_batch_size: 16` is the effective batch of 128 the config is written for. Asking
# for `--gres=gpu:h200:8 --nodes=1` is rejected at submit with "Requested node configuration is
# not available".
#
# Mirrors slurms/mrflow_train_helma.sh: the script re-execs itself under srun (one task per node)
# and derives torchrun's --node_rank from SLURM_NODEID, so single- and multi-node use one path.
#
#SBATCH --job-name=ccella_train
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/hnvme/workspace/y100dc19-mrflow-final/baselines/ccella_runs/logs/%x-%j.out

set -euo pipefail
unset SLURM_EXPORT_ENV

# sbatch spools the script to /var/tmp, so $0 is not in the repo -- use the submit dir.
REPO="${MRFLOW_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}}"
if [ ! -d "$REPO/baselines/ccella" ]; then
    echo "cannot locate the MRFlow repo (tried '$REPO'). Submit from the repo root, or set MRFLOW_REPO." >&2
    exit 2
fi
cd "$REPO"

CONFIG="${1:?usage: train.sh <config.yaml> [extra args...]}"
shift

source /hnvme/workspace/y100dc19-mrflow-final/venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# Compute nodes have no direct outbound route, so W&B needs the site proxy -- without it wandb.init
# hangs 90 s and kills the run. To log with no network at all, set WANDB_MODE=offline and
# `wandb sync <output_dir>/wandb/offline-run-*` afterwards.
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

GPUS_PER_NODE=4
NODES=${SLURM_JOB_NUM_NODES:-1}

if [ -z "${CCELLA_UNDER_SRUN:-}" ]; then
    # Gate, once, before anything is allocated: refuse to train on an incomplete or overlapping
    # cache. Nothing else notices -- the dataset globs whatever manifests exist, so 58 of 64 shards
    # would train quietly on 90% of the data. SKIP_VERIFY=1 only if you know what is missing.
    if [ -z "${SKIP_VERIFY:-}" ]; then
        python -m baselines.ccella.verify_cache --config "$CONFIG" --split train \
            --num_shards "${NUM_SHARDS:-64}" --skip_members
        python -m baselines.ccella.verify_cache --config "$CONFIG" --split val \
            --num_shards "${VAL_NUM_SHARDS:-4}" --skip_members
    fi
    echo "nodes: $NODES x $GPUS_PER_NODE GPUs -> world size $((GPUS_PER_NODE * NODES))"

    export CCELLA_UNDER_SRUN=1
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    # Re-exec by repo path, not "$0": the spooled copy is node-local, so on a second node srun
    # would fail with execve(): No such file. Through `bash -l` so a missing exec bit cannot kill
    # the job two seconds in.
    exec srun --ntasks-per-node=1 bash -l "$REPO/baselines/ccella/slurm/train.sh" "$CONFIG" "$@"
fi

echo "node ${SLURM_NODEID:-0}/$NODES on $(hostname), master $MASTER_ADDR:$MASTER_PORT"
# `torchrun` is NOT on PATH: this venv layers on a conda env that owns torch, and only python*
# lives in venv/bin. `python -m torch.distributed.run` is the same entry point and always resolves
# -- the same reason slurms/mrflow_train_helma.sh calls `python -m accelerate.commands.launch`.
python -m torch.distributed.run \
    --nnodes "$NODES" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --node_rank "${SLURM_NODEID:-0}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    -m baselines.ccella.train --config "$CONFIG" "$@"
