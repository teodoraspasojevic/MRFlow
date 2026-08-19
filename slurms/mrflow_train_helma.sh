#!/bin/bash -l
#
# Fine-tune MRFlow from the released CTFlow weights. 4 GPUs per node, any number of nodes.
#
#   sbatch slurms/mrflow_train_helma.sh
#   sbatch --nodes=4 slurms/mrflow_train_helma.sh          # multi-node, no other change needed
#   sbatch slurms/mrflow_train_helma.sh lvfm/configs/mrflow_STDiT-L2_16f8.yaml
#
# The script re-execs itself under srun (one task per node) and derives accelerate's machine_rank
# from SLURM_NODEID, so single- and multi-node use the same path -- unlike the CT pair of scripts,
# which split this across mnode_launcher_helma.sh + trainer_helma.sh.
#
# The MR config carries real paths, so there is no envsubst step and no container -- unlike the CT
# scripts, this runs the workspace venv directly. Requeue-safe: `resume_from_checkpoint: latest`
# picks up the newest checkpoint, and W&B resumes the same run id.
#
#SBATCH --gres=gpu:h200:4
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

unset SLURM_EXPORT_ENV

# sbatch spools the script to /var/tmp, so $0 is not in the repo -- use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

CONFIG=${1:-lvfm/configs/mrflow_STDiT-L2_16f8.yaml}
VENV=/hnvme/workspace/y100dc19-mrflow/venv

export PYTHONPATH=$PWD:$PYTHONPATH
# No PYTHONNOUSERSITE: this venv layers on ~/.local, where diffusers/transformers/tqdm live.
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Compute nodes have no direct outbound route, so W&B needs the site proxy: verified from an h200
# node, `wandb.Api().viewer` authenticates through this against ~/.netrc. Without it wandb.init
# hangs 90 s and kills the run. To log without any network, set WANDB_MODE=offline here and
# `wandb sync <output_dir>/wandb/offline-run-*` afterwards.
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1"

# Created up front: every rank calls load_checkpoint, which lists output_dir.
GPUS_PER_NODE=4
NODES=${SLURM_JOB_NUM_NODES:-1}

# Rank 0 of the allocation prepares shared state, then every node runs one accelerate launcher.
if [ -z "$MRFLOW_UNDER_SRUN" ]; then
    OUTPUT_DIR=$($VENV/bin/python -c "
from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').output_dir)")
    mkdir -p "$OUTPUT_DIR"
    echo "Output dir: $OUTPUT_DIR | nodes: $NODES x $GPUS_PER_NODE GPUs"

    export MRFLOW_UNDER_SRUN=1
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    exec srun --ntasks-per-node=1 "$0" "$CONFIG"
fi

echo "Node $SLURM_NODEID/$NODES on $(hostname), master $MASTER_ADDR:$MASTER_PORT"

LAUNCH_ARGS="--num_processes $((GPUS_PER_NODE * NODES)) --num_machines $NODES --mixed_precision bf16"
if [ "$((GPUS_PER_NODE * NODES))" -gt 1 ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS --multi_gpu"
fi
if [ "$NODES" -gt 1 ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS --machine_rank $SLURM_NODEID \
        --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"
fi

$VENV/bin/python -m accelerate.commands.launch $LAUNCH_ARGS lvfm/train.py --config "$CONFIG"
