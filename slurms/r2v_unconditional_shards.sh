#!/bin/bash -l
#
# Generate the frozen population with the **released** NV-Generate-MR-Brain: no adapter, no report,
# no cross-attention. The conditioning floor, one array task per shard.
#
#   sbatch --array=0-15 slurms/r2v_unconditional_shards.sh
#   R2V_CASES=baselines/cases-val-n100.json sbatch --array=0-15 slurms/r2v_unconditional_shards.sh
#
# Anything on the command line is passed through to generate_unconditional.py (--limit, --seed, ...).
# Score it exactly like any other baseline:
#   slurms/baseline_score.sh nvidia_stock "nvidia stock modality-only @ad5dca1"
#
# Read it on the HLIP report-alignment columns, where a report-blind generator should fall to
# chance. Its FID/FVD are **not** comparable to the conditioned rows -- an unconditional model is
# not penalised for ignoring the report, so a good FID here means nothing about conditioning.
#
# Cheaper than the conditioned arms: no text encoders to load and nothing to encode per case.
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=r2v_uncond
#SBATCH --output=logs/r2v_uncond_%A_%a.out
#SBATCH --error=logs/r2v_uncond_%A_%a.err

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/venv
UPSTREAM=$WS/baselines/upstream
NVIDIA_WS=${R2V_WORKSPACE:-/hnvme/workspace/y100dc19-nvidia-mri-brain}

CASES=${R2V_CASES:-baselines/cases-test-n100.json}
TAG=${R2V_TAG:-nvidia_stock}
OUT=$WS/baselines/runs/$TAG

ln -sfn "$UPSTREAM/R2V-MR-Generation" "$UPSTREAM/mrrate_r2v"

export PYTHONPATH=$PWD:$UPSTREAM:$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

# Fail before the queue wastes a GPU. The NVIDIA workspace expires 2026-10-13.
for p in "$NVIDIA_WS/models/diff_unet_3d_rflow-mr-brain_v0.pt" "$NVIDIA_WS/models/autoencoder_v1.pt"; do
    [[ -e "$p" ]] || { echo "MISSING: $p" >&2; exit 1; }
done

echo "stock (report-blind) | shard $SHARD/$NUM_SHARDS | cases $CASES | out $OUT/nifti"
$VENV/bin/python -m baselines.nvidia_r2v.generate_unconditional \
    --cases "$CASES" --out "$OUT/nifti" \
    --base_checkpoint "$NVIDIA_WS/models/diff_unet_3d_rflow-mr-brain_v0.pt" \
    --vae_checkpoint "$NVIDIA_WS/models/autoencoder_v1.pt" \
    --shard "$SHARD" --num_shards "$NUM_SHARDS" "$@"
