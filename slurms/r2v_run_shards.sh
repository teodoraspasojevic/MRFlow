#!/bin/bash -l
#
# Generate the frozen population with an R2V adapter, one array task per shard.
#
#   sbatch --array=0-15 slurms/r2v_run_shards.sh A
#   sbatch --array=0-15 slurms/r2v_run_shards.sh E
#
# Anything after the arm is passed through to generate.py (--limit, --seed, --overwrite, ...).
# R2V_CASES/R2V_PROMPTS select a different population and R2V_TAG the run directory, which is what
# `r2v_cfg_sweep.sh` uses to give every (arm, report cfg) cell its own volumes.
#
# Writes $OUT/nifti/<case_id>.nii.gz. Score them with:
#   slurms/baseline_score.sh nvidia_r2v_arm<X>_cfg<N> "<label>"
#
# **No separate venv.** Measured 2026-09-16: the whole `mrrate_r2v` stack -- monai 1.6.0,
# transformers 5.14.1, its MAISI UNet and the three text encoders -- imports cleanly in the MRFlow
# workspace venv, so this runs there. Only the upstream clone is added to PYTHONPATH, through a
# `mrrate_r2v` symlink because the clone's directory name carries a dash and is not importable.
#
# Cost: ~1.6 s/case generation plus a sliding-window decode, so 1,010 cases over 16 tasks is well
# under an hour per task. The fixed cost is model loading, not the case count.
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=r2v_generate
#SBATCH --output=logs/r2v_gen_%A_%a.out
#SBATCH --error=logs/r2v_gen_%A_%a.err

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

ARM=${1:?usage: r2v_run_shards.sh <A|E> [extra generate.py flags]}
shift

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/venv
UPSTREAM=$WS/baselines/upstream
NVIDIA_WS=${R2V_WORKSPACE:-/hnvme/workspace/y100dc19-nvidia-mri-brain}

# The population defaults to the committed test one; a sweep points both at the val pair instead.
CASES=${R2V_CASES:-baselines/cases-test-n100.json}
PROMPTS=${R2V_PROMPTS:-$WS/baselines/prompts-test-n100.json}

case "$ARM" in
    A) TAG=nvidia_r2v_armA_cfg7 ;;
    E) TAG=nvidia_r2v_armE_cfg3 ;;
    *) echo "unknown arm '$ARM' (expected A or E)" >&2; exit 1 ;;
esac
# R2V_TAG redirects the whole run elsewhere, so a smoke test cannot leave volumes in the real run
# directory -- where generate.py's resume would silently adopt them.
TAG=${R2V_TAG:-$TAG}
OUT=$WS/baselines/runs/$TAG

# The clone is imported as `mrrate_r2v`; the symlink exists because `R2V-MR-Generation` has a dash.
# Created here rather than assumed, so a fresh workspace works without a manual step.
ln -sfn "$UPSTREAM/R2V-MR-Generation" "$UPSTREAM/mrrate_r2v"

export PYTHONPATH=$PWD:$UPSTREAM:$PYTHONPATH
export PYTHONFAULTHANDLER=1
# Defaulted, not bare: outside sbatch SLURM_CPUS_PER_TASK is unset, and an empty OMP_NUM_THREADS
# makes OpenMP abort inside `import torch` ("Error #101: Out of heap memory") rather than warn.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
# The text encoders are staged snapshots; the encoder zoo resolves them from here. Offline so a
# node with no outbound route fails loudly instead of hanging on a Hub call.
export MRRATE_PRETRAINED_DIR=${MRRATE_PRETRAINED_DIR:-$NVIDIA_WS/pretrained}
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

# Fail before the queue wastes a GPU on a missing checkpoint. The NVIDIA workspace expires
# 2026-10-13; set R2V_WORKSPACE once its contents are staged elsewhere.
for p in "$PROMPTS" "$NVIDIA_WS/models/diff_unet_3d_rflow-mr-brain_v0.pt" \
         "$NVIDIA_WS/models/autoencoder_v1.pt" "$MRRATE_PRETRAINED_DIR"; do
    [[ -e "$p" ]] || { echo "MISSING: $p" >&2; exit 1; }
done

echo "arm $ARM | shard $SHARD/$NUM_SHARDS | out $OUT/nifti"
$VENV/bin/python -m baselines.nvidia_r2v.generate \
    --arm "$ARM" --cases "$CASES" --prompts "$PROMPTS" --out "$OUT/nifti" \
    --shard "$SHARD" --num_shards "$NUM_SHARDS" "$@"
