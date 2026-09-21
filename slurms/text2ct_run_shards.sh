#!/bin/bash -l
#
# Generate the frozen population with a fine-tuned Text2CT checkpoint, one array task per shard.
#
#   sbatch --array=0-15 slurms/text2ct_run_shards.sh 60000
#   sbatch --array=0-15 slurms/text2ct_run_shards.sh 60000 --guidance_scale 1.0
#
# The first argument is the checkpoint STEP; anything after it is passed through to generate.py.
# T2CT_CASES/T2CT_PROMPTS select a different population (default: the val pair, since checkpoint and
# guidance selection must not happen on test) and T2CT_TAG redirects the run directory.
#
# Writes $OUT/nifti/<case_id>.nii.gz. Score them with:
#   SPLIT=val slurms/baseline_score.sh text2ct_val_step<N> "<label>"
#
# **Its own venv.** Unlike nvidia_r2v, this cannot use the MRFlow workspace venv: Text2CT's vendored
# CLIP needs `transformers` 4.x and the MRFlow venv ships 5.x, where it does not import. See
# baselines/text2ct/README.md for how $WS/baselines/venv-text2ct was built.
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=t2ct_generate
#SBATCH --output=logs/t2ct_gen_%A_%a.out
#SBATCH --error=logs/t2ct_gen_%A_%a.err

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

STEP=${1:?usage: text2ct_run_shards.sh <checkpoint-step> [extra generate.py flags]}
shift

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/baselines/venv-text2ct
RUN=${T2CT_RUN:-$WS/baselines/runs/text2ct_mrrate_ft}
CKPT=$(printf "%s/checkpoint-%07d.pt" "$RUN" "$STEP")

# Checkpoint and guidance are chosen on val, never on the split the paper reports -- the same rule
# `baselines/nvidia_r2v` follows for its cfg sweep.
CASES=${T2CT_CASES:-baselines/cases-val-n100.json}
PROMPTS=${T2CT_PROMPTS:-$WS/baselines/prompts-val-n100.json}
TAG=${T2CT_TAG:-text2ct_val_step$STEP}
OUT=$WS/baselines/runs/$TAG

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
# The CLIP tokenizer/processor snapshot comes from the HF cache; the site proxy is what compute
# nodes need to reach it the first time.
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1"

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

# Fail before the queue wastes a GPU on a missing input.
for p in "$CKPT" "$CASES" "$PROMPTS" "$VENV/bin/python"; do
    [[ -e "$p" ]] || { echo "MISSING: $p" >&2; exit 1; }
done

echo "step $STEP | shard $SHARD/$NUM_SHARDS | out $OUT/nifti"
$VENV/bin/python -m baselines.text2ct.generate \
    --checkpoint "$CKPT" --cases "$CASES" --prompts "$PROMPTS" --out "$OUT/nifti" \
    --shard "$SHARD" --num_shards "$NUM_SHARDS" "$@"
