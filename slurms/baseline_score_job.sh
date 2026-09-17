#!/bin/bash -l
#
# Ingest one baseline run's NIfTI output and score it. The batch half of `baseline_score.sh`, and a
# `--dependency` target for a sweep that submits generation and scoring in one go.
#
#   sbatch slurms/baseline_score_job.sh <run-tag> "<label>" [extra main.py flags]
#
# Three steps in one job, because the middle one is what makes it a job:
#
#   1. ingest   <run>/nifti/*.nii.gz -> <run>/generated/*.npy + shard-0000.json
#   2. rm       the NIfTIs, ~24 MB and one inode per case, disposable once the manifest exists
#   3. combine  every metric over those volumes -> <run>/metrics.json
#
# Ingest reads 1,010 volumes at ~24 MB and writes as many again, which is why it is here rather
# than on the login node. It needs no GPU but shares the one this job already holds.
#
# `ingest.py` writes no shard-*.pt, so --combine always takes `score_cached`: one sequential pass
# over every case on one GPU. Measured budget is hours, not minutes -- hence the 6 h limit.
#
# SPLIT/CASES must match the population the run was generated over. `load_frozen_cases` refuses a
# mismatch, so a val population under --split test fails loudly instead of scoring the wrong set.
#
#SBATCH --gres=gpu:h200:1
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=r2v_score
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err

set -euo pipefail

unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

TAG=${1:?usage: baseline_score_job.sh <run-tag> "<label>" [extra main.py flags]}
LABEL=${2:?usage: baseline_score_job.sh <run-tag> "<label>" [extra main.py flags]}
shift 2

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/venv
OUT=$WS/baselines/runs/$TAG
SPLIT=${SPLIT:-test}
CASES=${CASES:-baselines/cases-$SPLIT-n100.json}
CONFIG=${CONFIG:-lvfm/configs/mrflow_STDiT-L2_16f8.yaml}

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false
# Compute nodes have no direct outbound route; wandb.init hangs 90 s and dies without the proxy.
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1"

echo "=== 1/3  ingest $TAG  ($SPLIT, $CASES)"
$VENV/bin/python -m baselines.common.ingest \
    --cases "$CASES" --nifti "$OUT/nifti" --out "$OUT" --label "$LABEL"

# **Refuse an incomplete cell, before the NIfTIs are gone.** Generation runs on `preempt`, where a
# cancelled array task is not requeued, and the score job fires on `afterany` regardless. Scoring
# anyway would write a metrics.json over a short population -- which the sweep's resume then skips,
# so the cell is quietly measured on fewer cases than its neighbours. Stopping here leaves the
# NIfTIs in place, so re-running r2v_cfg_sweep.sh regenerates only the gaps. ALLOW_MISSING accepts a
# cell whose shortfall is real generation failures rather than a lost task.
missing=$($VENV/bin/python -c "import json,sys; print(sum(c['status']=='missing' for c in json.load(open(sys.argv[1]))['cases']))" "$OUT/shard-0000.json")
if [ "$missing" -gt "${ALLOW_MISSING:-0}" ]; then
    echo "$missing cases have no volume -- not scoring. Re-run the generation array (it resumes" >&2
    echo "from the NIfTIs already here), or set ALLOW_MISSING=$missing to score as is." >&2
    exit 1
fi

echo "=== 2/3  drop the NIfTIs"
# `_generate-*.json` is the generation side's provenance -- the adapter, seed and both guidance
# scales -- and it lives in the NIfTI directory. Keep it; the 24 MB volumes are what goes.
mv -f "$OUT"/nifti/_generate-*.json "$OUT"/ 2>/dev/null || true
rm -rf "$OUT/nifti"

echo "=== 3/3  score"
# --ckpt is required positionally and is meaningless for a baseline; --label names the run.
$VENV/bin/python evaluation/main.py \
    --config "$CONFIG" --ckpt none --split "$SPLIT" --combine \
    --out "$OUT" --cases "$CASES" --label "$LABEL" "$@"

echo "done -> $OUT/metrics.json"
