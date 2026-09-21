#!/bin/bash -l
#
# Score every retained Text2CT checkpoint on the val population, one cell per checkpoint.
#
#   bash slurms/text2ct_ckpt_sweep.sh                 # all checkpoints in the run directory
#   bash slurms/text2ct_ckpt_sweep.sh 58000 60000     # just these steps
#   DRY_RUN=1 bash slurms/text2ct_ckpt_sweep.sh       # print the sbatch lines, submit nothing
#
# Each cell submits a 16-task generation array on `preempt` plus a dependent ingest-and-score pass
# on `h200`, writing $WS/baselines/runs/text2ct_val_step<N>/metrics.json. Cells that already have a
# metrics.json are skipped, so re-running resumes a partial sweep -- the same shape as
# `r2v_cfg_sweep.sh`.
#
# **Selection happens on val.** Checkpoint and guidance are hyper-parameters like any other, and
# choosing them on the split the paper reports is the mistake this repo moved MRFlow's own cfg sweep
# off test to avoid. Score the winner once on test with `slurms/baseline_score.sh`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WS=/hnvme/workspace/y100dc19-mrflow-final
RUN=${T2CT_RUN:-$WS/baselines/runs/text2ct_mrrate_ft}
SHARDS=${SHARDS:-16}
CASES=${CASES:-baselines/cases-val-n100.json}
PROMPTS=${PROMPTS:-$WS/baselines/prompts-val-n100.json}

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    mapfile -t STEPS < <(ls "$RUN"/checkpoint-*.pt 2>/dev/null \
        | sed -E 's/.*checkpoint-0*([0-9]+)\.pt/\1/' | sort -n)
fi
[ ${#STEPS[@]} -gt 0 ] || { echo "no checkpoints in $RUN" >&2; exit 1; }
echo "checkpoints: ${STEPS[*]}"

for STEP in "${STEPS[@]}"; do
    TAG=text2ct_val_step$STEP
    OUT=$WS/baselines/runs/$TAG
    LABEL="text2ct mrrate-ft step$STEP cfg5 val @887caa9"
    if [ -f "$OUT/metrics.json" ]; then
        echo "  $TAG: metrics.json exists, skipping"
        continue
    fi
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  sbatch --array=0-$((SHARDS-1)) slurms/text2ct_run_shards.sh $STEP"
        echo "  DEP=afterany:<gen> SPLIT=val slurms/baseline_score.sh $TAG \"$LABEL\""
        continue
    fi
    GEN=$(sbatch --parsable --array=0-$((SHARDS-1)) --time=02:00:00 \
        --export="ALL,T2CT_TAG=$TAG,T2CT_CASES=$CASES,T2CT_PROMPTS=$PROMPTS" \
        slurms/text2ct_run_shards.sh "$STEP")

    # `baseline_score_job.sh` directly, not `baseline_score.sh`: the latter refuses to submit until
    # $OUT/nifti exists, which it cannot before the generation array has run. Same reason
    # `r2v_cfg_sweep.sh` calls the job script directly.
    #
    # afterany, not afterok: a preempted task exits non-zero and afterok would strand the cell
    # forever. Completeness is checked inside the score job, which sees the ingest manifest and
    # refuses a short cell without deleting its NIfTIs -- so re-running this script fills the gaps.
    SCORE=$(sbatch --parsable --dependency=afterany:"$GEN" --job-name="score_$TAG" \
        --export="ALL,SPLIT=val,CASES=$CASES" \
        slurms/baseline_score_job.sh "$TAG" "$LABEL")
    echo "  $TAG: generation $GEN -> score $SCORE"
done
