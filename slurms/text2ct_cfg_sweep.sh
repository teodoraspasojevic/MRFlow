#!/bin/bash -l
#
# Sweep the report guidance scale for one Text2CT checkpoint, on the val population.
#
#   bash slurms/text2ct_cfg_sweep.sh 60000                 # the default grid
#   bash slurms/text2ct_cfg_sweep.sh 60000 1 3 7 10        # explicit scales
#   DRY_RUN=1 bash slurms/text2ct_cfg_sweep.sh 60000       # print, submit nothing
#
# Same shape as `r2v_cfg_sweep.sh`: each cell is a 16-task generation array on `preempt` plus a
# dependent ingest-and-score pass on `h200`, writing
# $WS/baselines/runs/text2ct_val_step<N>_cfg<S>/metrics.json. Cells with a metrics.json are skipped.
#
# **Why this sweep has to exist.** Text2CT ships report classifier-free guidance and this fine-tune
# kept it: 10% of training samples saw a zero context (`cfg.report_dropout_prob`, upstream's own
# `conditional_free_guidance`), so the unconditional branch is genuinely trained and the scale is a
# real knob. `baselines/nvidia_r2v` tuned its two arms to 7 and 4 on this same val population;
# scoring Text2CT only at upstream's default of 5.0 while the competitor runs at its tuned value
# would handicap it. One scale varies, everything else is held.
#
# **5.0 is deliberately not in the default grid** -- it is already measured for every checkpoint as
# `text2ct_val_step<N>/metrics.json` from `text2ct_ckpt_sweep.sh`. Include it explicitly to redo it.
#
# cfg 1.0 is the plain report-conditioned prediction, not the report switched off; `generate.py`
# skips the unconditional branch entirely there, so that cell is also the cheapest.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WS=/hnvme/workspace/y100dc19-mrflow-final
SHARDS=${SHARDS:-16}
CASES=${CASES:-baselines/cases-val-n100.json}
PROMPTS=${PROMPTS:-$WS/baselines/prompts-val-n100.json}

STEP=${1:?usage: text2ct_cfg_sweep.sh <checkpoint-step> [scales...]}
shift
SCALES=("$@")
[ ${#SCALES[@]} -gt 0 ] || SCALES=(1 3 7 10)

echo "checkpoint step $STEP | scales: ${SCALES[*]} (5.0 already in text2ct_val_step$STEP)"
for CFG in "${SCALES[@]}"; do
    TAG=text2ct_val_step${STEP}_cfg${CFG}
    OUT=$WS/baselines/runs/$TAG
    LABEL="text2ct mrrate-ft step$STEP cfg$CFG val @887caa9"
    if [ -f "$OUT/metrics.json" ]; then
        echo "  $TAG: metrics.json exists, skipping"
        continue
    fi
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  sbatch --array=0-$((SHARDS-1)) slurms/text2ct_run_shards.sh $STEP --guidance_scale $CFG"
        continue
    fi
    GEN=$(sbatch --parsable --array=0-$((SHARDS-1)) --time=02:00:00 \
        --export="ALL,T2CT_TAG=$TAG,T2CT_CASES=$CASES,T2CT_PROMPTS=$PROMPTS" \
        slurms/text2ct_run_shards.sh "$STEP" --guidance_scale "$CFG")
    SCORE=$(sbatch --parsable --dependency=afterany:"$GEN" --job-name="score_$TAG" \
        --export="ALL,SPLIT=val,CASES=$CASES" \
        slurms/baseline_score_job.sh "$TAG" "$LABEL")
    echo "  $TAG: generation $GEN -> score $SCORE"
done
