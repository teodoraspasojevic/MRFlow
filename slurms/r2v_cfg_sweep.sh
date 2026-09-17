#!/bin/bash -l
#
# Launch the R2V report-guidance sweep: one isolated run per (arm, report cfg) cell. This is a
# *submitter*, not a batch job -- run it on the login node and it submits 2 jobs per cell (the
# generation array, plus a dependent ingest-and-score pass).
#
#   bash slurms/r2v_cfg_sweep.sh
#   DRY_RUN=1 bash slurms/r2v_cfg_sweep.sh                # print the sbatch lines, submit nothing
#
# Overridable: ARMS, REP_SCALES, MOD_SCALE, SPLIT, TASKS, PREFIX. Anything on the command line is
# passed through to generate.py for every cell.
#
# **Modality guidance stays at 10.0 in every cell.** That is NVIDIA's own `cfg_guidance_scale` for
# mr-brain and what every R2V number on disk was produced at; only the report term is swept, so a
# cell differs from its neighbours in exactly one scale. The arithmetic is
# `D_00 + s_mod*(D_m0 - D_00) + s_rep*(D_mr - D_m0)` -- the same hierarchy MRFlow uses, so
# `s_rep = 1` is the plain report-conditioned term rather than no report at all (that is `s_rep = 0`,
# which collapses to NVIDIA's official behaviour and is not in the grid).
#
# **This sweep runs on val, and forces it if asked for test.** The A=7 / E=3 scales in README.md
# were chosen on test, i.e. selection on the reported split. This picks them on the same 1,002-case
# val population MRFlow's own checkpoint and cfg selection used, so the two models' scales are
# chosen the same way and test stays untouched for the one final scoring run.
#
# **Every cell gets its own run directory.** `generate.py` skips a case whose NIfTI already exists,
# so two cells sharing a directory would not collide loudly -- the second would adopt the first's
# volumes and score them under its own label. Per-cell directories are the whole defence, which is
# why the tag carries the split, the arm and the scale.
#
# Cost, measured on this stack: ~1.6 s/case generation plus a sliding-window decode, so a 16-task
# array is well under an hour per task; the fixed cost is model loading, not the case count. The
# scoring pass is the long pole -- one sequential GPU pass over 1,002 cases, budgeted at 6 h.
#
# Inodes, not disk, are the constraint on this account (102,400 hard, per user across /hnvme).
# A cell holds 1,000 `.npy` volumes once its NIfTIs are dropped, so the 8-cell grid is ~8,000.
#
# Cells whose metrics.json already exists are skipped, so re-running this resumes a partial sweep.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WS=/hnvme/workspace/y100dc19-mrflow-final
ARMS=${ARMS:-"A E"}
REP_SCALES=${REP_SCALES:-"1 4 7 10"}
MOD_SCALE=${MOD_SCALE:-10}
SPLIT=${SPLIT:-val}
TASKS=${TASKS:-16}
PREFIX=${PREFIX:-r2v}
COMMIT=$(git -C "$WS/baselines/upstream/R2V-MR-Generation" rev-parse --short HEAD)
DRY_RUN=${DRY_RUN:-0}

if [ "$SPLIT" = "test" ]; then
    echo "SPLIT=test refused: guidance scales are chosen here, and choosing them on the reported" >&2
    echo "split is selection on test. Running on val instead -- the same 1,002-case population" >&2
    echo "MRFlow's checkpoint and cfg selection used. Score the winner on test with baseline_score.sh." >&2
    SPLIT=val
fi

CASES=${CASES:-baselines/cases-$SPLIT-n100.json}
PROMPTS=${PROMPTS:-$WS/baselines/prompts-$SPLIT-n100.json}
for p in "$CASES" "$PROMPTS"; do
    [ -e "$p" ] || { echo "MISSING: $p -- run baselines.common.{cases,prompts} first" >&2; exit 1; }
done

submit() {  # print under DRY_RUN, otherwise submit. Either way stdout is only the job id, since
            # the caller captures it -- the echoed command goes to stderr.
    if [ "$DRY_RUN" = "1" ]; then
        printf '  sbatch %s\n' "$*" >&2
        echo "000000"
    else
        sbatch "$@"
    fi
}

n_cells=0
for arm in $ARMS; do
    for rep in $REP_SCALES; do
        tag="${PREFIX}_${SPLIT}_arm${arm}_rep${rep}"
        out=$WS/baselines/runs/$tag
        label="nvidia_r2v arm$arm rep$rep mod$MOD_SCALE $SPLIT @$COMMIT"

        if [ -f "$out/metrics.json" ]; then
            echo "$tag: metrics.json exists, skipping"
            continue
        fi

        echo "$tag -> $out"
        # Generation goes to `preempt`: measured ~5 min to start against ~2 days on h200, and a
        # cell is ~20 min per task, so the wait dominates the work. A cancelled task is not
        # requeued, but `baseline_score_job.sh` refuses to score a short cell and leaves its NIfTIs, so
        # re-running this script fills the gaps. Scoring itself stays on h200 -- `score_cached`
        # has no resume and would restart at case 0.
        job=$(submit --parsable --array=0-$((TASKS - 1)) -p preempt --time=01:00:00 \
            --export="ALL,R2V_TAG=$tag,R2V_CASES=$CASES,R2V_PROMPTS=$PROMPTS" \
            slurms/r2v_run_shards.sh "$arm" \
            --report_guidance_scale "$rep" --modality_guidance_scale "$MOD_SCALE" "$@")

        # afterany, not afterok: a preempted task exits non-zero and afterok would strand the cell
        # forever. Completeness is checked in `baseline_score_job.sh` instead, which sees the manifest.
        submit --parsable --dependency=afterany:"$job" --job-name="score_$tag" \
            --export="ALL,SPLIT=$SPLIT,CASES=$CASES" \
            slurms/baseline_score_job.sh "$tag" "$label" >/dev/null
        n_cells=$((n_cells + 1))
    done
done

echo
echo "$n_cells cells, $((n_cells * 2)) jobs (split $SPLIT, $TASKS tasks/cell, modality cfg $MOD_SCALE)."
echo "Results land in $WS/baselines/runs/${PREFIX}_${SPLIT}_arm*_rep*/metrics.json"
echo "Compare with: for f in $WS/baselines/runs/${PREFIX}_${SPLIT}_arm*/metrics.json; do echo -n \"\$f \"; \\"
echo "  python -c 'import json,sys; m=json.load(open(sys.argv[1]))[\"metrics\"]; \\"
echo "  print(m[\"fid_2d_inception\"], m[\"FVD_f16\"], m[\"n_scored_files\"])' \$f; done | sort -k2 -n"
