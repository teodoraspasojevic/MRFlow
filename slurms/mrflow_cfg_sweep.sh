#!/bin/bash -l
#
# Launch the hierarchical-CFG grid: one isolated evaluation per (modality, report) scale pair.
# This is a *submitter*, not a batch job -- run it on the login node and it submits 2 jobs per
# cell (an array that generates and scores, plus a dependent --combine that pools and logs to W&B).
#
#   bash slurms/mrflow_cfg_sweep.sh <config.yaml> <ckpt/denoiser_ema> <base out dir>
#   DRY_RUN=1 bash slurms/mrflow_cfg_sweep.sh ...        # print the sbatch lines, submit nothing
#
# Overridable: CELLS (or MOD_SCALES x REP_SCALES), SPLIT, N_PER_BUCKET, TASKS. Anything after the
# base out dir is passed through to evaluation/main.py for every cell.
#
# CELLS is an explicit list of mod:rep pairs, for the usual case where the grid is not a full
# product. An unguided baseline plus every combination of the guided scales is 10 cells, not the
# 16 that a {1,4,7,10} x {1,4,7,10} product would give:
#
#   CELLS="1:1 4:4 4:7 4:10 7:4 7:7 7:10 10:4 10:7 10:10"
#
# **Every cell MUST get its own --out, which is why this script exists.** `--out` defaults to
# <output_dir>/eval/<regime>-<split>, with no cfg scales in the path, and `main.py` skips a shard
# that already exists unless --overwrite. So pointing two cells at one directory does not collide
# loudly -- the second cell's tasks print "already written", skip generation entirely, and its
# --combine pass then pools the *first* cell's shards and logs them to W&B under the second cell's
# name. You get a full sweep table of identical numbers under different labels and no error
# anywhere. Per-cell directories are the whole defence.
#
# **--n_per_bucket, never --limit, for any number you intend to compare.** `select_cases` runs
# before the shard slice and uses no RNG, and per-case noise is seeded from the case id rather than
# its position, so `--split test --n_per_bucket 100` selects the identical 1,010 cases (1,000
# scored) with identical noise at ANY task count -- R2V-MR-Generation's population case for case.
# `--limit` has none of that: it applies per shard *after* the interleave, so the case set changes
# with the task count and two runs are not comparable.
#
# **Cost, measured rather than budgeted.** A full-body rollout of a real test case is 11.8 s/block
# over 9-13 blocks -- ~141 s, or ~3.3 min per case including scoring, at cfg 1/1 without compile.
# So a 1,000-case cell is ~54 GPU-hours unguided and ~85 guided (the three CFG branches go through
# the denoiser as one batched call, ~1.8x wall time, not the 3x the FLOP count suggests). The eval
# script's 6 h-per-task limit fits ~31 cases at TASKS=32 with room to spare, and --compile defaults
# on here, so expect less than the above.
#
# Two stages are still worth considering, because model selection on the split you report is a
# methodological problem even when the compute is affordable:
#
#   # stage 1 -- rank the grid, ~100 val cases per cell (the defaults)
#   bash slurms/mrflow_cfg_sweep.sh $CONFIG $CKPT $BASE/sweep
#
#   # stage 2 -- the winner only, on the exact population every other model was scored on
#   SPLIT=test N_PER_BUCKET=100 TASKS=32 MOD_SCALES=4 REP_SCALES=7 \
#       bash slurms/mrflow_cfg_sweep.sh $CONFIG $CKPT $BASE/final
#
# Running all 16 cells at SPLIT=test N_PER_BUCKET=100 works and is comparable everywhere, it just
# costs the full ~2,900 GPU-hours and picks the cfg pair on test.
#
# **Rank on FVD_f16, not FID.** At N_PER_BUCKET=10 (~100 scored cases) a case yields ~17-21 f16
# clips, so FVD_f16 rests on ~2,000 samples against 400 feature dimensions -- thin but usable for
# ordering. FID sees ~4,000 slices against 2,048 dimensions, which is too close to the dimension to
# rank on. Absolute values from a 100-case sweep are not comparable to the 1,000-case run either;
# only the ordering carries over.
#
# Cost: the 1/1 cell is cheaper per block than the rest, because at
# modality_cfg_scale == report_cfg_scale == 1 the sampler takes its single-conditional
# short-circuit instead of batching three branches. Not 3x cheaper though -- the three branches go
# through the denoiser as one batched call, and batching measured ~1.7x throughput on this trunk,
# so expect roughly 1.8x wall time rather than 3x.
#
# Cells whose metrics.json already exists are skipped, so re-running this resumes a partial sweep.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    # the header comment block is the usage text, so it cannot go stale
    awk '/^set -euo/{exit} NR>1 && /^#/{sub(/^# ?/, ""); print}' "$0"
    exit 1
fi

CONFIG=$1
CKPT=$2
BASE=$3
shift 3

MOD_SCALES=${MOD_SCALES:-"1 4 7 10"}
REP_SCALES=${REP_SCALES:-"1 4 7 10"}
# An explicit cell list wins over the product, so a non-rectangular grid needs no loop surgery.
CELLS=${CELLS:-}
SPLIT=${SPLIT:-val}
# --n_per_bucket rather than --limit: it caps each (modality, plane) bucket *before* the shards are
# cut, so the case list is identical at any task count and stays modality-balanced. --limit slices
# a seed-shuffled list instead, so its modality mix is whatever the split happens to give.
N_PER_BUCKET=${N_PER_BUCKET:-10}
TASKS=${TASKS:-8}
DRY_RUN=${DRY_RUN:-0}

EVAL=slurms/mrflow_eval_helma.sh
[ -f "$EVAL" ] || { echo "run me from the repo root: $EVAL not found" >&2; exit 1; }

submit() {  # print under DRY_RUN, otherwise submit. Either way stdout is only the job id, since
            # the caller captures it -- the echoed command goes to stderr.
    if [ "$DRY_RUN" = "1" ]; then
        printf '  sbatch %s\n' "$*" >&2
        echo "000000"
    else
        sbatch "$@"
    fi
}

if [ -z "$CELLS" ]; then
    for mod in $MOD_SCALES; do
        for rep in $REP_SCALES; do CELLS="$CELLS $mod:$rep"; done
    done
fi

n_cells=0
for pair in $CELLS; do
    mod=${pair%%:*}
    rep=${pair##*:}
    # %g formatting in main.py's W&B run name turns 1 into "mod1", so integer scales keep the
    # directory and the run name in step. Non-integer scales work but will not match.
    cell="mod${mod}-rep${rep}"
    out="$BASE/$cell"

    if [ -f "$out/metrics.json" ]; then
        echo "$cell: metrics.json exists, skipping"
        continue
    fi

    echo "$cell -> $out"
    job=$(submit --parsable --array=0-$((TASKS - 1)) "$EVAL" "$CONFIG" "$CKPT" \
        --split "$SPLIT" --n_per_bucket "$N_PER_BUCKET" --out "$out" \
        --modality_cfg_scale "$mod" --report_cfg_scale "$rep" "$@")

    # afterany, not afterok: one dead task must not strand the pool. Check n_scored_files in
    # metrics.json against the case count afterwards -- combine pools whatever shards it finds.
    submit --parsable --dependency=afterany:"$job" "$EVAL" "$CONFIG" "$CKPT" \
        --split "$SPLIT" --n_per_bucket "$N_PER_BUCKET" --out "$out" --combine \
        --modality_cfg_scale "$mod" --report_cfg_scale "$rep" "$@" >/dev/null
    n_cells=$((n_cells + 1))
done

echo
echo "$n_cells cells, $((n_cells * (TASKS + 1))) jobs (split $SPLIT, n_per_bucket $N_PER_BUCKET,"
echo "$TASKS tasks/cell). Results land in $BASE/mod*-rep*/metrics.json"
echo "Compare with: for f in $BASE/mod*-rep*/metrics.json; do echo -n \"\$f \"; \\"
echo "  python -c 'import json,sys; m=json.load(open(sys.argv[1]))[\"metrics\"]; \\"
echo "  print(m[\"FVD_f16\"], m[\"FID\"], m[\"n_scored_files\"])' \$f; done | sort -k2 -n"
