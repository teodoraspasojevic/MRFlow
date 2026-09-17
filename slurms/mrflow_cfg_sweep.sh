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
# its position, so `--split val --n_per_bucket 100` selects the identical 1,002 cases with identical
# noise at ANY task count -- the exact population the checkpoint selection under
# <eval>/ckptsel/*/checkpoint-* was scored on. `--limit` has none of that: it applies per shard
# *after* the interleave, so the case set changes with the task count and two runs are not
# comparable.
#
# **This sweep never runs on test, and forces val if asked to.** Guidance scales are a
# hyper-parameter: choosing them on the split the paper reports is selection on the test set, and
# the reported numbers stop being held out. Checkpoint selection already ran on val at
# --n_per_bucket 100, so a val sweep picks the cfg pair on the same 1,002 cases the checkpoint was
# picked on, and test stays untouched for the one final scoring run.
#
# **Cost, measured rather than budgeted.** A full-body rollout of a real test case is 11.8 s/block
# over 9-13 blocks -- ~141 s, or ~3.3 min per case including scoring, at cfg 1/1 without compile.
# So a 1,000-case cell is ~54 GPU-hours unguided and ~85 guided (the three CFG branches go through
# the denoiser as one batched call, ~1.8x wall time, not the 3x the FLOP count suggests). The eval
# script's 6 h-per-task limit fits ~31 cases at TASKS=32 with room to spare, and --compile defaults
# on here, so expect less than the above.
#
# **Every cell runs the full 1,002-case val population** (N_PER_BUCKET=100, TASKS=32 -- the
# defaults), so a cell is directly comparable with the checkpoint sweep and not only with the other
# cells. At the per-cell cost above that is ~1,330 GPU-hours for the 16-cell product (one cheap
# unguided cell plus 15 guided) and ~820 for the 10-cell list. Trim CELLS before trimming the
# population: a cheaper sweep on fewer cases changes what the numbers mean, a cheaper sweep on
# fewer cells does not.
#
#   # the full grid on the selection population
#   bash slurms/mrflow_cfg_sweep.sh $CONFIG $CKPT $BASE/sweep
#
#   # one cell, e.g. re-running a winner after a metric change
#   MOD_SCALES=4 REP_SCALES=7 bash slurms/mrflow_cfg_sweep.sh $CONFIG $CKPT $BASE/final
#
# **At the full population both numbers order; on a reduced one, rank on FID -- this reversed when
# the metrics moved to their reference protocols.** FVD now takes one clip per volume (StyleGAN-V's
# own protocol) rather than ~17-21 overlapping windows, so the clip count *is* the case count: 1,002
# cases give ~1,000 samples against 400 feature dimensions, usable but below the 2,048 StyleGAN-V
# quotes. Cut the population to N_PER_BUCKET=10 and it is ~100 samples against 400 -- a rank-
# deficient covariance that cannot order anything. fid_2d_inception went the other way, every slice
# instead of every 4th, so even ~100 cases give ~16,000 rows against 2,048 dimensions.
#
# So a cheap ordering pass with N_PER_BUCKET=10 (a tenth of the population, so ~130 GPU-hours
# for the 16-cell grid) is still possible, but
# read only the FID ordering off it, and not the absolute values: those are not comparable to a
# 1,000-case run.
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
if [ "$SPLIT" = "test" ]; then
    echo "SPLIT=test refused: guidance scales are chosen here, and choosing them on the reported" >&2
    echo "split is selection on test. Running on val instead -- the same population the checkpoint" >&2
    echo "selection used. Score the winner on test with slurms/mrflow_eval_helma.sh." >&2
    SPLIT=val
fi
# --n_per_bucket rather than --limit: it caps each (modality, plane) bucket *before* the shards are
# cut, so the case list is identical at any task count and stays modality-balanced. --limit slices
# a seed-shuffled list instead, so its modality mix is whatever the split happens to give.
N_PER_BUCKET=${N_PER_BUCKET:-100}
# 32 to match the checkpoint sweep's array size. Everything but IS is task-count invariant, and at
# ~31 cases per task a guided cell lands around 3 h, inside the eval script's 4:30 limit.
TASKS=${TASKS:-32}
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
echo "  print(m[\"FVD_f16\"], m[\"fid_2d_inception\"], m[\"n_scored_files\"])' \$f; done | sort -k2 -n"
