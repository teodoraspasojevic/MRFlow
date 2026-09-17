#!/bin/bash -l
#
# Submit the ingest-and-score job for one baseline run. Identical for every baseline -- that is the
# point of the contract in baselines/README.md.
#
#   slurms/baseline_score.sh nvidia_r2v_armA_cfg7 "nvidia_r2v armA cfg7 @ad5dca1"
#   SPLIT=val slurms/baseline_score.sh r2v_val_armA_rep7 "nvidia_r2v armA rep7 @ad5dca1"
#
# The work itself is `baseline_score_job.sh`: ingest the NIfTIs, drop them, then score every metric over the
# cached volumes. **Ingest runs inside that job rather than here**, because it reads and writes
# ~24 MB per case over 1,010 cases and a login node is the wrong place for it -- and because a
# sweep needs the whole chain to be one `--dependency` target.
#
# The combine pass reads the ground truth fresh from the MR-RATE archives and canonicalizes it in
# memory. Nothing about a baseline's run touches the reference, which is what makes every row of
# the table share one.
#
# Overridable: SPLIT (default test) and CASES (default baselines/cases-$SPLIT-n100.json), which
# must match the population the run was generated over. DEP passes an sbatch dependency through.

set -euo pipefail

TAG=${1:?usage: baseline_score.sh <run-tag> "<label>" [extra main.py flags]}
LABEL=${2:?usage: baseline_score.sh <run-tag> "<label>" [extra main.py flags]}
shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT=/hnvme/workspace/y100dc19-mrflow-final/baselines/runs/$TAG
SPLIT=${SPLIT:-test}
CASES=${CASES:-baselines/cases-$SPLIT-n100.json}
[[ -d "$OUT/nifti" ]] || { echo "no $OUT/nifti -- run the generation array first" >&2; exit 1; }

# Exported explicitly rather than left to sbatch's default ALL: which population a run is scored
# over is the one thing that must not depend on the submitting shell's environment.
sbatch --job-name="score_$TAG" ${DEP:+--dependency="$DEP"} \
    --export="ALL,SPLIT=$SPLIT,CASES=$CASES" \
    slurms/baseline_score_job.sh "$TAG" "$LABEL" "$@"

echo "When it lands: $OUT/metrics.json"
