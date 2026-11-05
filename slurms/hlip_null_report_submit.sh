#!/bin/bash
#
# The null-report experiment: how much of each model's HLIP alignment comes from the report.
#
#   DRY_RUN=1 bash slurms/hlip_null_report_submit.sh   # print the sbatch lines, submit nothing
#   bash slurms/hlip_null_report_submit.sh
#
# Submits, all at cfg 1/1 (report CFG means nothing without a report, and Text2CT has no modality
# CFG to match):
#   - null-report generations on the 100-study subset baselines/cases-test-null100.json:
#     MRFlow+CXR-BERT 180k, MRFlow+3ENC+trim+qk+new-lr 120k, MRFlow+3ENC+trim+old-lr 200k. Text2CT
#     follows once it is retrained: generate it with --null_report, and the job compares it if present;
#   - one CPU job, after all of them, comparing each null run with its model's full-report 1/1 run
#     (slurms/hlip_null_report_job.sh), then packing the null runs' volumes.
#
# The two 3ENC arms are generated from the MRFlow-trim-blank worktree, which builds their
# sectioned conditioning; the main repo would silently condition them on CXR-BERT.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
TRIMBLANK=$(realpath ../MRFlow-trim-blank)
WS=/hnvme/workspace/y100dc19-mrflow-final
EXP=$WS/experiments
OUT=$WS/eval/null_report
SUBSET=$REPO/baselines/cases-test-null100.json
GPU="--partition=preempt,h200 --gres=gpu:h200:1"
NULL="--split test --modality_cfg_scale 1 --report_cfg_scale 1 --null_report --score_collapsed --no-hlip"

submit() {   # prints the job id, or the command under DRY_RUN
    if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo DRY; else sbatch --parsable "$@"; fi
}

mkdir -p logs "$TRIMBLANK/logs"

CXR=$(submit $GPU --array=0-31 slurms/mrflow_eval_helma.sh \
    $EXP/mrflow_from_scratch/config.yaml $EXP/mrflow_from_scratch/checkpoint-180000/denoiser_ema \
    $NULL --cases $SUBSET --out $OUT/cxrbert_ckpt180k)

cd "$TRIMBLANK"
QK=$(submit $GPU --array=0-31 slurms/mrflow_eval_helma.sh \
    $EXP/mrflow_from_scratch_sectioned_qknorm_trim-v1/config.yaml \
    $EXP/mrflow_from_scratch_sectioned_qknorm_trim-v1/checkpoint-120000/denoiser_ema \
    $NULL --n_per_bucket 100 --study_subset $SUBSET --out $OUT/qknorm_trim-v1_ckpt120k)
OLDLR=$(submit $GPU --array=0-31 slurms/mrflow_eval_helma.sh \
    $EXP/mrflow_from_scratch_sectioned_trim-v1/config.yaml \
    $EXP/mrflow_from_scratch_sectioned_trim-v1/checkpoint-200000/denoiser_ema \
    $NULL --n_per_bucket 100 --study_subset $SUBSET --out $OUT/sectioned_trim-v1_ckpt200k)
cd "$REPO"

COMPARE=$(submit --dependency=afterok:$CXR:$QK:$OLDLR slurms/hlip_null_report_job.sh)

# Pack the null runs' volumes once the comparison has read them (verified before any delete).
RUNS=$WS/pack_generated/runs_null_report.txt
printf "%s\n" $OUT/cxrbert_ckpt180k $OUT/qknorm_trim-v1_ckpt120k $OUT/sectioned_trim-v1_ckpt200k > $RUNS
submit --dependency=afterok:$COMPARE $WS/pack_generated/pack_and_clean.sh $RUNS
