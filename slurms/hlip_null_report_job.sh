#!/bin/bash -l
#
# The null-report comparison, after slurms/hlip_null_report_submit.sh's generations: ingests the
# Text2CT null run, then scores every model's null run against its full-report cfg-1/1 run on the
# 100-study subset (evaluation/hlip_null_report.py), with trim on both sides. Writes
# <null run>/hlip_null_report.json. CPU only: HLIP-base is ~0.5 s per scan here.
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=hlip_null_report
#SBATCH --output=logs/hlip_null_%j.out
#SBATCH --error=logs/hlip_null_%j.err

set -euo pipefail
unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/venv
SUBSET=baselines/cases-test-null100.json
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export TOKENIZERS_PARALLELISM=false
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1"

T2CT_NULL=$WS/baselines/runs/text2ct_test_step56000_null
[ -f $T2CT_NULL/shard-0000.json ] || $VENV/bin/python -m baselines.common.ingest \
    --cases $SUBSET --nifti $T2CT_NULL/nifti --out $T2CT_NULL --label "Text2CT step56000 null report"

compare() {   # <full-report run> <null-report run>
    echo "=== $2"
    $VENV/bin/python evaluation/hlip_null_report.py --full "$1" --null "$2" \
        --study_subset $SUBSET --gt_bounds $WS/data/mrflow_flux_trim-v1 --trim_generated \
        --score_collapsed --workers 16 --out "$2/hlip_null_report.json"
}

compare $WS/baselines/runs/text2ct_test_step56000_cfg1 $T2CT_NULL
compare $WS/eval/final/scratch_old_ckpt180k/mod1-rep1 $WS/eval/null_report/cxrbert_ckpt180k
compare $WS/eval/final/qknorm_trim-v1_ckpt120k_mod1rep1 $WS/eval/null_report/qknorm_trim-v1_ckpt120k
compare $WS/eval/final/sectioned_trim-v1_ckpt200k/mod1-rep1 $WS/eval/null_report/sectioned_trim-v1_ckpt200k
