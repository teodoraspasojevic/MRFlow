# Baselines

Every baseline MRFlow is compared against, scored by `evaluation/main.py` on the same cases, the
same ground truth and the same extractors.

**This directory holds our code only.** The baselines themselves are cloned unmodified into the
workspace and pinned by commit below. They are not vendored here and not submodules: the three
upstream stacks cannot share a venv (GenerateCT wants an A100-era torch and its own two packages,
Text2CT a MONAI/MAISI stack, NV-Generate-MR-Brain its `.sif` containers), so one clone per venv in
the workspace is the only arrangement that runs. What lives here is the thin layer that makes their
output scoreable: the frozen population, the geometry conversion, the ingest, and one directory per
baseline recording how its volumes were produced.

```
common/cases.py        WHICH series to generate -> cases-<split>-n100.json  (committed)
common/prompts.py      the REPORT TEXT for those series -> $WS/prompts-*.json  (never committed)
common/canonicalize.py an external model's volume -> MRFlow's output grid
common/ingest.py       a baseline's NIfTI -> generated/*.npy + shard-NNNN.json
cases-test-n100.json   the frozen test population: 1,010 cases (1,000 scored + 10 MRA excluded)
cases-val-n100.json    the frozen val population: 1,002 cases (1,000 scored + 2 MRA excluded) --
                       case for case the set MRFlow's checkpoint and cfg selection ran on, so a
                       hyper-parameter chosen for a baseline is chosen the way MRFlow's was
<baseline>/            provenance and notes, one per baseline
```

**The shell scripts live in [`slurms/`](../slurms), with every other job in this repo.**
`baseline_score.sh` submits and `baseline_score_job.sh` is the job that ingests and scores;
generation is per baseline (`r2v_run_shards.sh`), as is any sweep over its settings
(`r2v_cfg_sweep.sh`). What stays here is the Python and the frozen populations.

**Guidance scales are chosen on val, never on test.** A baseline's inference settings are a
hyper-parameter like any other, and tuning them on the split the paper reports makes the reported
numbers selection-contaminated. `cases-val-n100.json` exists for that: sweep there, then score the
winner once on `cases-test-n100.json`.

**Ground truth is never preprocessed by a baseline and never written to disk.** A baseline
generates and stops. `score_cached` re-reads each reference straight out of the MR-RATE archives
at scoring time and canonicalizes it in memory, exactly as it does for MRFlow — which is what makes
every row of the table share one reference by construction rather than by agreement. `cases.py` and
`prompts.py` are the only two inputs a baseline gets, and `load_cases`/`load_prompts` are both
stdlib-only, so they read under any interpreter.

Upstream trees and weights: `/hnvme/workspace/y100dc19-mrflow-final/baselines/upstream/`.

| baseline | upstream | commit | cloned |
|---|---|---|---|
| `nvidia_r2v` | [teodoraspasojevic/R2V-MR-Generation](https://github.com/teodoraspasojevic/R2V-MR-Generation) | `ad5dca1` (2026-09-08) | 2026-09-16 |
| `generatect` | [ibrahimethemhamamci/GenerateCT](https://github.com/ibrahimethemhamamci/GenerateCT) | `2a81135` (2024-07-03) | 2026-09-16 |
| `text2ct` | [danielemolino/Text2CT](https://github.com/danielemolino/Text2CT) | `887caa9` (2026-09-15) | 2026-09-16 |

## The contract

A baseline is done when it has written two things into its results directory:

```
<out>/generated/<bucket>-<case_id>.npy    fp16, 1 mm isotropic, plane-first, 256^2, [0, 1]
<out>/shard-NNNN.json                     one row per case of the frozen population
```

That is the whole interface. `score_cached` in [`evaluation/main.py`](../evaluation/main.py) reads
exactly those two things plus the raw archives the manifest points at — no config, no checkpoint,
no model — so scoring a baseline is the same `--combine` pass that rescores MRFlow:

```bash
slurms/baseline_score.sh <run-tag> "<name> <settings> @<commit>"          # test, the default
SPLIT=val slurms/baseline_score.sh <run-tag> "<name> <settings> @<commit>"
```

`baseline_score.sh` only submits; `baseline_score_job.sh` is the job, and it ingests, drops the
NIfTIs and scores in one go. **Ingest runs inside that job, not on the login node** -- it reads and writes ~24 MB per
case over a thousand cases, and a sweep needs the whole chain to be one `--dependency` target.

`--label` is what keeps the W&B table readable: without it a baseline run is named from
`config.guidance`, i.e. MRFlow's sampler settings, which are not the baseline's. It also lands in
`metrics.json` — alongside `generators`, which `--combine` reads back out of the shard manifests,
so a results dir says what produced it even if the label was forgotten.

Three steps get you there, and only the middle one is baseline-specific:

```bash
# 1. the population, once, shared by every baseline and by MRFlow itself
#    (--split val --out baselines/cases-val-n100.json for the tuning population)
python -m baselines.common.cases --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
    --split test --n_per_bucket 100 --out baselines/cases-test-n100.json
#    MRFlow then rolls out over that same file rather than re-deriving one:
#    sbatch --array=0-31 slurms/mrflow_eval_helma.sh <config> <ckpt> \
#        --split test --cases baselines/cases-test-n100.json --out <out>

# 1b. the report text for those cases -- patient text, so it goes to the workspace
python -m baselines.common.prompts --cases baselines/cases-test-n100.json \
    --out $WS/baselines/prompts-test-n100.json

# 2. generate, in THAT baseline's venv, writing <case_id>.nii.gz per case
#    reads the two JSONs above and nothing else from MRFlow
sbatch slurms/<name>_run_shards.sh ...

# 3. ingest and score, back in the MRFlow venv
slurms/baseline_score.sh <run-tag> "<name> <settings> @<commit>"
```

## Four rules that make the table readable

**The population is a file, not a function.** `evaluation/main.py` derives its case list at runtime
from `list_series` + `select_cases` at a fixed seed. That is reproducible for one run, but a
baseline table is built over weeks, and a changed parquet or `max_series_test` would move the
population underneath it with nothing in any output to say so. `cases-test-n100.json` is committed
so the population is an artifact you can diff and cite. It holds 1,010 rows: the 1,000 scored cases
plus 10 MRA, which are carried rather than dropped so `n_excluded_out_of_scope_modality` still
reports them.

**A baseline uses its own preprocessing, its own text encoder and its own geometry.** Sharing our
dataset code for *generation* would mean measuring GenerateCT-running-on-MRFlow's-pipeline, which
is not GenerateCT. What is shared is narrow and deliberate: which cases, the raw report text handed
to each model, and the ground truth — and the last is safe by construction, because the evaluation
reads GT itself from the archives and no baseline ever supplies its own reference.

**NIfTI is the handoff.** All three upstream repos already write it, and its affine carries spacing
and orientation, so `ingest.py` reads a baseline's output with `read_canonical` — the *same*
function the ground-truth path uses — rather than each adapter restating what its model's axes
mean. It also means the generation side needs no MRFlow import at all. The intermediate NIfTIs are
disposable once the manifest is written; `generated/*.npy` (~21 MB per case) is what the evaluation
reads from then on.

**Geometry conversion happens in exactly one place.** `canonicalize_external` mirrors
`preprocess_volume` step for step — verified against it on real MR-RATE volumes: identical output
shapes and 0.99992 correlation, the residual being only `preprocess_volume`'s nonzero-voxel
intensity normalization. If a baseline needs a different framing, that is a `posterior_shift_mm`
argument and a line in its own README, never a second copy of this code.

## Still to wire up

- The GenerateCT and Text2CT generation drivers and their `slurms/<name>_run_shards.sh`. Each
  baseline's README says what its driver has to do; `nvidia_r2v`'s is written, the other two
  are not.
- `--combine` still requires an MRFlow `--config`, because it reads `wandb_args` for the project
  and group and validates the label mapping. Harmless for a baseline — nothing from that config
  reaches a metric — but it does mean a baseline is scored with an MRFlow config path on the
  command line, and `metrics.json` still carries a meaningless `ckpt` field next to the meaningful
  `label`.
