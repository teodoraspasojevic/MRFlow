# Evaluation

Roll out MRFlow over an MR-RATE split and score it with the **official VLM3D
`mr-volume-generation` metrics**. The metric code in [`challenge.py`](challenge.py) is a port of the
challenge's own scoring container, so the numbers here are the leaderboard's, not our own
definitions.

No preprocessed split is needed — every case is read straight out of the raw MR-RATE tars — so
`--split test` runs exactly like `--split val`.

## How to evaluate one model

Start with a quick check on one GPU, so a broken path or config fails in minutes instead of hours:

```bash
python evaluation/main.py --config <experiment>/config.yaml \
    --ckpt <experiment>/checkpoint-N/denoiser_ema --split val --limit 8
```

Then the real run — two commands. The first generates and scores, spread over many GPUs; the
second adds the pieces up. Submitting them together with `--dependency` means you type both once
and walk away:

```bash
CONFIG=<experiment>/config.yaml
CKPT=<experiment>/checkpoint-N/denoiser_ema
OUT=/hnvme/workspace/y100dc19-mrflow-final/eval/my_run

# 1. generate + score: 32 tasks in parallel, 32 cases each = 1024 cases
JOB=$(sbatch --parsable --array=0-31 slurms/mrflow_eval_helma.sh \
    $CONFIG $CKPT --split val --limit 32 --out $OUT)

# 2. pool the pieces into metrics.json and log to W&B, after step 1 has finished
sbatch --dependency=afterany:$JOB slurms/mrflow_eval_helma.sh \
    $CONFIG $CKPT --split val --combine --out $OUT
```

**`--limit` is per task**, so cases = array size × limit. Drop it to run the whole split.

### Which cases: `--n_per_bucket`

`--limit` slices `list_series`, which is shuffled on a fixed seed — reproducible, but whatever the
split's modality mix happens to be. `--n_per_bucket N` instead takes the first `N` cases of every
`(modality, plane)` bucket, each bucket ordered by `(study_uid, series_id)`. No RNG is involved at
all, every prefix is bucket-balanced, and the cap is applied **before** sharding, so the shard count
does not change which cases run.

**`--n_per_bucket 100` is the comparison population.** It is what `R2V-MR-Generation`'s
`select_eval_cases` does, at the `N_PER_BUCKET=100` every number in that repo was produced at. On
MR-RATE's `test` split it selects 1,010 cases, of which **1,000 are scored** — 10 of the 12 buckets
hold an in-scope modality, and the 10 MRA cases are counted as
`n_excluded_out_of_scope_modality` rather than dropped. Those 1,000 are R2V's 1,000, case for case:
`run_shard` calls `list_series` with `max_repeats=1` rather than the config's `mri.max_repeats`, so
one acquisition per (study, contrast, plane) is eligible — the same cohort R2V's
`series_selection="one_per_study_per_bucket"` builds, on the full split too (29,016 scored series
either way). With the config's `max_repeats: null` the duplicate acquisitions
(`t1w-raw-axi-2`, `-3`, …) crowd out other studies and only 868 of the 1,000 match. Its per-bucket
caveat carries over too: the 2.5D FID compares 512-d covariances, so at 100 per bucket only the
pooled numbers are trustworthy.

Sampler noise is seeded per case (`config.seed + case_id`), not per position, so a rerun — or the
same case under a different shard count — draws the same noise and scores the same. `config.seed`
is 42, the same base seed the R2V runs used.

**Why two commands.** Each array task scores its own slice and writes `shard-NNNN.pt`. None of them
knows it is the last to finish, so the totalling has to be a separate job that starts afterwards.
FID also cannot be averaged per shard — `combine` pools the raw features across all of them.

**Only `--combine` logs to W&B.** The array tasks write files and nothing else. The W&B run is named
after the config's `wandb_args.name`, so give a config copy its own name when the checkpoint is not
from that experiment, or the run borrows a training run's name.

Pass the config **saved into the experiment dir**, next to the checkpoints — it has the paths and
the architecture the checkpoint was trained with. `--out` defaults to `<output_dir>/eval/<regime>-<split>`;
pass it explicitly for a checkpoint that is not from a finished experiment, so the run does not
create that experiment's directory.

Everything lands in `$OUT`: one `shard-NNNN.pt` per task, `examples/*.mp4`, and `metrics.json`.
**Check `n_total_files` in `metrics.json` matches the case count you asked for** — `combine` pools
whatever shards it finds, so a task that died leaves a quietly smaller evaluation.

| file | what |
|---|---|
| [`challenge.py`](challenge.py) | vendored official container: modality scope, MSE/PSNR/SSIM, streaming 2.5D FID |
| [`__init__.py`](__init__.py) | `ChallengeAccumulator` — the official `score.py` aggregation, over pairs held in memory |
| [`main.py`](main.py) | the CLI: build a case, generate, score, write `metrics.json`, log to W&B |

## Preprocessing

**The rule: the ground truth reaches the metric in its released geometry.** The official metric
percentile-normalizes both volumes itself and resamples the *generated* one onto the ground truth's
shape. Resampling or normalizing the reference here would score the model against a different
target than the platform does.

| | what happens to it |
|---|---|
| ground truth | `load_native_volume`: reorient to RAS, transpose so the slice axis leads. **Nothing else** — no resample, no normalization, no crop or pad. |
| generated | nothing. Submitted as the model produced it: 256², 1 mm, `T` slices. |
| both, inside the metric | 0.5/99.5-percentile normalize to `[0, 1]`; the generated volume is `zoom`ed onto the ground truth's shape |

The one transformation applied to the ground truth is a **transpose**, which is filing, not
preprocessing: same voxel values, same three axis lengths, same spacing. It exists so that "slice
40" means the same anatomical cut in both volumes. Its order depends on how the series was
acquired, because the model rolls out along the slice axis and that has to be axis 0:

| plane | slices stack along | axis order |
|---|---|---|
| axial | head↔toe (`S`) | `S, R, A` |
| sagittal | left↔right (`R`) | `R, S, A` |
| coronal | front↔back (`A`) | `A, S, R` |

`read_canonical` and `plane_order` are shared with `preprocess_volume`, so a ground-truth volume and
a generated one cannot end up in different orders.

> **Spacing is `(S, R, A)` regardless of the array's order.** It is permuted once, in
> `read_canonical`, and never again — so for a coronal series `spacing[0]` is the `S` spacing while
> array axis 0 is `A`. Deliberate: spacing exists only to go into the conditioning text, which
> wants `(S, R, A)`. Never index it with an array axis.

## Prediction setup

Each case is derived from one read of its MR-RATE series:

- **Conditioning** — `encode_conditioning`, the same call preprocessing makes: frozen CXR-BERT over
  the study's report plus the series' `[MODALITY]`/`[PLANE]`/`[SPACING]` markers, pooled to one
  768-d token and L2-normalized. Identical to what the model trained against.
- **Regime** — one entry in `REGIMES`. The challenge is report-to-volume, so `full-body` is the
  default. Adding a regime is adding an entry; each receives a zero-argument `gt_latent()` so a
  regime that needs no ground truth never pays to encode one.

  | `--regime` | seed block | for |
  |---|---|---|
  | `full-body` | the black boundary token | the challenge task |
  | `gt-head` | the volume's own first block | diagnosing the rollout |

- **Rollout** — `LatentAutoregressiveGenerator`, the same one `auto_regressive_generate/main.py`
  builds: 201-step Euler per block, fp32, one case per rollout, stopping when a block matches the
  white boundary token. `--max_blocks` defaults to `max_slices / target_nframes` (20).

  Two speed switches, and the difference between them matters. **`--compile`** (on) compiles the
  denoiser with static shapes and leaves the sampler alone — measured, `torch.compile` contributes
  nothing to the numerical drift (1.91e-2 relative velocity difference with bf16 on, against
  1.90e-2 for bf16 alone) — so it is on by default; pass `--no-compile` for a smoke test, since the
  first two or three blocks of any run are spent compiling. **`--bf16`** (off) is ~3.2x faster per
  block and is the precision training itself ran in, but it does *not* reproduce an fp32 sample:
  one denoiser call differs by 1.9e-2 relative, and 201 steps x 6 blocks amplify that to a volume
  23.6 dB from its fp32 counterpart. So it stays off, and if you turn it on, turn it on for both
  sides of a comparison.

  Batching several volumes into one rollout is deliberately **not** available here, although it is
  another 1.7x: a batch draws all of its noise in one call, so noise could only be seeded per batch
  rather than per case, and two runs with different groupings would score different volumes.
- **Seeding** — per case, from the case id rather than its position, so a rerun and a run under a
  different shard count draw the same noise. Verified bit-identical per case.
- **Failures** — an unreadable series or a rollout that collapses to nothing is logged and counted
  as a missing output, never allowed to lose the shard.

**Sharding.** A full-body rollout is 20 blocks of 201 steps, so shard as wide as the queue allows.
Each task writes `shard-NNNN.pt` (its per-case scores plus its accumulated FID features); the
`--combine` pass pools them. FID is pooled at the feature level, so the per-plane distances are
computed over every shard's slices at once rather than averaged per shard.

## Metrics

| metric | what it is |
|---|---|
| `MSE_mean` / `PSNR_mean` / `SSIM_mean` | per case on the normalized pair, then averaged |
| `FID_2p5D_XY` / `_XZ` / `_YZ` | Frechet distance over squeezenet1_1 features of every 4th slice, per array axis |
| `FID_2p5D_Avg` | mean of the three |
| `dice` | a literal copy of `SSIM_mean` — the platform's primary-metric shim, not real Dice |
| `n_total_files` etc. | cases seen, scored, missing, and excluded by modality |

**Scope.** Only T1w/T2w/FLAIR/SWI are scored, matching the organizers' decision. Other modalities
are counted in `n_excluded_out_of_scope_modality` and skipped before generation.

**Missing cases are dropped from the MSE/PSNR/SSIM means**, not penalized with a worst-case value —
that is what the official aggregation actually does, despite what its per-case record suggests.

**`METRIC_KEYS` is also the reporting order** — FID average, PSNR, SSIM, MSE, the per-plane FIDs,
then `dice` and the counts — and it drives the console dump, `metrics.json` and the W&B table at
once.

Two caveats worth knowing. The FID plane labels are **nominal**: the official code names planes
after array axes, so with our layout `FID_2p5D_YZ` is the acquisition-plane view. It is applied
identically to real and generated volumes, so the metric is sound; the official container is no
better defined, since it loads each `.nii.gz` with no reorientation at all. And `n_total_files`
counts eligible MR-RATE series from `list_series` (report present, not derived, not a localizer, no
duplicate acquisitions), not files in the platform's ground-truth directory.

**Do not "improve" [`challenge.py`](challenge.py).** Its quirks are the leaderboard's arithmetic.
The only sanctioned additions are marked in the file: `raw_features`/`finalize_pooled` for
cross-shard FID pooling, and `_matrix_sqrt`, which drops a `sqrtm` kwarg scipy ≥ 1.17 removed.

## Output

```
<output_dir>/eval/<regime>-<split>/
├── shard-0000.pt ...        per-shard scores + FID features (input to --combine)
├── metrics.json             {"metrics": {...}, "per_case": [...], regime, split, ckpt}
└── examples/                a few ground-truth | generated mp4s (--examples)
```

W&B gets a `challenge_metrics` table, the same values in the run summary and as scalars, and the
example videos — matching the `R2V-MR-Generation` baseline's panels so the two models' runs read
side by side.
