# Evaluation

Roll out MRFlow over an MR-RATE split and score it with **two metric families at once**.

- **Challenge** — MSE/PSNR/SSIM and 2.5D FID, from [`challenge_metrics.py`](challenge_metrics.py),
  a port of the challenge's own scoring container. These numbers are the leaderboard's, not our own
  definitions, and the pair reaches them exactly as the platform would see it.
- **Paper** — slice-axis FID on Inception-v3 pool3, FVD over 16- and 64-slice I3D clips, and
  Inception Score, from [`paper_metrics.py`](paper_metrics.py). These put both volumes in the 1 mm
  isotropic / 256² grid the model was *trained* in, and describe the generative distribution rather
  than agreement with one particular patient's scan.

The two families are given **different arrays on purpose** — see [the geometry
contract](#the-geometry-contract).

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
| [`challenge_metrics.py`](challenge_metrics.py) | vendored official container: modality scope, MSE/PSNR/SSIM, streaming 2.5D FID. Do not "improve" it |
| [`paper_metrics.py`](paper_metrics.py) | ours: the 1 mm/256² canonicalization, slice-axis FID on Inception-v3 pool3, FVD over I3D clips, Inception Score |
| [`__init__.py`](__init__.py) | `ChallengeAccumulator` — feeds each metric family the geometry it is defined on, and pools shards |
| [`main.py`](main.py) | the CLI: build a case, generate, score, write `metrics.json`, log to W&B |
| [`../tests/test_evaluation_metrics.py`](../tests/test_evaluation_metrics.py) | 16 value tests: identical pairs score zero, ladders order, pooling is shard-invariant, the container's golden values hold |

## Preprocessing

**The rule: only one version of the pair is ever read from disk, and each metric family
canonicalizes it its own way.** `run_shard` loads the ground truth once, in its released geometry;
whatever a metric needs beyond that happens inside that metric's module.

| | what happens to it |
|---|---|
| ground truth, as loaded | `load_native_volume`: reorient to RAS, transpose so the slice axis leads. **Nothing else** — no resample, no normalization, no crop or pad. |
| generated, as produced | nothing. 256², 1 mm, `T` slices, however many the stop token emitted. |
| inside the **challenge** metrics | 0.5/99.5-percentile normalize to `[0, 1]`; the generated volume is `zoom`ed onto the ground truth's shape. The reference is never touched — resampling it would score against a different target than the platform does. |
| inside the **paper** metrics | both volumes canonicalized to 1 mm / 256², then normalized — see [the geometry contract](#the-geometry-contract) |

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

- **Conditioning** — `build_conditioner(mri, device)`, the same factory preprocessing uses, so
  whichever configuration `mri.conditioning` names is the one the model trained against: the
  study's report plus the series' `[MODALITY]`/`[PLANE]` markers, either pooled to one 768-d
  CXR-BERT token or split into three 2560-wide section tokens, then L2-normalized.
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

**Two families, two geometries.** The challenge metrics get the pair exactly as the leaderboard
would — ground truth in its released geometry, generated volume `zoom`ed onto it by the official
code. The paper metrics get both volumes canonicalized into the 1 mm isotropic / 256² grid the
model was *trained* in, because that is the distribution it was asked to match. Neither family may
be handed the other's arrays.

### Paper metrics ([`paper_metrics.py`](paper_metrics.py))

| metric | what it is |
|---|---|
| `FID` | Frechet distance over Inception-v3 **pool3 (2048-d)** features of every 4th acquisition-plane slice |
| `FVD_f16` / `FVD_f64` | Frechet distance over I3D features of **clips of 16 / 64 consecutive slices** (stride 8 / 32) |
| `IS_mean` / `IS_std` | Inception Score of the generated slices; no ground truth involved |
| `FID_thin` / `FID_thick`, `FVD_*_thin` / `_thick` | the same, split by the ground truth's **native** slice spacing (≤ 1.5 mm vs above) |
| `n_fid_slices`, `n_fvd_f16_clips`, `n_fvd_f64_clips` | samples behind each distance — read these against the 2048/400 feature dimension |
| `n_thin_cases` / `n_thick_cases` | cases in each stratum |

`FID` uses torchvision's Inception-v3, not the TF-ported `pt_inception-2015-12-05` that
`pytorch-fid` wraps, so it is comparable across our own runs and to other torchvision-based numbers
but not digit-for-digit to a paper quoting the TF port.

**Why clips.** FVD is not defined on a whole volume: the reference implementation cuts videos into
clips and each clip is one sample. `f16` is one generated block, so it reads within-block
continuity; `f64` is four, so it reads whether the rollout stays coherent across block boundaries —
the drift the CT paper's `FVD_f16`/`FVD_f128` pair was built to expose. `f128` does not fit here:
MR-RATE volumes are 143–200 slices at 1 mm, which would leave ~1 clip per case against 400 feature
dimensions. The strides overlap by half so the covariance has enough samples to be an estimate
(~20k `f16` and ~4k `f64` clips at 1,000 cases, against 1,000 under one-vector-per-volume).

**Why the strata.** MR-RATE mixes 2D thick-slice and 3D thin-slice acquisitions *inside every*
`(modality, plane)` bucket — the modal spacing covers only 33–71% of a bucket, and FLAIR/SAGITTAL
spans 0.5–6.5 mm — so contrast is not a proxy for geometry. Both volumes are on the same 1 mm grid
either way; what differs is whether the reference's through-plane detail was **acquired or
interpolated**. The model is told neither (`[SPACING]` is not in the conditioning and there is no
spacing class embedding), so it cannot know which to produce. A model that hedges between the two
shows up as too smooth on `thin` and too detailed on `thick`.

### Challenge metrics ([`challenge_metrics.py`](challenge_metrics.py))

| metric | what it is |
|---|---|
| `MSE_mean` / `PSNR_mean` / `SSIM_mean` | per case on the normalized pair, then averaged |
| `FID_2p5D_XY` / `_XZ` / `_YZ` | Frechet distance over squeezenet1_1 (512-d) features of every 4th slice, per array axis |
| `FID_2p5D_Avg` | mean of the three |
| `FID_T` | a literal copy of `FID_2p5D_YZ` — see below |
| `dice` | a literal copy of `SSIM_mean` — the platform's primary-metric shim, not real Dice |
| `n_total_files` etc. | cases seen, scored, missing, and excluded by modality |

`FID_T` is not a second computation. The official code slices array axis 0 under the label `YZ`, so
that number already **is** slice-axis FID on the container's own backbone; `FID_T` exposes it under
the name the paper uses, and the two are asserted equal. It is also the only official plane worth
quoting: `_XY` and `_XZ` slice along axes whose images *contain* the slice axis, so a 38-slice
reformat and a 200-slice one get squashed to 224² by very different factors, and those two planes
measure that distortion as much as they measure anatomy.

### The geometry contract

| | ground truth | generated |
|---|---|---|
| in-plane | resampled to 1 mm and cropped/padded to 256² with preprocessing's own 15 mm posterior shift, **anti-aliased** | already 1 mm / 256² — untouched |
| slice axis | resampled to 1 mm at its **native extent** (143–200 slices over MR-RATE) | untouched; the stop token decides the length |
| intensity | `_normalize01` after the geometry | `_normalize01` after the geometry |

Three deliberate choices in there:

- **Nothing is stretched to a fixed slice count.** Forcing both sides to one length would spread
  ~158 mm of anatomy over whatever that count implies, and would hide a rollout that generated the
  wrong amount of anatomy. Keeping native extent means a short rollout contributes fewer clips over
  different anatomy, which the metric can still see.
- **The ground truth is never upsampled beyond 1 mm, and the generation is never upsampled at all.**
  Resampling runs finer → coarser on both axes: the ground truth is in-plane finer than 1 mm
  (matrices 186–1024), the generation is finer along z (always 1 mm against a 0.5–6.8 mm reference).
  So nothing is invented on either side.
- **The in-plane resize is anti-aliased**, because a native ground truth shrinks by up to 2.7× on
  the way to 224² while a generated volume never shrinks at all. Un-filtered decimation folds that
  difference into the real features as false structure — measured, at 512→224 anti-aliasing changes
  the pixel std from 0.191 to 0.084, against 0.192 → 0.167 at 256→224. `antialias=` exists only for
  bilinear/bicubic 2D, so in-plane is a 2D resize over the slice batch and the slice axis is
  area-averaged separately (`adaptive_avg_pool3d`), which is anyway the right operation rather than
  a filter: a thick MR slice *is* approximately the integral of the tissue over its thickness.

**Read this before quoting the paper FVD.** 58% of MR-RATE is acquired above 1.5 mm slice spacing
(median 4.05 mm), and `preprocess_volume` trilinearly upsampled those to 1 mm *for training*. So on
the `thick` stratum the 1 mm reference is interpolated, and FVD there measures agreement with
interpolated data rather than with a real 1 mm acquisition. MR-RATE contains no real 1 mm T2w at all
to check against. That is a dataset limitation to disclose, not a metric to engineer around — and
it is why the `thin` stratum, where the reference was genuinely acquired at ≤1.5 mm, is the number
to lead with.

Shard files written under an older metric layout carry no `fid_raw`/`fvd_raw`, so `--combine`
raises on them rather than pooling a partial number. Re-run the array.

**The I3D weights are a download.** `load_i3d` caches the torchscript in the torch hub directory
(so `TORCH_HOME` moves it) on first use, over the proxy the SLURM script exports. It is already
cached on Helma, and a compute node can fetch it itself in ~2 s. Set `MRFLOW_I3D_PATH` to a
pre-baked copy on a node with no outbound route.

**Scope.** Only T1w/T2w/FLAIR/SWI are scored, matching the organizers' decision. Other modalities
are counted in `n_excluded_out_of_scope_modality` and skipped before generation.

**Missing cases are dropped from the MSE/PSNR/SSIM means**, not penalized with a worst-case value —
that is what the official aggregation actually does, despite what its per-case record suggests.

**`METRIC_KEYS` is also the reporting order** — `HEADLINE_KEYS` first (`FVD_f16`, `FVD_f64`, `FID`,
`FID_2p5D_Avg`, `IS_mean`, `PSNR_mean`, `MSE_mean`, `SSIM_mean`), then whatever each family has
left: the strata splits and sample counts, the per-plane FIDs and `dice`, and the file counts. It
drives the console dump, `metrics.json` and the W&B table at once.

One caveat on the counts: `n_total_files` counts eligible MR-RATE series from `list_series` (report
present, not derived, not a localizer, no duplicate acquisitions), not files in the platform's
ground-truth directory. And **`n_scored_files` counts in-scope cases, not successfully scored ones**
— it is `n_total - n_excluded`, so a missing case is included in it. The means average over
`n_scored_files - n_missing_outputs`.

**Do not "improve" [`challenge_metrics.py`](challenge_metrics.py).** All of it is the leaderboard's
arithmetic. The only sanctioned additions are marked in the file: `raw_features`/`finalize_pooled`
for cross-shard FID pooling, and `_matrix_sqrt`, which drops a `sqrtm` kwarg scipy ≥ 1.17 removed.
[`paper_metrics.py`](paper_metrics.py) imports `_normalize01` and `_matrix_sqrt` from it so there is
one definition of each; everything else there is ours and may be changed freely.

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
