# Evaluation

Roll out MRFlow over an MR-RATE split and score it with **two metric families at once**.

- **FID, FVD and Inception Score**, each computed the way its own reference implementation computes
  it — pytorch-fid, StyleGAN-V and torch-fidelity respectively. These are the numbers the work is
  read by, and they are the field's standard protocols *unmodified*. Both volumes reach them in the
  1 mm isotropic / 256² grid the model was trained in.
- **MSE/PSNR/SSIM and the 2.5D FID**, from [`challenge_metrics.py`](challenge_metrics.py), the
  VLM3D scoring container vendored. The challenge is over so these are nobody's reference any more,
  but they are still worth having and cost one extra pass over volumes already in memory. They see
  the pair exactly as the leaderboard did.

**The two families are given different arrays, and that separation is the point** — see [the
geometry contract](#the-geometry-contract). Nothing from the container may reach the reference
protocols.

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
caveat carries over too: a per-bucket FID or FVD compares covariances estimated from one bucket's
samples, so at 100 per bucket only the pooled numbers are trustworthy.

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
| [`paper_metrics.py`](paper_metrics.py) | the metrics: the 1 mm/256² canonicalization, then FID, FVD and IS by their reference protocols |
| [`fid_inception.py`](fid_inception.py) | pytorch-fid v0.3.0 verbatim — the TF-ported Inception and its Frechet distance. Do not edit |
| [`__init__.py`](__init__.py) | `EvalAccumulator` — canonicalizes each pair, accumulates features, pools shards |
| [`main.py`](main.py) | the CLI: build a case, generate, score, write `metrics.json`, log to W&B |
| [`challenge_metrics.py`](challenge_metrics.py) | the VLM3D scoring container, vendored: modality scope, MSE/PSNR/SSIM, streaming 2.5D FID. Do not "improve" it |
| [`../tests/test_evaluation_metrics.py`](../tests/test_evaluation_metrics.py) | value tests: the reference protocols hold, identical pairs score zero, ladders order, pooling is shard-invariant |

## Preprocessing

**The rule: the pair is read from disk once, in its released geometry, and each family
canonicalizes it its own way.** `run_shard` loads the ground truth and nothing else happens to it
there; whatever a metric needs beyond that happens inside that metric's module.

| | what happens to it |
|---|---|
| ground truth, as loaded | `load_native_volume`: reorient to RAS, transpose so the slice axis leads. **Nothing else** — no resample, no normalization, no crop or pad. |
| generated, as produced | nothing. 256², 1 mm, `T` slices, however many the stop token emitted. |
| inside the **reference-protocol** metrics | both volumes canonicalized to 1 mm / 256², percentile-normalized and quantized to uint8 — see [the geometry contract](#the-geometry-contract) |
| inside the **container** metrics | 0.5/99.5-percentile normalize to `[0, 1]`; the generated volume is `zoom`ed onto the ground truth's shape. The reference is never touched — resampling it would score against a different target than the platform did. |

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
- **Regime** — one entry in `REGIMES`. The task is report-to-volume, so `full-body` is the
  default. Adding a regime is adding an entry; each receives a zero-argument `gt_latent()` so a
  regime that needs no ground truth never pays to encode one.

  | `--regime` | seed block | for |
  |---|---|---|
  | `full-body` | the black boundary token | report-to-volume, the real task |
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
Each task writes `shard-NNNN.pt` (its per-case record plus its accumulated feature rows); the
`--combine` pass pools them. Every distance pools at the feature level, so the shard count never
changes it. `IS` is the exception: it shuffles on a fixed seed before splitting, which makes the
ten splits representative rather than contiguous blocks of shards, but a fixed permutation of a
reordered set is still a different assignment — `IS_mean` moves ~0.25% across a 1-vs-32 shard
change, `IS_std` a lot, so quote `IS_std` only between runs of the same shard count.

Feature volume is the one thing to size. Every slice is kept as a 2048-d row per side for FID plus
a 1008-d row for IS, so a 1,000-case run at ~158 slices per case pools to roughly **3 GB** in the
`--combine` process — fine on a compute node, worth knowing before running it on a login node.

## Metrics

**The three numbers the work is read by are computed the way their own reference implementations
compute them.** Where a reference exists it is used rather than reimplemented, and every place we
deviate is named. The references, read from source:

| metric | reference implementation | what it pins |
|---|---|---|
| `FID` | [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) **v0.3.0**, vendored verbatim as [`fid_inception.py`](fid_inception.py) | backbone `pt_inception-2015-12-05` (the TF port, **not** torchvision's ImageNet Inception), 2048-d final pool, images as `[0, 1]` floats the module itself resizes to 299² (bilinear, `align_corners=False`, no antialias) and rescales to `[-1, 1]`, **every image** in the set, `eps·I` only when `sqrtm` returns non-finite |
| `FVD_f16`, `FVD_f64` | [StyleGAN-V](https://github.com/universome/stylegan-v) `src/metrics/frechet_video_distance.py` | Kinetics-400 I3D torchscript at its 400-d pre-softmax layer, called `rescale=True, resize=True, return_features=True` so the detector does its own `x/255*2-1` and its own bilinear resize to 224², **one clip of `num_frames` consecutive frames per video, no overlap**, short videos discarded |
| `IS_mean`, `IS_std` | [`torch-fidelity`](https://github.com/toshas/torch-fidelity) `torch_fidelity/metric_isc.py` | logits (not probabilities), shuffled with `RandomState(2020).permutation(N)`, 10 contiguous splits at `i*N//splits`, KL in float64. The shuffle makes the splits representative; it does **not** make the value order-invariant |

Both references consume **uint8** images, so both volumes are percentile-normalized to `[0, 1]` and
then quantized to uint8 before any feature extractor sees them.

The container's MSE/PSNR/SSIM and 2.5D FID are computed alongside, on their own arrays, and are
marked **container** below — they follow no reference but the leaderboard's, see [the container
metrics](#the-container-metrics-and-why-they-stay-separate).

| metric | what it is |
|---|---|
| `FID` | Frechet distance over pool3 features of **every** acquisition-plane slice |
| `FVD_f16` | Frechet distance over I3D features, **one 16-slice clip per volume** — the standard protocol |
| `FVD_f64` | the same with 64-slice clips — **ours**, four generated blocks, reads coherence across block boundaries |
| `IS_mean` / `IS_std` | Inception Score of the generated slices; no ground truth involved |
| `FID_thin` / `FID_thick`, `FVD_*_thin` / `_thick` | the same, split by the ground truth's **native** slice spacing (≤ 1.5 mm vs above) — **ours** |
| `n_fid_slices_real` / `_fake`, `n_fvd_*_clips_real` / `_fake`, `n_is_slices` | samples behind each number — read these against the 2048/400 feature dimension |
| `n_thin_cases` / `n_thick_cases` | cases in each stratum |
| `MSE_mean` / `PSNR_mean` / `SSIM_mean` | **container**: per case on its own normalized pair, then averaged over the scored cases |
| `FID_2p5D_XY` / `_XZ` / `_YZ` / `_Avg` | **container**: Frechet distance over squeezenet1_1 (512-d) features of every 4th slice, per array axis, and their mean |

**Both sample counts are reported for every distance**, because nothing forces a rollout to the
reference's length: a generation that stopped early contributes fewer slices, and one shorter than
a clip contributes no clip at all. Equal counts are a property of the run, not of the code.

### What is ours, and must be declared when these are quoted

- **The geometry.** Both volumes are canonicalized into the model's own 1 mm isotropic / 256²
  training grid rather than the reference being left in its released geometry with the generation
  resampled onto it — see [the geometry contract](#the-geometry-contract). That is the distribution
  the model was asked to match, but it also means these numbers are not comparable to a paper that
  scored native-geometry references.
- **The slice axis.** `FID` and `IS` see slices of array axis 0 only — the acquisition plane, the
  axis the model rolls out along, so the images scored are the images generated.
- **The intensity window**, a 0.5/99.5 percentile per volume.
- **`FVD_f64`**, and the `thin`/`thick` strata. `FVD_f16` is the only number here computed by an
  unmodified reference protocol, which is why it leads the report.
- **The clip offset.** StyleGAN-V draws a random offset per video; we center it, because these
  "videos" are anatomically ordered rather than arbitrary and a random offset would add sampling
  noise without buying independence.

**Why the strata.** MR-RATE mixes 2D thick-slice and 3D thin-slice acquisitions *inside every*
`(modality, plane)` bucket — the modal spacing covers only 33–71% of a bucket, and FLAIR/SAGITTAL
spans 0.5–6.5 mm — so contrast is not a proxy for geometry. Both volumes are on the same 1 mm grid
either way; what differs is whether the reference's through-plane detail was **acquired or
interpolated**. The model is told neither (`[SPACING]` is not in the conditioning and there is no
spacing class embedding), so it cannot know which to produce. A model that hedges between the two
shows up as too smooth on `thin` and too detailed on `thick`.

**Sample counts are the thing to check first.** One clip per volume means the FVD sample count *is*
the case count: 1,000 cases give 1,000 clips against 400 feature dimensions, where StyleGAN-V
quotes 2,048. FID is comfortable by comparison — ~158 slices per case, so ~158k rows against 2,048
dimensions. Below a few hundred samples an FVD stops ordering reliably: measured on single-scale
synthetic volumes, noise at sigma 0.05/0.2/0.8 scored 3159/2841/840, i.e. backwards, and
identically so at N=8 and N=96.

`IS` is compressed for a different reason — ImageNet's 1,000 classes do not describe an MR slice,
and the posteriors carry 3.7 of a possible 6.9 nats — so it sits near 1.6–1.8 and is never a
natural-image number. Read it against another MR run or not at all.

### The container metrics, and why they stay separate

[`challenge_metrics.py`](challenge_metrics.py) is computed and logged, but it is **not** a
reference for anything: a squeezenet1_1 2.5D FID that nobody publishes against, `stride=4` slice
sampling, a per-slice intensity window, and per-case MSE/PSNR/SSIM against one particular patient's
scan, which a report-to-volume model has no reason to reproduce. They are reported because they are
cheap and were the challenge's own numbers, and they are read as such.

What may never happen is the reverse of the old arrangement: **none of those habits may reach
[`paper_metrics.py`](paper_metrics.py)**, whose entire value is that FID/FVD/IS there are the
standard protocols unmodified. Concretely — `compute_basic_metrics` and `FIDAccumulator` are handed
`real` and `produced` exactly as they arrived, the reference-protocol metrics are handed the
canonicalized pair, and `PAPER_KEYS`/`CHALLENGE_KEYS` record which family each key belongs to, which
is what says the arrays it was computed on. `test_the_challenge_family_scores_the_released_pair_not_the_canonicalized_one`
is the guard.

`dice` and `FID_T` are gone: both were aliases rather than computations (`dice` a copy of
`SSIM_mean` for the platform's primary-metric slot, `FID_T` a copy of `FID_2p5D_YZ` under the
paper's name). With a reference-protocol slice-axis `FID` now present, a second one under a third
name is only confusing. `challenge_metrics.py` still computes everything it ever did.

### The geometry contract

| | ground truth | generated |
|---|---|---|
| in-plane | resampled to 1 mm and cropped/padded to 256² with preprocessing's own 15 mm posterior shift, **anti-aliased** | already 1 mm / 256² — untouched |
| slice axis | resampled to 1 mm at its **native extent** (143–200 slices over MR-RATE) | untouched; the stop token decides the length |
| intensity | `normalize01` (0.5/99.5 percentile) after the geometry, then uint8 | the same |

Three deliberate choices in there:

- **Nothing is stretched to a fixed slice count.** Forcing both sides to one length would spread
  ~158 mm of anatomy over whatever that count implies, and would hide a rollout that generated the
  wrong amount of anatomy. Keeping native extent means a short rollout contributes fewer clips over
  different anatomy, which the metric can still see.
- **The ground truth is never upsampled beyond 1 mm, and the generation is never upsampled at all.**
  Resampling runs finer → coarser on both axes: the ground truth is in-plane finer than 1 mm
  (matrices 186–1024), the generation is finer along z (always 1 mm against a 0.5–6.8 mm reference).
  So nothing is invented on either side.
- **This in-plane resize is anti-aliased, and the feature extractors' own resizes are not.** They
  are different operations. Here a native ground truth shrinks by up to 2.7× on the way to 256²
  while a generated volume is not resized at all, and un-filtered decimation would fold that
  asymmetry into the real features as false structure — measured, at 512→224 anti-aliasing changes
  the pixel std from 0.191 to 0.084, against 0.192 → 0.167 at 256→224. By the time a slice reaches
  Inception or the I3D both sides are 256² and get the identical un-filtered bilinear resize their
  references specify. `antialias=` exists only for bilinear/bicubic 2D, so in-plane is a 2D resize
  over the slice batch and the slice axis is area-averaged separately (`adaptive_avg_pool3d`),
  which is anyway the right operation rather than a filter: a thick MR slice *is* approximately the
  integral of the tissue over its thickness.

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

**Scope.** Only T1w/T2w/FLAIR/SWI are scored. That was the challenge organizers' decision and is
now ours (`IN_SCOPE_MODALITIES` in [`__init__.py`](__init__.py)), kept unchanged so a number stays
comparable with the `R2V-MR-Generation` runs. Other modalities are counted in
`n_excluded_out_of_scope_modality` and skipped before generation.

**A missing case contributes no samples to the distances** — there is nothing to penalize a
distribution distance with — and is **dropped from the MSE/PSNR/SSIM means** rather than given a
worst-case value, which is what the official aggregation actually does. Watch `n_missing_outputs`
and the per-side sample counts.

**`METRIC_KEYS` is also the reporting order** — `HEADLINE_KEYS` first (`FVD_f16`, `FID`, `IS_mean`,
`FVD_f64`, then `FID_2p5D_Avg`, `PSNR_mean`, `SSIM_mean`, `MSE_mean`: the reference protocols lead,
`FVD_f16` first because it is the one computed by an unmodified reference end to end), then the
remainder of `PAPER_KEYS`, the remainder of `CHALLENGE_KEYS`, then the file counts. It drives the
console dump, `metrics.json` and the W&B table at once.

One caveat on the counts: `n_total_files` counts eligible MR-RATE series from `list_series` (report
present, not derived, not a localizer, no duplicate acquisitions), not files in the platform's
ground-truth directory. And **`n_scored_files` counts in-scope cases, not successfully scored ones**
— it is `n_total - n_excluded`, so a missing case is included in it. `n_scored_files -
n_missing_outputs` is how many volumes actually reached a metric.

**Do not edit [`fid_inception.py`](fid_inception.py).** It is pytorch-fid v0.3.0 verbatim, and the
point of copying it is that our FID is the same arithmetic on the same weights as every published
FID. It carries exactly one marked edit — a `_sqrtm` shim dropping the `disp=` kwarg scipy ≥ 1.17
removed (this venv is on 1.18, so the original raises).

**Do not "improve" [`challenge_metrics.py`](challenge_metrics.py) either.** All of it is the
leaderboard's own arithmetic, quirks included. The only sanctioned additions are marked in the file:
`raw_features`/`finalize_pooled` for cross-shard FID pooling, and `_matrix_sqrt`, which drops the
same `sqrtm` kwarg.

## Output

```
<output_dir>/eval/<regime>-<split>/
├── shard-0000.pt ...        per-shard scores + accumulated features (input to --combine)
├── metrics.json             {"metrics": {...}, "per_case": [...], regime, split, ckpt}
├── generated/               every rollout as fp16 .npy, plus index-NNNN.json per shard
└── examples/                one ground-truth | generated mp4 per bucket (--examples)
```

**`generated/` is on by default** (`--no-cache_generated` turns it off). Generation is the
expensive half of an evaluation — 20 blocks of 201 Euler steps per case — while every metric
downstream is minutes of feature extraction, so keeping the rollouts makes the next change to a
metric a rescore rather than another GPU-week. ~21 MB per case, so ~21 GB and 1,000 inodes for a
1,000-case run, well inside this account's 102,400 on `/hnvme`. fp16 is finer than the uint8 every
feature extractor quantizes to anyway (max error 2.4e-4 against a 3.9e-3 quantization step). Each
`index-NNNN.json` carries the modality, plane, spacing, archive and member for its shard's cases —
everything needed to rebuild a pair without re-deriving the population.

W&B gets a `metrics` table, the same values in the run summary and as scalars, and the
example videos — matching the `R2V-MR-Generation` baseline's panels so the two models' runs read
side by side.
