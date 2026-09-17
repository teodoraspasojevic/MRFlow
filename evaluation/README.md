# Evaluation

Roll out MRFlow over an MR-RATE split and score it with **four metric families at once**.

- **`fid_2d_inception`, FVD and Inception Score**, each computed the way its own reference
  implementation computes it — pytorch-fid, StyleGAN-V and torch-fidelity respectively. These are
  the numbers the work is read by, and they are the field's standard protocols *unmodified*. Both
  volumes reach them in the 1 mm isotropic / 256² grid the model was trained in.
- **`fid_3d_medicalnet` and `fid_2p5d_radimagenet_*`**, from [`medical_fid.py`](medical_fid.py):
  the same Fréchet arithmetic on medical-image backbones, following the protocol CCELLA and
  Alignment-to-Synthesis use. `fid_3d_medicalnet` is **volume-level** — one feature per volume, not
  per slice — and the 2.5D one scores the three anatomical planes separately.
- **`hlip_*`**, from [`hlip_metrics.py`](hlip_metrics.py): the MR-RATE-trained HLIP
  vision-language model asked whether the generated volume matches the report it was generated
  from, as a paired cosine and as report→volume retrieval. **The only metric here that reads the
  conditioning at all**; every other one would score the same if the reports had been shuffled.
- **MSE/PSNR/SSIM and the squeezenet 2.5D FID**, from
  [`challenge_metrics.py`](challenge_metrics.py), the VLM3D scoring container vendored. The
  challenge is over so these are nobody's reference any more, but they are still worth having and
  cost one extra pass over volumes already in memory. They see the pair exactly as the leaderboard
  did.

**The container family is given different arrays from everything else, and that separation is the
point** — see [the geometry contract](#the-geometry-contract). Nothing from the container may reach
the reference protocols.

**Three FIDs, three different metrics.** `fid_2d_inception` is slice-level on a natural-image
backbone, `fid_2p5d_radimagenet_*` slice-level on a radiology backbone in three planes,
`fid_3d_medicalnet` volume-level on a 3D medical backbone. Their scales are unrelated and none of
them converts into another. The old generic key `FID` is gone — it was always the 2D one.

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

# 2. score every cached volume into metrics.json and log to W&B, after step 1 finishes
sbatch --dependency=afterany:$JOB slurms/mrflow_eval_helma.sh \
    $CONFIG $CKPT --split val --combine --out $OUT
```

**`--limit` is per task**, so cases = array size × limit. Drop it to run the whole split.

Three knobs on the feature extractors: `--feature_batch_size` (slices per forward pass, default 32)
and `--hlip_batch_size` (volumes per forward pass, default 8) for a smaller GPU, and `--no-hlip` to
skip the HLIP tower entirely. Everything runs on CPU too, just slowly — `device="auto"` picks
whatever is there.

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

**Why two commands.** Each array task only *generates*: it caches its rollouts as `.npy` and
writes a `shard-NNNN.json` manifest. None of them knows it is the last to finish, and none of them
has seen the whole population, so scoring has to be a separate job that starts afterwards and reads
every cached volume. A Fréchet distance cannot be averaged per shard.

**Only `--combine` logs to W&B.** The array tasks write files and nothing else. The W&B run is named
after the config's `wandb_args.name`, so give a config copy its own name when the checkpoint is not
from that experiment, or the run borrows a training run's name.

Pass the config **saved into the experiment dir**, next to the checkpoints — it has the paths and
the architecture the checkpoint was trained with. `--out` defaults to `<output_dir>/eval/<regime>-<split>`;
pass it explicitly for a checkpoint that is not from a finished experiment, so the run does not
create that experiment's directory.

Everything lands in `$OUT`: `generated/*.npy` (the cache), one `shard-NNNN.json` manifest per task,
`examples/*.mp4`, and `metrics.json`. **Check `n_total_files` in `metrics.json` matches the case
count you asked for** — `--combine` scores whatever manifests it finds, so a task that died leaves
a quietly smaller evaluation.

| file | what |
|---|---|
| [`paper_metrics.py`](paper_metrics.py) | the 1 mm/256² canonicalization, then `fid_2d_inception`, FVD and IS by their reference protocols |
| [`medical_fid.py`](medical_fid.py) | `fid_3d_medicalnet` and `fid_2p5d_radimagenet_*`: weights, preprocessing, per-plane accumulators |
| [`medicalnet_resnet.py`](medicalnet_resnet.py) | MedicalNet's 3D ResNet encoder, vendored. Do not edit |
| [`hlip_metrics.py`](hlip_metrics.py) | `hlip_*`: the HLIP tower, report/volume embeddings, paired cosine and retrieval |
| [`fid_inception.py`](fid_inception.py) | pytorch-fid v0.3.0 verbatim — the TF-ported Inception and its Frechet distance. Do not edit |
| [`__init__.py`](__init__.py) | `EvalAccumulator` — canonicalizes each pair, accumulates features, `results()` turns them into the metrics dict |
| [`main.py`](main.py) | the CLI: build a case, generate, score, write `metrics.json`, log to W&B |
| [`challenge_metrics.py`](challenge_metrics.py) | the VLM3D scoring container, vendored: modality scope, MSE/PSNR/SSIM, streaming 2.5D FID. Do not "improve" it |
| [`../tests/test_evaluation_metrics.py`](../tests/test_evaluation_metrics.py) | value tests: the reference protocols hold, identical pairs score zero, ladders order, pooling is shard-invariant |
| [`../tests/test_medical_metrics.py`](../tests/test_medical_metrics.py) | value tests for the medical FIDs and HLIP: shapes, preprocessing symmetry, three-plane slicing, retrieval arithmetic, cache invalidation |

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
  different shard count draw the same noise.

  **It is not bit-reproducible, though, and the noise floor is worth knowing before you read a
  sweep.** Measured on the same 4 cases run twice (1 shard vs 2): identical rollout *lengths* —
  so the same noise draw and the same stop block — but the volumes sit ~50-56 dB apart and 6-14%
  of voxels differ, because the denoiser's GPU kernels are not deterministic across processes
  (cuBLAS/cuDNN autotuning, non-deterministic reductions) and 201 Euler steps x ~10 blocks
  accumulate it. That is a *far* smaller perturbation than the 23.6 dB `--bf16` costs, and it moves
  the metrics by ~0.13% on `FVD_f16` and ~5e-5 relative on `fid_2d_inception`.

  **So: a difference between two cells smaller than ~0.2% on FVD is noise, not signal.** Forcing
  `torch.use_deterministic_algorithms(True)` would fix it but has no kernel for the xformers
  attention the denoiser uses, and would invalidate every number measured so far.
- **Failures** — an unreadable series or a rollout that collapses to nothing is logged and counted
  as a missing output, never allowed to lose the shard.

**Sharding.** A full-body rollout is 20 blocks of 201 steps, so shard as wide as the queue allows.
Each task writes its rollouts to `generated/` and a `shard-NNNN.json` manifest; the `--combine`
pass reads every one of them. **The shard count cannot change a number**, because every distance is
computed once by a single process that has seen the whole population — there is no pooling contract
to keep. `IS_std` is no longer shard-dependent either.

Two things to size, and they pull in opposite directions:

- **Disk**, for the volume cache: ~21 MB and one inode per case, so a 1,000-case run is ~21 GB and
  1,000 inodes (against this account's 102,400 on `/hnvme`). Check before caching a full split.
- **Wall clock in `--combine`**, which is now a sequential pass over the whole population on one
  GPU: re-read and canonicalize each ground truth, then run six extractors over the pair. Budget
  hours for a 1,000-case run, against minutes when the feature rows were shipped per shard. That is
  the deliberate trade for never persisting a feature.

Peak memory in that process is the feature rows it holds: ~4.5 GB at 1,000 cases.

## Metrics

**The numbers the work is read by are computed the way their own reference implementations
compute them.** Where a reference exists it is used rather than reimplemented, and every place we
deviate is named. The references, read from source:

| metric | reference implementation | what it pins |
|---|---|---|
| `fid_2d_inception` | [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) **v0.3.0**, vendored verbatim as [`fid_inception.py`](fid_inception.py) | backbone `pt_inception-2015-12-05` (the TF port, **not** torchvision's ImageNet Inception), 2048-d final pool, images as `[0, 1]` floats the module itself resizes to 299² (bilinear, `align_corners=False`, no antialias) and rescales to `[-1, 1]`, **every image** in the set, `eps·I` only when `sqrtm` returns non-finite |
| `FVD_f16`, `FVD_f64` | [StyleGAN-V](https://github.com/universome/stylegan-v) `src/metrics/frechet_video_distance.py` | Kinetics-400 I3D torchscript at its 400-d pre-softmax layer, called `rescale=True, resize=True, return_features=True` so the detector does its own `x/255*2-1` and its own bilinear resize to 224², **one clip of `num_frames` consecutive frames per video, no overlap**, short videos discarded |
| `IS_mean`, `IS_std` | [`torch-fidelity`](https://github.com/toshas/torch-fidelity) `torch_fidelity/metric_isc.py` | logits (not probabilities), shuffled with `RandomState(2020).permutation(N)`, 10 contiguous splits at `i*N//splits`, KL in float64. The shuffle makes the splits representative; it does **not** make the value order-invariant |
| `fid_3d_medicalnet` | [MedicalNet](https://github.com/Tencent/MedicalNet) 3D ResNet-50, the protocol [CCELLA](https://github.com/grabkeem/CCELLA) ([arXiv:2506.10230](https://arxiv.org/html/2506.10230)) and Alignment-to-Synthesis ([arXiv:2506.00633](https://arxiv.org/html/2506.00633)) use | `resnet_50_23dataset.pth` (the 23-dataset pretrain), `layer4` map globally average-pooled to 2048-d, one vector **per volume**, the encoder frozen and in eval mode |
| `fid_2p5d_radimagenet_*` | [RadImageNet](https://github.com/BMEII-AI/RadImageNet) ResNet-50, the Report2CT ([arXiv:2509.14780](https://arxiv.org/pdf/2509.14780)) / Alignment-to-Synthesis 2.5D protocol | the official PyTorch `ResNet50.pt`, `layer4` map globally average-pooled to 2048-d, one vector **per slice**, three anatomical planes accumulated separately, `_mean` their unweighted mean |
| `hlip_*` | [HLIP](https://github.com/zch0414/hlip), built from its own [model card](https://huggingface.co/zch0414/clip-vit_base-scan_study-dualdinotxt1568) | the MR-RATE-trained checkpoint `zch0414/clip-vit_base-scan_study-dualdinotxt1568` at revision `6e4b8ab1`, its tokenizer, its `loader` preprocessing, its `model_configs/*.json` architecture, `encode_text` and the image prefix token each variant was trained against (`[:, 1]` findings, `[:, 0]` impression), all L2-normalized |

The three image references consume **uint8** images, so both volumes are percentile-normalized to
`[0, 1]` and then quantized to uint8 before Inception, the I3D or RadImageNet see them. MedicalNet
is the exception: it is a volume network whose own pipeline z-scores over the foreground, so it
gets that instead — see [`medical_fid.py`](medical_fid.py), which states every such decision.

The container's MSE/PSNR/SSIM and 2.5D FID are computed alongside, on their own arrays, and are
marked **container** below — they follow no reference but the leaderboard's, see [the container
metrics](#the-container-metrics-and-why-they-stay-separate).

| metric | what it is |
|---|---|
| `fid_2d_inception` | Frechet distance over Inception pool3 features of **every** acquisition-plane slice |
| `FVD_f16` | Frechet distance over I3D features, **one 16-slice clip per volume** — the standard protocol |
| `FVD_f64` | the same with 64-slice clips — **ours**, four generated blocks, reads coherence across block boundaries |
| `IS_mean` / `IS_std` | Inception Score of the generated slices; no ground truth involved |
| `fid_3d_medicalnet` | Frechet distance over MedicalNet features, **one per volume** — so the sample count is the case count, like FVD |
| `fid_2p5d_radimagenet_axial` / `_coronal` / `_sagittal` | Frechet distance over RadImageNet features of **every** slice of that anatomical plane |
| `fid_2p5d_radimagenet_mean` | the unweighted arithmetic mean of those three; a plane that is undefined drops out rather than dragging it to nan |
| `hlip_<variant>_volume_cosine_mean` / `_std` | cosine between each generated volume and **its own** text, for `variant` in `findings` / `impression`, each against the image token it was trained against |
| `hlip_<variant>_to_volume_r1` / `_r5` / `_r10` | text→volume retrieval: the fraction of queries with a correct volume in the top K |
| `hlip_<variant>_to_volume_r*_within_stratum` | the same, with candidates restricted to the query's own **modality *and* plane** — the harder and more informative number |
| `hlip_<variant>_positives_per_query_mean` | how many correct volumes each query had |
| `n_fid_2d_inception_slices_real` / `_fake`, `n_fvd_*_clips_real` / `_fake`, `n_is_slices`, `n_fid_3d_medicalnet_volumes_*`, `n_fid_2p5d_radimagenet_<plane>_*` | samples behind each number — read these against the 2048/400 feature dimension, and against `n_scored_files` for the study count |
| `n_hlip_<variant>_queries` / `_candidates` / `_pairs` | **read these before reading an R@K**: it is capped at `min(K, n_candidates)`, so R@5 and R@10 are uninformative below 5 and 10 candidates |
| `n_hlip_<variant>_excluded_no_text` / `_truncated` / `_token_collisions` | cases with no text for that variant (excluded, never scored against an empty string), strings past HLIP's 256-token context, and ids whose truncated tokens are indistinguishable |
| `MSE_mean` / `PSNR_mean` / `SSIM_mean` | **container**: per case on its own normalized pair, then averaged over the scored cases |
| `FID_2p5D_XY` / `_XZ` / `_YZ` / `_Avg` | **container**: Frechet distance over squeezenet1_1 (512-d) features of every 4th slice, per array axis, and their mean |

**Both sample counts are reported for every distance**, because nothing forces a rollout to the
reference's length: a generation that stopped early contributes fewer slices, and one shorter than
a clip contributes no clip at all. Equal counts are a property of the run, not of the code.

### The protocol, metric by metric

Everything a reader needs to know whether a number of ours is comparable with one of theirs.

| metric | extractor / pinned checkpoint | intensity in | resize / resample | channels | slices or frames used | dim | one sample is | weighting | known limitations |
|---|---|---|---|---|---|---|---|---|---|
| `fid_2d_inception` | pytorch-fid v0.3.0 `pt_inception-2015-12-05` | `[0, 1]` → uint8 → `[0,1]`, module rescales to `[-1,1]` | module's own 299², bilinear, no antialias | grey ×3 | **every** slice of array axis 0 | 2048 | one slice | longer volume → more rows | acquisition plane only; our 1 mm/256² geometry, not native |
| `fid_2p5d_radimagenet_{axial,coronal,sagittal}` | official RadImageNet PyTorch `ResNet50.pt` (sha256 `08629f7e…`) | `[0, 1]` → `2x − 1` | 224², bilinear | grey ×3, reversed to BGR | **every** slice of that anatomical axis | 2048 | one slice | longer volume → more rows | not interchangeable with CCELLA's RadImageNet FID (different input mapping) |
| `fid_2p5d_radimagenet_mean` | — | — | — | — | — | — | unweighted mean of the three FIDs above | — | planes are never pooled into one FID |
| `fid_3d_medicalnet` | MedicalNet `resnet_50_23dataset.pth` (sha256 `ff48a622…`) | z-score over nonzero voxels | trilinear to 128³ | 1 | **no slice selection**: the whole volume | 2048 | **one volume** | one row per study | 128³ stretches the slice axis, so length is not read here |
| `FVD_f16` / `FVD_f64` | StyleGAN-V Kinetics-400 I3D torchscript | `[0, 1]` → uint8 → `[0,255]`, detector rescales | detector's own 224², bilinear | grey ×3 | **16 / 64** consecutive frames, centred, one clip | 400 | one clip = one volume | one row per study | the frame count is part of the metric's definition and is deliberately *not* all-slices |
| `IS_mean` / `IS_std` | the same TF Inception, 1008-way logits | `[0, 1]` → uint8 | 299², bilinear | grey ×3 | **every** slice of array axis 0 | 1008 | one slice | longer volume → more rows | no ground truth involved |
| `hlip_findings_*` | HLIP `clip-vit_base-scan_study-dualdinotxt1568` @ `6e4b8ab1` (MR-RATE-trained) | `[0, 1]`, then the official loader's scalar ImageNet normalize | pad to square → 256² → **full-depth deterministic resample of the whole slice axis to 48** (`nearest-exact`) → centre crop 224² | 1 | all of them, resampled to **48**: an architectural constant (`img_size=(48,224,224)`, patch `(6,16,16)` → 1568 tokens), not a sampling policy | 768, `image_features[:, 1]` | **one volume = one study of one scan** | one row per study | `"This study looks like: " + findings`, HLIP's own template; identity is `study_uid` |
| `hlip_impression_*` | the same | the same | the same | 1 | the same | 768, `image_features[:, 0]` | the same | one row per study | `"This study shows: " + impression`, HLIP's own template; identity is `study_uid` |
| `FID_2p5D_*`, `MSE`/`PSNR`/`SSIM` | VLM3D container, squeezenet1_1 | container's per-slice window | 224² | grey ×3 | every 4th slice of each *array* axis | 512 | one slice | — | the leaderboard's own quirks, kept unchanged |

**The all-slice rule.** Every slice is used unless the extractor architecturally forbids it or
using all of them would redefine a named metric. The two exceptions are stated above and nowhere
else: HLIP's 48-slice input is fixed by its position embeddings, and FVD's 16/64 frames are part of
what "FVD" means — changing them would make the number incomparable with every published FVD.

**The paired real/generated slice counts are not equal, on purpose.** In the (S, R, A) frame a
volume acquired in plane P contributes its own `T` slices along P's axis and exactly 256 along the
other two, so a plane's real-minus-generated count *is* the summed rollout-length error of the
volumes acquired in that plane. Measured on a 4-case smoke run — per-case length errors +8
(coronal), +26 and +10 (sagittal), −6 (axial) — the plane deltas come out at exactly +8, +36 and
−6. Variable volume length is an intended model output: the stop token decides where a rollout
ends, and `preprocess_volume` never touches `T` for that reason. Equalizing the counts would hide a
real failure mode, so they are reported as they fall.

**Longer volumes weigh more in every slice-level FID, and that is not hidden.** A 158-slice
volume contributes 158 axial rows where a 200-slice one contributes 200. Both the study count
(`n_scored_files`) and the per-plane slice counts (`n_fid_2p5d_radimagenet_<plane>_real` / `_fake`,
`n_fid_2d_inception_slices_real` / `_fake`) are reported, so the weighting is visible in every run.
All volumes reach the extractors on the same 1 mm isotropic / 256² grid — only the slice count
varies — and any baseline compared against these must use the identical manifest and geometry.

### What is ours, and must be declared when these are quoted

- **The geometry.** Both volumes are canonicalized into the model's own 1 mm isotropic / 256²
  training grid rather than the reference being left in its released geometry with the generation
  resampled onto it — see [the geometry contract](#the-geometry-contract). That is the distribution
  the model was asked to match, but it also means these numbers are not comparable to a paper that
  scored native-geometry references.
- **The slice axis.** `fid_2d_inception` and `IS` see slices of array axis 0 only — the acquisition
  plane, the axis the model rolls out along, so the images scored are the images generated.
  `fid_2p5d_radimagenet_*` is the deliberate counterpart: it undoes that permutation and scores all
  three anatomical planes, so the two out-of-plane views are also read.
- **The intensity window**, a 0.5/99.5 percentile per volume.
- **`FVD_f64`**. `FVD_f16` is the only number here computed by an unmodified reference protocol,
  which is why it leads the report.
- **The clip offset.** StyleGAN-V draws a random offset per video; we center it, because these
  "videos" are anatomically ordered rather than arbitrary and a random offset would add sampling
  noise without buying independence.

**Sample counts are the thing to check first.** One clip per volume means the FVD sample count *is*
the case count: 1,000 cases give 1,000 clips against 400 feature dimensions, where StyleGAN-V
quotes 2,048. FID is comfortable by comparison — ~158 slices per case, so ~158k rows against 2,048
dimensions. Below a few hundred samples an FVD stops ordering reliably: measured on single-scale
synthetic volumes, noise at sigma 0.05/0.2/0.8 scored 3159/2841/840, i.e. backwards, and
identically so at N=8 and N=96.

`IS` is compressed for a different reason — ImageNet's 1,000 classes do not describe an MR slice,
and the posteriors carry 3.7 of a possible 6.9 nats — so it sits near 1.6–1.8 and is never a
natural-image number. Read it against another MR run or not at all.

### The medical-backbone FIDs

Same Fréchet arithmetic as everything else — `paper_metrics.pooled`, which is pytorch-fid's —
on two backbones trained on medical images instead of on ImageNet photographs. Both are handed the
pair already canonicalized to 1 mm / 256², so they differ from `fid_2d_inception` only in the
feature extractor and in what one sample is.

| | `fid_3d_medicalnet` | `fid_2p5d_radimagenet_*` |
|---|---|---|
| weights | `resnet_50_23dataset.pth`, sha256 `ff48a622…` — byte-identical to `pretrain/resnet_50_23dataset.pth` in MedicalNet's [official 2.8 GB release](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view), downloaded from the single-file mirror instead | `RadImageNet_pytorch/ResNet50.pt`, sha256 `08629f7e…`, from RadImageNet's [official PyTorch release](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view) |
| architecture | vendored as [`medicalnet_resnet.py`](medicalnet_resnet.py); the checkpoint holds no segmentation head, so it loads with nothing missing and nothing left over | torchvision `resnet50` truncated at `layer4`; the state dict is exactly that `nn.Sequential`, keys `backbone.0…backbone.7` |
| feature layer | `layer4` map, mean over `(D, H, W)` → 2048-d | `layer4` map, mean over `(H, W)` → 2048-d |
| one sample is | one **volume** | one **slice** |
| input | trilinear resample to 128³, then z-score over the **nonzero** voxels — MedicalNet's own `datasets/brains18.py` normalization, minus its random background fill, which would make the metric non-deterministic | `2x − 1` on the `[0, 1]` volume window, greyscale repeated to three identical channels, reversed to **BGR**, bilinear resize to 224² |
| geometry | `plane_order` undone, so every case reaches the 3D convolutions in the same (S, R, A) anatomical frame — a plane-first array is a different axis order per acquisition plane, and a 3D conv is not orientation-equivariant | `plane_order` undone too, so the planes are anatomical: axis 0 of (S, R, A) is axial, axis 1 sagittal, axis 2 coronal |
| sampling | every scored volume, whole, no slice selection | **every slice of every plane** |

Five decisions the references do not pin down, all made in [`medical_fid.py`](medical_fid.py)'s
docstring and repeated here because they change the number:

- **The anatomical frame.** Both extractors undo `plane_order` first. Without it the real
  distribution would be a mixture of three axis orders — an axially acquired series reaches the
  extractor as (S, R, A), a sagittal one as (R, S, A) — and neither a 3D convolution nor a plane
  label is invariant to that.
- **The fixed 128³ grid.** MedicalNet is fully convolutional, so the global pool would accept any
  size — but the receptive field would then cover a different fraction of the anatomy per case, and
  rollouts are not all the same length. CCELLA feeds it one fixed shape because its volumes are one
  shape by construction; here that has to be imposed. The cost: the slice axis is **stretched**, so
  a short rollout is rescaled rather than penalised for its length. Length is read from `FVD` and
  from the sample counts, not from here.
- **The MedicalNet intensity normalization.** CCELLA feeds `[0, 1]` volumes straight in; we follow
  MedicalNet's own data pipeline instead, which is the regime the network's batch-norm running
  statistics were estimated under.
- **The RadImageNet input mapping is the upstream one.** `pytorch_example.ipynb` reads an 8-bit
  image with `cv2.imread` and applies `(image − 127.5) * 2 / 255`, i.e. `[0, 255] → [−1, 1]`; on our
  `[0, 1]` representation that is `2x − 1`. No ImageNet mean subtraction and no torchvision
  normalization on top of it. The `[0, 1]` window is per *volume* — a per-slice min–max would erase
  how much signal each slice carries and would behave differently on real and generated artifacts.
  **CCELLA passes `[0, 1]` straight in and documents no reason for it**, so our RadImageNet FID is
  deliberately *not* numerically interchangeable with CCELLA's published one.
- **Every slice, in all three planes.** `sample_every_k = 1`, `center_slices = false`,
  `drop_empty = false`. The rows are never materialized: `paper_metrics.Moments` streams the mean
  and the second moment, so using every slice costs 33 MB per side per plane whatever the case
  count instead of the ~11 GB the rows would be at 1,000 cases.

**The two out-of-plane views carry a length signal.** Their slices are `T × 256`, and `T` differs
between a ground truth and a rollout, so the resize to 224² stretches the two sides by slightly
different factors. Nothing is padded or cropped to hide that — a model that systematically
generates short volumes should show it — but those two planes are therefore not a pure content
comparison. The plane matching the acquisition is.

The two downloads are cached in the torch hub directory beside the I3D and Inception weights
(`TORCH_HOME` moves them, `MRFLOW_MEDICALNET_PATH` / `MRFLOW_RADIMAGENET_PATH` override the
individual paths for a node with no outbound route) and are **hash-checked on every load**. Neither
ever falls back to random or ImageNet weights: a bad hash or a state-dict mismatch raises.

### The HLIP report–volume metrics

Every other metric in this file would give the same answer if the reports had been shuffled before
generation. These would not.

`hlip_<variant>_volume_cosine_mean` is the cosine between each generated volume and its own text,
both L2-normalized; `_std` is the spread over pairs. The retrieval numbers build the full
text×volume cosine matrix and ask, for each query, whether a correct volume is among the K
highest-scoring candidates.

- **Model.** [HLIP](https://github.com/zch0414/hlip) (TMLR 2026), checkpoint
  [`zch0414/clip-vit_base-scan_study-dualdinotxt1568`](https://huggingface.co/zch0414/clip-vit_base-scan_study-dualdinotxt1568)
  pinned at revision `6e4b8ab1a1330c59f64a72773c454e513591cf89` — the release upstream **trains on
  the MR-RATE training split**, with `--text-process-cfg "sentence and findings"`. Its larger
  sibling `clip-vit_large-scan_study-*` is a stronger brain-MRI tower but was trained on
  institutional data and only *evaluated* on MR-RATE; in-domain is worth more here than raw
  strength, and using it would mean scoring MR-RATE reports with a tower that never read one. The
  build is the model card's: its vendored `hlip` package registers the visual encoder with timm,
  its `model_configs/*.json` is the architecture, its tokenizer files are the tokenizer. Nothing
  pretrained is fetched for either tower, so **every weight comes from the checkpoint**, and
  `load_hlip` raises if any parameter did not — a randomly initialised tower would still produce
  plausible-looking cosines.
- **Input.** The model card's `loader`: `[0, 255]` ÷ 255, pad to square, bilinear resize to 256²,
  `nearest-exact` along the slice axis to 48, centre-crop to 224², normalize by the *scalar* means
  of the ImageNet channel statistics. A generated series is presented as a study holding one scan,
  `[1, 1, 1, 48, 224, 224]`. **No padding and no mask**: the checkpoint is built with
  `max_num_scans=0`, so it has no per-scan position embedding and no attention or padding mask
  anywhere — `num_scans` is read off the input's shape and every slot present is a real scan, which
  is why the official `StudyDataset` runs at `batch_size=1`. A blank scan must therefore never be
  fabricated to pad: `_scan2study` averages prefix tokens over the scan axis, so an empty slot would
  drag the study embedding. The 48-slice depth is an **architectural** cap (`img_size=(48,224,224)`
  at patch `(6,16,16)` → 1568 tokens) and the official loader's `nearest-exact` resample is what
  meets it. What the loader does **not** pin is the intensity mapping — upstream's scans are uint8
  tensors built by data-processing code the repo has since removed — so our 0.5/99.5 percentile
  window over nonzero voxels is not provably their transform. It is the same transform for every
  model scored here, so comparisons hold; absolute cosines are not comparable to HLIP's published
  numbers.
- **Two image embeddings, and each variant is scored against its own.** `dualdinotxt` means the
  visual head emits two prefix tokens (the trunk's `cls_token` and `reg_token`) and upstream trains
  them under **separate contrastive losses**: `image_features[:, 0]` against the impression
  sentence, `image_features[:, 1]` against the findings (`hlip_train/train.py`,
  `image_features_sentence` / `image_features_report`). `VARIANTS` holds that mapping and
  `signature()` records it. Crossing them is silent — the cosines stay in range and merely stop
  meaning what the key name says. Upstream's zero-shot scripts index `[:, 0]` unconditionally
  because every prompt they ship is a `"This study shows: …"` one.
- **Text: HLIP's own MR-RATE templates**, re-encoded by HLIP's own tokenizer and text tower; the
  generator's CXR-BERT *vector* never reaches this file. `findings` is
  `"This study looks like: " + findings` and `impression` is `"This study shows: " + impression` —
  upstream's `get_findings` / `get_impressions`, spaced after the colon where upstream is
  inconsistent about it. Both are identified by `study_uid`: one study's report is one report
  however many series it has. There is **no impression↔findings fallback**: a case missing a
  section is excluded from that variant alone and counted in `n_hlip_<variant>_excluded_no_text`,
  and strings past HLIP's 256-token context are counted in `n_hlip_<variant>_truncated`.
- **The generator's own conditioning string is no longer scored.** `hlip_condition_*` was a third
  variant holding the exact `[MODALITY] … [PLANE] …` + `[FINDINGS]` + `[IMPRESSION]` string the
  model was conditioned on. It is gone: that string is deliberately outside this tower's text
  distribution, and **42.7% of them ran past the 256-token context** (300 val series, median 238,
  p90 403, max 655), so the number reported truncation as much as generation quality.
  `conditioning_uid` survives in `main.py` as manifest provenance — no metric reads it.
- **Positives come from the study id, never from a row index or a string.** A candidate is correct
  for a query when it belongs to that same study; the diagonal is one such positive and is never
  excluded, and a study with several series has several. Queries are the distinct studies, so one
  study contributing several series does not become several identical queries — but it does get
  several chances to be hit, which `hlip_<variant>_positives_per_query_mean` reports. Two studies
  whose reports read alike stay two answers.
- **R@K is capped at `min(K, n_candidates)`** — a four-volume run reports R@5 = R@10 = 1.0, which
  means nothing. Read `n_hlip_<variant>_candidates` first, and use a production pool of hundreds;
  the smoke run checks plumbing only. For the same reason, a comparison across guidance settings
  should be restricted to the cases every setting generated: a collapsed rollout changes the
  candidate pool, not just the score.
- **Does it discriminate at all? Impression does; findings barely.** Measured on 57 real MR-RATE
  val series (ground truth, not rollouts), each variant against the token it is scored on:

  | variant | token | matched | shuffled | σ above shuffled | R@1 (chance 0.018) |
  |---|---|---|---|---|---|
  | `impression` | 0 | 0.1861 | 0.1415 | **4.13** | 0.035 |
  | `findings` | 1 | 0.0818 | 0.0672 | **1.39** | 0.053 |

  Both variants do score highest on the token upstream trained them against (findings reads 1.13σ
  on token 0 against 1.39 on token 1; impression 4.13 on token 0 against 4.03 on token 1) — the
  mapping is right, but the gap between tokens is far smaller than the gap between variants. So
  **read `hlip_impression_*` as the primary number.** The findings head is the one upstream's
  `train.py` trains through a `logit_bias` stand-in it marks FIXME, and every prompt HLIP ships for
  zero-shot work is a `"This study shows: …"` one, so the impression head is the better-exercised
  of the two.

  Even 4.1σ is a ceiling worth keeping in view: at n=57 impression's R@1 is twice chance and
  findings' three times, on *real* scans. A small `hlip_*` gap between two models is noise; only a
  large one is evidence.
  `tests/test_medical_metrics.py::test_matched_beats_shuffled_on_real_mr_rate_volumes` asserts the
  sign on impression at n=64 and reports findings without asserting, so a change that breaks the
  text or image path is caught without the suite going flaky over findings' effect size. At that n,
  1,000 permutations give **p=0.001** for impression (the floor the permutation count allows) and
  **p=0.044** for findings — both positive, an order of magnitude apart in confidence.

- **Retrieval depends on the size of the candidate pool** — more candidates is a harder retrieval —
  which is one more reason scoring is a single pass over the whole population.
- **Independence.** MRFlow conditions on CXR-BERT, so HLIP is an independent judge.
  `warn_if_not_independent` checks `config.mri.text_checkpoint` at start-up and prints a banner if
  that ever stops being true — a model scored by its own conditioning encoder is grading its own
  homework.

`--no-hlip` skips the tower and its 1.5 GB download; the `hlip_*` keys then read nan.

### What is cached

**The generated volumes, and nothing else. No features are ever written to disk.**

`generated/<bucket>-<case>.npy` is the rollout exactly as the model produced it — straight off
`decode_latent`, before any canonicalization, windowing or quantization — stored fp16. Beside it,
`shard-NNNN.json` lists every case the shard saw: its status (`generated`/`missing`/`excluded`) and
the archive, member, plane and spacing that find and orient its ground truth.

That makes a metric change a `--combine` away rather than another GPU-week, and it removes a whole
class of failure: there is no stale feature cache that could be pooled across a backbone change, so
nothing has to be invalidated. `metrics.json` still records which extractors produced it, under
`extractors` — the checkpoint hashes, feature layers, preprocessing and target shapes — so two runs
made under different settings are told apart by their own contents.

fp16 is finer than the uint8 every feature extractor quantizes to (max error 2.4e-4 against a
3.9e-3 quantization step). It does mean MSE/PSNR/SSIM see the fp16 round-trip, which is *why* a
rescore reproduces the original run exactly rather than approximately.

### The container metrics, and why they stay separate

[`challenge_metrics.py`](challenge_metrics.py) is computed and logged, but it is **not** a
reference for anything: a squeezenet1_1 2.5D FID that nobody publishes against, `stride=4` slice
sampling, a per-slice intensity window, and per-case MSE/PSNR/SSIM against one particular patient's
scan, which a report-to-volume model has no reason to reproduce. They are reported because they are
cheap and were the challenge's own numbers, and they are read as such.

What may never happen is the reverse of the old arrangement: **none of those habits may reach
[`paper_metrics.py`](paper_metrics.py) or [`medical_fid.py`](medical_fid.py)**, whose entire value
is that the distances there are standard protocols on standard backbones. Concretely —
`compute_basic_metrics` and `FIDAccumulator` are handed `real` and `produced` exactly as they
arrived, every other metric is handed the canonicalized pair, and
`PAPER_KEYS`/`MEDICAL_KEYS`/`HLIP_KEYS`/`CHALLENGE_KEYS` record which family each key belongs to,
which is what says the arrays it was computed on. `test_the_challenge_family_scores_the_released_pair_not_the_canonicalized_one`
is the guard.

`dice` and `FID_T` are gone: both were aliases rather than computations (`dice` a copy of
`SSIM_mean` for the platform's primary-metric slot, `FID_T` a copy of `FID_2p5D_YZ` under the
paper's name). With a reference-protocol slice-axis `fid_2d_inception` now present, a second one
under a third name is only confusing. `challenge_metrics.py` still computes everything it ever did.

**`FID_2p5D_*` and `fid_2p5d_radimagenet_*` are not two names for one number.** The first is the
container's: squeezenet1_1, 512-d, every 4th slice of an *array* axis, a per-slice intensity
window, on the released pair. The second is RadImageNet, 2048-d, 32 slices of an *anatomical*
plane, a per-volume window, on the canonicalized pair. They are kept apart for the same reason
everything else here is.

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
(median 4.05 mm), and `preprocess_volume` trilinearly upsampled those to 1 mm *for training*. So for
most of the split the 1 mm reference is interpolated, and FVD measures agreement with interpolated
data rather than with a real 1 mm acquisition. MR-RATE contains no real 1 mm T2w at all to check
against. That is a dataset limitation to disclose, not a metric to engineer around. Every case's
native slice spacing is written to `per_case` as `slice_mm`, so how much of a given run sits above
1.5 mm can be read off `metrics.json` rather than assumed.

A `shard-NNNN.json` is written only after its task finishes its cases, so a task that died leaves
no manifest and `--combine` cannot silently score a partial shard. Volumes already on disk are
reused when the task is re-run.

**Four weight downloads, all cached in the torch hub directory** (so `TORCH_HOME` moves them) on
first use, over the proxy the SLURM script exports: Inception (91 MB), the I3D torchscript
(51 MB), MedicalNet (185 MB) and RadImageNet's release zip (195 MB, one 94 MB model kept). HLIP is
a fifth, ~1.5 GB, into the Hugging Face cache. All are already cached on Helma, and a compute node
fetches them itself in seconds. On a node with no outbound route, point `MRFLOW_I3D_PATH`,
`MRFLOW_MEDICALNET_PATH` and `MRFLOW_RADIMAGENET_PATH` at pre-baked copies and set `HF_HUB_OFFLINE`
with a warm HF cache. The two Google-Drive downloads need `gdown`, which is in
[`requirements.txt`](../requirements.txt); HLIP needs `open-clip-torch`.

**Scope.** Only T1w/T2w/FLAIR/SWI are scored. That was the challenge organizers' decision and is
now ours (`IN_SCOPE_MODALITIES` in [`__init__.py`](__init__.py)), kept unchanged so a number stays
comparable with the `R2V-MR-Generation` runs. Other modalities are counted in
`n_excluded_out_of_scope_modality` and skipped before generation.

**A missing case contributes no samples to the distances** — there is nothing to penalize a
distribution distance with — and is **dropped from the MSE/PSNR/SSIM means** rather than given a
worst-case value, which is what the official aggregation actually does. Watch `n_missing_outputs`
and the per-side sample counts.

**`METRIC_KEYS` is the reporting order** — and the only one: it drives the W&B table, the console
dump and `metrics.json` alike. **Every score first, then every count**, so a reader sees the
numbers that mean something before the bookkeeping that qualifies them:

1. `FVD_f16`, `FVD_f64`
2. `fid_2d_inception`, `fid_3d_medicalnet`, `fid_2p5d_radimagenet_{mean,axial,coronal,sagittal}`,
   `FID_2p5D_{Avg,XY,XZ,YZ}`
3. HLIP, most important first and grouped by metric across the three variants so the same number
   reads side by side: `_volume_cosine_mean`, `_to_volume_r{1,5,10}`, then the harder
   `_within_stratum` triple, then `_volume_cosine_std` and `_positives_per_query_mean`
4. `IS_mean`, `IS_std`
5. `PSNR_mean`, `MSE_mean`, `SSIM_mean`
6. every `n_*`: the per-metric sample counts in the same family order, then the run-level file
   counts

44 scores and 37 counts. **Both sample counts are reported for every distance** — nothing forces a
rollout to the reference's length, so **unequal counts are expected** and equalizing them would
hide a real failure mode. `n_hlip_<variant>_{queries,candidates}` matter for a different reason —
an R@K is capped at `min(K, n_candidates)`, so a tiny run reports R@10 = 1.0.

One caveat on the counts: `n_total_files` counts eligible MR-RATE series from `list_series` (report
present, not derived, not a localizer, no duplicate acquisitions), not files in the platform's
ground-truth directory. And **`n_scored_files` counts in-scope cases, not successfully scored ones**
— it is `n_total - n_excluded`, so a missing case is included in it. `n_scored_files -
n_missing_outputs` is how many volumes actually reached a metric.

**Do not edit [`medicalnet_resnet.py`](medicalnet_resnet.py) either.** Same rule, same reason: it
is MedicalNet's encoder, and copying it is what makes our 3D FID the same arithmetic on the same
weights as every FID quoted against MedicalNet.

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
├── generated/               every rollout as fp16 .npy -- the only cache
└── examples/                one ground-truth | generated mp4 per bucket (--examples)
```

**`generated/` is the evaluation's only cache; no features are ever written.** Generation is the
expensive half of an evaluation — 20 blocks of 201 Euler steps per case — while every metric
downstream is minutes of feature extraction, so keeping the rollouts makes the next change to a
metric a rescore rather than another GPU-week. ~21 MB per case, so ~21 GB and 1,000 inodes for a
1,000-case run, well inside this account's 102,400 on `/hnvme`. fp16 is finer than the uint8 every
feature extractor quantizes to anyway (max error 2.4e-4 against a 3.9e-3 quantization step). Each
`shard-NNNN.json` carries the modality, plane, spacing, archive and member for its shard's cases —
everything `--combine` needs to rebuild a pair without re-deriving the population.

W&B gets a `metrics` table, the same values in the run summary and as scalars, and the
example videos — matching the `R2V-MR-Generation` baseline's panels so the two models' runs read
side by side.
