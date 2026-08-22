# Evaluation

Roll out MRFlow over an MR-RATE split and score it with the **official VLM3D
`mr-volume-generation` metrics**. The metric code in [`challenge.py`](challenge.py) is a port of the
challenge's own scoring container, so the numbers here are the leaderboard's, not our own
definitions.

No preprocessed split is needed — every case is read straight out of the raw MR-RATE tars — so
`--split test` runs exactly like `--split val`.

```bash
# one array task per shard, then one pass to pool them
sbatch --array=0-15 slurms/mrflow_eval_helma.sh <experiment>/config.yaml \
    <experiment>/checkpoint-N/denoiser_ema --split val
sbatch slurms/mrflow_eval_helma.sh <config> <ckpt> --split val --combine

# quick local check, one GPU, 8 cases
python evaluation/main.py --config <experiment>/config.yaml \
    --ckpt <experiment>/checkpoint-N/denoiser_ema --split val --limit 8
```

Pass the config **saved into the experiment dir**, next to the checkpoints — it has the paths and
the architecture the checkpoint was trained with.

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
  builds: 201-step Euler per block, stopping when a block matches the white boundary token.
  `--max_blocks` defaults to `max_slices / target_nframes` (20).
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
