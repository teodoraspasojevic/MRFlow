# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CTFlow (ICCV 2025 VLM3D workshop): a Spatial-Temporal DiT trained with **flow matching** to synthesize 3D CT volumes **block-by-block, auto-regressively**, conditioned on a CT-report text embedding. Everything happens in FLUX-VAE latent space — the model never sees pixels during training.

**MRFlow** is the MRI branch of the same code: the identical STDiT + flow-matching trunk trained on MR-RATE brain MRI, conditioned on a frozen report embedding, still in FLUX-VAE latent space. It is not a separate package — MR support is `echosyn/common/mrrate.py`, one extra dataset class, one preprocessing script, and its own configs. Both paths share `lvfm/train.py` unchanged. Two report conditionings exist, selected by `mri.conditioning` — one pooled CXR-BERT token, or three sectioned 2560-wide tokens; see **Text** under Architecture.

## Running things

There is no package install (`no setup.py`/`pyproject.toml`), no test suite, and no linter config. Always run from the repo root with the root on `PYTHONPATH` — `echosyn` and `auto_regressive_generate` are imported as top-level packages.

Training (single node, 4 GPUs):
```bash
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
    lvfm/train.py --config lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all.yaml
```
Add `--no_wandb` to run with wandb in disabled mode (the tracker is still initialized).

Multi-node SLURM: `sbatch slurms/mnode_launcher_helma.sh` → `srun slurms/trainer_helma.sh` per node. Inference sweep: `sbatch slurms/submit_val_infer.sh` → 64 ranks of `slurms/infer_worker.sh`.

### MRFlow on Helma

Preprocess first (task 0 also writes the boundary latents, so let it finish), then train:
```bash
sbatch --array=0-3  slurms/mrflow_preprocess_helma.sh val   --zip
sbatch --array=0-63 slurms/mrflow_preprocess_helma.sh train --zip
python tools/mrflow_verify.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
    --split train --num_shards 64          # gate: exits 1 if anything is missing
sbatch slurms/mrflow_train_helma.sh
```

`--zip` is not optional on this account — see the inode gotcha below. Anything after the split is
passed through to `preprocess_mrrate.py`, so `--overwrite`, `--limit` and `--embeddings_only` work
the same way.

**Changing the conditioning does not mean re-encoding the volumes.** The report is the only input
that changed, so `--embeddings_only` re-encodes just the embeddings into `mri.embedding_root`,
leaving the ~6 TB of latents under `mri.dataset_root` alone. Measured on an A100, the sectioned
conditioning is **123 ms per series** (three encoders over three sections) against 17 ms for the
pooled one, so the train split is **~20 GPU-hours** plus the tar reads, against the ~290 the volume
pass costs — 15x cheaper, not free. One 64-task array is ~18 min per task. The dataset joins the two trees on `sample_id` and raises at
construction if any encoded series has no embedding; `mrflow_verify.py` checks the same thing when
the config declares an `embedding_root`, so it stays the gate:

```bash
# The preprocess script's $1 is the split, so its config is an env override -- unlike the train
# and eval scripts, which take it as $1. Without MRFLOW_CONFIG it runs the pooled config, whose
# mri block has no embedding_root, and --embeddings_only refuses to start.
sbatch --export=ALL,MRFLOW_CONFIG=lvfm/configs/mrflow_from_scratch_sectioned.yaml \
    --array=0-63 slurms/mrflow_preprocess_helma.sh train --zip --embeddings_only
python tools/mrflow_verify.py --config lvfm/configs/mrflow_from_scratch_sectioned.yaml \
    --split train --num_shards 64
sbatch --nodes=16 slurms/mrflow_train_helma.sh lvfm/configs/mrflow_from_scratch_sectioned.yaml
```

**Always run `tools/mrflow_verify.py` between preprocessing and training.** Nothing else notices an
array task that never finished: the dataset globs `manifest/*.csv` and raises only on *zero* rows,
so 63 of 64 shards trains quietly on 98% of the data. The verifier recomputes the expected series
list (deterministic — `list_series` shuffles on a fixed seed, then truncates), compares it against
the manifests and the zip central directories, and exits non-zero on a missing manifest, an
artifact a manifest references but no container holds, a volume under `2 * target_nframes` slices,
a bundle no manifest points at (a task killed between `store.close()` and `write_manifest`), or a
missing boundary latent. It needs no GPU and decodes no volumes. `--num_shards` is required, not
inferred: a missing *final* shard is invisible if you only look at the manifests that exist.
Multi-node needs no second script — `sbatch --nodes=4 slurms/mrflow_train_helma.sh` re-execs under `srun` and derives `machine_rank` from `SLURM_NODEID`.

**This account has h200 GPUs but no h100 allocation.** Always request `--partition=h200 --gres=gpu:h200:N`. An h100 request fails two different ways: `-p h100` is rejected at submit ("Invalid account or account/partition combination"), while `-p preempt --gres=gpu:h100:1` is accepted but pends forever on `AssocGrpGRES` — which reads like a quota problem but is really the wrong GRES type. Everything runs from the workspace venv at `/hnvme/workspace/y100dc19-mrflow-final/venv` (no container). Unlike the CT configs, `lvfm/configs/mrflow_STDiT-L2_16f8.yaml` carries real paths and needs no `envsubst`.

Single-sample inference:
```bash
python auto_regressive_generate/main.py --config <experiment>/config.yaml \
    --ckpt <checkpoint-N>/denoiser_ema --embedding <emb>.pt \
    --output out_frames/ --type full-body   # or gt-head / block-wise (last two need --gt-latent)
```

### Challenge evaluation

`evaluation/` scores a checkpoint with the official VLM3D `mr-volume-generation` metrics. It needs
no preprocessed split — every case comes out of the raw MR-RATE archives in one read — so `test`
runs exactly like `val`:

```bash
sbatch --array=0-15 slurms/mrflow_eval_helma.sh <experiment>/config.yaml \
    <experiment>/checkpoint-N/denoiser_ema --split val
sbatch slurms/mrflow_eval_helma.sh <config> <ckpt> --split val --combine   # after the array
```

The array size is the shard count (`SLURM_ARRAY_TASK_COUNT`), each task writes
`<out>/shard-NNNN.pt`, and `--combine` pools them into `<out>/metrics.json` and one W&B run.
Sharding matters: a full-body rollout is 20 blocks of 201 Euler steps, so a 2,000-case val sweep is
days on one GPU. [`evaluation/README.md`](evaluation/README.md) covers the pipeline and the metrics.

**The rollout is one case at a time, fp32, 201 Euler steps — everywhere.** `evaluation/main.py`,
`auto_regressive_generate/main.py` and `submission/predict.py` all drive
`LatentAutoregressiveGenerator` the same way, and the only speed switches are `use_compile`
(`--compile`, on in evaluation, off in the debug script, `MRFLOW_USE_COMPILE` in the container) and
`use_bf16` (`--bf16`, off in both scripts, `MRFLOW_USE_BF16` in the container). The asymmetry is
deliberate: measured, `torch.compile` adds nothing to the numerical drift (`1.91e-2` relative
velocity difference with bf16 on, against `1.90e-2` for bf16 alone), while bf16 puts a rollout
`23.6 dB` from its fp32 counterpart — same distribution, different draw. Batched rollouts were
tried (another ~1.7x per volume, saturating by batch 2) and **removed**, because a batch draws all
of its noise in one call: noise can then only be seeded per batch, so a rerun under a different
grouping generates different volumes. bf16 applies to the denoiser only; the VAE encode/decode
stays fp32 in every path.

### Configs are not directly runnable

`lvfm/configs/*.yaml` contain **shell** variables (`${LATTE_TRAIN_DATA_ROOT}`, `${LATTE_EMBEDDING_ROOT}`, `${LATTE_VALID_DATA_ROOT}`, `${LATTE_VALID_EMBEDDING_ROOT}`) that OmegaConf will *not* resolve. `trainer_helma.sh` runs `envsubst` over the config into a per-node temp copy before launching. Running `train.py` on a raw config only works if those four vars are exported in the environment — otherwise OmegaConf raises on interpolation. `vae.pretrained` and `output_dir` are also literal `/path/to/...` placeholders that must be edited. The SLURM scripts likewise carry `YOUR_PROXY` / `YOUR_EMAIL` / `/path/to/tmi_container.sif` placeholders.

At inference, pass the config **saved into the experiment dir** (`save_checkpoint` writes `config.yaml` next to the checkpoints) — it has the substitutions already baked in.

## Architecture

**Flow direction is reversed from the usual convention.** `z_0` is the clean latent and `z_1` is noise; `t=0` is data, `t=1` is noise. Training ([lvfm/train.py:294-307](lvfm/train.py#L294-L307)) samples `t ~ U(0,1)`, builds `z_t = (1-t)·z_0 + (ε+(1-ε)t)·z_1`, and regresses the velocity `u = (1-ε)·z_1 - z_0` with plain MSE. Sampling therefore integrates `odeint` from `t=1.0 → 0.0`. Any change to one side must mirror the other.

**Block-pair conditioning.** [`LatentBlockDataset`](echosyn/common/datasets.py) returns `image` = block at `[t, t+16)` (the condition), `video` = block at `[t+16, t+32)` (the target), plus an L2-normalized text embedding. It samples `t=0` with 50% probability to bias toward volume starts. The keys `image`/`video` are legacy echo-synthesis names — here they mean *previous block* / *next block*, not stills vs. clips.

**Conditioning path.** [`DiffuserSTDiT.forward`](echosyn/common/models.py#L1492) concatenates `cond_image` onto `x` along the channel dim, which is why every config sets `in_channels: 32` (16 target + 16 condition) with `out_channels: 16`. The text embedding goes through cross-attention with `caption_channels: 768`, `model_max_length: 1` (exactly one token). Inside [`STDiTBlock.forward`](echosyn/common/models.py#L1074), each block runs spatial attention over `S` tokens, then temporal attention over `T` tokens, then cross-attention, then MLP — temporal positional embedding is injected only in block 0.

**Auto-regressive rollout.** [`LatentAutoregressiveGenerator.generate`](auto_regressive_generate/__init__.py#L103) seeds with a zero-image latent block (or a GT block in `gt-head` mode), then repeatedly conditions on the last 16 latent frames and Euler-integrates a new block (201 steps). Stopping is **visual, not learned**: the model is trained to emit all-white frames at the end of a volume, and `is_stop_frame`/`trim_stop_frames` compare against a pre-encoded latent of an all-ones image with `eps=0.1`. The `overlap=8` default splits each freshly generated block into two entries in the list, so the next step's condition window straddles the boundary.

**The MR path reuses all of the above.** Only the data changes, so `lvfm/train.py` has no MRI branch. [`echosyn/common/mrrate.py`](echosyn/common/mrrate.py) holds every MR-specific piece and `lvfm/preprocess_mrrate.py` is its one CLI:

- **Volumes.** MR-RATE ships as webdataset tars read in place (no extraction), indexed by its own `series.parquet`. `preprocess_volume` does RAS canonicalize → 1 mm trilinear resample → percentile-normalize over **nonzero** voxels to `[0, 1]` → permute so the acquisition plane's slice axis leads → center crop/pad in-plane to 256². `T` is never touched: variable volume length is the point of the autoregressive formulation. `[0, 1]` is the same range CTFlow normalized CT to, and that is **measured, not stylistic** — see the gotcha below. There is no Hounsfield scale and no HU window to convert, hence no `* 306`.
- **Boundaries are learned tokens.** Both are the ends of the pixel range, read from the config's top-level `black_value`/`white_value` — `0.0`/`1.0`, the same values CTFlow used, since MR is normalized to the same `[0, 1]`. **Training and inference get them by different routes, as in CTFlow.** Preprocessing VAE-encodes the pair once into `<root>/boundary/{black,white}.pt` as `(mean, std)`, and that is what `MRRateLatentBlockDataset` feeds its `start`/`end` examples; `LatentAutoregressiveGenerator` never reads that cache — it re-encodes both frames at init through `encode_image`, whose `latent_dist.sample()` already draws, so its tokens are `latent_channels` wide and the `sample_latents` call on the white one is a no-op. Both routes are draws from the same posterior: measured, an encoded stop token agrees with the cached one on 99.93% of elements within the `eps=0.1` stop threshold.
- **Text is one of two named conditioning configurations,** selected by `mri.conditioning` and tabled in `CONDITIONINGS` (`mrrate.py`). The choice fixes the encoders, the pooling, the token count and the width together, and `check_conditioning` refuses a config whose `caption_channels`/`model_max_length` disagree with it — so a mismatch fails in seconds, not at step 1 on 16 nodes. `build_conditioner` is the only place a config becomes a live encoder; every caller (preprocessing, evaluation, the container, `tools/ctflow_transfer_check.py`) goes through it, so no two paths can drift in tokenizer, pooling or section order.

  - **`cxr_bert_cls`** (default, and the released checkpoint's own): frozen CXR-BERT, CLS-pooled to **one 768-d token**. One encode of one string — the `[MODALITY]/[PLANE]/[SPACING]` prefix followed by `[FINDINGS]`/`[IMPRESSION]` — and its CLS state is the embedding. An earlier version also pooled the prefix on its own and added it back as a unit vector, because in-string markers are diluted (~0.0002 cosine against ~0.03 for report content); that re-add and its `mri.marker_weight` knob are gone, so contrast and plane are present but faint. The prefix is **kept** alongside the class embeddings below, so embeddings encoded before those existed stay usable.
  - **`report2ct_style_meta`** (arm E of `../R2V-MR-Generation`, itself Report2CT-style): **three 2560-wide tokens**, one per section. `findings`, `impression` and `acquisition` are each pooled independently — masked mean, never averaging padding — by MedEmbed-large (1024), Bio_ClinicalBERT (768) and CXR-BERT (768), and those three vectors are concatenated on the feature axis. Both orders are part of the contract and may never be sorted or inferred: `SECTIONS` fixes the sequence axis, the encoder tuple fixes the feature axis, and a silently permuted 2560-vector is invisible in a loss curve. Three tokens rather than one is the point — cross-attention over a single key is degenerate, since a softmax over one element is identically 1, so a pooled conditioning reaches the denoiser as a per-channel bias that is the same at every voxel and its `q`/`k` projections never receive gradient. Two deliberate differences from the pooled path: findings and impression carry **no** bracket marker (the sequence position identifies them, so a marker would only be a constant in every vector), and the acquisition section *is* the markers — `acquisition_prefix`'s string and nothing else. An absent section is the **empty string**, not a dropped token: every encoder still emits its CLS/SEP pair, so the row is a constant the model can learn as "missing" while the token count stays 3. That is Report2CT's own behaviour, and it keeps MRFlow's untested `mask` branch in `STDiT.forward` out of the picture.

  Neither configuration leaks modality into the modality guidance term: `v_modality - v_base` has the report dropped in *both* branches, and the null replaces the whole tensor, acquisition token included, so no report text reaches it at all (`tests/test_conditioning.py::test_modality_term_cannot_see_the_report`). The only real requirement is that the caller keep the two conditions consistent — `--modality`/`--plane` set the class ids while `--embedding` is a pre-encoded vector whose prefix was baked at preprocessing time, and nothing can check they agree, because a pooled embedding cannot be read back.

- **Modality and plane are class conditions as well as text.** `MODALITY_TO_ID` / `PLANE_TO_ID` in `mrrate.py` are literal tables — the single source of truth, stable across splits, workers, runs and checkpoints; ids may never move. Id 0 of the modality table is `CFG_NULL` (the guidance null, NVIDIA's `unknown: 0`), and `UNKNOWN` is a separate real category. Plane has **no** null: it is a geometry condition, always supplied, never dropped, never guided. `MRRateLatentBlockDataset` resolves both at construction (so an unmapped label fails immediately) and returns them as scalar longs; `STDiT.forward` adds `nn.Embedding` lookups to the timestep embedding, MAISI's `class_embedding` pattern, so they modulate every block through `t_block` and the output through `final_layer`. Both embeddings are **zero-initialized**, so a checkpoint predating them predicts identically at step 0.

- **Semantic condition states.** `config.condition_states` samples one of three per sample — `full` 0.80, `modality_only` 0.10 (real modality, null report), `semantic_null` 0.10 (CFG_NULL modality, null report) — rather than two independent Bernoulli draws, which at 0.1/0.1 would leave only 1% fully-null, and that branch is the base of the guidance expansion. Must sum to 1. `p_drop_conditionning` is unrelated: it drops the previous block, an autoregressive trick, not semantic CFG.

- **Hierarchical CFG at inference.** `LatentAutoregressiveGenerator.velocity` batches three branches into one denoiser call — `(CFG_NULL, null report)`, `(modality, null report)`, `(modality, report)` — and combines `v_base + s_mod·(v_mod − v_base) + s_rep·(v_full − v_mod)`. The latent, timestep, `cond_image` and plane are the same tensor repeated in all three. At `modality_cfg_scale == report_cfg_scale == 1` the sum is exactly `v_full`, so a single conditional call is made instead.
- **Volumes are one series, reports are one study**, so there is one embedding per *series*, not per study — which is why one pass writes both artifacts.

**Evaluation runs two metric families over the same rollouts, on deliberately different arrays.** [`evaluation/challenge_metrics.py`](evaluation/challenge_metrics.py) is a port of the challenge's `mrgen_evaluation` (`modality_filter.py` + `metrics_basic.py` + `fid_2p5d.py`, comments translated, math untouched) and gets the pair exactly as the leaderboard would; [`evaluation/paper_metrics.py`](evaluation/paper_metrics.py) is ours and gets both volumes canonicalized into the model's own 1 mm isotropic / 256² training grid. [`evaluation/__init__.py`](evaluation/__init__.py) holds `ChallengeAccumulator`, which feeds each family the geometry that family is defined on; [`evaluation/main.py`](evaluation/main.py) generates the pairs. **Never give one family the other's arrays.** Three consequences shape the code:

- **The ground truth reaches the metric untouched.** `load_native_volume` RAS-reorients and permutes plane-first — the same axis order the model rolls out in — and does nothing else. The metric percentile-normalizes both volumes itself and `zoom`s the *generated* one onto the ground truth's shape, so resampling or normalizing the reference here would score against a different target than the leaderboard does. `read_canonical` / `plane_order` are shared with `preprocess_volume` so the two orientations cannot drift.
- **`METRIC_KEYS` is also the reporting order** — `HEADLINE_KEYS` first (`FVD_f16`, `FVD_f64`, `FID`, `FID_2p5D_Avg`, `IS_mean`, `PSNR_mean`, `MSE_mean`, `SSIM_mean`, deliberately interleaving the two families so a run leads with the numbers it is read by), then the remainder of `PAPER_KEYS` (`IS_std`, the per-stratum splits, the clip/case counts), the remainder of `CHALLENGE_KEYS` (the per-plane FIDs, `FID_T`, `dice`), then the file counts — and it drives the W&B table, the summary and the console dump at once. `PAPER_KEYS`/`CHALLENGE_KEYS` still say which family a key belongs to, which is what decides the arrays it was computed on. `dice` is a literal copy of `SSIM_mean` — the platform's primary-metric shim, not real Dice. `FID_T` is a literal copy of `FID_2p5D_YZ`, not a second computation: the official code slices array axis 0 under the label "YZ", so that number already *is* slice-axis FID on the container's squeezenet backbone. Unlike `_XY`/`_XZ`, whose reformats contain the slice axis and so get squashed to 224² by a different factor on each side, an axis-0 slice is `(H, W)` and carries no slice-count distortion — it is the only official plane worth quoting.
- **The paper metrics are ours and describe a different question.** `paper_metrics.py` puts both volumes in the model's training grid — ground truth in-plane resampled to 1 mm and cropped/padded to 256² with preprocessing's own posterior shift, slice axis resampled to 1 mm at its **native extent** (143–200 slices over MR-RATE, never stretched to a fixed count); the generated volume is already exactly that. Then `_normalize01` on both. `FID` is torchvision Inception-v3 pool3 (2048-d) over acquisition-plane slices — not the TF-ported `pt_inception-2015-12-05`, so it is comparable across our runs but not digit-for-digit to a paper quoting that port. `FVD` is I3D over **clips of consecutive slices**, `f16` (stride 8) reading within-block continuity and `f64` (stride 32) reading coherence across four block boundaries — the drift the CT paper's `FVD_f16`/`FVD_f128` pair exposes; `f128` does not fit, since 143–200-slice volumes would leave ~1 clip per case against 400 feature dimensions. Every distance is reported with its row/clip count, because that against the feature dimension is what says whether the covariance is an estimate. Never quote any of these as a challenge metric.

**Config-driven instantiation.** Nearly every component (denoiser, optimizer, scheduler, VAE, dataloader) is built from a `{target, args}` YAML stanza via `instantiate_class_from_config` / `instantiate` in [echosyn/common/__init__.py](echosyn/common/__init__.py). To swap a model or scheduler, change the config's `target`, not the training script.

`echosyn/common/models.py` also holds unused-by-this-pipeline architectures (`SegUnet2DModel`, `EDM2UNet`, `UNetSTIC`, `DiffuserSTDiTSC`) carried over from the EchoSyn lineage; `train.py` has `UNetSTIC` special-cases for xformers.

## Gotchas

- `echosyn.common` is imported as `from echosyn.common import *`; `rearrange`, `torch`, etc. reach the other modules through that star import. Adding an import there changes downstream namespaces.
- `scale_latents` / `unscale_latents` mutate their argument **in place** (`-=`, `*=`). Don't call them twice on the same tensor.
- **MR is normalized to `[0, 1]`, and `[-1, 1]` is not an equivalent choice.** Measured with `tools/ctflow_transfer_check.py` (20 val series): the CTFlow checkpoint's zero-shot flow loss relative to the target velocity's own magnitude is `0.106` on `[0, 1]` latents and `2.78` on `[-1, 1]` ones — and `1.0` is the predict-zero baseline, so at `[-1, 1]` the init is *worse than useless*. Standardizing `[-1, 1]` latents to mean 0 / std 1 does **not** help (`2.85`), because the VAE encoder is nonlinear: an affine change in pixel space is not an affine change in latent space, so the two ranges are structurally different representations, not rescalings of each other. VAE reconstruction is a tie (PSNR 39.12 vs 38.92), so `[0, 1]` costs nothing. Re-run that check if the VAE, `init_from`, or preprocessing ever changes.
- **Do not replace FLUX's `scaling_factor` / `shift_factor` with measured latent statistics** while `init_from` points at a CTFlow checkpoint. `get_vae_scaler` reads `0.3611`/`0.1159` from the VAE's own `config.json`; standardizing on measured moments instead made transfer slightly *worse* in both ranges (`0.106 → 0.116`), because the trunk was trained with exactly those factors. Computing your own statistics is the right move for a scratch run, not for this fine-tune.
- The four pixel-range constants — `decode_scale`, `decode_shift`, `black_value`, `white_value` — are **required top-level config keys with no defaults in code**, so a config missing them raises instead of silently decoding at the wrong scale. MR uses `255`/`0`/`0.0`/`1.0` (`[0, 1]`, no window to convert); the CT configs carry `306`/`0`/`0.0`/`1.0`, where `306` is **not** a data range: CT pixels are `[0, 1]` over an HU window of `(-1000, 1400)`, and `306` rescales that to the challenge's `(-1000, 1000)` so the uint8 clamp lands exactly on HU +1000 (`306 * 2000/2400 = 255`). Experiment configs saved before this moved out of code (e.g. `experiments/smoke_overfit/config.yaml`) need the keys added by hand before inference.
- **Preprocessing needs `--zip` on Helma.** One `.pt` per latent plus one per embedding is 2 files
  per series, so the configured 122k series is ~244k files against this account's **102,400 inode
  hard limit** on `/hnvme` (a per-*user* limit across the whole filesystem — workspaces have no
  project quota of their own, and directories and symlinks count too). Without the flag the train
  array dies ~19k series in, having burned ~200 GPU-hours, and each shard silently writes an
  under-reporting manifest. `--zip` puts a shard's artifacts in one `ZIP_STORED` archive
  (`LatentStore` in `mrrate.py`), taking the run to ~140 inodes; a zip's central directory records
  every member's offset, so `read_bundled` still reads one 5.4 MB member in ~2 ms without
  unpacking. The manifest's `zip` column says which layout was written and `_load` branches on it,
  so loose-file datasets — and manifests written before the column existed — keep working
  untouched. The trade-off: resume is per-shard rather than per-series, so a killed task re-encodes
  its ~1,875 series. Measured on the 64-task train array: median 42 min per task, 53 min
  wall-clock, ~45 GPU-hours for the whole split (val 4x10 min, test 16x~35 min).
- MR latents are stored **fp16**, unscaled, and as the VAE posterior's `(mean, std)` concatenated on the channel axis — so every stored tensor is `2 * latent_channels` wide and **`sample_latents: true` is mandatory**, matching the regime CTFlow itself trained under. Setting it false does not error: it silently takes the first 16 channels (the mean) and trains on one frozen encoding per volume. Everything that reads a latent must sample before scaling — `scale_latents` on a 32-channel tensor would scale the stds too. `MRRateLatentBlockDataset` casts to float on load. Storing both parameters doubles the split to ~1.5 TB, on top of `mri.max_repeats` / `mri.max_series_*` — the defaults keep 120k of 576k eligible train series, still ~30 epochs at a 60k-step fine-tune.
- **FLUX's VAE is very nearly deterministic**, so sampling buys less than it looks like. Measured on a real val series: latent std 3.50 against a posterior std of median 4e-5 and max 0.044 — a noise-to-signal ratio of **0.0004**. The draw is a ~0.04% perturbation, not a meaningful augmentation. It is kept because it matches CTFlow's training regime and costs no extra compute, but the doubled storage buys very little regularization.
- **`mri.conditioning`, `caption_channels` and `model_max_length` are one decision in three places.** `check_conditioning` (called first thing in `train.py`) refuses a config where they disagree, so the failure is a message in the first seconds rather than a caption-projection error at step 1 on 16 nodes. `mri.conditioning` itself defaults to `cxr_bert_cls` and `mri.text_root` to the parent of `mri.text_checkpoint` — the same directory the sectioned encoders are staged in — so every config written before sectioned conditioning existed, including the ones saved next to released checkpoints, keeps working untouched.
- **The dataset L2-normalizes the whole embedding, not each token.** `embedding.norm(p=2)` on a `[3, 2560]` tensor is the Frobenius norm, so the rows and the per-encoder blocks keep their relative magnitudes. Those magnitudes are **not** balanced — measured per-element RMS is 0.556 (MedEmbed-large) / 0.411 (Bio_ClinicalBERT) / 0.191 (CXR-BERT), i.e. an energy split of roughly 67 / 27 / 6 % — and that is deliberate: neither Report2CT nor the reference arm E equalizes it either. Per-token normalization would instead discard how much a section says.
- **The three sectioned encoders are staged, not downloaded.** `TEXT_ENCODERS` maps each to a directory under `mri.text_root` (`MedEmbed-large-v0.1`, `Bio_ClinicalBERT`, `BiomedVLP-CXR-BERT-specialized`, ~2.3 GB together) and every load is `local_files_only`. CXR-BERT keeps its `bert_shim` loader — `AutoTokenizer` on that directory demands `trust_remote_code` and aborts. Bio_ClinicalBERT is staged with no `tokenizer_config.json`, so its tokenizer defaults to `do_lower_case=True` against a *cased* vocab (`FLAIR` → `fl ##air`); that is left alone on purpose, because it is how the reference implementation was trained.
- `mri.min_slices` must stay at least `2 * block_size`; the dataset assumes every row can produce an interior pair and has no short-volume fallback.
- `init_from` loads **weights only** (fresh optimizer/scheduler/EMA/step) and is applied before `accelerator.prepare`, so a `resume_from_checkpoint` restore still wins over it. The CTFlow EMA checkpoint matches every key except `model.{modality,plane}_embedder.weight`, which postdate it; `load_init_weights` allows exactly those two and **raises** on anything else missing or unexpected, so a real config divergence cannot pass as a warning. Never widen that to an unrestricted `strict=False`. **`init_from` cannot cross conditionings**: a sectioned caption path is `[3, 2560]` where a pooled one is `[1, 768]`, and `load_state_dict` refuses a size mismatch even at `strict=False`. That is why the sectioned run trains from scratch — the same recipe as `mrflow_from_scratch.yaml`, so the two runs differ only in the conditioning — and it is a loud failure, not a silent one.
- **Do not put conditioning state in a buffer.** `y_embedding` (the learned null report) used to be `register_buffer("y_embedding", nn.Parameter(...))`, which type-checks and is checkpointed but is invisible to the optimizer and to EMA — and `EMAModel.save_pretrained` rebuilds the model from its config and copies only `parameters()`, so **every buffer in a `denoiser_ema/` save is re-randomized**. Measured before the fix: two saves of one run differed (`sum` −1.5595 vs +0.0641) while the `denoiser/` saves agreed exactly. It is a real `nn.Parameter` now (same state-dict key), and `named_buffers()` should stay exactly `{pos_embed, pos_embed_temporal}` — both deterministic. `tests/test_conditioning.py::test_only_deterministic_buffers_remain` guards this.
- `get_vae_scaler` silently falls back to mean=0/std=1 (printing a warning) if `<vae.pretrained>/config.json` is unreadable, which quietly changes the latent scale.
- Validation in `train.py` uses adaptive `odeint` (default dopri5) with `timesteps=[1.0, 0.0]`, while inference uses explicit `method="euler"` with 201 steps — validation samples are not directly comparable to inference output.
- The dataset takes `embedding[0].unsqueeze(0)` → `[1, D]`, but `main.py` only does `.unsqueeze(0)` on the loaded file. If an embedding `.pt` holds more than one token, inference feeds a different token count than training used.
- Checkpoint retention is via `checkpoints_to_keep` (an explicit step list, plus the latest) — everything else is deleted at each checkpointing step, so raising `checkpointing_steps` frequency without editing that list still discards intermediate checkpoints.
- Resuming reads `wandb_args.id` back from `<output_dir>/config.yaml`; `if_fine_tuned: true` loads weights but resets `global_step`, optimizer, and scheduler.
- **Do not "improve" `evaluation/challenge_metrics.py`.** All of it is the leaderboard's own arithmetic, quirks included: `stride=4` slice sampling, squeezenet1_1 rather than InceptionV3, a 0.5/99.5-percentile window recomputed per slice, and a missing case dropped from the MSE/PSNR/SSIM means instead of penalized. The only sanctioned additions are marked in the file — `raw_features`/`finalize_pooled` for cross-shard FID pooling, and `_matrix_sqrt`, which drops the `disp=` kwarg scipy ≥ 1.17 removed (this venv is on 1.18, so the official file cannot run here unpatched). Cross-shard pooling agrees with a single process to ~2e-7 relative, not bit-exactly: `np.cov` over reordered float32 rows accumulates differently. `paper_metrics.py` imports `_normalize01` and `_matrix_sqrt` from it so there is one definition of each; everything else there is ours and may change freely.
- **Clips, not volumes, are the FVD sample — which is what makes its covariance an estimate.** One 400-d I3D vector per *clip*: measured on MR-RATE, a 1 mm volume yields ~17–21 `f16` clips and 3–4 `f64` clips, so 1,000 cases give roughly 20k and 4k clips against 400 dimensions instead of the 1,000 that one-vector-per-volume gave. Real and generated are clipped identically, so a case contributes the same count on both sides **unless the rollout came out a different length** — which is the error a fixed-length resize used to hide. Below ~400 samples the number stops ordering: measured on single-scale synthetic volumes, noise at sigma 0.05/0.2/0.8 scored 3159/2841/840, i.e. backwards, and identically so at N=8 and N=96; on a broadband phantom the same ladder is monotone, which is why the ordering tests use one. IS is compressed for a different reason — ImageNet's 1,000 classes do not describe an MR slice, and the posteriors carry 3.7 of a possible 6.9 nats — so it sits near 1.6–1.8, never a natural-image number, and `IS_std` is **not** shard-count invariant (measured 0.069 at one shard against 0.029 at 32, while `IS_mean` moved 0.25%).
- **`tests/test_evaluation_metrics.py` is the regression net for both metric families** (`pytest tests/test_evaluation_metrics.py`, ~7.5 min on a login node, seconds on a GPU node): an identical pair must score 0 for FID and both FVDs, a repeated slice must score IS exactly 1.0, one clip must read nan rather than 0, FID/FVD must order a broadband noise ladder, clip pooling must be invariant across shard counts, an empty stratum must read nan, and `test_official_basic_metrics_still_score_what_they_did` pins the vendored container's MSE/PSNR/SSIM to measured golden values so a leaderboard number cannot move unnoticed. Note `tests/` is gitignored, so this file does not travel with the repo.
- A shard file written under an older metric layout carries no `fid_raw`/`fvd_raw`, so `--combine` raises on it rather than pooling a partial number. Re-run the array. The I3D torchscript is a first-use download into the torch hub cache (`MRFLOW_I3D_PATH` overrides it for a node with no outbound route); it is already cached on Helma, and a compute node can fetch it itself in ~2 s over the proxy the eval script exports.
- The model can only roll out on **GPU**: `DiffuserSTDiT`'s cross-attention goes through xformers' `memory_efficient_attention`, which has no CPU kernel, so a CPU evaluation run fails per case and scores every one as missing rather than erroring out.
- `logs/`, `*.pt`, `*.safetensors`, and `checkpoint-*/` are gitignored — checkpoints never belong in a commit.
