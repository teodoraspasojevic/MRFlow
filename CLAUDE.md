# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CTFlow (ICCV 2025 VLM3D workshop): a Spatial-Temporal DiT trained with **flow matching** to synthesize 3D CT volumes **block-by-block, auto-regressively**, conditioned on a CT-report text embedding. Everything happens in FLUX-VAE latent space — the model never sees pixels during training.

**MRFlow** is the MRI branch of the same code: the identical STDiT + flow-matching trunk fine-tuned on MR-RATE brain MRI, conditioned on a frozen CXR-BERT report embedding, still in FLUX-VAE latent space. It is not a separate package — MR support is `echosyn/common/mrrate.py`, one extra dataset class, one preprocessing script, and its own config. Both paths share `lvfm/train.py` unchanged.

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
sbatch slurms/mrflow_train_helma.sh
```

`--zip` is not optional on this account — see the inode gotcha below. Anything after the split is
passed through to `preprocess_mrrate.py`, so `--overwrite` and `--limit` work the same way.
Multi-node needs no second script — `sbatch --nodes=4 slurms/mrflow_train_helma.sh` re-execs under `srun` and derives `machine_rank` from `SLURM_NODEID`.

**This account has h200 GPUs but no h100 allocation.** Always request `--partition=h200 --gres=gpu:h200:N`. An h100 request fails two different ways: `-p h100` is rejected at submit ("Invalid account or account/partition combination"), while `-p preempt --gres=gpu:h100:1` is accepted but pends forever on `AssocGrpGRES` — which reads like a quota problem but is really the wrong GRES type. Everything runs from the workspace venv at `/hnvme/workspace/y100dc19-mrflow/venv` (no container). Unlike the CT configs, `lvfm/configs/mrflow_STDiT-L2_16f8.yaml` carries real paths and needs no `envsubst`.

Single-sample inference:
```bash
python auto_regressive_generate/main.py --config <experiment>/config.yaml \
    --ckpt <checkpoint-N>/denoiser_ema --embedding <emb>.pt \
    --output out_frames/ --type full-body   # or gt-head / block-wise (last two need --gt-latent)
```

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
- **Boundaries are learned tokens.** Both are the ends of the pixel range, read from the config's top-level `black_value`/`white_value` — `0.0`/`1.0`, the same values CTFlow used, since MR is normalized to the same `[0, 1]`. Both are VAE-encoded once into `<root>/boundary/{black,white}.pt` and are what `MRRateLatentBlockDataset` uses for its `start`/`end` examples; `LatentAutoregressiveGenerator` reuses that cache when it exists and re-encodes from the config values otherwise.
- **Text.** Frozen CXR-BERT, CLS-pooled to one 768-d token, loaded as a stock `BertModel` (its `bert.*` weights are plain BERT) rather than with `trust_remote_code`. `encode_conditioning` pools the report and the `[MODALITY]/[PLANE]` markers **separately** and sums them as unit vectors: one MR-RATE report maps to ~12 series of different contrast, and markers left inside the report string move the pooled vector by only ~0.0002 cosine against ~0.03 for report content, so the model would be blind to which contrast to generate. `mri.marker_weight: 0.0` restores report-only conditioning.
- **Volumes are one series, reports are one study**, so there is one embedding per *series*, not per study — which is why one pass writes both artifacts.

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
  its ~1,875 series (~4 h).
- MR latents are stored **fp16** and unscaled; `MRRateLatentBlockDataset` casts to float on load. A full MR-RATE split would be ~9 TB in fp32, hence `mri.max_repeats` / `mri.max_series_*` — the defaults keep 120k of 576k eligible train series, still ~30 epochs at a 60k-step fine-tune.
- `mri.min_slices` must stay at least `2 * block_size`; the dataset assumes every row can produce an interior pair and has no short-volume fallback.
- `init_from` loads **weights only** (fresh optimizer/scheduler/EMA/step) and is applied before `accelerator.prepare`, so a `resume_from_checkpoint` restore still wins over it. The CTFlow EMA checkpoint key-matches `DiffuserSTDiT` exactly (474/474), so any missing/unexpected keys mean the configs diverged.
- `get_vae_scaler` silently falls back to mean=0/std=1 (printing a warning) if `<vae.pretrained>/config.json` is unreadable, which quietly changes the latent scale.
- Validation in `train.py` uses adaptive `odeint` (default dopri5) with `timesteps=[1.0, 0.0]`, while inference uses explicit `method="euler"` with 201 steps — validation samples are not directly comparable to inference output.
- The dataset takes `embedding[0].unsqueeze(0)` → `[1, D]`, but `main.py` only does `.unsqueeze(0)` on the loaded file. If an embedding `.pt` holds more than one token, inference feeds a different token count than training used.
- Checkpoint retention is via `checkpoints_to_keep` (an explicit step list, plus the latest) — everything else is deleted at each checkpointing step, so raising `checkpointing_steps` frequency without editing that list still discards intermediate checkpoints.
- Resuming reads `wandb_args.id` back from `<output_dir>/config.yaml`; `if_fine_tuned: true` loads weights but resets `global_step`, optimizer, and scheduler.
- `logs/`, `*.pt`, `*.safetensors`, and `checkpoint-*/` are gitignored — checkpoints never belong in a commit.
