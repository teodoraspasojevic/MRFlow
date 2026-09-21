# Text2CT

End-to-end 3D latent diffusion: a generation-oriented 3D-CLIP text encoder, a volumetric VAE, and
cross-attention conditioning in latent space — explicitly no cascaded super-resolution, which is
its stated advantage over earlier text-to-CT work. Architecturally the closest to MRFlow, and the
most interesting comparison for that reason.

    upstream   https://github.com/danielemolino/Text2CT  @ 887caa9 (2026-09-15)
    clone      $WS/baselines/upstream/Text2CT
    weights    huggingface.co/dmolino/text2ct-weights

## Read the upstream README before planning this one

Two things it says that change the work:

- **The released code is the original-preprint release, not the BMVC 2026 version.** The BMVC
  paper's contribution — text-level structured hard negatives in the 3D-CLIP encoder, and updated
  checkpoints — is not in this commit, and the README's own date for it is unfilled. So `887caa9`
  reproduces the *preprint* model. Pin the commit in the table and say which paper it is, or re-pin
  when they publish.
- Its evaluation protocol is the one **our own medical FIDs already follow** —
  `evaluation/medical_fid.py`'s `fid_3d_medicalnet` and `fid_2p5d_radimagenet_*` are the
  CCELLA / Alignment-to-Synthesis protocol. That is a real advantage here: this baseline and
  MRFlow are already being scored the way this paper scores, so those two numbers are directly
  readable against the published ones (modulo our geometry, which is ours and declared).

## What has to happen before it is a baseline

**Chest CT → MR-RATE brain MRI fine-tuning.** Three components could each be
frozen or trained (3D-CLIP, VAE, diffusion UNet); the honest default is to fine-tune the diffusion
model and keep the released encoder and VAE, and to state it. That is also **exactly MRFlow's own
recipe** — a CTFlow trunk fine-tuned on MR-RATE with a frozen CXR-BERT and a frozen FLUX VAE — so
one adaptation protocol covers both sides of the table, which is a far better answer to "did you
handicap the baseline?" than an argument about compute budget.

**The VAE needs no adaptation and no experiment: it is the same file our NVIDIA baseline uses.**

    sha256 1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3
      Text2CT   models/autoencoder_epoch273.pt          (huggingface.co/dmolino/text2ct-weights)
      NVIDIA    $WS_nvidia/models/autoencoder_v1.pt     (../nvidia_r2v/)
      MAISI     bundles/maisi_ct_generative/models/autoencoder.pt

All three are byte-identical, and the `autoencoder_def` in `configs/config_rflow.json` matches the
MAISI bundle's on 16 of 17 fields (`num_splits` 2 vs 4 is a sliding-inference memory knob, not
weights). So despite the `-mr-brain` naming only NVIDIA's *UNet* is MR-specific, and Text2CT's
frozen reconstruction ceiling on MR-RATE **is** the one our challenge model already established.
Its decode path confirms the input convention transfers: `diff_model_demo.py:194` maps
`[0, 1] → [-1000, 1000]` HU only at decode, so the latent space is over `[0, 1]` volumes, which is
what `preprocess_volume` already produces for MR.

**That ceiling has been measured, and it is low.** `ablations/vae_ceiling.py` over the 1,002-case val
population: MAISI reconstructs MR-RATE at **26.19 dB / 0.465 SSIM** against FLUX's **36.70 dB / 0.982**,
distributions non-overlapping, the gap present in every modality and inside the brain mask, and not a
precision artifact. MAISI compresses 16x per voxel (4x4x4 spatial, 4 channels) where FLUX compresses
4x (8x8 spatial, 16 channels, no depth compression at all). Both baselines therefore generate against
a ceiling ~10.5 dB below MRFlow's, which has to be declared — and reframed, because it is the
mechanism rather than only a confound: monolithic 3D generation *forces* a depth-compressing 3D VAE,
while block-autoregressive generation is what permits a 2D one.

**The four conditioning inputs you inherit with the UNet**, from `configs/config_rflow.json`:

| input | released state | call for MR |
|---|---|---|
| `include_body_region` | **`false`** | nothing to do — the `top/bottom_region_index` in `config_diff_model.json` is dead config |
| `num_class_embeds: 128`, modality | `diff_model_train.py:313` hardcodes `torch.ones(...)` | keep id 1; it is the only index with trained weights |
| `include_spacing_input: true` | trained at `0.75/0.75/3.0` | pass the real grid spacing, `0.5/0.5/1.5`; inside MAISI's range, outside the constant CT-RATE saw |
| `scale_factor = 1.0287` | CT-derived latent scaling | **keep, do not recompute** — same call as MRFlow's FLUX factors under a fine-tune |

The scheduler is `RFlowScheduler`, so Text2CT and MRFlow share the rectified-flow objective family
and the comparison isolates the monolithic 3D UNet against block-wise autoregressive STDiT.

**Which leaves the text encoder as the only open transfer question**, measured in
[`ablations/diagnose_text_encoder.py`](../../ablations/diagnose_text_encoder.py) — see
`ablations/results/text2ct_transfer_diagnostics/report.md`.

`scripts/sample.py`'s `check_input` constrains `output_size` to `[256, 384, 512]` in-plane and
`[128...768]` on the third axis, with spacing in `[0.5, 3.0]` / `[0.5, 5.0]` — a MAISI-derived
interface, the same lineage as the NVIDIA model in `../nvidia_r2v/`. 256 in-plane at 1 mm is
reachable directly, which is convenient but not required: `ingest.py` resamples from the affine
either way.

## Adapter: what still has to be written

`slurms/text2ct_run_shards.sh` + a driver that, for each case of `baselines/cases-test-n100.json`, encodes that
study's report with **Text2CT's own** 3D-CLIP encoder, samples, decodes through its VAE, and saves
`<case_id>.nii.gz` with a correct affine. Its own venv in the workspace.

---

# Fine-tuning Text2CT on MR-RATE

`fine_tune.py` is the entry point. Everything it needs lives in this directory; nothing here
imports `echosyn`, `evaluation` or `lvfm`, because the Text2CT stack runs in its own venv.

```
config.py           defaults, YAML merge, validation        configs/mrrate_finetune.yaml
mrrate_data.py      MR-RATE access, preprocessing,          configs/smoke.yaml
                    modality vocabulary, cache, dataset
model.py            builds the three components; freezing    test_finetune.py
text_encoder.py     the frozen 3D-CLIP report encoder
prepare_data.py     MR-RATE -> cached latents + embeddings
upstream_bridge.py  the updater to/from the unmodified clone
fine_tune.py        the training loop
```

## What was audited, and what it says

**Commit.** `887caa9e14ef5e20155c8df159e464ef49bfed17`, 2026-09-15, "Merge pull request #6 from
cosbidev/main" -- verified with `git log`, matching the table above. This is the
**original-preprint release**, not the BMVC version: the hard-negative contrastive encoder is not
in this tree.

| what | where | value |
|---|---|---|
| text encoder | `core/models/encoders/clip.py:171` `FrozenCLIP3D` | CLIP ViT-L/14 text tower, position embeddings widened to **512 tokens**, `CLIP3DModel` projection, output L2-normalized `[B, 1, 768]` |
| autoencoder | `configs/config_rflow.json` `autoencoder_def` | MONAI `AutoencoderKlMaisi`, 4 latent channels, **4x compression per axis**. sha256 `1f8a7a05...` -- byte-identical to MAISI's and to the NVIDIA baseline's. Needs one library patch to be usable at 512 in-plane; see below |
| UNet | `configs/config_rflow.json` `diffusion_unet_def` | MONAI `DiffusionModelUNetMaisi`, 4 levels `[64,128,256,512]`, cross-attention `768`, `num_class_embeds: 128`, `include_spacing_input: true`, `include_body_region: false` |
| scheduler | `configs/config_rflow.json` `noise_scheduler` | `RFlowScheduler`, 1000 steps, uniform timestep sampling, resolution transform on. **v-prediction only** |
| volume grid (CT) | `scripts/preprocess_ctrate.py:54` | `512 x 512 x 128` at `0.75/0.75/3.0` mm, RAS, HU clipped to `[-1000, 1000]`, then `ScaleIntensityRanged` to `[0, 1]` at `scripts/diff_model_create_training_data.py:60` |
| latent shape (CT) | `scripts/diff_model_create_training_data.py:181` | `[4, 128, 128, 32]` |
| report text | `scripts/save_embeddings_ctrate.py:81` | `f"Findings: {findings} Impression: {impressions}"`, one string, one token out |
| what training reads | `scripts/diff_model_train.py:107-136` | **cached** VAE latents (NIfTI), **cached** text embeddings (`.npy`), a per-volume JSON for spacing. No raw volumes, no encoders in the training process |
| trainable | `scripts/diff_model_train.py:230` | `unet.parameters()` only; the VAE and CLIP are not constructed at all during training |
| optimizer | `scripts/diff_model_train.py:230,244,544` | Adam, lr `1e-4`, `PolynomialLR(power=2.0)`, **L1** loss, fp16 `autocast` + `GradScaler`, batch 4, 200 epochs |
| accumulation / clipping | -- | **neither exists** upstream |
| distributed | `scripts/diff_model_train.py:525` | `partition_dataset` once at startup, then a plain shuffling `DataLoader` per rank -- no `DistributedSampler`, so a rank only ever sees its own partition |
| validation | -- | **none**. `n_epochs`, a train-loss print, and that is all |
| checkpointing | `scripts/diff_model_train.py:409` | every `save_epoch_freq` epochs; resume via `continue_training_from` |

**Three upstream defects this adaptation does not inherit.** `load_unet`
(`scripts/diff_model_train.py:163,168`) references an undefined name `checkpoint` on the branch
that handles its *own* checkpoints, so resuming from a Text2CT-saved file raises `NameError`; it
also loads with `strict=False`, which would swallow a real architecture mismatch.
`load_training_state` (`:190`) assigns `starting_training_step = checkpoint_unet["grad_scaler"]`
and then discards it, so the step counter never resumes. The released checkpoint is a *bare*
state dict, so `load_unet` takes the working branch and the release itself loads fine.

**One thing that looks like a defect and is not.** `scripts/diff_model_demo.py:114-116` nulls the
spacing tensor when `args.diffusion_unet_def["include_top_region_index_input"]` is falsey. That key
is the *unresolved* string `"@include_body_region"` -- `load_config` does no MONAI interpolation --
which is truthy, so spacing is passed after all. Fragile, but the released model does receive
spacing at inference.

## What is Text2CT's code, and what is ours

The reviewer question this has to answer is "how much of this is still Text2CT?". Component by
component:

| | source | how |
|---|---|---|
| diffusion UNet | **upstream** | built by `scripts/utils.define_instance` -- *imported*, not copied -- from `configs/config_rflow.json` |
| VAE | **upstream** | same path, same JSON |
| noise scheduler | **upstream** | same path; `RFlowScheduler`, v-prediction |
| 3D-CLIP text encoder | **upstream** | `core.cfg_helper.model_cfg_bank` + `core.models.common.get_model` build upstream's `FrozenCLIP3D`; we add only a tokenizer shim and `freeze()` |
| training objective | restated | `z_t` from `add_noise`, target `z_0 - eps`, L1, `latent * scale_factor` -- identical to `diff_model_train.py:327-380` |
| report string | restated | one f-string from `save_embeddings_ctrate.py:81` |
| report CFG | restated | 0.1 dropout, zero-vector null, same modality both branches |
| **training loop** | **ours** | see below |
| **cache layout + dataset** | **ours** | see below |
| MR-RATE access, modality vocabulary, freezing | **ours** | upstream has none of it |

The four restated items each have a test that reads `887caa9`'s own source and fails if the pinned
commit moves (`test_training_objective_still_matches_upstreams` and friends). Nothing about the
model, the encoders or the scheduler can drift, because they are constructed by upstream's code.

### Proving the loop is equivalent, by running the original

`upstream_bridge.py` is the updater between this package and the unmodified clone, and it exists so
the equivalence claim is a run rather than an argument.

**One direction re-emits our cache in upstream's own contract.**

```bash
python -m baselines.text2ct.prepare_data --config baselines/text2ct/configs/smoke.yaml \
    --split val --shard 0 --num_shards 1 --limit 16 --upstream_layout $WS/text2ct_upstream_equiv
# then, with NO patch, NO import hook and NO edit to the pinned clone:
cd $WS/baselines/upstream/Text2CT && python scripts/diff_model_train.py \
    --env_config  $WS/text2ct_upstream_equiv/environment.json \
    --model_config $WS/text2ct_upstream_equiv/config_diff_model.json \
    --model_def   configs/config_rflow.json --num_gpus 1 --no_amp
```

`write_upstream_layout` reproduces `diff_model_train.py:508-521`'s path arithmetic exactly -- latent
`.nii.gz` in `(X, Y, Z, C)`, info `.json` carrying `spacing`, cond `.npy` under the
`_impression_xgem_3D` suffix, and a zero-byte stub at the `data_base_dir` path that upstream only
ever `os.path.exists`-checks (`:511`) and never opens.
`test_upstream_layout_is_readable_by_upstreams_own_path_arithmetic` replays that derivation and
asserts every file it lands on exists, and that the latent reloads through upstream's own
`LoadImaged` + `EnsureChannelFirstd` in `(C, X, Y, Z)`.

**The other direction drives upstream's real function on our data.**
`test_our_loss_equals_upstreams_own_train_one_epoch` imports `diff_model_train.train_one_epoch` --
not a transcription of it -- runs it over one batch at `lr=0` (so the code path executes without
moving a weight, asserted by checksum), and compares its loss against `fine_tune.diffusion_loss` on
the same tensors under the same seed. They agree to `rel=1e-6`. Report dropout is set to 0 and the
modality forced to upstream's hardcoded `ct` = 1, because those two are precisely where this
adaptation differs on purpose; everything else must not.

**Measured, on 16 real MR-RATE val series** (job 873784, one h200). Upstream's script ran to
completion with no modification: it loaded `unet_rflow_200ep.pt`, read all 16 volumes through its own
MONAI transform chain, and trained one epoch at batch 1.

| | mean loss over 16 single-sample steps |
|---|---|
| `scripts/diff_model_train.py`, unmodified | **0.9510** |
| `fine_tune.py`, same 16 series, same `scale_factor` | **1.0046** (std 0.098) |

The 5.6% gap is what two independent noise draws over 16 samples give -- the standard error of the
difference is ~0.035, so the gap is ~1.5 sigma -- and upstream's `PolynomialLR` had decayed its lr to
~4e-10 by the last iteration while ours held at 1e-4. **The end-to-end run demonstrates that the
layout and the data are right; the bit-level equivalence is the unit test above**, which removes the
sampling by fixing the seed and the batch.

**One thing the run exposes**: `diff_model_train.py:537` calls `calculate_scale_factor`, which sets
`scale_factor = 1 / std(z)` from the first batch on **every** run, ignoring the released `1.0287`.
That is right for training from MAISI and wrong for a fine-tune, and it is why
`model.recompute_scale_factor` here raises instead of doing it quietly.

**Why the cache layout is not upstream's.** `diff_model_train.py:508-521` derives three paths per
series -- a latent `.nii.gz`, an info `.json` and a cond `.npy`. Over the 575,328-series train split
that is **1.73M files**, against this account's **81,000-inode hard limit** on `/hnvme` (currently
63,000 used, ~18,000 free): a 96x overshoot. One `ZIP_STORED` bundle per shard is the only layout
that fits, and once the layout differs, `prepare_data` inside `diff_model_train.py` cannot read it.

**Why the training loop is not upstream's.** Two of this adaptation's requirements live inside it and
cannot be configured from outside:

- **Modality conditioning.** `diff_model_train.py:313` is `modality_tensor = torch.ones(...)`. A
  hardcoded constant is the opposite of a modality condition.
- **An effective batch of 128.** Upstream has no gradient accumulation, so its effective batch is
  `batch_size x world_size`. At this grid micro-batch 12 is the memory ceiling, so 8 GPUs cap out at
  96. (16 GPUs at batch 8 would reach 128 -- at twice the allocation.)

It also has no gradient clipping, no validation, and a resume path that raises `NameError`
(`:163`, `:168` reference an undefined `checkpoint`) and discards the step counter (`:190`). Gradient
accumulation, clipping, validation loss and a correct resume were all explicitly required of this
work, so they had to be added; upstream's loop is ~40 lines of objective wrapped in ~200 lines of
scaffolding, and it is the scaffolding that changed.

## The MR-RATE data contract

One MR-RATE series becomes one Text2CT training sample. Each step names the MRFlow function it was
copied from; `mrrate_data.py` is a vendored copy, not an import, because `echosyn` will not import
under a pre-v5 `transformers`.

```
tar member (NIfTI bytes)
  -> read_canonical          RAS reorient, (S, R, A) float32 + native spacing   echosyn read_canonical
  -> target_spacing_sra      0.5/0.5/1.5 mm, coarse axis on the plane's         NEW (anisotropic grid)
                             stacking axis: S axial, R sagittal, A coronal
  -> trilinear resample       one step, from native -- no intermediate grid     echosyn preprocess_volume
  -> _normalize               0.5/99.5 percentile of NONZERO voxels -> [0, 1]   echosyn _normalize
  -> plane_order permute      acquisition plane's slice axis leads              echosyn plane_order
  -> _crop_pad                in-plane 512^2, 15 mm posterior shift             echosyn _crop_pad
  -> _fit_slices              centre crop / zero pad the slice axis to 128      NEW (monolithic UNet)
  -> transpose (1, 2, 0)      to Text2CT's (X, Y, Z), slice axis last           NEW (3D VAE input)
     = float32 (512, 512, 128) in [0, 1]
  -> vae.encode_stage_2_inputs under inference_mode
     = float16 (4, 128, 128, 32), UNSCALED           ~4.19 MB per series

report.json (one per STUDY, many series share it)
  -> format_report            "Findings: ... Impression: ..."   (upstream's string, verbatim)
  -> CLIPTokenizer            truncation at 512 tokens
  -> FrozenCLIP3D.encode_text pooled, projected, L2-normalized
     = float32 (1, 768)                              ~3 KB per series
```

At training time the dataset returns `latent [4,128,128,32] f32`, `context [1,768] f32`,
`class_label [] i64`, `spacing [3] f32` = `[50, 50, 150]` (the grid's mm **times 1e2**, which is the
scale `scripts/diff_model_train.py:111` trains `spacing_layer` on). `fine_tune.diffusion_loss`
multiplies the latent by `scale_factor = 1.0287` -- upstream's `images = images * scale_factor`.

**No HU window is applied.** MRI has no Hounsfield scale, so
`scripts/preprocess_ctrate.py:42`'s clip to `[-1000, 1000]` has no MR counterpart. The `[0, 1]`
range is preserved, which is what the latent space is over (`diff_model_demo.py:194` maps
`[0, 1] -> [-1000, 1000]` HU only *at decode*) and what MRFlow independently measured for MR.

**Failure handling.** A series with fewer than `min_native_slices` after resampling, non-finite
voxels, an unreadable member, or an empty report raises a typed error (`VolumeUnusable`,
`ReportMissing`) and is counted and skipped by `prepare_data.py` rather than written. The kept
count and the per-type drop counts are printed per shard.

### Which grid, and why it is Text2CT's rather than MRFlow's

The released UNet was trained on a **single** CT grid, from `scripts/preprocess_ctrate.py:41-58`:

```
image  512 x 512 x 128   spacing 0.75 / 0.75 / 3.0 mm   FOV 384 x 384 x 384 mm
latent   4 x 128 x 128 x 32   ->  one latent voxel = 3.0 x 3.0 x 12.0 mm   anisotropy 1:4
```

(Note that script resamples to that spacing and *then force-resizes* to `(128, 512, 512)` rather
than cropping, so the recorded affine is nominal and the true per-case spacing drifts by ~20%. The
model saw a soft spacing distribution, not a hard one.)

Adopting MRFlow's 1 mm isotropic 256^2 grid would have been convenient -- it is the evaluation
grid, so nothing is resampled at ingest -- but it is **our** geometry, not this baseline's, and the
`baselines/` README's own rule is that a baseline brings its own. Concretely it would have put the
UNet on a `(4, 64, 64, 40)` latent whose voxels are `4 x 4 x 4` mm: a quarter of the in-plane tokens
it was trained with, and a through-plane scale 3x finer than anything it has seen. The NVIDIA
baseline makes the symmetric choice, deriving its grid from NVIDIA's published FOV table.

So the grid here reproduces **Text2CT's own latent shape**, `(4, 128, 128, 32)`. What is not kept is
its spacing: `0.75/0.75/3.0` is a 384 mm field of view, roughly 40% air for a brain, and it would
mean 3x through-plane upsampling at ingest. `0.5/0.5/1.5` puts the same latent over
`256 x 256 x 192` mm. Measured over the 575,328-series train manifest (depth at 1 mm: min 140,
median 162, p90 192):

| in-plane | slice | image | latent | depth FOV | brains fully covered | median fill | ingest resample | MB/series | full split |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | **1.5** | 512x512x128 | **4x128x128x32** | 192 mm | **91.5%** | **84.4%** | 1.5x up | 4.19 | 2.41 TB |
| 0.5 | 2.0 | 512x512x128 | 4x128x128x32 | 256 mm | 99.7% | 63.3% | 2.0x up | 4.19 | 2.41 TB |
| 0.75 | 3.0 | 512x512x128 | 4x128x128x32 | 384 mm | 100% | 42.2% | 3.0x up | 4.19 | 2.41 TB |
| *(rejected)* 1.0 | 1.0 | 256x256x160 | 4x64x64x40 | 160 mm | 48.7% | -- | none | 1.31 | 0.75 TB |

`slice_mm` is the one free parameter and 1.5 dominates: same latent, same cost, least padding, least
upsampling against the 1 mm reference. It is a one-line config change if you want a wider FOV.

**What keeping the latent shape actually buys** is the self-attention token count. `attention_levels:
[false, false, true, true]` puts attention at 2 and 3 downsamples, so on a `128 x 128 x 32` latent it
runs over `32 x 32 x 8 = 8,192` and `16 x 16 x 4 = 1,024` tokens -- **exactly** the counts the
released weights were trained with, since CT-RATE produced the same latent shape. The 1 mm grid's
`64 x 64 x 40` latent would have given 2,560 and 320: a third of the tokens, at a different physical
scale per token. Convolutions transfer across sizes; attention trained at one sequence length and a
different receptive scale is the part that does not.

**Two constraints the grid satisfies.** Every image axis is a multiple of 32 -- the UNet has four
levels so its latent must be divisible by 8, and the latent is the image `// 4`
(`R2V-MR-Generation/data/geometry.py`'s `UNET_SPATIAL_MULTIPLE`, verified empirically on this same
architecture). And `config.check_input_problems` restates `scripts/sample.py:25`'s own validator --
in-plane in `[256, 384, 512]`, depth in `[128, 256, ...]`, spacing in `[0.5, 3.0]` / `[0.5, 5.0]` --
so a grid the released sampler would refuse fails at config time. The 1 mm isotropic grid fails it
(160 is not an allowed depth); this one passes.

**The anisotropy is plane-aware.** `target_spacing_sra` puts `slice_mm` on the acquisition plane's
stacking axis -- S for axial, R for sagittal, A for coronal -- because `preprocess_volume` resamples
*before* it permutes. An isotropic grid hides this; an anisotropic one blurs the wrong axis and
nothing downstream notices, so it has its own test.

**What the extra in-plane resolution buys: nothing at evaluation time.** MR-RATE is natively
~0.7-1.0 mm in-plane, so 0.5 mm is interpolation, and `canonicalize_external` resamples the
generation back to 1 mm / 256^2 for scoring. It is there to put the UNet on the token count it was
trained with, not to add detail. Through-plane, 1.5 mm is genuinely coarser than the 1 mm reference,
and that loss is real and declared.

### One library patch: MONAI's MAISI autoencoder at 512 in-plane

Encoding a `(1, 1, 512, 512, 128)` volume takes **29.8 s** with stock MONAI, against 0.45 s at 384
in-plane. That is not compute: `MaisiGroupNorm3D.forward` (autoencoderkl_maisi.py:89) and
`MaisiConvolution._concatenate_tensors` (:214) both branch on `max(size) < 500` and, above that
threshold, concatenate their activations **on the CPU**, one chunk at a time, with a
`torch.cuda.empty_cache()` and a `gc.collect()` between chunks. Text2CT's grid is exactly 512, so it
lands on the wrong side of a hardcoded constant.

`model.set_fast_maisi_concat` replaces both concatenations with `torch.cat` on the GPU. Measured
on one h200:

| | encode time | peak | max abs z_mu diff |
|---|---|---|---|
| stock MONAI | 29.83 s | 15.4 GiB | -- |
| group-norm concat on GPU | 6.51 s | 16.5 GiB | 0.0 |
| **both concats on GPU** | **0.55 s** | 16.5 GiB | **0.0** |

**53.8x, bit-identical** -- `z_mu` and `z_sigma` agree exactly, because a concatenation is a
concatenation. Without it, caching the train split is ~5,000 GPU-h instead of ~280, which would make
this grid infeasible rather than merely expensive. `model.vae_fast_concat: false` restores stock
behaviour, and `test_fast_concat_is_bit_identical_to_stock_monai` is the guard. No weights and no
arithmetic change; only two `.to("cpu")` calls disappear.

Two related notes. `save_mem=False` is a further ~35% on grids below the threshold and is likewise
bit-identical, but it makes no difference at 512 (29.90 s vs 30.03 s) because the host round-trip
dominates. And `num_splits` -- 4 in `config_rflow.json`, 2 in the MAISI bundle -- is close to
irrelevant here: 4 -> 1 is 30.9 s -> 25.7 s and costs 15.4 -> 120.3 GiB, so it stays at the released 4.

**Caching is mandatory, in zips.** Two artifacts per series over the train split is >1M files
against this account's inode hard limit on `/hnvme` (81,000, with 63,000 already used -- the
epilogue of any job prints the live figure). `LatentStore` writes one `ZIP_STORED` archive per
`(split, shard)` and the manifest's `zip` column says so, close to MRFlow's layout:

```
<cache_root>/artifacts/train-0000.zip … train-0063.zip, val-0000.zip … val-0003.zip
                 <sample_id>.latent.npy   fp16 (4,128,128,32)   4.19 MB
                 <sample_id>.text.npy     fp32 (1,768)          ~3 KB
<cache_root>/manifest/train-0000.csv … , val-*.csv
<cache_root>/cache_meta.json
```

**The split is in the bundle name, and that is not cosmetic.** It was `bundle-NNNN.zip` at first,
so `train` shard 3 and `val` shard 3 were the same path; two array tasks opened it in `"w"` mode and
truncated each other's archive **in place**, while each still wrote a perfectly valid-looking
manifest. Caught on the first production launch, before any training consumed it.
`test_two_splits_never_share_a_bundle` is the guard.

`cache_meta.json` fingerprints the grid, the percentiles, the report sections and the sha256 of
both frozen checkpoints; `fine_tune.py` refuses to train on a cache whose fingerprint does not
match the config.

## Modality conditioning: the existing hook, reused

Text2CT already has generic class conditioning and it is already inside the UNet. MONAI's
`DiffusionModelUNetMaisi` builds `nn.Embedding(num_class_embeds, time_embed_dim)` and **adds the
lookup to the timestep embedding** (`diffusion_model_unet_maisi.py:319-323`), which is the design
this adaptation would otherwise have had to add. Text2CT sets `num_class_embeds: 128` and then uses
exactly one index: `scripts/diff_model_train.py:313` hardcodes `torch.ones(...)`.

So nothing is added to the architecture. What is decided is **which** indices and **what they start
at**:

- The ids are MAISI's own, from `R2V-MR-Generation/models/nvidia_configs/modality_mapping.json` --
  the vocabulary the 128-row table was sized for. `T1w=9 (mri_t1)`, `T2w=10 (mri_t2)`,
  `FLAIR=11 (mri_flair)`, `MRA=16 (mri_mra)`, `SWI=20 (mri_swi)`, `UNKNOWN=8 (mri)`,
  `CFG_NULL=0 (unknown)`. `ct = 1` is reserved and never reassigned.
- `modality_init: ct_row` (default) copies the **trained CT row** into every MR id, so at step 0 the
  fine-tune is bit-identical to the released model conditioned on CT
  (`test_ct_row_init_reproduces_the_pretrained_model_at_step_zero`). `zeros` is available and is
  what a from-scratch head would use, but it is *not* the pretrained function: the pretrained model
  always added row 1. Measured on the release, `||row_1|| = 2.18` against `0.07` for an unused row.
- `class_embedding` is a UNet parameter, so it trains with the UNet and needs no separate group.
- `modality_to_id` **raises** on anything unmapped -- there is no fallback id.
- The vocabulary is written into every checkpoint under `modality_vocabulary`, and resume refuses a
  checkpoint whose vocabulary differs.

**Plane is not conditioned at all.** Text2CT has exactly one class input and it is spent on
modality; the report string stays upstream's verbatim, so plane is not in the text either. This is a
deliberate limitation with a real consequence, stated again in the paper list: the grid is identical
for all three acquisition planes, so nothing tells the model which one to produce, and a generation
cannot be *asked* for a sagittal volume -- it draws from whatever marginal over planes the training
data induces. The evaluation buckets cases by `(modality, plane)`, so a case whose ground truth is
sagittal will usually be matched against an axial-looking generation.

The alternative that needs no conditioning input is to carry plane in the **geometry**, as the
NVIDIA baseline does: a per-`(modality, plane)` output shape, so an axial request is a short-D
volume and a sagittal request a short-H one. See "Which grid" below.

`data.modality_prefix` (default **false**) optionally prepends `"Brain MRI, T1-weighted. "` to the
report. It exists to test whether the frozen text tower adds anything on top of the class
embedding; it is off because that string is off-distribution for a CT-trained encoder and the class
embedding already carries the signal.

## Report classifier-free guidance: already there, unchanged

Text2CT implements report CFG and this adaptation changes none of it.

- **Training dropout exists**: `config_diff_model.json` sets `conditional_free_guidance: 0.1` and
  `scripts/diff_model_train.py:335-337` zeroes the context for that fraction of samples.
- **The null is a zero vector**, not a learned token -- both in training (above) and in sampling
  (`scripts/diff_model_infer.py:248`, `torch.zeros_like(impression)`).
- **Sampling is the standard form**, `scripts/diff_model_infer.py:262`:
  `out = uncond + s * (cond - uncond)`.
- **The guidance scale is configurable**: `guidance_scale: 5.0` in both environment configs.
- **Modality is already the same in both branches**: the unconditional inputs are a `.copy()` of the
  conditional ones taken *after* `class_labels` was set, so only the context differs.

That is exactly the form this task specifies, so nothing was invented. `cfg.report_dropout_prob`
defaults to upstream's `0.1` and `cfg.guidance_scale` to upstream's `5.0`.

**Modality CFG is off.** `cfg.modality_dropout_prob: 0.0` and `cfg.modality_guidance_scale: 1.0`,
so modality is always supplied and never guided -- it is acquisition metadata, not optional
semantics. The code path exists (it swaps in `CFG_NULL`, never a zero tensor, because a class
embedding is a lookup) and is experimental. Note that `CFG_NULL = 0` is **not** a trained null in
this checkpoint: row 0 carries a norm of 1.88 from MAISI pretraining, so a modality-CFG experiment
would be guiding against MAISI's `unknown` class rather than against nothing.

## What `fine_tune.py` adds over `scripts/diff_model_train.py`

Gradient accumulation and clipping; a non-finite-gradient skip with a consecutive-failure abort
(bf16 runs without a `GradScaler`, so nothing else would drop a bad update, and one nan reaches
every weight through `clip_grad_norm_`); a `DistributedSampler` with `set_epoch`, so shuffling
crosses ranks; a validation loss on a held-out split at a fixed noise seed; step-based
checkpointing with a resume that actually restores the step, the optimizer and the scaler; bf16 as
well as fp16; and the freezing assertions. Checkpoints keep upstream's key names
(`unet_state_dict`, `scale_factor`), so `scripts/diff_model_infer.py` loads one unchanged.

`model.scale_factor` stays at the released `1.0287`. Upstream *recomputes* it as `1/std(z)` from
the first batch (`scripts/diff_model_train.py:209`), which is right for a scratch run and wrong
here -- the trunk was trained with that number. `recompute_scale_factor` raises rather than doing
it quietly; this is the same call MRFlow makes about FLUX's `scaling_factor`.

## Freezing

The VAE and the report encoder are frozen four ways -- `eval()`, `requires_grad_(False)`, excluded
from the optimizer, and forwarded under `inference_mode` -- and `freeze()` also replaces `.train()`
with a no-op so a stray `.train()` from a wrapper cannot re-enable dropout or running statistics.
`FrozenGuard` checksums every frozen module at load and re-checks after training.

In the default cached-artifact mode the report encoder is **not resident at all**, which is a
stronger guarantee than freezing it; `data.load_text_encoder: true` loads and freezes it so the
assertions cover it too. The VAE is loaded whenever it is on disk, so the parameter report and the
checksum always cover a real module.

## Running it

Everything runs from the repo root with the root on `PYTHONPATH`, in the Text2CT venv (below).

```bash
WS=/hnvme/workspace/y100dc19-mrflow-final
VENV=$WS/baselines/venv-text2ct
CFG=baselines/text2ct/configs/mrrate_finetune.yaml

# 1. bounded self-check FIRST -- builds every component, trains three steps on synthetic latents,
#    runs the freezing assertions and one CFG sampling pass. Needs no cache and no MR-RATE.
$VENV/bin/python -m baselines.text2ct.fine_tune --config $CFG --smoke --steps 3

# 2. cache the latents and the report embeddings. 1.76 s per series measured, so 64 tasks over the
#    575,328-series train split is ~4.5 h each; val is four tasks of ~3 min. 2.41 TB in total.
sbatch --array=0-3 --gres=gpu:h200:1 --partition=h200 --time=04:00:00 \
  --wrap "cd \$SLURM_SUBMIT_DIR && PYTHONPATH=\$PWD $VENV/bin/python \
    -m baselines.text2ct.prepare_data --config $CFG \
    --split val --shard \$SLURM_ARRAY_TASK_ID --num_shards 4"
sbatch --array=0-63 --gres=gpu:h200:1 --partition=h200 --time=08:00:00 \
  --wrap "cd \$SLURM_SUBMIT_DIR && PYTHONPATH=\$PWD $VENV/bin/python \
    -m baselines.text2ct.prepare_data --config $CFG \
    --split train --shard \$SLURM_ARRAY_TASK_ID --num_shards 64"

# 3. the self-check again, now against the real cache -- this is what verifies the cache metadata
$VENV/bin/python -m baselines.text2ct.fine_tune --config $CFG --smoke --steps 3
```

**The full fine-tune** -- 60,000 optimizer updates at an effective batch of 128, 8 h200 over two
nodes, ~7.5 h. Save as e.g. `slurms/text2ct_finetune_helma.sh` and `sbatch` it (this directory holds
no shell scripts, by the `baselines/` README's rule that the job scripts live in `slurms/`):

```bash
#!/bin/bash -l
#SBATCH --gres=gpu:h200:4 --partition=h200 --nodes=2 --ntasks-per-node=1
#SBATCH --cpus-per-task=32 --time=24:00:00
#SBATCH --output=logs/text2ct_ft_%j.out --error=logs/text2ct_ft_%j.err
unset SLURM_EXPORT_ENV
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=$PWD:$PYTHONPATH TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
# `python -m torch.distributed.run`, not `torchrun`: the venv layers on the MRFlow one and has no
# console scripts of its own. c10d rendezvous assigns the node rank, so none is passed.
srun --export=ALL /hnvme/workspace/y100dc19-mrflow-final/baselines/venv-text2ct/bin/python \
  -m torch.distributed.run \
  --nnodes "$SLURM_JOB_NUM_NODES" --nproc_per_node 4 \
  --rdzv_backend c10d --rdzv_id "$SLURM_JOB_ID" --rdzv_endpoint "$MASTER:29500" \
  -m baselines.text2ct.fine_tune \
  --config baselines/text2ct/configs/mrrate_finetune.yaml
```

`--resume latest` is the default, so a requeued job continues from the newest checkpoint in
`output_dir`. Resume restores the step, the optimizer and the scaler, but not the position inside
an epoch -- a requeue re-reads its epoch from the start, so at most a fraction of one epoch of
samples is seen twice. Over 13.35 epochs that is noise, and the alternative (persisting the
sampler's cursor) buys less than the complexity costs. On an 80 GiB card add `--set train.micro_batch_size=4 train.gradient_accumulation_steps=4`,
which is the same effective batch and the same 60,000 steps.

Running the regression net:

```bash
$VENV/bin/python -m pytest baselines/text2ct/test_finetune.py -v      # needs one GPU
```

## Environment

The upstream tree needs `transformers < 5` for its vendored CLIP, which the MRFlow venv does not
have. `$WS/baselines/venv-text2ct` is a `--system-site-packages` venv over the MRFlow one with
`transformers==4.49.0`, `opencv-python-headless`, `vector-quantize-pytorch`, `ema_pytorch`,
`beartype`, `ftfy` and `easydict` layered on:

```bash
python -m venv --system-site-packages $WS/baselines/venv-text2ct
$WS/baselines/venv-text2ct/bin/pip install transformers==4.49.0 opencv-python-headless \
    vector-quantize-pytorch==1.1.2 ema_pytorch==0.7.7 beartype ftfy easydict
```

`transformers==4.30.1` (upstream's pin) has no Python 3.12 wheel for its `tokenizers<0.14`
dependency, and 4.49 breaks the *vendored* slow tokenizer's constructor. `text_encoder.py`
therefore substitutes the installed `transformers.CLIPTokenizer`, which reads the same
`openai/clip-vit-large-patch14` vocab and merges and produces the same ids; the vendored
`CLIP3DModel` -- the actual text tower -- is untouched.

## Compute: matching MRFlow's budget

MRFlow's own MR-RATE adaptation is the comparator, and its budget is on disk in
`experiments/mrflow_from_ct_checkpoint/config.yaml` (job 798342, completed 2026-08-30):

```
max_train_steps 60,000     batch_size 16/GPU    gradient_accumulation 1    8 x h200 (2 nodes)
  -> effective global batch = 16 x 1 x 8 = 128
  -> 575,328 train series / 128 = 4,494 optimizer steps per epoch (drop_last)
  -> 60,000 / 4,494 = 13.35 epochs,  60,000 x 128 = 7.68M samples seen
lr 1e-4 (5-arm swept at this batch)  warmup 500  bf16  max_grad_norm 1.0  10.8 h, ~86 GPU-h
```

**The step count is matched; the epoch count follows from it, and "samples seen" is not
comparable.** One MRFlow sample is a 16-slice latent block *pair* drawn at random from a series, so
the same series yields a different crop every epoch; one sample here is a whole volume, identical
every epoch. With `mrrate.max_repeats: null` this config trains on the same 575,328 series at the
same effective batch for the same 60,000 updates, i.e. the same 13.35 epochs -- which is as close
as the two formulations can be brought.

### Measured on one h200 (this grid, bf16, AdamW, `unet` 217,659,524 parameters)

| micro-batch | s / micro-step | peak GiB | samples/s |
|---|---|---|---|
| 1 | 0.130 | 12.7 | 7.7 |
| 2 | 0.204 | 22.4 | 9.8 |
| 4 | 0.358 | 41.8 | 11.2 |
| 6 | 0.517 | 61.2 | 11.6 |
| **8** | **0.672** | **80.6** | **11.9** |
| 12 | 0.984 | 119.4 | 12.2 |

Throughput saturates by micro-batch 8, and 12 is the last that fits in 143 GiB. Preprocessing,
measured over 12 real val series **with the concat patch**: 0.51 s archive read + 0.47 s
resample/normalize + 0.78 s VAE encode = **1.76 s/series**, i.e. ~281 GPU-h and **2.41 TB** for the
full train split at 4.19 MB per latent. Median depth fill 82%, as predicted.

### Three configurations, all 60,000 steps at an effective batch of 128

| | GPUs | micro-batch | accum | effective | peak | wall clock | GPU-h | epochs |
|---|---|---|---|---|---|---|---|---|
| conservative | 8 | 4 | 4 | 128 | 41.8 GiB | ~24 h | ~192 | 13.35 |
| **recommended** | **8** | **8** | **2** | **128** | **80.6 GiB** | **~22 h** | **~180** | **13.35** |
| larger memory | 16 | 8 | 1 | 128 | 80.6 GiB | ~11 h | ~180 | 13.35 |

None fits an 80 GiB card at micro-batch 8; the conservative row is the one that does (41.8 GiB).
Wall clocks exclude the DDP all-reduce (217M bf16 parameters per step) and dataloading, which is one
4.19 MB read per sample.

**This is ~2x MRFlow's training budget** (180 GPU-h against 86) at the same optimizer-step count,
effective batch and training population, because one Text2CT sample is a whole `(4, 128, 128, 32)`
volume while one MRFlow sample is a `(16, 16, 32, 32)` block pair. Preprocessing is ~281 GPU-h
against MRFlow's ~45, and storage 2.41 TB against ~1.5 TB. That is the price of putting the UNet on
the grid it was trained at rather than on ours, and it is the right way round for a baseline: the
comparison should not be able to be dismissed as under-training or a resolution handicap.

### Learning rate: why 1e-4 and not a scaled value

Upstream trains at `lr 1e-4`, batch 4, one GPU. Naively scaling that to an effective batch of 128
gives `3.2e-3` linearly or `5.7e-4` by the square root. **Neither is applicable here**, for two
separate reasons:

- Batch-scaling rules exist to hold the *per-sample* step size constant as gradient noise falls, and
  they are derived for from-scratch SGD. A fine-tune's goal is the opposite: stay in the basin the
  pretrained weights are in. A larger batch should be spent on a cleaner gradient, not a bigger step.
- Adam and AdamW normalize by the gradient's own second moment, so the update magnitude is already
  roughly scale-free. What a larger batch changes is the variance of that estimate, not the step
  length, and the linear rule is an SGD result that does not carry over cleanly.

The anchor that *is* applicable is empirical and from this repository: MRFlow swept five learning
rates **at exactly this effective batch, with a pretrained trunk** and chose `1e-4`. Text2CT's UNet
is 218M parameters against STDiT-L's 512M, so if anything it tolerates that value more easily. It is
also upstream's own pretraining lr, which makes it a provably stable value for this network -- and
at 32x the batch it is a strictly *more* conservative per-sample step than pretraining used.

Warmup is 500 steps (short: the trunk is pretrained, not random) and the schedule is cosine to
`1e-6`, both MRFlow's. `scheduler: poly` reproduces upstream's `PolynomialLR(power=2.0)` for anyone
who wants the method-faithful shape; it has no warmup, which is the wrong start for an already
converged UNet.

`optimizer: adamw, weight_decay: 1e-2` follows MRFlow rather than upstream's plain Adam with no
decay -- the one-protocol-for-both-sides argument. Over 60,000 steps at `lr ~1e-4` the decoupled
decay shrinks weights by roughly 6%. `optimizer: adam, train.weight_decay: 0` is the faithful
alternative and is one `--set` away.

**Parameter counts** (all of the UNet trains, including `class_embedding`):

```
unet           trainable  217,659,524      (includes class_embedding: 128 x 256 = 32,768)
vae               frozen   20,944,897
text_encoder      frozen  785,818,378      (text tower 123,984,896 + vision tower 661,833,481)
                                            not resident at all in the default cached mode
```

## What has actually been run

Every command below was executed; nothing in this section is projected.

These were run at the final grid (512 x 512 x 128 @ 0.5/0.5/1.5, latent `(4, 128, 128, 32)`).

| check | result |
|---|---|
| `pytest baselines/text2ct/test_finetune.py` on one h200 | **66 passed, 1 skipped** (the skip needs `echosyn`, which does not import under `transformers` 4.x) |
| that one test in the MRFlow venv | **passed** -- the vendored `list_series` returns the identical series ids and tar members as `echosyn.common.mrrate.list_series` for the same split, cap and seed |
| `fine_tune.py --smoke` with no cache | 3 steps on synthetic latents, frozen VAE verified unchanged, one CFG sampling pass finite |
| `prepare_data.py` on 16 real MR-RATE val series, 2 shards | 16 kept, 0 dropped; **5 files total** (2 bundles, 2 manifests, `cache_meta.json`) |
| preprocessing throughput, 12 real val series | 1.76 s/series (0.51 read + 0.47 resample + 0.78 VAE encode), median depth fill 82% |
| the MONAI concat patch | 29.83 s -> 0.55 s at 512 x 512 x 128, `z_mu` and `z_sigma` **bit-identical** |
| `fine_tune.py --smoke` on that real cache, 2-GPU DDP | 4 steps, loss 1.10 -> 0.84, grad norm 0.3-0.9, no skipped steps, frozen VAE unchanged |
| resume, 2-GPU DDP | picked up at step 4 from `checkpoint-0000004.pt`, continued to 6 |
| cache-metadata guard | changing `volume.num_slices` refused the run with a diff of the two settings |
| `scripts/diff_model_train.py` **unmodified** on an emitted layout, 16 real val series | ran to completion, released checkpoint loaded, one epoch, mean loss 0.9510 against our 1.0046 on the same data |
| `test_our_loss_equals_upstreams_own_train_one_epoch` | upstream's imported `train_one_epoch` and `fine_tune.diffusion_loss` agree to `rel=1e-6` on the same batch and seed |
| 2-rank DDP on 2 h200 | `torch.distributed.run --nproc_per_node 2`: effective batch 4 (1 x 2 x 2), the `DistributedSampler` split 16 series into 8 micro-batches per rank, 4 steps, checkpoint, then a clean resume to step 6. Frozen VAE unchanged on both runs |

Parameter report as printed by the run:

```
parameters by component:
  unet           trainable  217,659,524   frozen            0
  vae            trainable            0   frozen   20,944,897
  TOTAL          trainable  217,659,524   frozen   20,944,897
modality vocabulary: {'CFG_NULL': 0, 'UNKNOWN': 8, 'T1w': 9, 'T2w': 10, 'FLAIR': 11,
                      'MRA': 16, 'SWI': 20} (init=ct_row)
```

A checkpoint is 2.6 GB (fp32 weights plus Adam's two moments), so `keep_checkpoints: 4` is ~10.5 GB.

## Generating and scoring the frozen population

Same contract as every other baseline (`baselines/README.md`): `generate.py` writes
`$WS/baselines/runs/<tag>/nifti/<case_id>.nii.gz` and stops; the score job ingests those, drops
them, and scores with **the evaluation code unchanged**.

```bash
# every retained checkpoint, on val -- generation array + dependent score job per cell
bash slurms/text2ct_ckpt_sweep.sh
DRY_RUN=1 bash slurms/text2ct_ckpt_sweep.sh        # print the sbatch lines, submit nothing
bash slurms/text2ct_ckpt_sweep.sh 60000            # one checkpoint

# or by hand
sbatch --array=0-15 slurms/text2ct_run_shards.sh 60000
SPLIT=val slurms/baseline_score.sh text2ct_val_step60000 "text2ct mrrate-ft step60000 cfg5 val"
```

Cells that already have a `metrics.json` are skipped, so re-running resumes a partial sweep, and
`generate.py` skips a case whose NIfTI exists, so a preempted array task resumes rather than redoing
its shard. MRA cases are skipped outright -- the evaluation excludes that modality -- and `ingest.py`
still records them as excluded.

**Selection happens on val, never on test.** Checkpoint and guidance scale are hyper-parameters like
any other; `baselines/cases-val-n100.json` exists precisely so they can be chosen without
contaminating the reported split. Score the winner once on `cases-test-n100.json`.

### What the driver has to get right

Each of these is a silent failure -- plausible metrics, never an error:

- **The report string is rebuilt from the checkpoint's own config**, not from a default, so it is
  byte-identical to what the fine-tune trained on. Plane is absent, by design.
- **The modality id comes from the checkpoint's recorded `modality_vocabulary`**, and a mismatch
  against the current table raises. A class id means nothing without the table it was trained under.
- **No HU mapping.** `diff_model_demo.py:194` maps the decoder's `[0, 1]` to `[-1000, 1000]` HU --
  a CT convention. MR has no Hounsfield scale and this fine-tune's targets were `[0, 1]`, so the
  decoder output is written as-is. This is the analogue of `nvidia_r2v`'s `postprocess=False`.
- **The sampling loop is upstream's.** `diff_model_infer.py:run_inference` reproduced call for call,
  with `ReconModel` and `dynamic_infer` imported from the pinned clone rather than restated.
- **The geometry is inverted, not re-derived.** `save_generated_nifti` undoes `preprocess_volume`
  step for step so `ingest.py`'s `read_canonical` -- the same function the ground-truth path uses --
  recovers the orientation the model generated in. Two tests: an exact round trip on a deliberately
  non-cubic array for all three planes, and a real MR-RATE series round-tripped through the
  evaluation's own ingest geometry, which lands at **0.985 correlation** against MRFlow's grid. A
  transposed axis leaves every shape intact and correlation near zero, so the second test is the one
  that matters.
- **Seeding is off the case id**, as `evaluation/main.py` and the R2V driver both do, so a rerun at a
  different `--num_shards` is the same volume.

Measured: ~40 s/case (30 Euler steps plus upstream's sliding-window decode at 512x512x128), so 1,002
cases over 16 tasks is well under an hour each. One smoke case ingests to a `(192, 256, 256)` fp16
`[0, 1]` cache entry -- 128 slices at 1.5 mm becoming 192 at 1 mm is the resample landing where it
should.

## Declaring this in the paper

Call it **"Text2CT adapted to MR-RATE"**, never an unchanged reproduction. What changed:

1. **Fine-tuned**, from the preprint release's `unet_rflow_200ep.pt`, on MR-RATE brain MRI. The VAE
   and the 3D-CLIP encoder are the released CT ones, frozen -- the same protocol as MRFlow's own
   adaptation (a CTFlow trunk, a frozen CXR-BERT, a frozen FLUX VAE).
2. **Modality conditioning added** through the UNet's existing MAISI class-embedding input, which
   the released model used for a single constant (`ct`). The MR ids are MAISI's own.
3. **Plane is not conditioned**, in either the class input or the text. The output grid is the same
   for all three acquisition planes, so a generation cannot be requested in a given plane and the
   model draws from the marginal over planes. Since the evaluation buckets by `(modality, plane)`,
   most cases are scored against a reference acquired in a plane the generation was not asked for.
   This is a handicap relative to MRFlow, which conditions on plane explicitly, and must be said.
4. **Preprocessing replaced**: no HU window, and MR percentile normalization over nonzero voxels.
   The grid keeps the released model's **latent shape** `(4, 128, 128, 32)` and changes its
   spacing, `0.75/0.75/3.0` (a 384 mm CT field of view) to `0.5/0.5/1.5` (256 x 256 x 192 mm, brain
   sized). In-plane 0.5 mm is interpolation from MR-RATE's native ~0.7-1.0 mm -- it buys token
   count, not detail -- while 1.5 mm through-plane is genuinely coarser than the 1 mm reference the
   evaluation scores against, a real resolution loss.
5. **Optimization changed**: AdamW with decay, cosine-with-warmup, gradient accumulation and
   clipping, in place of Adam + `PolynomialLR` with neither -- chosen to match MRFlow's fine-tune
   budget rather than upstream's from-MAISI schedule.
6. **The VAE ceiling is not ours and must be stated**: `ablations/vae_ceiling.py` measures MAISI at
   26.19 dB / 0.465 SSIM on MR-RATE against FLUX's 36.70 dB / 0.982. Both MAISI-based baselines
   generate against a ceiling ~10.5 dB below MRFlow's.
7. **The released code is the preprint model**, not the BMVC one; the hard-negative contrastive
   encoder is not in `887caa9` and was not reimplemented.
8. **The spacing conditioning is out of the trained distribution.** `include_spacing_input: true`
   and the released model only ever saw `0.75/0.75/3.0` mm (every CT-RATE volume was resampled to
   it), so the MR grid's `1/1/1` mm is a value `spacing_layer` was never fine-tuned on. It does
   become constant again during this fine-tune, so the layer adapts; the deviation is that the
   *released* weights for it are CT-specific.
9. **Sampling depth is fixed at 128 slices of 1.5 mm (192 mm)** where MRFlow rolls out to a learned
   length. 91.5% of MR-RATE brains fit inside that field of view; the rest are centre-cropped, and
   the median volume fills 84% of it, so a generation carries some blank depth by construction. The
   per-plane real-vs-generated slice counts the evaluation reports therefore differ for a reason
   that is not a length error.
10. **Not done, and not claimed**: no hard-negative mining, no retraining of the contrastive
    image/text encoders, and no re-derivation of the latent `scale_factor`.
