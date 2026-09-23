# CCELLA on MR-RATE

CCELLA (Class-Conditioned Efficient Large Language model Adapter) adapted from 3D prostate T2 to
MR-RATE brain/spine MRI, with **the smallest change to upstream that the data and the storage
allow**. The architecture, the training step, the optimizer, the schedule, the noise objective, the
MAISI autoencoder and the FLAN-T5 report representation are all upstream's and are called, not
reimplemented.

    upstream   https://github.com/grabkeem/CCELLA
    commit     619dfbb6f5c20eaa6724a970485eb0a95c22146f  ("Revise README", 2026-05-12)
    paper      Grabke, Taati, Haider. IEEE TBME 2026. arXiv:2506.10230, doi:10.1109/TBME.2025.3648426
    clone      /hnvme/workspace/y100dc19-mrflow-final/baselines/upstream/CCELLA
    licence    Apache-2.0 (code); NVIDIA non-commercial (the MAISI autoencoder weights)

**Nothing in this directory duplicates upstream.** What lives here is the MR-RATE reader, the zip
cache, the 14-label plumbing, the driver that runs upstream's step under a global-step schedule,
and one 21-line patch.

---

## 1. What is exactly original

| piece | where it comes from |
|---|---|
| The `CCELLA_LDM` architecture: perceiver-resampler cascade, six blocks, 256 x 768 adapter tokens, MAISI U-Net | `scripts/text_ldm.py`, subclassed in [`model.py`](model.py) |
| The training step: noise draw, timestep draw, `add_noise`, U-Net call, L1 objective, classification term, `GradScaler`, gradient clip, `lr_scheduler.step()` | `scripts.diff_model_train_all.train_one_epoch`, called from [`train.py`](train.py) |
| Optimizer (`create_optimizer`, one AdamW group -- `multiple_lr` is in none of upstream's released configs), LR schedule (`create_lr_scheduler`, `PolynomialLR` power 2), latent scale factor (`calculate_scale_factor`), DDPM scheduler and model construction (`define_instance`, `load_unet_text`) | `scripts/diff_model_train_all.py`, `scripts/utils.py` |
| Volume preprocessing: RAS orientation, `ScaleIntensityRangePercentiles(0, 99.5, clip)`, `encode_stage_2_inputs`, `ReconModelRaw`'s `z / scale_factor` decode | MONAI's own transform and upstream's own modules, via [`volume.py`](volume.py) |
| Report representation: FLAN-T5-XXL `T5EncoderModel`, `T5Tokenizer(truncation_side="left")`, `encode_plus(padding="max_length", truncation=True, max_length=512)`, `last_hidden_state` | `scripts/data_processing/gen_json_maisi_merged.py`, via [`text.py`](text.py) |
| The conditioning order: timestep embedding **+** modality class embedding, then concat pathology, then concat spacing; the `* 1e2` conditioning scale | `diffusion_model_unet_maisi_prostate.py::_get_time_and_class_embedding` / `_get_input_embeddings`, untouched |
| The frozen autoencoder | `model_maisi_autoencoder_epoch273_alternative.pt`, sha256 `1f8a7a05...` -- byte-identical to the file Text2CT and NV-Generate-MR-Brain already use in this workspace, so nothing was downloaded |

## 2. Every modification, and why

### 2.1 The one upstream edit

[`upstream.patch`](upstream.patch) — **+21 / −3 lines, one function**,
`scripts/diff_model_train_all.py::train_one_epoch`.

| hunk | change | why |
|---|---|---|
| signature | `step_hook=None` | `train_one_epoch` runs a whole epoch and returns. Step-based checkpointing at an explicit step list, validation at a step interval, per-step W&B logging and an exact-step resume all need control *inside* the loop. No dependency injection reaches inside a function that does not yield. |
| `loss_torch = torch.zeros(2)` → `zeros(5)` | two more accumulator slots | the diffusion and classification terms have to be reported separately |
| the U-Net call | `modality_id=train_data["modality_id"].to(device)` | modality is structured metadata and an explicit control, not text |
| `isnull.unsqueeze(1).repeat(1,2)` → `repeat(1, class_loss.shape[-1])` | mask as wide as the loss | the hard-coded `2` is upstream's class count; 14 labels need 14 |
| `clip_grad_norm_(...)` | keep the returned norm | it is already computed; logging it costs nothing |
| after `lr_scheduler.step()` | call `step_hook`, break on `False` | the hook above |
| before the class term is added | `diffusion_loss = loss.detach()` | name the diffusion term before it is summed with the classification term |

Every line that samples noise, calls the U-Net, forms the loss, steps the scaler and steps the
scheduler is untouched. [`upstream.py`](upstream.py)`::verify_upstream` checks the checkout is
exactly the pinned commit **plus** this patch — it reverse-applies the patch and fails on a drifted
checkout, a missing patch or a stale one; `require_upstream()` runs before preprocessing and before
training, and `test_ccella.py` asserts the patch touches only that file and has not grown.

To apply it to a fresh clone:

```bash
git -C /hnvme/workspace/y100dc19-mrflow-final/baselines/upstream/CCELLA \
    apply /path/to/MRFlow/baselines/ccella/upstream.patch
```

### 2.2 The model: a subclass with three replacements

[`model.py`](model.py)`::CCELLAMRRate` subclasses `CCELLA_LDM` and replaces:

1. **the six `ClassAdaptor` output layers, 2 → 14.** Upstream constructs `ClassAdaptor()` with its
   defaults inside `PerceiverResamplerCascade.__init__` (`text_ldm.py:288`), so `num_classes`
   cannot be threaded through the constructor. The **fusion classifier needs no change** — it is
   already `nn.Linear(layers*num_classes, num_classes)`, so passing `num_classes=14` sizes it to
   84 → 14 by itself.
2. **`unet.pirads_layer`, input width 2 → 14.** `diffusion_model_unet_maisi_prostate.py:160`
   builds it with a literal `2`; it is rebuilt by calling upstream's own
   `_create_embedding_module(14, time_embed_dim)`.
3. **`forward`: `softmax` → `sigmoid`, and `modality_id` passed as `class_labels`.**

Nothing else. `configs/ccella_def.json` is upstream's `configs/CCELLA_def.json` with **exactly
three values** changed:

```
-  "_target_": ".scripts.text_ldm.CCELLA_LDM",     +  "_target_": "baselines.ccella.model.CCELLAMRRate",
-  "num_classes": 2,                               +  "num_classes": 14,
                                                   +  "num_class_embeds": 7
```

### 2.3 Deviations that the data forces

| # | upstream | here | why it could not stay |
|---|---|---|---|
| D1 | binary PI-RADS, softmax + `FocalLoss(alpha=0.75, use_softmax=True)` | 14 non-exclusive groups, `BCEWithLogitsLoss(pos_weight, reduction="none")` | 41.4% of train studies carry `PP_Unspecific_bucket` and 18.7% `PP_Neurodegenerative`, often together, up to 11 at once. A softmax forces a simplex and turns "which ones" into "which one". Unreduced output keeps upstream's masking lines and its `sum over classes / n_unmasked` reduction working verbatim. |
| D2 | `pirads` multiplied by `1e2` in the dataloader | the **target** is left at 0/1; `spacing` keeps the `* 1e2` | a BCE target must lie in `[0, 1]`. Upstream's scale is harmless on a softmax cross-entropy target (it just multiplies the term by 100) but is undefined for BCE. The *conditioning* vector keeps the `* 1e2` scale, so the U-Net's input distribution is unchanged. |
| D3 | `text_class_pred_weight = 1e-4` against a 2-class target scaled by 100 | **unchanged at 1e-4**, against 14 unscaled BCE terms | not tuned, and deliberately not tuned on any split. The two effective magnitudes land within about 2x of each other: upstream ≈ `1e-4 x 100 x 2 x focal(~0.1)` ≈ 1e-3 per sample, here ≈ `1e-4 x 14 x BCE(~0.3)` ≈ 4e-4. Both losses are logged separately so the actual ratio is visible in W&B. |
| D4 | volumes already resampled to a fixed size outside the pipeline | trilinear to 1 mm isotropic + centred crop/pad to `256 x 256 x 224` | MR-RATE ships native-space volumes in tars; upstream's README states the fixed size as a precondition, so for this dataset it has to be made explicit. |
| D5 | per-case voxel size as the `spacing` condition | the **native acquisition** spacing | ours is fixed at 1 mm by construction and would carry no information; native spacing is what varies and what says a 5 mm-slice acquisition stays blurred through-plane. |
| D6 | one `.nii.gz` per latent, one `.npy` per report, in a directory tree | zip shards | 575,328 series is over half a million inodes against an 81,000 hard limit on `/hnvme`. |
| D7 | latents and embeddings stored fp32 | stored fp16, cast to float on load | halves ~1 TB to ~0.5 TB. FLAN-T5 hidden states and MAISI latents both sit well inside fp16 range. |
| D8 | epoch-based checkpointing, TensorBoard, no validation loop | global-step checkpointing, W&B, validation on a fixed subset | required here; implemented through the one `step_hook`, with a `NullWriter` absorbing upstream's `add_scalar` calls. |
| D9 | `gen_json_maisi_merged.py` encodes reports | replaced by [`text.py`](text.py) + [`prepare_data.py`](prepare_data.py) | that script **does not run as released**: it indexes a list with a string key (`train_dict['text']` inside a loop over `entry`), twice. |
| D10 | `synthetic_datagen.py` writes generated volumes | not used | its save path is prostate-specific (`imkey` ∈ {axt2,b1600,adc}, a `* 962` intensity rescale, SimpleITK flips). Generation for the baseline table is **not part of this implementation** — see §11. |
| D11 | `tokenizer.encode_plus(...)` | `tokenizer(...)` | transformers v5 removed `encode_plus`. For a single string it always delegated to `__call__`, and every keyword is upstream's, unchanged. Forced by the environment, not a semantic change. |

---

## 3. The 14 labels

**Source** — the local, authoritative artifact, produced by
`MR-RATE/contrastive-pretraining/scripts/eval_labels/build_merged_group_labels.py --source majority`:

```
MR-RATE/contrastive-pretraining/scripts/eval_labels/splits_merged_majority/
    mrrate_merged_labels.csv    study_uid + 14 binary columns   (97,896 rows)
    splits.csv                  batch_id,patient_uid,study_uid,split
    group_definitions.json      group -> member pathologies
```

Three models (Claude Opus 4.7, GPT-5.5, Nemotron-3 Super 120B) label each study over MR-RATE's own
37 SNOMED/RadLex pathologies; a pathology is positive when a strict majority of the *present* votes
agree; a group is the logical **OR** of its members. Two of the 37 (`Empty sella syndrome`,
`Hyperostosis of skull`) belong to no group and are excluded upstream.

**This is not the 37-label schema.** `<study>/labels.json` in the archives and the dataset's
`pathology_labels/mrrate_labels.csv` are the single-model (Qwen3.5-35B) 37 categories — a different
artifact with a different vote rule. No mapping between them is invented here;
[`labels.py`](labels.py) reads the 14-column CSV and **asserts its header** against `LABELS_14`
rather than trusting it.

**Order is the CSV header order and may never move** — a head's output channel is meaningless
without it, and a checkpoint outlives a run. It is written into every checkpoint and checked on
resume.

| # | label | members |
|---|---|---|
| 0 | `PP_Cerebrovascular` | infarction, hemorrhage, lacunar infarct, micro-hemorrhage, cavernous hemangioma, subdural hemorrhage, aneurysm, watershed infarct |
| 1 | `PP_Neoplastic` | metastasis, meningioma, glioma, pituitary adenoma, schwannoma |
| 2 | `PP_Neurodegenerative` | cerebral atrophy, ventriculomegaly, cerebellar degeneration |
| 3 | `PP_Spinal` | disc herniation, cord compression, foraminal stenosis, spinal stenosis, vertebral hemangioma |
| 4 | `PP_Cystic_developmental` | arachnoid cyst, pineal cyst, cavum septi pellucidi, mega cisterna magna, Chiari, Rathke's cleft cyst, choroid plexus cyst, lipoma |
| 5 | `PP_Infectious` | mastoiditis, chronic mastoiditis |
| 6 | `PP_Inflammatory` | demyelinating disease |
| 7 | `PP_Unspecific_bucket` | gliosis, cerebral edema, encephalomalacia |
| 8 | `BP_Atrophies` | *= PP_Neurodegenerative* |
| 9 | `BP_Contrast_enhancing_intracranial` | *= PP_Neoplastic* |
| 10 | `BP_Infectious_lesions` | *= PP_Infectious* |
| 11 | `BP_Edematous_lesions` | infarction, lacunar infarct, watershed infarct, demyelinating disease |
| 12 | `BP_Hemorrhagic_lesions` | hemorrhage, micro-hemorrhage, cavernous hemangioma, encephalomalacia |
| 13 | `BP_Cystic_lesions` | arachnoid cyst, pineal cyst, cavum septi pellucidi, mega cisterna magna, Rathke's cleft cyst, choroid plexus cyst |

**Three PP/BP pairs are identical by construction** (8≡2, 9≡1, 10≡5 — same member lists), and the
audit confirms they are identical in the data for all 97,896 rows. They contribute three duplicate
columns to the loss and three duplicate entries to any macro average. This is a property of the
label artifact, not of this code, and it is left alone rather than silently de-duplicated.

### 3.1 Coverage and prevalence audit

Produced by [`audit_labels.py`](audit_labels.py); full machine-readable output in
[`label_audit.json`](label_audit.json). No identifiers, no report text.

```bash
python -m baselines.ccella.audit_labels --out baselines/ccella/label_audit.json
```

| split | studies w/ report | with all 14 labels | % | with none | eligible series | series w/o labels | all-negative studies |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 88,864 | 88,582 | 99.68% | 282 | 575,328 | 1,157 | 41.3% |
| val | 3,775 | 3,764 | 99.71% | 11 | 23,358 | 38 | 44.4% |
| test | 5,561 | 5,550 | 99.80% | 11 | 34,452 | 50 | 42.7% |

- **Partial labels do not exist in this schema.** The upstream builder resolves every vote into a
  0/1 before writing and *skips* a study no model saw, so a row is always complete and missingness
  is per study. The mask is shaped per label anyway, so a future source with per-label uncertainty
  needs no change.
- **No duplicate or conflicting rows**; `read_label_table` raises on either.
- **0 study ids in the label table that MR-RATE does not know**, **0 studies with labels but no
  report**, **0 split disagreements** between the label artifact's `splits.csv` and MR-RATE's own.
- **No constant or near-constant label**: the rarest, `PP_Spinal`, is 2.05% of train.
- Positives per study (train): 0 → 36,569 · 1 → 15,066 · 2 → 12,696 · 3 → 12,759 · 4 → 3,412 ·
  5 → 5,309 · 6 → 1,539 · 7 → 864 · 8 → 290 · 9 → 52 · 10 → 24 · 11 → 2.

| label | train n+ | train % | val % | test % | `pos_weight` |
|---|---:|---:|---:|---:|---:|
| `PP_Cerebrovascular` | 8,916 | 10.07 | 9.17 | 10.56 | 8.94 |
| `PP_Neoplastic` | 7,382 | 8.33 | 7.76 | 7.89 | 11.00 |
| `PP_Neurodegenerative` | 16,595 | 18.73 | 15.94 | 18.77 | 4.34 |
| `PP_Spinal` | 1,814 | 2.05 | 1.35 | 1.62 | **20.00 (capped)** |
| `PP_Cystic_developmental` | 8,353 | 9.43 | 9.43 | 9.15 | 9.60 |
| `PP_Infectious` | 4,349 | 4.91 | 4.73 | 4.54 | 19.37 |
| `PP_Inflammatory` | 2,721 | 3.07 | 1.89 | 1.62 | **20.00 (capped)** |
| `PP_Unspecific_bucket` | 36,692 | 41.42 | 39.51 | 41.51 | 1.41 |
| `BP_Atrophies` | 16,595 | 18.73 | 15.94 | 18.77 | 4.34 |
| `BP_Contrast_enhancing_intracranial` | 7,382 | 8.33 | 7.76 | 7.89 | 11.00 |
| `BP_Infectious_lesions` | 4,349 | 4.91 | 4.73 | 4.54 | 19.37 |
| `BP_Edematous_lesions` | 7,599 | 8.58 | 7.04 | 7.48 | 10.66 |
| `BP_Hemorrhagic_lesions` | 6,745 | 7.61 | 6.62 | 7.55 | 12.13 |
| `BP_Cystic_lesions` | 7,768 | 8.77 | 8.69 | 8.52 | 10.40 |

### 3.2 Loss and masking policy

- `BCEWithLogitsLoss(pos_weight, reduction="none")` → `[B, 14]`, which upstream's own two masking
  lines then reduce as `sum(masked) / n_unmasked`, i.e. a per-sample **sum over labels**, mean over
  samples. That is upstream's convention, kept.
- **`pos_weight` = `neg/pos` per label, from the train split only, capped at 20.** Uncapped,
  `PP_Spinal` would ask for 47.8 and `PP_Inflammatory` for 31.6, letting a handful of positives
  dominate a term that is only auxiliary. The computed vector is logged and written into every
  checkpoint, so the weighting a checkpoint was trained under is recoverable from the checkpoint.
- **A missing label is never a negative.** A study with no label row gets an all-zero mask; the
  sample still trains the diffusion objective and contributes nothing to the classification loss.
  The mask travels in the `text_isnull` field — upstream reads that field in exactly one place, to
  exclude a sample from the classification term, which is the same meaning. The number of masked
  samples per step is logged as `train/labelled_fraction`.
- Logits are raw for the loss; `sigmoid` is applied only for the conditioning vector and for
  metrics.

---

## 4. The report string

```
"<findings>\n\n<impression>"
```

Empty sections are dropped; a study with neither yields `""`, upstream's null-report case. That is
the smallest mapping of MR-RATE's five sections onto upstream's "the plaintext radiology report for
that exam" — one string, no bracketed markers of the kind MRFlow adds.

**Not in the string**, each for a stated reason:

- **modality** — structured metadata, and it reaches the U-Net as `class_labels`. Putting it in the
  report and predicting it back would make the auxiliary head trivial.
- **spacing** — already an explicit U-Net input (`spacing_tensor`).
- **plane** — not represented at all; see §5.
- **the 14 labels** — the prediction target. They are derived *from* the findings upstream of this
  repository, which is exactly CCELLA's own situation with PI-RADS, but no label name and no label
  value is inserted.

`test_ccella.py` asserts all four absences on real section text.

Each **study** is encoded once and every series of that study references the same embedding.
Measured on the frozen 1,010-case test population, 11.0% of `findings + impression` exceed 512 T5
tokens (median 327, p95 613) and are truncated head-first, upstream's `truncation_side="left"`.

---

## 5. Volume preprocessing and latent geometry

```
tar member -> nib.as_closest_canonical (== Orientationd axcodes="RAS")
           -> trilinear to 1 mm isotropic                     [D4]
           -> guard: any resampled axis < 32 voxels -> skip    (AFTER the resample, see below)
           -> centred crop/pad to 256 x 256 x 224             [D4]
           -> ScaleIntensityRangePercentiles(0, 99.5, b_min=0, b_max=1, clip=True)   (upstream's)
           -> autoencoder.encode_stage_2_inputs               (upstream's, frozen MAISI)
           -> fp16 (C, X, Y, Z) = 4 x 64 x 64 x 56            [D7]
```

- The intensity window is **CCELLA's 0/99.5 over all voxels**, not MRFlow's 0.5/99.5 over nonzero
  voxels. The two give different arrays; this baseline uses CCELLA's.
- Padding is zero and happens before the percentile scaling, as it must, since upstream also scales
  over its own already-cropped fixed-size volume. With `lower=0` the low anchor is the minimum,
  which background already is.
- **The short-volume guard runs *after* the resample, never before.** A 26-slice axial stack at
  6 mm is 156 mm of anatomy and is perfectly usable; judging it on its native slice count discards
  ~a third of MR-RATE (measured: 30-38% of series rejected). On the 1 mm grid `min_extent_voxels`
  is therefore also a threshold in millimetres, and it matches MRFlow's `mri.preprocess.min_slices`.
  With the guard in the right place, **0 of 2,000 val series are skipped**.
- **The grid was chosen by measurement, not by analogy.** Brain-mask bounding boxes over 400
  sampled train series give a brain extent of (p95 / max) 150 / 176 mm L-R, 174 / 256 mm P-A and
  147 / 256 mm I-S, with the brain centre offset from the FOV centre by -21..0 mm P-A (MR-RATE is
  defaced, so the front of the FOV is empty) and -9..+27 mm I-S (the FOV reaches into the neck). A
  **centred** crop therefore keeps the whole brain for:

  | box | whole brain kept | latent | fp16 / series |
  |---|---:|---|---:|
  | 192 x 224 x 192 | 95.8% | 48 x 56 x 48 | 1.03 MB |
  | 224 x 256 x 192 | 95.8% | 56 x 64 x 48 | 1.31 MB |
  | 256 x 256 x 192 | 95.8% | 64 x 64 x 48 | 1.50 MB |
  | **256 x 256 x 224** | **99.5%** | **64 x 64 x 56** | **1.75 MB** |
  | 256 x 256 x 256 | 99.5% | 64 x 64 x 64 | 2.00 MB |

  The binding axis is **I-S**, not the in-plane axes: 192 x 224 in-plane scores exactly the same as
  256 x 256, and 256 in depth buys nothing over 224. The 256^2 in-plane is kept anyway because it is
  MRFlow's own evaluation grid, which makes `canonicalize_external` a pure depth operation when a
  generated volume is scored. Every axis is divisible by 32 (MAISI /4, U-Net /8); `config.validate`
  refuses a grid that is not.
- **No crop shift is applied**, unlike MRFlow's `posterior_shift_mm: 15`. The box is large enough
  that the measured P-A and I-S offsets do not clip it; a shift would be a framing parameter
  upstream does not have.
- Upstream stores `(X, Y, Z, C)` because it round-trips through NIfTI + `EnsureChannelFirstd`; we
  store `.npy` inside a zip, so channel-first round-trips as-is and the pair of transposes cancels.
  The tensor the U-Net sees is identical.
- **Plane is not represented.** CCELLA's chain reorients to RAS and lands every volume on one
  isotropic grid, so the acquisition plane is not an axis-order variable the way it is in MRFlow
  (where `plane_order` permutes the array); it survives only as through-plane blur, plus whatever
  the native `spacing` condition carries. That is a real limitation of adopting upstream's
  preprocessing unchanged. **No plane embedding and no plane-prediction objective was added** — it
  would be a conditioning channel upstream does not have. It is reported, not fixed.

---

## 6. Cache format

```
<cache_root>/artifacts/<split>-<shard>.latents.zip   ZIP_STORED, one member per series
<cache_root>/artifacts/<split>-<shard>.reports.zip   ZIP_STORED, one member per STUDY
<cache_root>/manifest/<split>-<shard>.csv            one row per series
<cache_root>/cache_meta.json                         the fingerprint, written by shard 0
```

Manifest columns: `sample_id, split, study_key, series_key, latent_zip, latent_member, report_zip,
report_member, modality, modality_id, plane, spacing_mm, grid, labels, label_mask`. Keys are
`sha1(study)[:16]` / `sha1(study|series)[:16]` — **no identifier and no text ever reaches a
manifest**.

- One task owns its two archives, so no lock is needed. Both are written to `.tmp` and `os.replace`d
  only after `close()`; the manifest is written last and is the commit record. A killed task leaves
  nothing a later run reads as complete, and re-running the shard is safe.
- `ZIP_STORED`, not deflate: members are dense fp16 arrays, and storing uncompressed keeps a
  member's bytes contiguous so a read is one seek.
- Reports are per study because a FLAN-T5-XXL hidden state is 4.19 MB and MR-RATE averages ~7
  series per study: per-series would cost 2.3 TB instead of 344 GB.
- **Shards are split by study, not by series**, and that is what makes the point above true.
  `list_series` shuffles series, so slicing that list directly scatters a study's ~7 series over
  every shard and the per-shard dedup stops working -- measured on the train split, 64 contiguous
  series slices touch **543,917** studies between them against **82,150** distinct ones, i.e. 6.6x
  the FLAN-T5 forwards and **2.28 TB** of embeddings instead of 336 GB. `shard_slice` groups first,
  which keeps the determinism and the mixing at the cost of unequal series counts per shard
  (measured 4,304 / 9,345 / 10,828 min / median / max).
- **The fingerprint** records the upstream commit, the volume block, the text/tokenizer block, the
  autoencoder sha256, the latent channel count, the label source id, the **label order** and the
  sha256 of the label CSV. `check_cache_meta` names every differing key and refuses to train; a
  cache with no `cache_meta.json` is refused outright. `text.encoder_path` is deliberately excluded
  — moving the weights must not invalidate a cache.

**Measured** (smoke run, 24 series): 7 files total. **Projected** for the full run at
`256 x 256 x 224`:

| artifact | per unit | train | val | total |
|---|---:|---:|---:|---:|
| latents (fp16, 4 x 64 x 64 x 56) | 1.75 MB / series | 0.96 TB | 40 GB | 1.00 TB |
| reports (fp16, 512 x 4096) | 4.19 MB / study | 344 GB | 14 GB | 358 GB |
| checkpoints (model + AdamW moments) | ~6.9 GB each | 5 x | — | 35 GB |
| **inodes** | — | 64 x 2 zips + 64 manifests | 4 x 2 + 4 | **~205 files** |

≈ **1.4 TB and ~205 inodes.**

---

## 7. Conditioning order

Upstream already implements the required order; this adaptation only supplies the ids.

```
base_temb   = time_embed(timestep_embedding(t))            # 256
temb        = base_temb + class_embedding(modality_id)     # elementwise ADD  -- _get_time_and_class_embedding
pathology   = sigmoid(classifier(hstack(6 x head))) * 1e2  # 14, independent probabilities
combined    = cat(temb, pirads_layer(pathology),           # concat            -- _get_input_embeddings
                        spacing_layer(spacing * 1e2))      # concat, last
                                                           # -> 768 = 3 x 256
```

- `modality_id` indexes `echosyn.common.mrrate.MODALITY_TO_ID` — MRFlow's own table, imported not
  copied: `CFG_NULL 0, T1w 1, T2w 2, FLAIR 3, SWI 4, MRA 5, UNKNOWN 6`. Ids may never move.
- **The predicted pathology vector is not detached.** The diffusion loss flows back through
  `pirads_layer` → `classifier` → the six heads. Verified by test — note that every resnet block's
  `conv2` and the output conv are `zero_module`-initialised upstream, so at step 0 *every*
  conditioning gradient is exactly zero, including the plain timestep's; that is identity
  initialisation, not a dead path, and the test perturbs those convs before probing.
- `sigmoid`, not `softmax`: the test asserts the conditioning vector sums to more than the
  conditioning scale, which a softmax could not.

---

## 8. Environment

Everything runs in the existing MRFlow venv, `/hnvme/workspace/y100dc19-mrflow-final/venv` — CCELLA
needs no separate environment. Checked: `monai 1.6.0`, `diffusers 0.39.0`, `transformers 5.14.1`,
`torch 2.5.1`, `nibabel`, `pandas`, `openpyxl`, `scipy`, `scikit-image`, `scikit-learn 1.9.0`,
`cv2`, `imageio`, `wandb`, `sentencepiece` all present.

Two were missing and are reported here rather than assumed:

- **`tensorboard`** — imported at the top of `scripts/diff_model_train_all.py`. Installed
  (2.21.0) with the user's approval. No event file is ever written: a `NullWriter` absorbs
  upstream's `add_scalar` calls and everything goes to W&B.
- **`SimpleITK`** — only `synthetic_datagen.py` needs it, which this adaptation does not use. Not
  installed.

**FLAN-T5-XXL** is fetched (42 GB, safetensors only) to
`/hnvme/workspace/y100dc19-mrflow-final/models/flan-t5-xxl` (`text.encoder_path`); `text.encoder`
stays the portable identity `google/flan-t5-xxl` that goes into the fingerprint, so moving the
weights does not invalidate a cache. `T5EncoderModel` reports `lm_head.weight` as unexpected, which
is correct — the encoder half does not have one.

---

## 9. Workspace recommendation

**No new workspace. Use a separate cache directory inside `y100dc19-mrflow-final`** —
`baselines/ccella_cache`, which is the configured default.

| consideration | finding |
|---|---|
| byte quota | none on `/hnvme` for this workspace (`SoftQ`/`HardQ` both report `0.0K`); the filesystem has ~861 TB free against 4.3 PB. ~1.4 TB is not a constraint. |
| inode quota | **75K used against 61K soft / 81K hard, with ~6 days of grace already running.** This is a pre-existing condition, not something CCELLA creates: zip sharding costs ~205 files. |
| does sharding solve it | yes, completely. Per-file storage would be ~1.2 M inodes and is not survivable; at 205 files the cache is invisible to the quota. |
| expiry | `y100dc19-mrflow-final` expires 2026-11-24 (63 days) with 3 extensions available — comfortably longer than a 60,000-step run plus scoring. A second workspace would add a second expiry to track. |
| isolation | everything CCELLA needs is already in this workspace: the pinned clone, the MAISI autoencoder (shared with Text2CT and NV-Generate), the venv, and the other baselines' caches. Splitting would put cross-workspace absolute paths into configs for no benefit. |
| drawback | the cache and the run share an expiry and a filesystem with everything else. Acceptable; `data.cache_root` is one config key, so moving later is trivial. |

**The inode grace timer is the one thing to watch**, and it is independent of this baseline.

---

## 10. Running it

### Preprocess

```bash
# val first (small), then train. Task 0 writes cache_meta.json -- let it finish before training.
sbatch --array=0-3  baselines/ccella/slurm/preprocess.sh baselines/ccella/configs/mrrate_ccella.yaml val
sbatch --array=0-63 baselines/ccella/slurm/preprocess.sh baselines/ccella/configs/mrrate_ccella.yaml train

# one shard, interactively
python -m baselines.ccella.prepare_data --config baselines/ccella/configs/mrrate_ccella.yaml \
    --split val --shard 0 --num_shards 4
```

A shard whose manifest exists is skipped; `--overwrite` redoes it, `--limit N` caps it.

### Train

```bash
sbatch baselines/ccella/slurm/train.sh baselines/ccella/configs/mrrate_ccella.yaml

# single GPU, no W&B
python -m baselines.ccella.train --config baselines/ccella/configs/mrrate_ccella.yaml --no_wandb
```

### Resume

```bash
sbatch baselines/ccella/slurm/train.sh baselines/ccella/configs/mrrate_ccella.yaml --resume
python -m baselines.ccella.train --config <cfg> --resume /path/to/checkpoint-40000.pt
```

`--resume` with no argument takes the latest checkpoint in `train.output_dir`. Model, optimizer,
scheduler, AMP scaler, scale factor and RNG are restored and the run continues from the saved
`global_step`: **no optimizer update is repeated or skipped**. What is not restored is the position
inside an epoch's shuffle — the partial epoch restarts from its beginning, so a few samples may be
seen twice across the boundary; `epoch` and `step_in_epoch` are recorded in the checkpoint. Saving
over an existing checkpoint is refused unless `--overwrite_checkpoints` is passed.

### Validation

Runs automatically every `validation.every` steps. To change the cost:

```bash
python -m baselines.ccella.train --config <cfg> \
    --set validation.every=5000 validation.subset=512 validation.visualize=false
```

### Tests

```bash
pytest baselines/ccella/test_ccella.py -q          # 53 tests, ~15 s, CPU only
python -m baselines.ccella.audit_labels            # rewrites label_audit.json
```

### What has actually been run

| check | result |
|---|---|
| `pytest baselines/ccella/test_ccella.py` | **53 passed**, CPU only |
| label audit over all three splits | §3.1 |
| preprocessing, 8 val + 16 train series, `__zeros__` encoder, h200 | 7 cache files, 0 skipped, fingerprint written |
| preprocessing, 6 val series, **real FLAN-T5-XXL**, h200 | latent `(4,16,16,16)` fp16, text `(512,4096)` fp16 std 0.115, one report member per study |
| 4 training steps + validation + preview video, W&B disabled | checkpoint-4 written |
| resume → 8 steps | `global_step == scheduler.last_epoch == 8` at both checkpoints: **no update repeated, none skipped** |
| checkpoint contents | U-Net + ELLA + fusion classifier + modality embedding, optimizer, scheduler, scaler, RNG, scale factor, label order, `pos_weight` |

Nothing beyond that has been run: **the model is not trained** and no quality claim is made.

---

## 11. W&B, checkpoints, known limitations

**W&B** — rank 0 only, `online` / `offline` / `disabled` (`--no_wandb`, `WANDB_MODE`, or
`wandb.mode`). Project `MRFlow`, group `baselines`, run name `ccella_mrrate`, `resume="allow"` —
the same conventions `echosyn.common.init_trackers` uses for MRFlow itself.

| training (every `train.log_every` steps) | validation (every `validation.every` steps) |
|---|---|
| `train/loss`, `train/diffusion_loss`, `train/class_loss` | `val/total_loss`, `val/diffusion_loss`, `val/class_loss` |
| `train/lr_group{i}` for every optimizer group | `val/auroc_macro`, `val/auroc_micro`, `val/ap_macro`, `val/ap_micro` |
| `train/grad_norm` | `val/auroc/<label>`, `val/ap/<label>`, `val/prevalence/<label>` |
| `train/step_time_s`, `train/samples_per_s` | `val/n_samples_scored`, `val/n_labels_scored` |
| `train/labelled_fraction`, `train/epoch` | `validation` — a `wandb.Video` of ground truth ǀ generated |

The preview uses `label_frames` and `wandb.Video` from `echosyn.common`, the same helpers
`lvfm/train.py::log_validation` uses, so a CCELLA run and an MRFlow run look alike. Fixed
validation rows, fixed seeds. The caption carries modality, plane, spacing, the top predicted
pathology probabilities and the true labels — **never report text, patient id or study id**.
A label with no positives (or no negatives) in the subset is reported as undefined and left out of
the macro average, and `val/n_labels_scored` says how many survived.

**Checkpoints** at exactly the configured global steps — `[20000, 40000, 50000, 55000, 60000]` for a
60k run, `[20000, 40000, 60000, 80000, 100000, 110000, 120000]` for 120k. `validate_checkpoint_steps`
requires strictly increasing, positive, inside the run, and ending on the final step; a nonstandard
`max_train_steps` needs an explicit schedule rather than a guessed one. Each checkpoint holds the
full model (U-Net + ELLA + the 14-label heads + the modality embedding), optimizer, scheduler, AMP
scaler, `global_step`, `epoch`, `step_in_epoch`, scale factor, RNG states, the effective config, the
label order and the `pos_weight` vector.

### Known limitations

1. **The labels are study-level.** One study contributes up to 83 series, and every one of them
   carries the same 14-vector. A finding described in the report may be visible in only one
   sequence or plane — a small posterior-fossa lesion is not in a sagittal T1 of the same study.
   The auxiliary head is therefore trained against a target that is partly unobservable in the
   volume it is conditioned on. This is the same situation as CCELLA's own PI-RADS labels being
   assigned per study, but MR-RATE's multi-sequence studies make it much more pronounced.
2. **The labels are LLM-derived from the reports** (a 3-model majority over an LLM labelling of the
   findings), so the auxiliary task is closer to "read the report" than to "read the image".
   CCELLA's PI-RADS is partly biopsy-derived and so partly unreachable from text; ours is not. Its
   own head reached accuracy 0.81 on that task, which is the number `val/auroc_*` should be read
   against.
3. **Three of the 14 columns are exact duplicates of three others** (§3), so macro metrics
   over-weight neurodegeneration, neoplasm and infection by a factor of two.
4. **Plane is not conditioned on** (§5).
5. **No generation or scoring path.** The baseline contract in [`../README.md`](../README.md) —
   `generated/*.npy` plus `shard-NNNN.json`, then `slurms/baseline_score.sh` — is **not
   implemented here**. Training and validation are; turning a checkpoint into scoreable volumes is
   the remaining work.
6. **Not trained.** Everything below a few-step smoke run is unverified: no convergence, no loss
   curve, no quality claim.
