# Text2CT

End-to-end 3D latent diffusion: a generation-oriented 3D-CLIP text encoder, a volumetric VAE, and
cross-attention conditioning in latent space — explicitly no cascaded super-resolution, which is
its stated advantage over GenerateCT. Architecturally the closest of the three to MRFlow, and the
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

**Chest CT → MR-RATE brain MRI fine-tuning**, as with GenerateCT. Three components could each be
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
what `preprocess_volume` already produces for MR. Note the consequence for the table — two of three
baselines now share a latent space and a decoder, and MRFlow's is the odd one out (FLUX), so a
reconstruction-ceiling row per autoencoder is worth one job to pre-empt "how much of the margin is
the VAE?".

**The four conditioning inputs you inherit with the UNet**, from `configs/config_rflow.json`:

| input | released state | call for MR |
|---|---|---|
| `include_body_region` | **`false`** | nothing to do — the `top/bottom_region_index` in `config_diff_model.json` is dead config |
| `num_class_embeds: 128`, modality | `diff_model_train.py:313` hardcodes `torch.ones(...)` | keep id 1; it is the only index with trained weights |
| `include_spacing_input: true` | trained at `0.75/0.75/3.0` | pass real spacing; 1 mm iso is out of the trained distribution but in MAISI's range |
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
