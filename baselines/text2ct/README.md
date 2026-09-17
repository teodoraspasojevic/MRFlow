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
model and keep the released encoder and VAE, and to state it.

`scripts/sample.py`'s `check_input` constrains `output_size` to `[256, 384, 512]` in-plane and
`[128...768]` on the third axis, with spacing in `[0.5, 3.0]` / `[0.5, 5.0]` — a MAISI-derived
interface, the same lineage as the NVIDIA model in `../nvidia_r2v/`. 256 in-plane at 1 mm is
reachable directly, which is convenient but not required: `ingest.py` resamples from the affine
either way.

## Adapter: what still has to be written

`slurms/text2ct_run_shards.sh` + a driver that, for each case of `baselines/cases-test-n100.json`, encodes that
study's report with **Text2CT's own** 3D-CLIP encoder, samples, decodes through its VAE, and saves
`<case_id>.nii.gz` with a correct affine. Its own venv in the workspace.
