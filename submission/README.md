# MRFlow submission -- VLM3D `mr-volume-generation`

Packages MRFlow's STDiT + flow-matching autoregressive generator (this repo) as a Forithmus
`mr-volume-generation` Docker submission. Structurally mirrors
`R2V-MR-Generation/submission/` (same platform contract, checkpoint/resume, DDP, native-spacing
trick) but drives MRFlow's own model instead of the NVIDIA-MAISI-style adapter stack.

## Layout

```
submission/
  predict.py               entry point: /input/prompts.json -> /output/*.nii.gz
  entrypoint.sh             symlinks /weights into /opt/app/models, GPU sanity check, launches predict.py
  Dockerfile                 thin image: code + baked config.yaml + native_spacing_table.json
  native_spacing.py          slab-averages the generated slice axis to a realistic thickness (see below)
  native_spacing_table.json  MR-RATE's own per-(modality, plane) slice-thickness survey (ported from
                              R2V-MR-Generation, model-agnostic -- both projects preprocess the same
                              raw archives)
  config.yaml                the training config saved next to checkpoint-60000 (labels, guidance
                              defaults, mri.preprocess.*); vae.pretrained/mri.text_checkpoint are
                              overridden at runtime once the weight dirs are resolved
  make_weights_zip.sh        assembles weights.zip from /vol/.../VLM3D-MICCAI-2026/models
  package_image.sh           docker save -> classic (non-OCI) tarball, what Forithmus's validator wants
  mock_input/prompts.json    a 6-case local test fixture (sagittal/axial/coronal + one oblique case)
```

Weights (checkpoint, VAE, text encoder -- ~2.7 GB) ship separately via `--weights`, never baked into
the image; the trained checkpoint itself lives at
`/vol/idea_ramses/va47zasy/VLM3D-MICCAI-2026/models/mrflow/checkpoint-60000/denoiser_ema` (copied
from Helma's `/hnvme/workspace/y100dc19-mrflow-final/experiments/mrflow_from_ct_checkpoint/`).

## Build and test locally

```bash
cd /vol/idea_ramses/va47zasy/VLM3D-MICCAI-2026/MRFlow
docker buildx build --platform linux/amd64 -f submission/Dockerfile -t mrflow-vlm3d-challenge:latest .
./submission/make_weights_zip.sh                       # -> submission/weights/mrflow_weights.zip
docker run --rm --gpus all \
    -v "$PWD/submission/mock_input:/input:ro" \
    -v /tmp/mrflow_out:/output \
    -v /tmp/mrflow_ckpt:/checkpoint \
    -v "$PWD/submission/weights:/weights:ro" \
    mrflow-vlm3d-challenge:latest
./submission/package_image.sh mrflow-vlm3d-challenge:latest mrflow_submission.tar.gz
```

Final artifacts (image tarball + weights.zip) go to
`/vol/idea_ramses/va47zasy/VLM3D-MICCAI-2026/submission/` alongside the earlier R2V submissions,
under distinct names (`mrflow_submission.tar.gz`, `weights/mrflow_weights.zip`) so neither overwrites
the other.

## Design notes (what's MRFlow-specific here)

- **Output naming.** `filename = f"{case_id}.nii.gz"` where `case_id` is `input_image_name` verbatim
  (including any `-2`/`-3` repeat suffix) -- the evaluator pairs by this name only.
- **Modality/plane decoding.** `modality_plane_for()` regex-matches `..._<modality>-raw-<plane>` out
  of the case id (same pattern R2V-MR-Generation's own `conditioning.py` uses) and resolves both
  through `echosyn.common.mrrate.MODALITY_ALIASES`/`PLANE_ALIASES` -- the single source of truth
  also used at training and by `modality_to_id`/`plane_to_id`, so a code means the same thing in the
  class-id path and in the conditioning text. Oblique acquisitions (`obl`, 7/690 entries) resolve to
  `AXIAL`: MRFlow's plane embedding has no oblique class (`num_plane_classes: 3`), and `mrrate.py`'s
  own `PLANE_ALIASES` already treats an oblique acquisition as a tilted axial.
- **Report conditioning.** The challenge hands over one flat report string; training's
  `encode_conditioning` expects a `{findings, impression}` dict (MR-RATE's own `report.json` shape).
  `split_sections()` (ported from R2V-MR-Generation's `predict.py`) regex-splits the flat string back
  into that shape before encoding, so the acquisition-prefix + report text the model sees at
  inference is the same shape it was trained against.
- **Axis order / NIfTI affine.** The generated array is always `(T, H, W)` with the slice axis
  leading, whatever the plane (`echosyn.common.mrrate.plane_order`'s convention -- SRA for axial,
  RSA for sagittal, ASR for coronal). `build_affine()` in `predict.py` constructs the corresponding
  4x4 affine: a signed permutation of RAS with every sign positive (no axis is ever flipped in
  preprocessing, so none is flipped here either) -- an honestly-labeled, non-identity affine.
  Verified locally: `nib.aff2axcodes()` on a written volume reports `('S','R','A')` for an axial
  case and `('R','S','A')` for a sagittal one, and `nib.as_closest_canonical()` reorients it to a
  plain identity-direction affine, matching the challenge's "if you generate in RAS, an
  identity-direction affine is correct" for the *unpermuted* case.
- **Native-slice-count matching.** `native_spacing.py`, ported from R2V-MR-Generation's own
  `Dockerfile.native_spacing` variant (there measured as a large FID win): the official
  `FID_2p5D` metric never resamples a shape mismatch away (only `compute_basic_metrics` does), so a
  generated volume's always-~1mm slice grid is otherwise scored against native MR-RATE 2D
  acquisitions (commonly 4-6.5mm) on a mismatched grid. `to_native_grid()` slab-averages the
  generated volume's slice axis (always axis 0 here, unlike R2V's fixed-XYZ convention) down to a
  thickness drawn from MR-RATE's own per-(modality, plane) distribution, preserving the physical FOV
  to the millimetre and never upsampling. On by default (`MRFLOW_NATIVE_SPACING_MODE=sample`); set
  to `off` to write the generated 1mm grid unchanged, as an A/B control for a future submission.
- **Guidance.** `MRFLOW_MODALITY_CFG_SCALE=1.0` / `MRFLOW_REPORT_CFG_SCALE=1.0` (baked into the
  Dockerfile) is the plain conditional prediction -- config.guidance's own default, and the fast
  path in `LatentAutoregressiveGenerator.velocity` (one denoiser call instead of three). A future
  submission sweeps these by changing the two `ENV` lines and rebuilding.
