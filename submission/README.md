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
  Dockerfile                 thin image: code + baked config.yaml + helpers/
  helpers/                   grid-rewriting post-processors, each with the survey it draws from --
                              numpy/scipy only, no MRFlow imports, each switchable off at runtime
                              (mirrors R2V-MR-Generation/submission/helpers/):
    native_spacing.py          slab-averages the generated slice axis to a realistic thickness (see below)
    native_spacing_table.json  MR-RATE's own per-(modality, plane) slice-thickness survey (ported from
                                R2V-MR-Generation, model-agnostic -- both projects preprocess the same
                                raw archives)
    inplane_resample.py        spline-upsamples the two in-plane axes to a realistic pixel spacing (below)
    inplane_spacing_table.json MR-RATE's own per-(modality, plane) in-plane pixel-spacing survey (ported
                                verbatim from R2V-MR-Generation for the same reason)
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
- **Native-slice-count matching.** `helpers/native_spacing.py`, ported from R2V-MR-Generation's own
  `Dockerfile.native_spacing` variant (there measured as a large FID win): the official
  `FID_2p5D` metric never resamples a shape mismatch away (only `compute_basic_metrics` does), so a
  generated volume's always-~1mm slice grid is otherwise scored against native MR-RATE 2D
  acquisitions (commonly 4-6.5mm) on a mismatched grid. `to_native_grid()` slab-averages the
  generated volume's slice axis (always axis 0 here, unlike R2V's fixed-XYZ convention) down to a
  thickness drawn from MR-RATE's own per-(modality, plane) distribution, preserving the physical FOV
  to the millimetre and never upsampling. On by default (`MRFLOW_NATIVE_SPACING_MODE=sample`); set
  to `off` to write the generated 1mm grid unchanged, as an A/B control for a future submission.
- **Native in-plane-resolution matching.** `helpers/inplane_resample.py`, the same trick on the other two
  axes, ported from R2V-MR-Generation's own `inplane_resample.py` (there measured separately from
  the slice axis: FID_2p5D_Avg 43.90 -> 35.37 on its 64-case local set, SSIM/PSNR/MSE unchanged).
  MRFlow's in-plane grid is a fixed 256² at 1mm (`mri.preprocess.inplane_size`/`target_spacing`, a
  preprocessing choice), while MR-RATE's own survey puts every bucket's median in-plane spacing at
  0.45-0.8mm -- *finer* than we generate, so this direction is an upsample where the slice axis was
  a reduction. `to_inplane_grid()` cubic-spline upsamples axes 1 and 2 (always the in-plane pair
  here, unlike R2V's per-plane `INPLANE_AXES_XYZ` lookup) to a spacing drawn from that
  distribution, again preserving the physical FOV and never coarsening. Its own switch
  (`MRFLOW_INPLANE_MODE=sample`, `off` for the A/B control) rather than being folded into
  `MRFLOW_NATIVE_SPACING_MODE`, since the two were measured independently. Interpolation cannot add
  texture the model never produced; what it buys is that the resample happens once, deliberately,
  instead of as a side effect of the evaluator's own `zoom(order=1)` shape-match. The cost is disk,
  not time: ~3.6x the pixels on average (1.6x-6.2x by bucket) and ~3.5x the written bytes at the
  median draw, since interpolated floats compress worse than the uint8 levels `decode_latent`
  emits -- measured 8.5 MB -> 30 MB for a 61x256x256 float32 volume, up to 90 MB at the 0.40mm
  tail, and `/output` and `/checkpoint` each hold a copy. `MRFLOW_OUTPUT_DTYPE=float16` halves that
  losslessly (11-bit mantissa exactly represents 0-255) if disk becomes the binding constraint.
- **Guidance.** `MRFLOW_MODALITY_CFG_SCALE=1.0` / `MRFLOW_REPORT_CFG_SCALE=1.0` (baked into the
  Dockerfile) is the plain conditional prediction -- config.guidance's own default, and the fast
  path in `LatentAutoregressiveGenerator.velocity` (one denoiser call instead of three). A future
  submission sweeps these by changing the two `ENV` lines and rebuilding.
- **Sampler switches, and one stale knob.** The container runs with `MRFLOW_USE_BF16=1` and
  `MRFLOW_USE_COMPILE=1` -- both **on**, unlike `evaluation/main.py`, which defaults `--bf16` off so
  its scores stay comparable with everything already measured. Batched inference was **removed**
  (the rollout is one case at a time again), but `ARG BATCH_SIZE` / `MRFLOW_BATCH_SIZE` are still
  set in the Dockerfile and nothing reads them -- a `--build-arg BATCH_SIZE=...` silently does
  nothing.
