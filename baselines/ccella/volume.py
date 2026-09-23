"""CCELLA's volume preprocessing and MAISI encoding, reading from MR-RATE's archives.

**Upstream's chain, preserved** (`scripts/data_processing/diff_model_create_training_data.py`):

    LoadImaged -> EnsureChannelFirstd -> Orientationd(axcodes="RAS") -> EnsureTyped(float32)
      -> ScaleIntensityRangePercentilesd(lower=0, upper=99.5, b_min=0, b_max=1, clip=True)
      -> autoencoder.encode_stage_2_inputs(x[None])
      -> z.squeeze().cpu().numpy().transpose(1, 2, 3, 0)     # (C, X, Y, Z) -> (X, Y, Z, C)

The intensity step is MONAI's own transform, imported and called here rather than reimplemented, so
its percentile arithmetic is upstream's byte for byte. Note it is **0 / 99.5 over all voxels** --
not MRFlow's 0.5 / 99.5 over nonzero voxels. The two produce different arrays and this baseline
uses CCELLA's.

The autoencoder is the frozen MAISI VQ-VAE upstream names in `models/put_models_here.txt`. The file
in this workspace is byte-identical to it (sha256 `1f8a7a05...`, the same file Text2CT and
NV-Generate-MR-Brain use), so nothing is downloaded and the checkpoint is hashed into the cache
fingerprint.

**Three things must differ, and only these three.**

1. *The reader.* Upstream walks a directory of `.nii.gz` files; MR-RATE is webdataset tars. The
   bytes are pulled with `echosyn.common.mrrate.read_member` and handed to nibabel in memory. No
   step of the chain above changes -- `nib.as_closest_canonical` is precisely what
   `Orientationd(axcodes="RAS")` does.

2. *The resample and the fixed grid.* Upstream's README states its input is "already resampled to
   the desired (fixed) image size"; that precondition is outside its code, so for native-space
   MR-RATE it has to be made explicit. Trilinear to 1 mm isotropic, then a centred crop/pad to
   `volume.grid`. Padding is zero, i.e. background, and happens *before* the percentile scaling --
   as it must, since upstream also scales over its own already-cropped fixed-size volume. With
   `lower=0` the low anchor is the minimum, which background already is, so added zeros move it
   nowhere; the 99.5 anchor sits among bright voxels and is likewise insensitive to them.

3. *The spacing the model is conditioned on.* Upstream stores each volume's own voxel size, which
   varies case to case there. Ours is fixed at 1 mm by construction and would carry no information
   at all, so the **native acquisition spacing** is passed instead -- the quantity that actually
   varies across MR-RATE and the one that says a 5 mm-slice acquisition stays blurred through-plane
   after resampling. The `* 1e2` scale upstream's dataloader applies is unchanged.

**Plane is not represented.** Because the chain reorients to RAS and lands every volume on one
isotropic grid, the acquisition plane is not an axis-order variable the way it is in MRFlow, where
`plane_order` permutes the array. Here it survives only as through-plane blur. That is a real
limitation of adopting CCELLA's preprocessing unchanged, and it is reported rather than fixed: a
plane embedding would be a conditioning channel upstream does not have.
"""

from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


class VolumeUnusable(ValueError):
    """Too small, empty, or unreadable -- skipped with a manifest note, never silently zeroed."""


def read_ras(nii_bytes):
    """NIfTI bytes -> (RAS float32 array in `(X, Y, Z)`, native `(x, y, z)` spacing in mm).

    `nib.as_closest_canonical` is MONAI's `Orientationd(axcodes="RAS")`; the array keeps nibabel's
    own axis order, which is the order upstream's `LoadImaged` hands to the autoencoder.
    """
    payload = gzip.decompress(nii_bytes) if nii_bytes[:2] == b"\x1f\x8b" else nii_bytes
    img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(payload))
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    np.nan_to_num(data, copy=False)
    if data.ndim != 3:
        raise VolumeUnusable(f"expected a 3D volume, got shape {data.shape}")
    return np.ascontiguousarray(data), spacing


def resample_isotropic(volume, spacing, target_mm):
    """Trilinear to `target_mm` per axis. The step upstream assumes has already happened."""
    scale = [s / t for s, t in zip(spacing, target_mm)]
    size = [max(1, int(round(n * f))) for n, f in zip(volume.shape, scale)]
    if size == list(volume.shape):
        return volume
    tensor = torch.from_numpy(volume)[None, None]
    out = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    return out[0, 0].numpy()


def crop_pad(volume, grid):
    """Centred crop/pad to exactly `grid`, padding with zero (background)."""
    out = np.zeros(tuple(grid), dtype=np.float32)
    src_slices, dst_slices = [], []
    for n, m in zip(volume.shape, grid):
        keep = min(n, m)
        src_start = (n - keep) // 2
        dst_start = (m - keep) // 2
        src_slices.append(slice(src_start, src_start + keep))
        dst_slices.append(slice(dst_start, dst_start + keep))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def scale_intensity(volume, lower, upper, clip):
    """Upstream's `ScaleIntensityRangePercentilesd`, the MONAI transform itself."""
    from monai.transforms import ScaleIntensityRangePercentiles

    transform = ScaleIntensityRangePercentiles(
        lower=lower, upper=upper, b_min=0.0, b_max=1.0, clip=clip)
    return np.asarray(transform(volume[None]))[0].astype(np.float32)


def preprocess_volume(nii_bytes, config):
    """MR-RATE NIfTI bytes -> (volume on the fixed grid in `[0, 1]`, native spacing in mm)."""
    spec = config["volume"]
    volume, spacing = read_ras(nii_bytes)
    volume = resample_isotropic(volume, spacing, spec["spacing_mm"])
    # **After** the resample, never before. A 26-slice axial stack at 6 mm is 156 mm of anatomy and
    # is perfectly usable; judging it on its native slice count would throw away a third of MR-RATE.
    # On the 1 mm grid this threshold is therefore also a threshold in millimetres, and it matches
    # MRFlow's own `mri.preprocess.min_slices`.
    if min(volume.shape) < spec["min_extent_voxels"]:
        raise VolumeUnusable(
            f"resampled shape {volume.shape} has an axis below min_extent_voxels "
            f"{spec['min_extent_voxels']} (native {[round(v, 2) for v in spacing]} mm)")
    volume = crop_pad(volume, spec["grid"])
    if not np.isfinite(volume).all():
        raise VolumeUnusable("non-finite voxels after resampling")
    if float(volume.max()) <= float(volume.min()):
        raise VolumeUnusable("constant volume")
    volume = scale_intensity(volume, spec["percentile_lower"], spec["percentile_upper"],
                             spec["clip"])
    return volume, spacing


def build_autoencoder(config, device):
    """The frozen MAISI autoencoder, built by upstream's own `define_instance` and def block."""
    from .config import upstream_namespace
    from .upstream import upstream_module

    define_instance = upstream_module("scripts.utils").define_instance
    args = upstream_namespace(config)
    model = define_instance(args, "autoencoder_def").to(device)
    state = torch.load(config["model"]["autoencoder_path"], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.inference_mode()
def encode_volume(autoencoder, volume, device):
    """Upstream's encode: `encode_stage_2_inputs` under autocast, then `(C, X, Y, Z)` -> fp16.

    Upstream then transposes to `(X, Y, Z, C)` because it stores a NIfTI and reloads it with
    `EnsureChannelFirstd`. We store `.npy` inside a zip, so the channel-first array round-trips as
    it is and the pair of transposes cancels. The tensor the U-Net sees is identical.
    """
    tensor = torch.from_numpy(volume)[None, None].to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        latent = autoencoder.encode_stage_2_inputs(tensor)
    return latent.squeeze(0).to(torch.float16).cpu().numpy()


@torch.inference_mode()
def decode_latent(autoencoder, latent, device):
    """Latent `(C, X, Y, Z)` -> volume in `[0, 1]`, for validation previews."""
    tensor = torch.as_tensor(latent, dtype=torch.float32, device=device)[None]
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        volume = autoencoder.decode_stage_2_outputs(tensor)
    return volume[0, 0].float().clamp(0, 1).cpu().numpy()
