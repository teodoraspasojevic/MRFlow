"""MR data: class labels, volume preprocessing, VAE encoding, and latent storage and manifests."""

import csv
import gzip
import hashlib
import io
import os
import zipfile
from collections import OrderedDict
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Array axis order is (S, R, A); each plane's slices stack along the axis below.
PLANE_TO_STACK_AXIS = {"AXIAL": 0, "SAGITTAL": 1, "CORONAL": 2}

AP_AXIS = 2  # anterior-posterior is (D, H, W) axis 2


### Class labels ###

# Fixed class ids; never reorder them. Id 0 is the classifier-free-guidance null.
MODALITY_TO_ID = {
    "CFG_NULL": 0,
    "T1w": 1,
    "T2w": 2,
    "FLAIR": 3,
    "SWI": 4,
    "MRA": 5,
    "UNKNOWN": 6,
}
CFG_NULL_MODALITY_ID = MODALITY_TO_ID["CFG_NULL"]
NUM_MODALITY_CLASSES = len(MODALITY_TO_ID)

# Every spelling the pipeline can hand us: MR-RATE's own column values and lowercase codes.
MODALITY_ALIASES = {
    "t1w": "T1w", "t1": "T1w",
    "t2w": "T2w", "t2": "T2w",
    "flair": "FLAIR",
    "swi": "SWI", "swan": "SWI",
    "mra": "MRA",
    "unknown": "UNKNOWN",
}

# Deliberately its own table, not derived from PLANE_TO_STACK_AXIS: that one is a preprocessing
# detail (which array axis to lead with) and may change, while a class id may never move.
PLANE_TO_ID = {"AXIAL": 0, "SAGITTAL": 1, "CORONAL": 2}
NUM_PLANE_CLASSES = len(PLANE_TO_ID)

# Plane has no null class; unknown spellings fall back to axial, as in `plane_order`.
PLANE_ALIASES = {
    "axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL",
    "obl": "AXIAL", "OBLIQUE": "AXIAL", "UNKNOWN": "AXIAL",
}


def modality_to_id(modality):
    """Modality string -> class id. Raises on anything not in the tables above."""
    name = MODALITY_ALIASES.get(str(modality).strip().lower(), str(modality).strip())
    if name not in MODALITY_TO_ID:
        raise KeyError(
            f"unmapped modality {modality!r}: add it to MODALITY_TO_ID or MODALITY_ALIASES in "
            f"mrflow/mri.py. Known: {sorted(MODALITY_TO_ID)}"
        )
    return MODALITY_TO_ID[name]


def plane_to_id(plane):
    """Plane string -> class id. Raises on anything not in the tables above."""
    name = str(plane).strip()
    name = PLANE_ALIASES.get(name, PLANE_ALIASES.get(name.lower(), name))
    if name not in PLANE_TO_ID:
        raise KeyError(
            f"unmapped plane {plane!r}: add it to PLANE_TO_ID or PLANE_ALIASES in "
            f"mrflow/mri.py. Known: {sorted(PLANE_TO_ID)}"
        )
    return PLANE_TO_ID[name]


def label_counts(rows):
    """Manifest rows -> (per-modality, per-plane, per-pair) counts, for the dataset's startup log."""
    modality, plane, pair = {}, {}, {}
    for row in rows:
        m, p = row["modality"], row["plane"]
        modality[m] = modality.get(m, 0) + 1
        plane[p] = plane.get(p, 0) + 1
        pair[(m, p)] = pair.get((m, p), 0) + 1
    return modality, plane, pair


### Artifact access ###

# (pid, path) -> open archive. Keyed on pid so a forked dataloader worker never inherits the
# parent's file descriptor and its read position.
_HANDLES = OrderedDict()


def _cached(path, opener, max_open=8):
    key = (os.getpid(), path)
    if key in _HANDLES:
        _HANDLES.move_to_end(key)
        return _HANDLES[key]
    _HANDLES[key] = opener(path)
    while len(_HANDLES) > max_open:
        _, old = _HANDLES.popitem(last=False)
        old.close()
    return _HANDLES[key]


def read_bundled(bundle_path, member):
    """Raw bytes of one member of a latent bundle. See LatentStore."""
    return _cached(bundle_path, zipfile.ZipFile).read(member)


def load_artifact(root, row, key):
    """One manifest row's latent or embedding, from its bundle or as a loose file."""
    if row.get("zip"):
        blob = read_bundled(os.path.join(root, row["zip"]), row[key])
        return torch.load(io.BytesIO(blob), map_location="cpu")
    return torch.load(os.path.join(root, row[key]), map_location="cpu")


def sample_id(study_uid, series_id):
    """Stable, identifier-free artifact name. series_ids repeat across studies, so they collide on
    their own, and `{study}_{series}` would put identifiers into filenames."""
    return hashlib.sha1(f"{study_uid}|{series_id}".encode()).hexdigest()[:16]


### Volume preprocessing ###


class VolumeTooShort(ValueError):
    """Fewer than `min_slices` after resampling -- too short to form a block pair."""


def _normalize(data, lower=0.5, upper=99.5):
    """Rescale the [lower, upper] percentile of the nonzero voxels to [0, 1].

    Percentiles rather than min-max: MR-RATE mixes uint16 and float32 series with a >100x dynamic
    range difference, and a min-max would key on bright-vessel and fat outliers.
    """
    mask = data != 0
    if mask.any():
        low, high = np.percentile(data[mask], [lower, upper])
    else:
        low, high = float(data.min()), float(data.max())
    data = np.clip(data, low, high)
    if high - low > 1e-8:
        data = (data - low) / (high - low)
    else:
        data = np.zeros_like(data)
    return data.astype(np.float32)


def _crop_pad(volume, size, shift_voxels=0, shift_axis=None):
    """Center crop/pad the two in-plane axes of a (T, A, B) array to size x size.

    T is left untouched -- variable volume length is the whole point of the autoregressive
    formulation. `shift_voxels` moves the window toward lower indices on the anterior-posterior
    axis: MR-RATE is already defaced, so a plain center crop keeps a band of removed face and
    loses posterior brain. Background after normalization is 0, so that is the pad value.
    """
    starts, sizes = [], []
    for axis, cur in enumerate(volume.shape[1:]):
        if axis == shift_axis:
            start = cur // 2 - int(shift_voxels) - size // 2
            start = max(0, min(start, max(cur - size, 0)))
        else:
            start = max((cur - size) // 2, 0)
        starts.append(start)
        sizes.append(min(size, cur - start))

    volume = volume[:, starts[0]:starts[0] + sizes[0], starts[1]:starts[1] + sizes[1]]
    pads = [((size - s) // 2, size - s - (size - s) // 2) for s in volume.shape[1:]]
    return np.pad(volume, ((0, 0), pads[0], pads[1]), constant_values=0.0)


def read_canonical(nii_bytes):
    """NIfTI bytes -> (RAS-canonical array in (S, R, A) axis order, native spacing). No resampling."""
    payload = gzip.decompress(nii_bytes) if nii_bytes[:2] == b"\x1f\x8b" else nii_bytes
    img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(payload))
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))  # (X, Y, Z) -> (D, H, W)

    data = np.asarray(img.get_fdata(), dtype=np.float32)
    np.nan_to_num(data, copy=False)
    return np.ascontiguousarray(data.transpose(2, 0, 1)), spacing  # (R, A, S) -> (S, R, A)


def plane_order(plane):
    """The (S, R, A) axis permutation that puts the acquisition plane's slice axis first, so the
    leading axis is the one the model rolls out along. Unknown/oblique planes go to axial, the
    majority plane.

    The resulting order is therefore plane-dependent -- SRA axial, RSA sagittal, ASR coronal -- and
    is what both a preprocessed volume and a generated one are in."""
    stack_axis = PLANE_TO_STACK_AXIS.get(plane, 0)
    return (stack_axis,) + tuple(a for a in (0, 1, 2) if a != stack_axis)


def preprocess_volume(nii_bytes, plane, target_spacing=(1.0, 1.0, 1.0), inplane_size=256,
                      posterior_shift_mm=15.0, max_slices=320, min_slices=32, normalize=True):
    """NIfTI bytes -> ([1, T, S, S] float32 in [0, 1], native spacing in mm).

    RAS reorientation, resampling to `target_spacing`, normalization over nonzero voxels, slice
    axis of the acquisition plane first, center crop/pad to S x S. `normalize=False` skips the
    intensity step.
    """
    data, spacing = read_canonical(nii_bytes)

    shape = [max(1, round(data.shape[i] * spacing[i] / target_spacing[i])) for i in range(3)]
    if shape != list(data.shape):
        data = F.interpolate(torch.from_numpy(data)[None, None], size=shape,
                             mode="trilinear", align_corners=False)[0, 0].numpy()

    if normalize:
        data = _normalize(data)

    order = plane_order(plane)
    data = np.ascontiguousarray(data.transpose(order))

    # A-P is in-plane unless it is itself the stacking axis (coronal), where the shift is moot.
    shift_axis = None if order[0] == AP_AXIS else order[1:].index(AP_AXIS)
    shift_voxels = round(posterior_shift_mm / target_spacing[AP_AXIS])
    data = _crop_pad(data, inplane_size, shift_voxels, shift_axis)

    if data.shape[0] > max_slices:
        # A >max_slices series is an over-long FOV rather than extra anatomy, so keep the centre.
        start = (data.shape[0] - max_slices) // 2
        data = data[start:start + max_slices]
    if data.shape[0] < min_slices:
        raise VolumeTooShort(f"{data.shape[0]} slices < min_slices={min_slices}")

    return torch.from_numpy(data[None].astype(np.float32)), spacing


### VAE ###


def encode_volume(vae, volume, batch_size=32, dtype=torch.float16):
    """[1, T, S, S] in [0, 1] -> [2C, T, S/8, S/8]: the VAE posterior's mean and std, unscaled.

    Slices are encoded independently by the 2D FLUX VAE, repeated to RGB.
    """
    slices = volume[0].unsqueeze(1).repeat(1, 3, 1, 1)  # (T, 3, S, S)
    out = []
    with torch.no_grad():
        for chunk in slices.split(batch_size):
            dist = vae.encode(chunk.to(vae.device, vae.dtype)).latent_dist
            out.append(torch.cat([dist.mean, dist.std], dim=1).to(dtype).cpu())
    return torch.cat(out).permute(1, 0, 2, 3).contiguous()


def encode_boundary(vae, value, inplane_size=256):
    """A constant-valued image through the identical encode path -> [2C, s, s].

    Sequence boundaries are learned tokens, not a length head: generation seeds with an all-black
    block and stops when a produced block matches all-white. Both must therefore live in exactly
    the same latent distribution as real data. Both are the ends of the pixel range, which is
    [0, 1] for MR as it was for CT, so black is 0 and white is 1 -- the same two values CTFlow used.
    """
    volume = torch.full((1, 1, inplane_size, inplane_size), float(value))
    return encode_volume(vae, volume)[:, 0]


### Artifact storage ###


class LatentStore:
    """Writes preprocessing artifacts, one file each or (`bundle=True`) one uncompressed zip per shard.

    The zip keeps the number of files small; a killed shard is re-encoded entirely.
    """

    def __init__(self, root, name, bundle=False, bundle_dir="latents"):
        self.root = root
        self.zip_path = os.path.join(bundle_dir, f"{name}.zip") if bundle else None
        if bundle:
            path = os.path.join(root, self.zip_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._tmp = f"{path}.tmp.{os.getpid()}"
            self._zf = zipfile.ZipFile(self._tmp, "w", zipfile.ZIP_STORED, allowZip64=True)

    def member(self, kind, sid):
        """Path of one artifact, relative to the bundle or, loose, to the dataset root."""
        if self.zip_path:
            return f"{kind}/{sid}.pt"
        return os.path.join(kind, sid[:2], f"{sid}.pt")  # fanout keeps Lustre directories small

    def write(self, member, tensor):
        if self.zip_path:
            buf = io.BytesIO()
            torch.save(tensor, buf)
            self._zf.writestr(member, buf.getvalue())
            return
        path = os.path.join(self.root, member)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(tensor, tmp)
        os.replace(tmp, path)  # atomic, so a killed task never leaves a half-written artifact

    def done(self, *members):
        """True if all of these are already written, so their series can be skipped."""
        return self.zip_path is None and all(
            os.path.exists(os.path.join(self.root, m)) for m in members)

    def read(self, member):
        return torch.load(os.path.join(self.root, member), map_location="cpu")

    def close(self):
        if self.zip_path:
            self._zf.close()
            os.replace(self._tmp, os.path.join(self.root, self.zip_path))  # atomic


### Manifest ###

# `zip` is the bundle the two paths are members of, or empty when they are loose files.
MANIFEST_FIELDS = ("sample_id", "split", "modality", "plane", "n_slices",
                   "latent_path", "embedding_path", "zip")
EMBEDDING_MANIFEST_FIELDS = ("sample_id", "split", "embedding_path", "zip")


def read_manifest(root, split):
    """Every row for one split. Shard manifests are globbed, so there is no merge step."""
    rows = []
    for path in sorted(glob(os.path.join(root, "manifest", "*.csv"))):
        with open(path, newline="") as f:
            rows += [r for r in csv.DictReader(f) if r["split"] == split]
    return rows


def write_manifest(path, rows, fields=MANIFEST_FIELDS):
    """One CSV per preprocessing shard; the dataset globs them, so there is no merge step."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
