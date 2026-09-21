"""MR-RATE for Text2CT: series listing, reports, volume preprocessing, latent cache, dataset.

**This is a vendored copy, not an import.** The Text2CT stack needs `transformers < 5` for its
vendored CLIP, so it runs in its own venv where `echosyn` does not import. Every geometry step
below is copied from `echosyn/common/mrrate.py` and the docstring says which function -- if that
file changes, this one has to be re-checked against it. Two differences are deliberate and both are
about the target grid rather than the transformation:

* MRFlow keeps `T` variable (the point of an autoregressive model) and caps it at 320. Text2CT's
  UNet is monolithic 3D, so every volume is cropped/padded to exactly `num_slices`.
* MRFlow hands the VAE a plane-first `[1, T, S, S]` stack of 2D slices. Text2CT's VAE is 3D and its
  scripts feed nibabel `(X, Y, Z)` with the slice axis last, so the array is transposed once more.

Nothing about the intensity path changes: RAS canonicalize, 1 mm trilinear resample, 0.5/99.5
percentile normalization over **nonzero** voxels to `[0, 1]`. That range is what Text2CT's own
decode path implies (`scripts/diff_model_demo.py:194` maps `[0, 1] -> [-1000, 1000]` HU *at decode*,
so the latent space is over `[0, 1]` volumes) and what MRFlow measured for MR. There is no
Hounsfield scale in MRI, so the CT clip to `[-1000, 1000]` in
`scripts/preprocess_ctrate.py:42` has no MR counterpart and is dropped.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import random
import tarfile
import zipfile
from collections import OrderedDict
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


### Geometry conventions (echosyn/common/mrrate.py) ###

PLANE_TO_STACK_AXIS = {"AXIAL": 0, "SAGITTAL": 1, "CORONAL": 2}
PLANE_CODES = {"axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL"}
SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}
AP_AXIS = 2  # anterior-posterior is (D, H, W) axis 2

PLANE_ALIASES = {
    "axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL",
    "obl": "AXIAL", "OBLIQUE": "AXIAL", "UNKNOWN": "AXIAL",
}
PLANES = ("AXIAL", "SAGITTAL", "CORONAL")


### Modality vocabulary ###

# Text2CT's UNet carries MAISI's class-conditioning table verbatim -- `num_class_embeds: 128` in
# `configs/config_rflow.json` -- and MAISI publishes what those 128 indices mean
# (`R2V-MR-Generation/models/nvidia_configs/modality_mapping.json`). Text2CT itself only ever used
# one of them: `scripts/diff_model_train.py:313` hardcodes `torch.ones(...)`, i.e. `ct` = 1.
#
# So the MR ids below are not invented. They are MAISI's own MR entries, which is the vocabulary
# the embedding table was sized for, and they leave `ct` = 1 untouched. Ids may never move: a
# checkpoint's `class_embedding.weight` is indexed by them.
PRETRAINED_CT_CLASS_ID = 1          # `ct` in MAISI's table; the only row Text2CT trained
NUM_CLASS_EMBEDS = 128              # must match diffusion_unet_def.num_class_embeds

MODALITY_TO_ID = {
    "CFG_NULL": 0,   # MAISI `unknown`; used only by the optional modality-CFG branch
    "UNKNOWN": 8,    # MAISI `mri`   -- a real category: MR of unrecorded contrast
    "T1w": 9,        # MAISI `mri_t1`
    "T2w": 10,       # MAISI `mri_t2`
    "FLAIR": 11,     # MAISI `mri_flair`
    "MRA": 16,       # MAISI `mri_mra`
    "SWI": 20,       # MAISI `mri_swi`
}
CFG_NULL_MODALITY_ID = MODALITY_TO_ID["CFG_NULL"]

# Every spelling the pipeline can hand us, as in echosyn/common/mrrate.py.
MODALITY_ALIASES = {
    "t1w": "T1w", "t1": "T1w",
    "t2w": "T2w", "t2": "T2w",
    "flair": "FLAIR",
    "swi": "SWI", "swan": "SWI",
    "mra": "MRA",
    "unknown": "UNKNOWN",
}

# What a modality id reads as in the conditioning text, when the acquisition prefix is on.
MODALITY_TEXT = {"T1w": "T1-weighted", "T2w": "T2-weighted", "FLAIR": "FLAIR", "SWI": "SWI",
                 "MRA": "MR angiography", "UNKNOWN": "MRI"}


def modality_name(modality):
    """Any spelling -> the canonical name. Raises on anything unmapped."""
    name = MODALITY_ALIASES.get(str(modality).strip().lower(), str(modality).strip())
    if name not in MODALITY_TO_ID:
        raise KeyError(
            f"unmapped modality {modality!r}: add it to MODALITY_TO_ID / MODALITY_ALIASES in "
            f"baselines/text2ct/mrrate_data.py. Known: {sorted(MODALITY_TO_ID)}"
        )
    return name


def modality_to_id(modality):
    """Modality string -> MAISI class id. Raises rather than guessing."""
    return MODALITY_TO_ID[modality_name(modality)]


def plane_name(plane):
    """Any spelling -> AXIAL / SAGITTAL / CORONAL. Oblique and unknown go to axial, as
    `plane_order` itself does, so the label and the array layout always agree."""
    name = str(plane).strip()
    name = PLANE_ALIASES.get(name, PLANE_ALIASES.get(name.lower(), name))
    if name not in PLANES:
        raise KeyError(f"unmapped plane {plane!r}. Known: {sorted(PLANES)}")
    return name


def modality_vocabulary():
    """The mapping as it is written into every checkpoint and every cache manifest."""
    return {"modality_to_id": dict(MODALITY_TO_ID),
            "pretrained_ct_class_id": PRETRAINED_CT_CLASS_ID,
            "num_class_embeds": NUM_CLASS_EMBEDS}


### Archive access (echosyn/common/mrrate.py) ###

_HANDLES = OrderedDict()


def _cached(path, opener, max_open=8):
    key = (os.getpid(), path)   # keyed on pid so a forked worker never shares a read position
    if key in _HANDLES:
        _HANDLES.move_to_end(key)
        return _HANDLES[key]
    _HANDLES[key] = opener(path)
    while len(_HANDLES) > max_open:
        _, old = _HANDLES.popitem(last=False)
        old.close()
    return _HANDLES[key]


def read_member(archive_path, member):
    """Raw bytes of one tar member."""
    fobj = _cached(archive_path, lambda p: tarfile.open(p, mode="r:")).extractfile(member)
    if fobj is None:
        raise IOError(f"{member} is not a regular file in {archive_path}")
    return fobj.read()


def read_bundled(bundle_path, member):
    """Raw bytes of one member of a cache bundle. See LatentStore."""
    return _cached(bundle_path, zipfile.ZipFile).read(member)


def read_report(archive_path, study_uid):
    """One study's report, sections kept separate."""
    return json.loads(read_member(archive_path, f"{study_uid}/report.json"))


def sample_id(study_uid, series_id):
    """Stable, identifier-free artifact name -- the same hash MRFlow and `baselines/common/cases.py`
    use, so a case id means the same series in every table."""
    return hashlib.sha1(f"{study_uid}|{series_id}".encode()).hexdigest()[:16]


def list_series(raw_root, split, max_repeats=1, max_series=None, seed=0):
    """Eligible series for one split, from MR-RATE's own parquet indices.

    Same selection as `echosyn.common.mrrate.list_series`: derived series and localizers dropped,
    studies with no report dropped, deterministic shuffle before the cap. Keeping it identical is
    what makes "the same training data" true across the table.
    """
    import pyarrow.parquet as pq

    studies = pq.read_table(os.path.join(raw_root, "studies.parquet"),
                            columns=["study_uid", "split", "has_report"])
    with_report = {studies["study_uid"][i].as_py() for i in range(studies.num_rows)
                   if studies["has_report"][i].as_py() and studies["split"][i].as_py() == split}

    table = pq.read_table(
        os.path.join(raw_root, "series.parquet"),
        columns=["study_uid", "series_id", "split", "shard_name", "modality", "plane", "repeat",
                 "is_derived", "is_localizer", "image_present", "tar_member_path"])
    col = {name: table.column(name) for name in table.column_names}
    split_dir = SPLIT_DIRS.get(split, split)

    series = []
    for i in range(table.num_rows):
        if col["split"][i].as_py() != split or not col["image_present"][i].as_py():
            continue
        if col["is_derived"][i].as_py() or col["is_localizer"][i].as_py():
            continue
        study = col["study_uid"][i].as_py()
        if study not in with_report:
            continue
        if max_repeats and (col["repeat"][i].as_py() or 0) > max_repeats:
            continue
        plane = col["plane"][i].as_py()
        series.append({
            "study_uid": study,
            "series_id": col["series_id"][i].as_py(),
            "modality": col["modality"][i].as_py() or "UNKNOWN",
            "plane": PLANE_CODES.get(plane, plane or "UNKNOWN"),
            "archive": os.path.join(raw_root, split_dir, col["shard_name"][i].as_py() + ".tar"),
            "member": col["tar_member_path"][i].as_py(),
        })

    random.Random(seed).shuffle(series)
    return series[:max_series] if max_series else series


### Conditioning text ###


class ReportMissing(ValueError):
    """Every requested section is empty -- there is nothing to condition on."""


def format_report(report, sections=("findings", "impression"), modality=None,
                  modality_prefix=False):
    """The string Text2CT's 3D-CLIP encoder is handed.

    By default this is `scripts/save_embeddings_ctrate.py:81`'s format verbatim --
    `"Findings: {...} Impression: {...}"` -- because that is the text distribution the released
    encoder was contrastively trained on, and reformatting it would change what the frozen encoder
    means before a single UNet weight moved.

    **Modality does not reach the model through the text.** It is a class id on the UNet's own
    `class_embedding` input (see `model.seed_modality_embeddings`), which is the conditioning hook
    this adaptation uses. `modality_prefix=True` additionally prepends the contrast in words, as a
    knob for testing whether the frozen text tower adds anything on top of the class embedding; it
    is off because the prefix is off-distribution for a CT-trained encoder and the class embedding
    already carries the signal.

    **Plane is not conditioned at all** -- neither here nor as a class id. Text2CT has exactly one
    class input and it is spent on modality. The consequence is real and is stated in the README:
    a generation cannot be asked for a sagittal volume, because the grid is identical for all three
    planes and nothing else tells the model which one to produce.

    Raises `ReportMissing` when every requested section is empty: ~9% of MR-RATE studies have no
    impression, and a caller asking for impression only must see that rather than train on
    `"Impression: "`.
    """
    parts = []
    for section in sections:
        text = (report.get(section) or "").strip()
        if text:
            parts.append(f"{section.capitalize()}: {text}")
    if not parts:
        raise ReportMissing(f"no text in sections {sections}")

    body = " ".join(parts)
    if not modality_prefix:
        return body
    contrast = MODALITY_TEXT.get(modality_name(modality), "MRI") if modality else "MRI"
    return f"Brain MRI, {contrast}. {body}"


### Volume preprocessing ###


class VolumeUnusable(ValueError):
    """The series cannot become a training sample: too few slices, degenerate, or non-finite."""


def _normalize(data, lower=0.5, upper=99.5):
    """`echosyn.common.mrrate._normalize`: the [lower, upper] percentile of the **nonzero** voxels
    to [0, 1]. Nonzero because MR-RATE is defaced and zero-padded -- background would otherwise own
    every percentile -- and percentiles rather than min-max because the archive mixes uint16 and
    float32 series with a >100x dynamic range difference."""
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
    """`echosyn.common.mrrate._crop_pad`: center crop/pad the two in-plane axes of a (T, A, B)
    array. `shift_voxels` moves the window toward lower indices on the anterior-posterior axis --
    MR-RATE is defaced, so a plain center crop keeps a band of removed face and loses posterior
    brain. Background after normalization is 0, so that is the pad value."""
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


def _fit_slices(volume, num_slices):
    """Center crop or symmetrically zero-pad axis 0 to exactly `num_slices`.

    Text2CT's UNet is monolithic, so the depth is fixed where MRFlow's is not. Centering keeps the
    brain: an over-long MR-RATE FOV is extra neck and scalp at the ends, not extra anatomy.
    """
    cur = volume.shape[0]
    if cur > num_slices:
        start = (cur - num_slices) // 2
        return volume[start:start + num_slices]
    if cur < num_slices:
        before = (num_slices - cur) // 2
        return np.pad(volume, ((before, num_slices - cur - before), (0, 0), (0, 0)),
                      constant_values=0.0)
    return volume


def read_canonical(nii_bytes):
    """`echosyn.common.mrrate.read_canonical`: NIfTI bytes -> (RAS-canonical (S, R, A) float32
    array, native (D, H, W) spacing in mm). No axis is ever flipped; nothing is resampled."""
    payload = gzip.decompress(nii_bytes) if nii_bytes[:2] == b"\x1f\x8b" else nii_bytes
    img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(payload))
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))  # (X, Y, Z) -> (D, H, W)

    data = np.asarray(img.get_fdata(), dtype=np.float32)
    np.nan_to_num(data, copy=False)
    return np.ascontiguousarray(data.transpose(2, 0, 1)), spacing  # (R, A, S) -> (S, R, A)


def plane_order(plane):
    """`echosyn.common.mrrate.plane_order`: the (S, R, A) permutation that puts the acquisition
    plane's slice axis first -- SRA axial, RSA sagittal, ASR coronal."""
    stack_axis = PLANE_TO_STACK_AXIS.get(plane_name(plane), 0)
    return (stack_axis,) + tuple(a for a in (0, 1, 2) if a != stack_axis)


def target_spacing_sra(plane, inplane_mm, slice_mm):
    """The (S, R, A) voxel size that realises `(inplane_mm, inplane_mm, slice_mm)` on the output.

    This exists because the target grid is **anisotropic**, and the coarse axis is the acquisition
    plane's stacking axis -- S for axial, R for sagittal, A for coronal. `preprocess_volume`
    resamples before it permutes, so the spacing it resamples with has to be stated in the
    pre-permutation frame, and getting it wrong is silent: an isotropic grid hides the error and an
    anisotropic one blurs the wrong axis.
    """
    order = plane_order(plane)
    spacing = [float(inplane_mm)] * 3
    spacing[order[0]] = float(slice_mm)
    return tuple(spacing)


def preprocess_volume(nii_bytes, plane, inplane_mm=0.5, slice_mm=1.5, inplane_size=512,
                      num_slices=128, posterior_shift_mm=15.0, min_native_slices=32,
                      percentiles=(0.5, 99.5)):
    """MR-RATE NIfTI bytes -> (float32 `(inplane, inplane, num_slices)` in [0, 1], native spacing,
    slice count before the depth was fitted).

    RAS canonicalize -> trilinear resample to the plane-aware target spacing -> percentile-normalize
    over nonzero voxels -> permute plane-first -> in-plane center crop/pad with the posterior shift
    -> fit the slice axis -> transpose to Text2CT's `(X, Y, Z)` layout, slice axis last.

    The returned array is what `scripts/diff_model_create_training_data.py` would hand
    `autoencoder.encode_stage_2_inputs` for a CT, minus the HU window: in-plane first two axes,
    stacking axis last, values in [0, 1]. At the default grid it is `(512, 512, 128)` at
    `0.5/0.5/1.5` mm -- Text2CT's own `(4, 128, 128, 32)` latent shape over a brain-sized
    `256 x 256 x 192` mm field of view.

    The resample is a single trilinear step from the **native** spacing, never a detour through an
    intermediate grid, so the in-plane upsampling costs one interpolation and no compounding.
    """
    data, spacing = read_canonical(nii_bytes)
    if not np.isfinite(data).all():
        raise VolumeUnusable("non-finite voxels after canonicalization")

    target = target_spacing_sra(plane, inplane_mm, slice_mm)
    shape = [max(1, round(data.shape[i] * spacing[i] / target[i])) for i in range(3)]
    if shape != list(data.shape):
        data = F.interpolate(torch.from_numpy(data)[None, None], size=shape,
                             mode="trilinear", align_corners=False)[0, 0].numpy()

    data = _normalize(data, *percentiles)

    order = plane_order(plane)
    data = np.ascontiguousarray(data.transpose(order))
    if data.shape[0] < min_native_slices:
        raise VolumeUnusable(f"{data.shape[0]} slices < min_native_slices={min_native_slices}")

    # A-P is in-plane unless it is itself the stacking axis (coronal), where the shift is moot.
    shift_axis = None if order[0] == AP_AXIS else order[1:].index(AP_AXIS)
    data = _crop_pad(data, inplane_size, round(posterior_shift_mm / target[AP_AXIS]), shift_axis)
    native_slices = data.shape[0]
    data = _fit_slices(data, num_slices)

    return (np.ascontiguousarray(data.transpose(1, 2, 0), dtype=np.float32), spacing,
            native_slices)


def save_generated_nifti(volume, plane, path, inplane_mm=0.5, slice_mm=1.5):
    """A generated `(X, Y, Z)` volume -> a RAS NIfTI that `read_canonical` inverts exactly.

    This is the **inverse of `preprocess_volume`'s geometry**, and it is the one place a baseline can
    scramble anatomy without erroring: `baselines/common/ingest.py` reads the file with
    `read_canonical` -- the same function the ground-truth path uses -- so the affine and the axis
    order here are what tell the evaluation which way is up. Written wrong, every metric still
    computes and every number is meaningless.

    Undoing `preprocess_volume` step for step, in reverse:

        (X, Y, Z)      the model's output, in-plane axes first, slice axis last
        -> transpose(2, 0, 1)        back to plane-first (T, A, B)
        -> inverse plane_order       back to (S, R, A)
        -> transpose(1, 2, 0)        back to nibabel's canonical (R, A, S)
        affine = diag(spacing_R, spacing_A, spacing_S, 1)

    `test_generated_nifti_round_trips_through_read_canonical` asserts the identity exactly, and
    `test_a_round_tripped_real_volume_still_looks_like_the_same_brain` asserts it on real anatomy --
    a transposed axis survives the first test's shape check only if the array is cubic, so both are
    needed.
    """
    import nibabel as nib

    order = plane_order(plane)
    inverse = np.argsort(order)
    plane_first = np.asarray(volume).transpose(2, 0, 1)
    sra = plane_first.transpose(inverse)
    ras = np.ascontiguousarray(sra.transpose(1, 2, 0), dtype=np.float32)

    spacing = target_spacing_sra(plane, inplane_mm, slice_mm)   # (S, R, A)
    affine = np.diag([spacing[1], spacing[2], spacing[0], 1.0])  # nibabel zooms are (R, A, S)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nib.save(nib.Nifti1Image(ras, affine=affine), path)
    return path


### Artifact cache ###

# Preprocessing writes two artifacts per series (a latent and a text embedding). At the full
# 575,328-series train split that is 1.15M files, against this account's 102,400-inode hard limit
# on /hnvme -- so, as in MRFlow, a shard's artifacts go into one uncompressed zip and the manifest
# records which layout was written. A zip's central directory holds every member's offset, so a
# single latent still reads without unpacking.

MANIFEST_FIELDS = ("sample_id", "split", "modality", "plane", "n_native_slices", "native_spacing",
                   "latent_path", "embedding_path", "zip")


class LatentStore:
    """Where preprocessing puts its `.npy` artifacts: one zip per shard, or loose files.

    Every path it hands back -- the bundle name and each member -- is relative to `cache_root`, so
    a manifest row means the same thing wherever the cache is mounted and `load_artifact` needs
    nothing but the root.
    """

    SUBDIR = "artifacts"

    def __init__(self, cache_root, shard, split, bundle=True):
        # The bundle name carries the SPLIT as well as the shard. Without it, `train` shard 3 and
        # `val` shard 3 are the same path, and two array tasks running concurrently truncate each
        # other's archive in place -- silently, since each still writes its own manifest.
        self.cache_root = cache_root
        self.bundle = bundle
        self.name = os.path.join(self.SUBDIR, f"{split}-{shard:04d}.zip")
        os.makedirs(os.path.join(cache_root, self.SUBDIR), exist_ok=True)
        if bundle:
            self._zip = zipfile.ZipFile(os.path.join(cache_root, self.name), "w",
                                        zipfile.ZIP_STORED)

    def save(self, member, array):
        buf = io.BytesIO()
        np.save(buf, array, allow_pickle=False)
        if self.bundle:
            self._zip.writestr(member, buf.getvalue())
            return member
        path = os.path.join(self.SUBDIR, member)
        with open(os.path.join(self.cache_root, path), "wb") as handle:
            handle.write(buf.getvalue())
        return path

    @property
    def zip_name(self):
        return self.name if self.bundle else ""

    def close(self):
        if self.bundle:
            self._zip.close()


def load_artifact(root, row, key):
    """One manifest row's latent or embedding, from its bundle or as a loose file."""
    if row.get("zip"):
        blob = read_bundled(os.path.join(root, row["zip"]), row[key])
        return np.load(io.BytesIO(blob), allow_pickle=False)
    return np.load(os.path.join(root, row[key]), allow_pickle=False)


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(root, split):
    """Every shard manifest for one split, concatenated in filename order."""
    rows = []
    for path in sorted(glob(os.path.join(root, "manifest", f"{split}-*.csv"))):
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


### Cache metadata ###

# The cache is only valid for the settings it was written under: change the grid, the percentiles,
# the report sections or either frozen checkpoint and every stored array means something else.
# `cache_meta.json` records all of it and `check_cache_meta` refuses a mismatch rather than
# training on stale arrays.

CACHE_META_NAME = "cache_meta.json"
CACHE_FORMAT_VERSION = 1


def file_digest(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_meta(config, vae_ckpt, clip_ckpt):
    """The fingerprint of everything that decides what a cached artifact contains."""
    volume, data = config["volume"], config["data"]
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "volume": {k: volume[k] for k in ("inplane_mm", "slice_mm", "inplane_size", "num_slices",
                                          "posterior_shift_mm", "percentiles",
                                          "min_native_slices")},
        "text": {"sections": list(data["report_sections"]),
                 "modality_prefix": bool(data["modality_prefix"]),
                 "max_length": data["text_max_length"]},
        "vae_sha256": file_digest(vae_ckpt),
        "clip_sha256": file_digest(clip_ckpt),
        "modality": modality_vocabulary(),
    }


def _comparable(meta):
    """Everything but the checkpoint digests, which are compared separately and reported by name."""
    return {k: v for k, v in meta.items() if not k.endswith("_sha256")}


def check_cache_meta(cache_root, expected, strict=True):
    """Raise unless the cache at `cache_root` was written under `expected`."""
    path = os.path.join(cache_root, CACHE_META_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. Rebuild the cache with baselines/text2ct/prepare_data.py, or pass "
            f"--ignore_cache_meta if you know the arrays match the config.")
    with open(path) as handle:
        found = json.load(handle)

    problems = []
    if _comparable(found) != _comparable(expected):
        for key in sorted(set(found) | set(expected)):
            if key.endswith("_sha256"):
                continue
            if found.get(key) != expected.get(key):
                problems.append(f"  {key}: cache={found.get(key)!r} config={expected.get(key)!r}")
    for key in ("vae_sha256", "clip_sha256"):
        if found.get(key) != expected.get(key):
            problems.append(f"  {key}: cache={found.get(key)} config={expected.get(key)}")
    if problems and strict:
        raise ValueError("cached artifacts were written under different settings:\n"
                         + "\n".join(problems))
    return problems


### Dataset ###


class Text2CTLatentDataset(torch.utils.data.Dataset):
    """One cached MR-RATE series -> one Text2CT training sample.

    Returns the four tensors the UNet is called with, plus the loss target's ingredients:

        latent      [C, X, Y, Z]  float32, the frozen VAE posterior sample, **unscaled**
        context     [1, D]        float32, the frozen 3D-CLIP report embedding
        class_label []            int64, the MAISI modality id
        spacing     [3]           float32, the output grid's spacing in mm x 1e2

    `spacing` carries the `* 1e2` that `scripts/diff_model_train.py:111` applies, because the
    trained `spacing_layer` expects that scale. Every cached volume is on the same grid, so this is
    a constant -- it is still passed per sample so the tensor's meaning does not depend on the
    training script.
    """

    def __init__(self, cache_root, split, spacing_mm=(0.5, 0.5, 1.5), limit=None):
        self.cache_root = cache_root
        self.rows = read_manifest(cache_root, split)
        if not self.rows:
            raise RuntimeError(f"no {split!r} rows in {cache_root}/manifest/{split}-*.csv")
        if limit:
            self.rows = self.rows[:limit]
        # Resolved once, in the main process: an unmapped label fails here, not in a worker.
        self.class_labels = [modality_to_id(row["modality"]) for row in self.rows]
        self.spacing = np.array(spacing_mm, dtype=np.float32) * 1e2

    def __len__(self):
        return len(self.rows)

    def counts(self):
        out = {}
        for row in self.rows:
            key = f"{modality_name(row['modality'])}/{plane_name(row['plane'])}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def __getitem__(self, idx):
        row = self.rows[idx]
        latent = load_artifact(self.cache_root, row, "latent_path")
        context = load_artifact(self.cache_root, row, "embedding_path")
        return {
            "latent": torch.from_numpy(np.ascontiguousarray(latent)).float(),
            "context": torch.from_numpy(np.ascontiguousarray(context)).float(),
            "class_label": torch.tensor(self.class_labels[idx], dtype=torch.long),
            "spacing": torch.from_numpy(self.spacing.copy()),
            "sample_id": row["sample_id"],
        }
