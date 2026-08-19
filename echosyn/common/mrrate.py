"""MR-RATE support: raw shard access, report formatting, volume preprocessing, VAE encoding.

MR-RATE ships as webdataset tars of per-study NIfTI series plus a study-level `report.json`,
indexed by `series.parquet`. Members are read straight out of the tars -- nothing is extracted.

Volumes reach the VAE as [1, T, S, S] in [0, 1], the same range CTFlow normalized CT to. That is
not cosmetic: the CTFlow checkpoint this fine-tunes from transfers 27x better to [0, 1] latents than
to [-1, 1] ones at matched latent scale, and standardizing cannot close the gap -- the VAE encoder
is nonlinear, so an affine change in pixel space is not an affine change in latent space
(tools/ctflow_transfer_check.py). MRI has no Hounsfield scale, so there is no HU window and no
`* 306` (CTFlow's factor rescales its (-1000, 1400) window to the challenge's (-1000, 1000)):
intensities are normalized per volume over *nonzero* voxels only, because MR-RATE images are defaced
and zero-padded and background would otherwise dominate every percentile.
"""

import csv
import gzip
import hashlib
import json
import os
import random
import tarfile
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Internal axis order is (D, H, W) = (S, R, A); nibabel canonical RAS is (X, Y, Z) = (R, A, S).
# AXIAL slices stack along S, SAGITTAL along R, CORONAL along A. Getting this permute wrong is
# silent for an isotropic cube and scrambles anatomy otherwise.
PLANE_TO_STACK_AXIS = {"AXIAL": 0, "SAGITTAL": 1, "CORONAL": 2}
PLANE_CODES = {"axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL"}
SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}

AP_AXIS = 2  # anterior-posterior is (D, H, W) axis 2


### Archive access ###

# (pid, path) -> TarFile. Keyed on pid so a forked dataloader worker never inherits the parent's
# file descriptor and its read position.
_HANDLES = OrderedDict()


def _archive(path, max_open=8):
    key = (os.getpid(), path)
    if key in _HANDLES:
        _HANDLES.move_to_end(key)
        return _HANDLES[key]
    _HANDLES[key] = tarfile.open(path, mode="r:")
    while len(_HANDLES) > max_open:
        _, old = _HANDLES.popitem(last=False)
        old.close()
    return _HANDLES[key]


def read_member(archive_path, member):
    """Raw bytes of one tar member."""
    fobj = _archive(archive_path).extractfile(member)
    if fobj is None:
        raise IOError(f"{member} is not a regular file in {archive_path}")
    return fobj.read()


### Series index and reports ###


def list_series(raw_root, split, max_repeats=1, max_series=None, seed=0):
    """Eligible series for one split, from the dataset's own parquet indices.

    Derived series and localizers are dropped, as are studies with no report -- the report is the
    entire conditioning signal. Geometry columns are ignored on purpose: that index's axis
    convention is unverifiable, so spacing is always re-read from the NIfTI header.

    `max_repeats` drops duplicate acquisitions of the same contrast and plane (MR-RATE numbers them
    `t1w-raw-axi-2`, `-3`, ...), which is 576k -> 478k train series on its own. `max_series` then
    caps the total, because the full split is ~9 TB of latents and ~1300 GPU-hours to encode, while
    a 60k-step fine-tune at an effective batch of 64 only sees 3.8M samples -- 120k series is
    already ~30 epochs.

    The list is shuffled deterministically before capping, so the cap stays representative (parquet
    order is by shard, hence by study) and every preprocessing array task gets a mixed workload.
    """
    import pyarrow.parquet as pq

    studies = pq.read_table(
        os.path.join(raw_root, "studies.parquet"),
        columns=["study_uid", "split", "has_report"],
    )
    with_report = {
        studies["study_uid"][i].as_py()
        for i in range(studies.num_rows)
        if studies["has_report"][i].as_py() and studies["split"][i].as_py() == split
    }

    table = pq.read_table(
        os.path.join(raw_root, "series.parquet"),
        columns=["study_uid", "series_id", "split", "shard_name", "modality", "plane", "repeat",
                 "is_derived", "is_localizer", "image_present", "tar_member_path"],
    )
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


def read_report(archive_path, study_uid):
    """One study's report, sections kept separate."""
    return json.loads(read_member(archive_path, f"{study_uid}/report.json"))


def format_report(report, sections=("findings", "impression")):
    """The report half of the conditioning string.

    Sections are bracketed because the raw MR-RATE text already contains `Findings:` headings, so a
    prose heading would be indistinguishable from content. Empty sections are dropped rather than
    emitted as a bare marker -- impression is missing for ~9% of studies, and a bare marker would
    teach the model it carries no information.
    """
    return "\n".join(f"[{s.upper()}] {report[s].strip()}"
                     for s in sections if (report.get(s) or "").strip())


def acquisition_prefix(modality, plane, spacing=None):
    """The structured half, and what every conditioning string starts with.

    Values come from the series index and the NIfTI header, never parsed out of the report.
    `spacing` is the **native** (D, H, W) = (S, R, A) voxel size in mm, i.e. the acquisition's own
    geometry rather than the 1 mm grid everything is resampled onto -- that is the informative
    version, since a 5 mm-slice acquisition stays visibly blurred through-plane after resampling.
    """
    parts = [f"[MODALITY] {modality}", f"[PLANE] {plane}"]
    if spacing is not None:
        parts.append("[SPACING] " + " ".join(f"{float(v):.2f}" for v in spacing))
    return " ".join(parts)


def encode_conditioning(tokenizer, model, report, modality, plane, spacing=None, marker_weight=1.0,
                        max_length=512, sections=("findings", "impression")):
    """Report + acquisition markers -> one pooled [1, D] conditioning vector.

    The string is `acquisition_prefix` followed by the report sections, so conditioning always
    starts with modality, plane and spacing.

    MR-RATE reports are study-level but the target is a single series, so one report maps to up to
    ~12 volumes of wildly different contrast and plane, and the markers are the *only* signal
    saying which one to generate. Leaving them inside the report string does not survive pooling: a
    ~60-character prefix inside a ~2000-character report moves the pooled CLS vector by a cosine of
    0.0002, while report content spans ~0.03 across studies -- roughly 170x weaker, so the model
    would be conditioned on which study but blind to which contrast.

    So the prefix is *also* pooled on its own and added back as a unit vector. At
    `marker_weight=1.0`, contrast/plane separates same-study series about as strongly as report
    content separates studies (measured cosine 0.976 vs 0.972). Set 0.0 to keep only the plain
    prefix-then-report string.

    Stays a single 768-d token, so STDiT's caption path and the released checkpoint's weights are
    untouched.
    """
    prefix = acquisition_prefix(modality, plane, spacing)
    full = encode_report(tokenizer, model, f"{prefix}\n{format_report(report, sections)}", max_length)
    if not marker_weight:
        return full
    return _unit(full) + marker_weight * _unit(
        encode_report(tokenizer, model, prefix, max_length))


def _unit(x):
    return x / (x.norm(p=2) + 1e-6)


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


def preprocess_volume(nii_bytes, plane, target_spacing=(1.0, 1.0, 1.0), inplane_size=256,
                      posterior_shift_mm=15.0, max_slices=320, min_slices=32):
    """MR-RATE NIfTI bytes -> ([1, T, S, S] float32 in [0, 1], native (D, H, W) spacing in mm).

    RAS canonicalize -> trilinear resample to `target_spacing` -> intensity normalize -> permute
    so the acquisition plane's slice axis leads -> center crop/pad in-plane to S x S -> cap T.

    The native spacing is returned because it goes into the conditioning text; it describes the
    acquisition, not the resampled output (which is `target_spacing` for every volume).

    At 1 mm one voxel is one mm in every volume, so anatomy is the same real-world size everywhere.
    Over all MR-RATE series ~94% already fit inside a 256 mm frame and are only padded; the rest
    lose a thin rim (median 4 mm).
    """
    payload = gzip.decompress(nii_bytes) if nii_bytes[:2] == b"\x1f\x8b" else nii_bytes
    img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(payload))
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))  # (X, Y, Z) -> (D, H, W)

    data = np.asarray(img.get_fdata(), dtype=np.float32)
    np.nan_to_num(data, copy=False)
    data = np.ascontiguousarray(data.transpose(2, 0, 1))  # (R, A, S) -> (S, R, A)

    shape = [max(1, round(data.shape[i] * spacing[i] / target_spacing[i])) for i in range(3)]
    if shape != list(data.shape):
        data = F.interpolate(torch.from_numpy(data)[None, None], size=shape,
                             mode="trilinear", align_corners=False)[0, 0].numpy()

    data = _normalize(data)

    stack_axis = PLANE_TO_STACK_AXIS.get(plane, 0)  # unknown/oblique -> axial, the majority plane
    order = (stack_axis,) + tuple(a for a in (0, 1, 2) if a != stack_axis)
    data = np.ascontiguousarray(data.transpose(order))

    # A-P is in-plane unless it is itself the stacking axis (coronal), where the shift is moot.
    shift_axis = None if stack_axis == AP_AXIS else order[1:].index(AP_AXIS)
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
    """[1, T, S, S] in [0, 1] -> [C, T, S/8, S/8], unscaled.

    FLUX's VAE is 2D, so slices become the batch axis and the single MR channel is repeated to RGB
    here -- the only place that happens. There is no compression along the slice axis, so
    T_latent == T_image and the released CTFlow STDiT stays shape-compatible.

    The posterior mean is stored, not a sample, so a config sets `sample_latents: false` and the
    stored latents are already `latent_channels` wide. Latents are stored *unscaled*: train.py
    applies `scale_latents` itself, exactly as it does for CT.

    Stored as fp16 by default: a full split is ~9 TB in fp32, the model trains in bf16 anyway, and
    the dataset casts back to float on load.
    """
    slices = volume[0].unsqueeze(1).repeat(1, 3, 1, 1)  # (T, 3, S, S)
    out = []
    with torch.no_grad():
        for chunk in slices.split(batch_size):
            dist = vae.encode(chunk.to(vae.device, vae.dtype)).latent_dist
            out.append(dist.mean.to(dtype).cpu())
    return torch.cat(out).permute(1, 0, 2, 3).contiguous()


def encode_boundary(vae, value, inplane_size=256):
    """A constant-valued image through the identical encode path -> [C, s, s].

    Sequence boundaries are learned tokens, not a length head: generation seeds with an all-black
    block and stops when a produced block matches all-white. Both must therefore live in exactly
    the same latent distribution as real data. Both are the ends of the pixel range, which is
    [0, 1] for MR as it was for CT, so black is 0 and white is 1 -- the same two values CTFlow used.
    """
    volume = torch.full((1, 1, inplane_size, inplane_size), float(value))
    return encode_volume(vae, volume)[:, 0]


### Text ###


def build_text_encoder(checkpoint, device, dtype=torch.float32):
    """Frozen CXR-BERT -> (tokenizer, model).

    Loaded as a stock `BertModel` rather than with `trust_remote_code`: CXR-BERT declares a custom
    model type whose repo code would otherwise execute, while its `bert.*` weights are plain BERT.
    `.from_pretrained` on the named tokenizer class, never `BertTokenizerFast(vocab_file=...)` --
    the constructor form builds a WordPiece model that emits one [UNK] per word without raising.
    """
    from transformers import BertConfig, BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    config = BertConfig.from_pretrained(checkpoint, local_files_only=True)
    model = BertModel.from_pretrained(checkpoint, config=config, local_files_only=True,
                                      add_pooling_layer=False)
    model.eval().to(device, dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def encode_report(tokenizer, model, text, max_length=512):
    """-> [1, D] pooled CLS state, unnormalized.

    STDiT cross-attends exactly one caption token (`model_max_length: 1`), so the report becomes a
    single pooled vector. Stored unnormalized; the dataset L2-normalizes at load, as CT does.
    """
    batch = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length,
                      return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        tokens = model(**batch).last_hidden_state
    return tokens[:, 0].float().cpu()


### Manifest ###

MANIFEST_FIELDS = ("sample_id", "split", "modality", "plane", "n_slices",
                   "latent_path", "embedding_path")


def write_manifest(path, rows):
    """One CSV per preprocessing shard; the dataset globs them, so there is no merge step."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
