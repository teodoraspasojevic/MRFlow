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

# Internal axis order is (D, H, W) = (S, R, A); nibabel canonical RAS is (X, Y, Z) = (R, A, S).
# AXIAL slices stack along S, SAGITTAL along R, CORONAL along A. Getting this permute wrong is
# silent for an isotropic cube and scrambles anatomy otherwise.
PLANE_TO_STACK_AXIS = {"AXIAL": 0, "SAGITTAL": 1, "CORONAL": 2}
PLANE_CODES = {"axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL"}
SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}

AP_AXIS = 2  # anterior-posterior is (D, H, W) axis 2


### Class labels ###

# Modality and plane reach the model as integer class ids, the way NVIDIA's
# `configs/modality_mapping.json` feeds `class_labels` into MAISI. These tables are the single
# source of truth: they are literals, so an id means the same thing in every split, every worker,
# every run and every checkpoint. Never sort, enumerate or hash at runtime.
#
# Id 0 is the classifier-free-guidance null, as NVIDIA's `unknown: 0` is. UNKNOWN is a *real*
# category -- `list_series` emits it when the series index has no modality -- so it gets its own id.
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

# Every spelling the pipeline can hand us: MR-RATE's own column values, and the lowercase codes
# the challenge's `{study}_{modality}-raw-{plane}` case ids use.
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

# Plane is a geometry condition with no null and no unknown class. Both spellings below fall back
# to axial because that is what `plane_order` itself does with them, so the label and the array
# layout always agree.
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
            f"echosyn/common/mrrate.py. Known: {sorted(MODALITY_TO_ID)}"
        )
    return MODALITY_TO_ID[name]


def plane_to_id(plane):
    """Plane string -> class id. Raises on anything not in the tables above."""
    name = str(plane).strip()
    name = PLANE_ALIASES.get(name, PLANE_ALIASES.get(name.lower(), name))
    if name not in PLANE_TO_ID:
        raise KeyError(
            f"unmapped plane {plane!r}: add it to PLANE_TO_ID or PLANE_ALIASES in "
            f"echosyn/common/mrrate.py. Known: {sorted(PLANE_TO_ID)}"
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


### Archive access ###

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


def read_member(archive_path, member):
    """Raw bytes of one tar member."""
    fobj = _cached(archive_path, lambda p: tarfile.open(p, mode="r:")).extractfile(member)
    if fobj is None:
        raise IOError(f"{member} is not a regular file in {archive_path}")
    return fobj.read()


def read_bundled(bundle_path, member):
    """Raw bytes of one member of a latent bundle. See LatentStore."""
    return _cached(bundle_path, zipfile.ZipFile).read(member)


def load_artifact(root, row, key):
    """One manifest row's latent or embedding, from its bundle or as a loose file."""
    if row.get("zip"):
        blob = read_bundled(os.path.join(root, row["zip"]), row[key])
        return torch.load(io.BytesIO(blob), map_location="cpu")
    return torch.load(os.path.join(root, row[key]), map_location="cpu")


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


SECTIONS = ("findings", "impression", "acquisition")


def report_sections(report, modality, plane, spacing=None):
    """The three sectioned-conditioning sections, in `SECTIONS` order.
    """
    return (
        (report.get("findings") or "").strip(),
        (report.get("impression") or "").strip(),
        acquisition_prefix(modality, plane, spacing),
    )


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


def read_canonical(nii_bytes):
    """MR-RATE NIfTI bytes -> (RAS-canonical (S, R, A) float32 array, native (D, H, W) spacing).

    Everything both the training and the evaluation path share, and nothing either of them can
    skip: the reorientation and the header read. Neither resamples nor rescales anything.

    The volume stays **RAS-oriented** -- no axis is ever flipped. `(S, R, A)` names the array's
    axis *order*, which is nibabel's canonical `(R, A, S)` transposed once so the slice axis of an
    axial acquisition (the majority plane) leads; `plane_order` then permutes it per plane.
    """
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
    is what both a preprocessed volume and a generated one are in, which is why the evaluation can
    compare them voxel for voxel."""
    stack_axis = PLANE_TO_STACK_AXIS.get(plane, 0)
    return (stack_axis,) + tuple(a for a in (0, 1, 2) if a != stack_axis)


def load_native_volume(nii_bytes, plane):
    """MR-RATE NIfTI bytes -> (raw float32 volume in the model's axis order, native spacing).

    The evaluation reference: reoriented to RAS and permuted plane-first so its axes mean the same
    thing as a generated volume's, and *nothing else*. No resample, no normalization, no crop or
    pad -- the official challenge metric percentile-normalizes both volumes itself and resamples
    the generated one onto this shape, so preprocessing the ground truth here would score the
    model against a different target than the leaderboard does.

    The two return values are in **different frames**, as they are out of `preprocess_volume`: the
    array is plane-first (SRA axial, RSA sagittal, ASR coronal) while the spacing stays (S, R, A).
    That is deliberate -- spacing exists only to go into the conditioning text, and
    `acquisition_prefix` wants (S, R, A), so permuting it to match the array would change the
    string the model was trained against. Do not index it with an array axis.
    """
    data, spacing = read_canonical(nii_bytes)
    return np.ascontiguousarray(data.transpose(plane_order(plane))), spacing


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
    data, spacing = read_canonical(nii_bytes)

    shape = [max(1, round(data.shape[i] * spacing[i] / target_spacing[i])) for i in range(3)]
    if shape != list(data.shape):
        data = F.interpolate(torch.from_numpy(data)[None, None], size=shape,
                             mode="trilinear", align_corners=False)[0, 0].numpy()

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
    """[1, T, S, S] in [0, 1] -> [2C, T, S/8, S/8], unscaled (mean and std, in that order).

    FLUX's VAE is 2D, so slices become the batch axis and the single MR channel is repeated to RGB
    here -- the only place that happens. There is no compression along the slice axis, so
    T_latent == T_image and the released CTFlow STDiT stays shape-compatible.

    Both posterior parameters are stored, so every read draws a fresh sample via `sample_latents`
    -- hence the 2C width and `sample_latents: true`. Stored unscaled and fp16, as CT does.
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


### Text ###

# Available encoders
TEXT_ENCODERS = {
    "cxr_bert": ("BiomedVLP-CXR-BERT-specialized", "bert_shim"),
    "medembed_large": ("MedEmbed-large-v0.1", "auto"),
    "bio_clinicalbert": ("Bio_ClinicalBERT", "auto"),
}

# Available conditioning configurations.
# A. "cxr_bert_cls" - report is encoded with CXR-BERT and only its CLS token is used -> [1, 768]
# B. "report2ct_style_meta" - 3 report sections are encoded separately, each is encoded with three different encoders
#    and the resulting vectors are concatenated (1024 MedEmbed-large + 768 Bio_ClinicalBERT + 768 CXR-BERT, in that order) -> [3, 2560]
CONDITIONINGS = {
    "cxr_bert_cls": {
        "encoders": ("cxr_bert",), "pooling": "cls", "sections": None, "tokens": 1, "dim": 768,
    },
    "report2ct_style_meta": {
        "encoders": ("medembed_large", "bio_clinicalbert", "cxr_bert"),
        "pooling": "mean", "sections": SECTIONS, "tokens": len(SECTIONS), "dim": 2560,
    },
}

DEFAULT_CONDITIONING = "cxr_bert_cls"


def build_conditioner(mri, device, dtype=torch.float32):
    """The one place a config becomes a live `TextConditioner`.

    `mri.conditioning` defaults to the released checkpoint's pooled CXR-BERT, and the encoder root
    falls back to the parent of the single-encoder `mri.text_checkpoint` -- which is the same
    directory the sectioned encoders are staged in. Both defaults exist so that a config written
    before sectioned conditioning (including the ones saved next to released checkpoints) keeps
    working untouched, and neither can be silently wrong: `check_conditioning` refuses to start a
    run whose denoiser caption shape disagrees with the choice.
    """
    return TextConditioner(
        mri.get("conditioning", DEFAULT_CONDITIONING),
        mri.get("text_root") or os.path.dirname(mri.text_checkpoint),
        device, dtype, max_length=mri.text_max_length,
    )


def check_conditioning(config):
    """Refuse a config whose denoiser is not shaped for the conditioning it selects.

    Static -- no encoder is loaded -- so `lvfm/train.py` can call it without a text encoder, which
    it never builds: it trains on the embeddings preprocessing already wrote. A config with no
    `mri` block is a CT run and has nothing to check.
    """
    if not config.get("mri"):
        return
    name = config.mri.get("conditioning", DEFAULT_CONDITIONING)
    if name not in CONDITIONINGS:
        raise ValueError(f"unknown conditioning {name!r}. Choose from: {sorted(CONDITIONINGS)}")
    spec, args = CONDITIONINGS[name], config.denoiser.args
    if (args.caption_channels, args.model_max_length) != (spec["dim"], spec["tokens"]):
        raise RuntimeError(
            f"conditioning {name!r} produces [{spec['tokens']}, {spec['dim']}] tokens, but the "
            f"denoiser is configured for caption_channels={args.caption_channels}, "
            f"model_max_length={args.model_max_length}."
        )


class TextConditioner:
    """Report -> conditioning tokens `[N, D]`, under one named configuration.

    Holds the frozen encoders, so preprocessing, evaluation and the submission container build the
    conditioning the same way and cannot drift in tokenizer, pooling or section order. Stored
    unnormalized; the dataset L2-normalizes the whole tensor at load, as CT does.
    """

    def __init__(self, name, root, device, dtype=torch.float32, max_length=512):
        if name not in CONDITIONINGS:
            raise ValueError(f"unknown conditioning {name!r}. Choose from: {sorted(CONDITIONINGS)}")
        spec = CONDITIONINGS[name]
        self.name = name
        self.sections = spec["sections"]
        self.pooling = spec["pooling"]
        self.tokens = spec["tokens"]
        self.dim = spec["dim"]
        self.max_length = max_length
        self.encoders = [self._load(n, root, device, dtype) for n in spec["encoders"]]

        width = sum(m.config.hidden_size for _, m in self.encoders)
        if width != self.dim:
            raise RuntimeError(
                f"conditioning {name!r} built {width} channels wide, but the table says {self.dim} "
                f"-- a staged snapshot's hidden size is not what this configuration was defined "
                f"against. Check {root}."
            )

    @staticmethod
    def _load(name, root, device, dtype):
        from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel,
                                  BertTokenizerFast)

        directory, loader = TEXT_ENCODERS[name]
        path = os.path.join(root, directory)
        if loader == "bert_shim":
            tokenizer = BertTokenizerFast.from_pretrained(path, local_files_only=True)
            config = BertConfig.from_pretrained(path, local_files_only=True)
            model = BertModel.from_pretrained(path, config=config, local_files_only=True,
                                              add_pooling_layer=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                      trust_remote_code=False)
            config = AutoConfig.from_pretrained(path, local_files_only=True,
                                                trust_remote_code=False)
            # These are MLM checkpoints with no trained pooler; letting transformers add one would
            # hand back a randomly initialized `pooler_output`.
            model = AutoModel.from_pretrained(path, config=config, local_files_only=True,
                                              trust_remote_code=False, add_pooling_layer=False)
        model.eval().to(device, dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        return tokenizer, model

    def _pool(self, tokenizer, model, texts):
        """`[len(texts), D]` for one encoder. Padding is excluded from the mean, never averaged in."""
        batch = tokenizer(texts, add_special_tokens=True, truncation=True,
                          max_length=self.max_length, padding=True, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            tokens = model(**batch).last_hidden_state
        if self.pooling == "cls":
            return tokens[:, 0].float().cpu()
        weights = batch["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        return ((tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)).float().cpu()

    def encode(self, report, modality, plane, spacing=None):
        """-> `[tokens, dim]`, one row per section (or the single pooled vector).

        The acquisition markers always reach the encoder: as the head of the joined string for a
        pooled configuration, as their own section for a sectioned one. Markers are faint at report
        length (~0.0002 cosine against ~0.03 for report content), which is what the sectioned
        configuration's third token exists to fix.

        No preprocessing or inference path passes `spacing`, so `[SPACING]` is omitted there (only
        `tools/ctflow_transfer_check.py` still does): it described the *native* geometry
        while the stored latents are all 1 mm isotropic, its (S, R, A) frame disagreed with the
        plane-permuted array, and the challenge cannot supply it at inference. It was also
        measurably inert -- an 8x wrong value moved the normalized embedding by 7e-5 cosine, against
        ~0.09 for a different patient's report. Pass a spacing here to put it back.
        """
        if self.sections is None:
            texts = [f"{acquisition_prefix(modality, plane, spacing)}\n{format_report(report)}"]
        else:
            texts = list(report_sections(report, modality, plane, spacing))
        # One pass per encoder over every section at once, then concatenate on the feature axis.
        # Each encoder pools its own tokenization: three tokenizers have no token-level
        # correspondence to align, which is what makes independent pooling the only way to fuse
        # them. Batching the sections rather than encoding them one at a time is equivalent up to
        # roundoff -- masked-mean pooling excludes the padding, so a section's vector depends on
        # its own tokens only, but the batch's padded width changes the matmul shapes and with them
        # the last bits (measured max 1.6e-6 absolute, 1.8e-7 relative, cosine 1.0 to 10 places).
        embedding = torch.cat([self._pool(t, m, texts) for t, m in self.encoders], dim=-1)
        assert embedding.shape == (self.tokens, self.dim), embedding.shape
        return embedding


### Artifact storage ###


class LatentStore:
    """Where preprocessing puts its .pt artifacts.

    `bundle=False` writes one file per artifact, as CTFlow does. `bundle=True` puts a whole shard
    into a single zip instead: a full MR-RATE split is ~244k loose files, past the inode quota of a
    typical cluster account, while one zip per shard is one file and still random-access, because a
    zip's central directory records every member's offset. Members are stored uncompressed -- fp16
    latents do not compress, so deflate would only cost CPU.

    The cost is resume granularity: loose artifacts are skipped individually on a re-run, while a
    bundle is written fresh, so a killed shard re-encodes all of its series.

    `bundle_dir` is the directory the zip goes in, which a re-embedding pass points elsewhere so its
    archives are not mistaken for latent bundles. Nothing reads it back: the manifest's `zip` column
    records the full relative path.
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

# What a re-embedding pass can honestly write: it decodes no volume, so it knows neither `n_slices`
# nor whether the volume was long enough to keep. The dataset joins on `sample_id` against the
# latent manifest, which stays the authority on which series exist.
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
