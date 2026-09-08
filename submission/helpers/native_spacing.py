"""Slab-average a generated volume down to a realistic clinical slice count before saving it.

Ported from `R2V-MR-Generation/submission/helpers/native_spacing.py` (that project's own writeup
of *why* this matters -- a large, measured FID win -- lives there; the short version: the
official `FID_2p5D` metric never resamples, so a generated volume's own slice grid is part of
what it measures, and MRFlow, like R2V's NVIDIA baseline, always generates near-isotropic ~1 mm
slices while real MR-RATE 2D acquisitions are commonly 4-6.5 mm. Matching the *count* -- not the
content -- of slices closes that gap.

**What's ported unchanged:** `NativeSpacingTable` and `_area_average`. The survey itself
(`native_spacing_table.json`) is a property of MR-RATE's raw archives, not of either model, so R2V's
table -- built from the very same `raw_root` this repo's `mrflow_STDiT-L2_16f8.yaml` points at -- is
reused verbatim rather than resurveyed.

**What's rewritten:** the array-axis logic. R2V's model always samples in a fixed
`(X, Y, Z) = (R, A, S)` order and picks the slice axis per plane (`SLICE_AXIS_XYZ`). MRFlow's
`preprocess_volume`/`plane_order` (`echosyn/common/mrrate.py`) instead always permutes so the
*slice* axis leads (axis 0), whatever the plane -- SRA for axial, RSA for sagittal, ASR for
coronal -- which is also the axis order `LatentAutoregressiveGenerator.generate` rolls out and
`decode_latent` returns. So here the slice axis is always 0, spacing is always 1 mm going in (the
model's own training grid), and there is no per-plane axis lookup to do.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: Below this many surveyed series a bucket's shares are noise; `resolve` falls through to the
#: pooled fallbacks. See build_native_spacing_table.py (R2V-MR-Generation) for the real thin cases.
MIN_BUCKET_SERIES = 200

#: How many thickness entries to sample from, renormalised. 8 (every stored entry), not 3 -- see
#: R2V-MR-Generation/submission/helpers/native_spacing.py's own note: truncation drops thick-slice
#: geometries, the exact ones this exists to reproduce.
DEFAULT_TOP_K = 8

MODES = ("sample", "mode", "off")
DEFAULT_MODE = "sample"

TABLE_SCHEMA = "mrrate-native-spacing/2"


# --------------------------------------------------------------------------- the table

class NativeSpacingTable:
    """The surveyed slice-thickness distribution per (modality, plane) bucket.

    Built by R2V-MR-Generation's `build_native_spacing_table.py` from the MR-RATE **train** split's
    own shard indexes -- the test ground truth is neither available nor permitted, and train/val/
    test are the same hospital and the same scanners. Unchanged from that project: modality and
    plane strings here are MRFlow's own canonical names (`T1w`/`T2w`/`FLAIR`/`SWI`,
    `AXIAL`/`SAGITTAL`/`CORONAL`), which is exactly what the table's keys already are.
    """

    PLANE_CODE = {"AXIAL": "axi", "SAGITTAL": "sag", "CORONAL": "cor"}

    def __init__(self, payload: dict, top_k: int = DEFAULT_TOP_K, mode: str = DEFAULT_MODE):
        if payload.get("schema") != TABLE_SCHEMA:
            raise ValueError(f"table schema {payload.get('schema')!r} != {TABLE_SCHEMA!r}")
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} must be one of {MODES}")
        self.payload = payload
        self.buckets = payload["buckets"]
        self.top_k = max(1, int(top_k))
        self.mode = mode

    @classmethod
    def load(cls, path, top_k: int = DEFAULT_TOP_K, mode: str = DEFAULT_MODE) -> "NativeSpacingTable":
        return cls(json.loads(Path(path).read_text()), top_k=top_k, mode=mode)

    def resolve(self, modality: str, plane: str) -> tuple[str, dict]:
        """`(bucket_key_used, bucket_info)`, most specific bucket first."""
        if plane not in self.PLANE_CODE:
            raise ValueError(f"unknown plane {plane!r}; known: {sorted(self.PLANE_CODE)}")
        chain = [f"{modality}|{self.PLANE_CODE[plane]}", f"{modality}|__pooled__",
                 "__pooled__|__pooled__"]
        for key in chain:
            info = self.buckets.get(key)
            if info and info["n_series"] >= MIN_BUCKET_SERIES and info["thickness_mm"]:
                return key, info
        raise KeyError(f"no usable bucket for ({modality!r}, {plane!r}); tried {chain}")

    def draw(self, modality: str, plane: str, case_id: str) -> dict:
        """`{thickness_mm, bucket, mode}` for one case.

        Deterministic in `case_id` alone -- not a call counter, a DDP rank, or `hash()` (which
        Python salts per process) -- so a rank or a resumed run redraws the same thickness for the
        same case, and a resumed nifti agrees with `done.json` about what it should look like.
        """
        bucket, info = self.resolve(modality, plane)
        entries = info["thickness_mm"][: self.top_k]
        if self.mode == "mode":
            return {"thickness_mm": float(entries[0]["value"]), "bucket": bucket, "mode": self.mode}

        u = int.from_bytes(hashlib.sha256(case_id.encode("utf-8")).digest()[:8], "big")
        u /= float(1 << 64)
        total = sum(e["share"] for e in entries)
        cumulative = 0.0
        chosen = entries[-1]        # floating-point backstop if `u` lands past the last boundary
        for entry in entries:
            cumulative += entry["share"] / total
            if u < cumulative:
                chosen = entry
                break
        return {"thickness_mm": float(chosen["value"]), "bucket": bucket, "mode": self.mode}


# --------------------------------------------------------------------------- resampling

def _area_average(volume: np.ndarray, axis: int, n_out: int) -> np.ndarray:
    """Exact box-filter average along one axis, for arbitrary non-integer ratios.

    Ported unchanged from R2V-MR-Generation: output sample k is the mean of the input over
    `[k*n_in/n_out, (k+1)*n_in/n_out)`, computed by linearly interpolating the cumulative sum at the
    slab edges -- exact for the piecewise-constant signal a voxel grid is. Naive block-averaging
    cannot do this: most `n_in -> n_out` ratios here are non-integer.
    """
    n_in = volume.shape[axis]
    if n_out == n_in:
        return volume
    moved = np.moveaxis(volume, axis, 0).astype(np.float64, copy=False)
    cumulative = np.concatenate(
        [np.zeros((1,) + moved.shape[1:], dtype=np.float64), np.cumsum(moved, axis=0)], axis=0)
    edges = np.linspace(0.0, float(n_in), n_out + 1)
    lo = np.clip(np.floor(edges).astype(np.int64), 0, n_in - 1)
    frac = (edges - lo).reshape((-1,) + (1,) * (moved.ndim - 1))
    integral = cumulative[lo] * (1.0 - frac) + cumulative[lo + 1] * frac
    widths = np.diff(edges).reshape((-1,) + (1,) * (moved.ndim - 1))
    out = (integral[1:] - integral[:-1]) / widths
    return np.moveaxis(out, 0, axis).astype(volume.dtype, copy=False)


def plan_slices(n_in: int, thickness_mm: float) -> tuple[int, float]:
    """`(n_out, slice_spacing_mm)`. The generated grid is always 1 mm through-plane going in.

    The FOV is preserved exactly and `thickness_mm` is a target, not a guarantee: the slice count
    must be a whole number, so `n = round(fov / thickness)` and the realised thickness is
    `fov / n`. **A drawn thickness finer than the generated grid is ignored** -- upsampling the
    slice axis cannot create detail the model never produced, so a draw asking for more slices than
    generated returns the grid unchanged (n_out = n_in, spacing = 1.0).
    """
    fov_mm = float(n_in) * 1.0
    n = max(1, int(round(fov_mm / thickness_mm)))
    if n >= n_in:
        return n_in, 1.0
    return n, fov_mm / n


def to_native_grid(volume: np.ndarray, thickness_mm: float) -> tuple[np.ndarray, float, dict]:
    """`(resampled_volume, slice_spacing_mm, info)`. `volume` is `(T, H, W)`, slice axis first --
    the axis order `LatentAutoregressiveGenerator.decode_latent` returns, for every plane.

    One resample, on axis 0 only, and only ever a reduction. The two in-plane axes -- and every
    voxel value within a slice -- come back untouched, for the evaluator to resample.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D (T, H, W) volume, got shape {volume.shape}")
    n_out, spacing = plan_slices(volume.shape[0], thickness_mm)
    out = _area_average(volume, axis=0, n_out=n_out)
    info = {"source_slices": int(volume.shape[0]), "target_slices": int(n_out),
            "slice_spacing_mm": round(float(spacing), 4)}
    return out, spacing, info
