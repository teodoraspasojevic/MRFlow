"""Cubic-spline upsample a generated volume's two in-plane axes to a realistic clinical pixel
spacing -- the companion to `native_spacing.py`'s slice-axis reduction, one axis pair over.

Ported from `R2V-MR-Generation/submission/helpers/inplane_resample.py`, whose own writeup carries
the measurement: on that project's 64-case local set against real ground truth, the `spline` method
moved `FID_2p5D_Avg` 43.90 -> 35.37 with SSIM/PSNR/MSE unchanged. The mechanism is the same one
`native_spacing.py` exploits: `FID_2p5D` never resamples a shape mismatch away (only
`compute_basic_metrics` does), so the grid we write is part of what it measures. MRFlow's in-plane
grid is a fixed 256^2 at 1 mm -- `mri.preprocess.inplane_size` / `target_spacing`, a preprocessing
choice, never checked against real data -- while MR-RATE's own survey (`inplane_spacing_table.json`)
puts every bucket's median in-plane spacing at 0.45-0.8 mm, i.e. 1.25-2x *finer* than we generate.
Interpolation cannot create texture the model never produced, but it is a better estimate of the
in-between values than the evaluator's own `zoom(order=1)` shape-match, and it makes the resample
happen once, deliberately, rather than as a side effect of scoring.

**What's ported unchanged:** `InplaneSpacingTable` and the cubic-spline resample. The survey is a
property of MR-RATE's raw archives, not of either model, so R2V's `inplane_spacing_table.json` --
built from the very same raw archives this repo preprocesses -- is reused verbatim rather than
resurveyed, exactly as `native_spacing_table.json` is.

**What's rewritten:** the array-axis logic, the same rewrite `native_spacing.py` needed. R2V's model
samples in a fixed `(X, Y, Z) = (R, A, S)` order and looks the in-plane pair up per plane
(`INPLANE_AXES_XYZ`). MRFlow's `preprocess_volume`/`plane_order` always permutes so the *slice* axis
leads, whatever the plane, so here the in-plane axes are always `(1, 2)` and always 1 mm on a 256
grid going in, and there is no per-plane axis lookup to do.

**What's dropped:** R2V's `sitk_bspline` and `zero_fill` methods. The first is the same cubic
B-spline through SimpleITK's `ResampleImageFilter`, which existed there only to ablate against
scipy's -- they came out equivalent, so scipy's won for costing one fewer pinned dependency. The
second is a diagnostic control (shape-matched, gaps left at 0) that R2V's own predict script never
offers as a submission choice.

**Upsample only, FOV preserved exactly**, same invariant `native_spacing.py` holds for the slice
axis: this changes how finely the same physical area is sampled, not how much of it is covered, and
a drawn spacing coarser than the generated 1 mm is ignored. No crop candidate exists -- every
surveyed bucket's real in-plane spacing is finer than generated, never coarser.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .native_spacing import DEFAULT_MODE, DEFAULT_TOP_K, MIN_BUCKET_SERIES, MODES

TABLE_SCHEMA = "mrrate-inplane-spacing/1"


# --------------------------------------------------------------------------- the table

class InplaneSpacingTable:
    """The surveyed in-plane pixel-spacing distribution per (modality, plane) bucket.

    Built by R2V-MR-Generation's `build_inplane_spacing_table.py` from the MR-RATE **train** split,
    for the same reason its slice-thickness sibling is: the test ground truth is neither available
    nor permitted, and train/val/test are the same hospital and the same scanners. Same
    resolve/draw contract as `native_spacing.NativeSpacingTable` -- pooled fallback below
    `MIN_BUCKET_SERIES`, per-case draw deterministic in `case_id` alone -- reading `inplane_mm`
    entries instead of `thickness_mm` ones.
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
    def load(cls, path, top_k: int = DEFAULT_TOP_K,
             mode: str = DEFAULT_MODE) -> "InplaneSpacingTable":
        return cls(json.loads(Path(path).read_text()), top_k=top_k, mode=mode)

    def resolve(self, modality: str, plane: str) -> tuple[str, dict]:
        """`(bucket_key_used, bucket_info)`, most specific bucket first."""
        if plane not in self.PLANE_CODE:
            raise ValueError(f"unknown plane {plane!r}; known: {sorted(self.PLANE_CODE)}")
        chain = [f"{modality}|{self.PLANE_CODE[plane]}", f"{modality}|__pooled__",
                 "__pooled__|__pooled__"]
        for key in chain:
            info = self.buckets.get(key)
            if info and info["n_series"] >= MIN_BUCKET_SERIES and info["inplane_mm"]:
                return key, info
        raise KeyError(f"no usable bucket for ({modality!r}, {plane!r}); tried {chain}")

    def draw(self, modality: str, plane: str, case_id: str) -> dict:
        """`{inplane_mm, bucket, mode}` for one case.

        Deterministic in `case_id` alone, same reasoning as `NativeSpacingTable.draw` (a rank or a
        resumed run must redraw the same geometry for the same case). The `inplane:` prefix on the
        hashed key is what keeps this draw independent of that one, rather than perfectly rank-
        correlated with it for every case.
        """
        bucket, info = self.resolve(modality, plane)
        entries = info["inplane_mm"][: self.top_k]
        if self.mode == "mode":
            return {"inplane_mm": float(entries[0]["value"]), "bucket": bucket, "mode": self.mode}

        u = int.from_bytes(hashlib.sha256(f"inplane:{case_id}".encode("utf-8")).digest()[:8], "big")
        u /= float(1 << 64)
        total = sum(e["share"] for e in entries)
        cumulative = 0.0
        chosen = entries[-1]        # floating-point backstop if `u` lands past the last boundary
        for entry in entries:
            cumulative += entry["share"] / total
            if u < cumulative:
                chosen = entry
                break
        return {"inplane_mm": float(chosen["value"]), "bucket": bucket, "mode": self.mode}


# --------------------------------------------------------------------------- resampling

def _fit_shape(array: np.ndarray, target_shape) -> np.ndarray:
    if tuple(array.shape) == tuple(target_shape):
        return array
    slices = tuple(slice(0, min(s, t)) for s, t in zip(array.shape, target_shape))
    fitted = np.zeros(target_shape, dtype=array.dtype)
    fitted[slices] = array[slices]
    return fitted


def _spline_upsample(volume: np.ndarray, target_shape) -> np.ndarray:
    """Cubic-spline (`order=3`) resample onto `target_shape`. The slice axis's factor is exactly
    1.0 by construction, so `zoom` leaves it -- and every value along it -- untouched."""
    from scipy.ndimage import zoom

    factors = [target_shape[axis] / volume.shape[axis] for axis in range(volume.ndim)]
    out = zoom(volume.astype(np.float64, copy=False), factors, order=3, mode="nearest")
    # A cubic spline rings, so it overshoots the source range at sharp edges -- here the brain/air
    # boundary, by a fraction of a level. R2V got this clipped for free by casting back to its
    # int16 pixel dtype; `decode_latent` hands us uint8 levels in a float array, so the written
    # NIfTI would otherwise carry values outside the [0, 255] range everything else here holds.
    np.clip(out, volume.min(), volume.max(), out=out)
    # `zoom`'s rounding can land one voxel off the planned shape; the FOV/spacing plan is the
    # source of truth, so crop or edge-pad the rare off-by-one back to it rather than propagate it.
    return _fit_shape(out, target_shape).astype(volume.dtype, copy=False)


def plan_inplane(n_in: int, target_mm: float) -> tuple[int, float]:
    """`(n_out, spacing_mm)` for one in-plane axis. The generated grid is always 1 mm in-plane.

    The FOV is preserved exactly and `target_mm` is a target, not a guarantee: the pixel count must
    be a whole number, so `n = round(fov / target_mm)` and the realised spacing is `fov / n`. **A
    drawn spacing coarser than (or equal to) the generated grid is ignored** -- this only ever adds
    resolution, the mirror of `plan_slices`'s refusal to invent slices the model never produced.
    """
    fov_mm = float(n_in) * 1.0
    n = max(1, int(round(fov_mm / target_mm)))
    if n <= n_in:
        return n_in, 1.0
    return n, fov_mm / n


def to_inplane_grid(volume: np.ndarray,
                    target_mm: float) -> tuple[np.ndarray, tuple[float, float], dict]:
    """`(resampled_volume, inplane_spacing_mm, info)`. `volume` is `(T, H, W)`, slice axis first --
    the axis order `LatentAutoregressiveGenerator.decode_latent` returns, for every plane.

    One resample, on axes 1 and 2 only, and only ever an upsample. The slice axis -- and the slab
    thickness `to_native_grid` may already have given it -- comes back untouched.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D (T, H, W) volume, got shape {volume.shape}")
    height, height_spacing = plan_inplane(volume.shape[1], target_mm)
    width, width_spacing = plan_inplane(volume.shape[2], target_mm)
    target_shape = (int(volume.shape[0]), height, width)

    out = volume if target_shape == volume.shape else _spline_upsample(volume, target_shape)
    info = {"source_inplane": (int(volume.shape[1]), int(volume.shape[2])),
            "target_inplane": (height, width),
            "inplane_spacing_mm": (round(height_spacing, 4), round(width_spacing, 4))}
    return out, (height_spacing, width_spacing), info
