"""An external model's volume on the grid `evaluation/main.py` caches MRFlow's own rollouts in.

The one conversion every baseline needs and the one place it happens. A baseline generates on
whatever grid it was trained on -- GenerateCT on 201x512x512, NV-Generate-MR-Brain on a per-bucket
FOV at roughly 1 mm, Text2CT on its VAE's own latent-derived shape -- while a cached MRFlow volume
is 1 mm isotropic, plane-first and 256^2 in-plane. `canonicalize_generated` assumes that shape and
only windows intensity, so a baseline that skipped this would be scored on a different geometry
than the model it is a baseline for, which is the one thing a baseline table may not do.
"""

import numpy as np
import torch
import torch.nn.functional as F

from echosyn.common.mrrate import AP_AXIS, _crop_pad, plane_order
from evaluation.paper_metrics import INPLANE, TARGET_MM


def canonicalize_external(volume, spacing, plane, posterior_shift_mm=0.0):
    """A baseline's volume in MRFlow's output geometry. Geometry only -- intensity is left alone.

    `volume` is RAS-canonical in (S, R, A) axis order and `spacing` its voxel size in mm on those
    same three axes -- i.e. exactly what `read_canonical` returns, before `plane_order` permutes
    it. `ingest.py` gets a baseline's NIfTI into that form by reading it with `read_canonical`
    itself, so the orientation logic is shared with the ground-truth path rather than restated
    here, and the two cannot drift.

    Mirrors `preprocess_volume` step for step, minus the intensity normalization: trilinear
    resample to 1 mm isotropic, permute so the acquisition plane's slice axis leads, center
    crop/pad the two in-plane axes to 256. T is never capped -- `cache_generated` does not cap it
    either, and a rollout that generated the wrong amount of anatomy should stay visible.

    **Intensity is deliberately untouched.** `canonicalize_generated` windows every cached volume
    with `normalize01` at scoring time, so a baseline emitting Hounsfield units and one emitting
    [0, 1] score identically; rescaling here would be a second, invisible normalization applied to
    some models and not others.

    **`posterior_shift_mm` defaults to 0, and that is not an oversight.** The 15 mm shift belongs
    to `preprocess_volume`, where it decides how a *native* volume is framed before the model ever
    sees it -- MR-RATE is defaced, so a plain center crop keeps removed face and loses posterior
    brain. A generated volume already carries whatever framing its model learned, and
    `canonicalize_generated` correspondingly shifts an MRFlow rollout by nothing. Set this only
    for a baseline whose output is provably framed like a native MR-RATE volume rather than like
    its own training distribution, and say so in that baseline's README.
    """
    shape = [max(1, round(n * mm / TARGET_MM)) for n, mm in zip(volume.shape, spacing)]
    if shape != list(volume.shape):
        tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
        volume = F.interpolate(tensor, size=shape, mode="trilinear",
                               align_corners=False)[0, 0].numpy()

    order = plane_order(plane)
    volume = np.ascontiguousarray(volume.transpose(order))
    # A-P is in-plane unless it is itself the stacking axis (coronal), where the shift is moot.
    shift_axis = None if order[0] == AP_AXIS else order[1:].index(AP_AXIS)
    return _crop_pad(volume, INPLANE, round(posterior_shift_mm / TARGET_MM), shift_axis)
