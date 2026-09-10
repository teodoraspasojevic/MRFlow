"""Paper metrics: FID over the slice axis, FVD over I3D clips, Inception Score.

None of these is scored by any VLM3D container -- `challenge_metrics.py` holds that, and nothing
here may reach back into it. These exist because the challenge's metric set is a poor description
of a generative model: `FID_2p5D` uses a squeezenet1_1 512-d feature nobody publishes against, and
MSE/PSNR/SSIM are voxel-level comparisons against one particular patient's scan, which a
report-to-volume model has no reason to reproduce.

**The geometry contract.** Both volumes reach every metric here in the geometry the model was
*trained* in -- 1 mm isotropic, 256^2 in-plane -- because that is the distribution the model was
asked to match:

    ground truth   `canonicalize_gt`: in-plane resampled to 1 mm and cropped/padded to 256^2 with
                   preprocessing's own posterior shift, slice axis resampled to 1 mm at its native
                   extent (143-200 slices over MR-RATE). NOT stretched to a fixed slice count.
    generated      already exactly this by construction, so `canonicalize_generated` only checks.
    both           `_normalize01` after the geometry, so the two are normalized on identical grids.

Two consequences worth knowing before reading a number:

  interpolated z   58% of MR-RATE is acquired above 1.5 mm slice spacing (median 4.05 mm), and
                   `preprocess_volume` trilinearly upsampled those to 1 mm for training. So for
                   thick-slice series the 1 mm reference is interpolated, and FVD there measures
                   agreement with interpolated data rather than with a real 1 mm acquisition.
                   MR-RATE contains no real 1 mm T2w at all to check against.
  no fixed length  neither side is padded or stretched to a common slice count, so a rollout that
                   stops early contributes fewer clips over different anatomy rather than being
                   silently rescaled to fit. `f16`/`f64` clips span 16 mm and 64 mm on both sides
                   by construction.

`FID` here is torchvision's Inception-v3 pool3 (2048-d), not the TF-ported `pt_inception-2015-12-05`
that `pytorch-fid` uses, so it is comparable across our own runs and to other torchvision-based
numbers, but not digit-for-digit to a paper quoting the TF port.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from scipy import stats

from echosyn.common.mrrate import AP_AXIS, _crop_pad, plane_order
from evaluation.challenge_metrics import _matrix_sqrt, _normalize01

### Geometry: the model's own training grid ###

TARGET_MM = 1.0
INPLANE = 256
POSTERIOR_SHIFT_MM = 15.0  # preprocess_volume's value; MR-RATE is defaced


def slice_spacing(spacing, plane):
    """The ground truth's slice-axis voxel size in mm.

    `spacing` is native (S, R, A) while a `load_native_volume` array is plane-permuted, so its
    array axis 0 is (S, R, A) axis `plane_order(plane)[0]`. Indexing `spacing` with an array axis
    is the standing trap in this codebase; `save_case` does the same realignment.
    """
    return spacing[plane_order(plane)[0]]


def _resample_inplane(vol, new_hw):
    """Resample the two in-plane axes of a `(T, H, W)` volume, anti-aliased.

    Slices become the channel axis so this is a 2D bilinear resize, which is the only mode
    `antialias=` supports -- and it is needed, because a native GT shrinks by up to 2.7x here while
    a generated volume never shrinks at all, and un-filtered decimation would fold that difference
    into the real features as false structure.
    """
    if tuple(vol.shape[1:]) == tuple(new_hw):
        return vol
    t = torch.from_numpy(vol)[None]
    t = F.interpolate(t, size=tuple(new_hw), mode="bilinear", align_corners=False, antialias=True)
    return t[0].numpy()


def _resample_z(vol, new_t):
    """Resample the slice axis of a `(T, H, W)` volume to `new_t`.

    Area-average when decimating, linear when interpolating. `antialias=` does not exist for
    trilinear, and averaging is anyway the right operation rather than a filter: a thick MR slice
    *is* approximately the integral of the tissue over its thickness, so collapsing n fine slices
    into one is how that slice would have been acquired.
    """
    if new_t == vol.shape[0]:
        return vol
    t = torch.from_numpy(vol)[None, None]
    if new_t < vol.shape[0]:
        t = F.adaptive_avg_pool3d(t, (new_t,) + tuple(vol.shape[1:]))
    else:
        t = F.interpolate(t, size=(new_t,) + tuple(vol.shape[1:]), mode="trilinear",
                          align_corners=False)
    return t[0, 0].numpy()


def canonicalize_gt(real, spacing, plane):
    """A `load_native_volume` ground truth in the model's training geometry, `[0, 1]`.

    `spacing` is the native (S, R, A) voxel size while `real` is already plane-permuted, so it is
    indexed through `plane_order` -- the same realignment `save_case` makes. Indexing it with an
    array axis is the standing trap in this codebase.

    The slice axis goes to 1 mm at the volume's *own* extent. Stretching it to a fixed count would
    spread ~158 mm of anatomy over whatever that count implies, and would also hide a rollout that
    generated the wrong amount of anatomy.
    """
    axis_mm = [spacing[i] for i in plane_order(plane)]
    new_shape = [max(1, round(n * mm / TARGET_MM)) for n, mm in zip(real.shape, axis_mm)]

    vol = _resample_inplane(real, new_shape[1:])
    order = plane_order(plane)
    shift_axis = None if order[0] == AP_AXIS else order[1:].index(AP_AXIS)
    vol = _crop_pad(vol, INPLANE, round(POSTERIOR_SHIFT_MM / TARGET_MM), shift_axis)
    vol = _resample_z(vol, new_shape[0])
    return _normalize01(vol)


def canonicalize_generated(produced):
    """The generated volume, `[0, 1]`. Already 1 mm isotropic and 256^2 by construction, so this
    only normalizes -- the crop/pad is a no-op guard for a config with a different `inplane_size`."""
    if produced.shape[1:] != (INPLANE, INPLANE):
        produced = _crop_pad(produced, INPLANE)
    return _normalize01(produced)


### Strata ###

# Split on the ground truth's *native* slice spacing, not on modality. MR-RATE mixes 2D
# thick-slice and 3D thin-slice acquisitions inside every (modality, plane) bucket -- measured, the
# modal spacing covers only 33-71% of a bucket and FLAIR/SAGITTAL spans 0.5-6.5 mm -- so contrast
# is not a proxy for geometry. The split is a diagnostic rather than a correction: both volumes are
# already on the same 1 mm grid either way. What differs is whether the reference's z detail was
# acquired or interpolated, and the model is told neither (`[SPACING]` is not in the conditioning
# and there is no spacing class embedding), so it cannot know which to produce. A model that hedges
# between the two shows up as too smooth on `thin` and too detailed on `thick`.
THIN_MAX_MM = 1.5
STRATA = ("thin", "thick")


def stratum_for(slice_mm):
    return "thin" if slice_mm <= THIN_MAX_MM else "thick"


def _empty_sides():
    return {s: {"real": [], "fake": []} for s in STRATA}


def _stack(sides_per_stratum, dim):
    empty = np.zeros((0, dim), np.float32)
    return {stratum: {side: np.concatenate(chunks) if chunks else empty
                      for side, chunks in sides.items()}
            for stratum, sides in sides_per_stratum.items()}


### Feature-extractor input ###


def _to_network_input(slices, size):
    """`(N, H, W)` in `[0, 1]` -> `(N, 3, size, size)` in `[-1, 1]`.

    One anti-aliased resize and one rescale, no per-slice renormalization: the volume was already
    normalized as a whole by `canonicalize_*`, which is what a slice of a real acquisition looks
    like. (The container's per-slice window is a different metric and stays in its own file.)
    """
    t = torch.from_numpy(np.ascontiguousarray(slices))[None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return (t[0] * 2.0 - 1.0).unsqueeze(1).repeat(1, 3, 1, 1)


def _frechet(feats_a, feats_b):
    """Frechet distance between two feature sets, with the always-on ridge the FVD reference uses.

    `challenge_metrics.frechet_distance` regularizes only after `sqrtm` has already returned
    something non-finite; this adds `eps * I` scaled by the traces up front, which is what
    `ct_challenges/ctgen_evaluation/FVD/fvd_pytorch_model.compute_fvd` does and what keeps a
    rank-deficient covariance from silently producing NaN.
    """
    mu_a, sigma_a = feats_a.mean(axis=0), np.cov(feats_a, rowvar=False)
    mu_b, sigma_b = feats_b.mean(axis=0), np.cov(feats_b, rowvar=False)

    eps = 1e-6 * np.trace(sigma_a + sigma_b) / sigma_a.shape[0]
    identity = np.eye(sigma_a.shape[0], dtype=sigma_a.dtype)
    sigma_a, sigma_b = sigma_a + eps * identity, sigma_b + eps * identity

    m = np.square(mu_a - mu_b).sum()
    s = _matrix_sqrt(np.dot(sigma_a, sigma_b))
    return float(np.real(m + np.trace(sigma_a + sigma_b - s * 2)))


### FID over the slice axis (Inception-v3 pool3, 2048-d) ###

_FID_DIM = 2048
_FID_INPUT = 299


class _InceptionPool3(nn.Module):
    """Inception-v3 truncated to its 2048-d pooled feature -- the layer FID is defined over.

    `transform_input=False` with a `[-1, 1]` input is the reference-port convention, and dropping
    `fc` is what turns the classifier into a feature extractor. torchvision forces `aux_logits`
    on when weights are given, but `eager_outputs` returns the single tensor in eval mode.
    """

    def __init__(self, device):
        super().__init__()
        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        self.net = tv_models.inception_v3(weights=weights, transform_input=False)
        self.net.fc = nn.Identity()
        self.eval()
        self.to(device)

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)


class SliceFIDAccumulator:
    """FID over slices of the acquisition plane only -- array axis 0, the axis the model rolls out
    along, so the images scored are the images generated.

    The official `FID_2p5D` also slices axes 1 and 2, whose reformats contain the slice axis and so
    get squashed to 224^2 by a different factor on each side whenever the two volumes differ in
    length. Those two planes measure that distortion as much as they measure anatomy; this one
    cannot, because an axis-0 slice is `(H, W)`.
    """

    def __init__(self, device="auto", stride=4, batch_size=32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        self.model = _InceptionPool3(device)
        self._feats = _empty_sides()

    @torch.no_grad()
    def _features(self, vol):
        kept, out = vol[:: self.stride], []
        for chunk in np.split(kept, range(self.batch_size, len(kept), self.batch_size)):
            x = _to_network_input(chunk, _FID_INPUT).to(self.device)
            out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out)

    def add_pair(self, real, fake, stratum):
        self._feats[stratum]["real"].append(self._features(real))
        self._feats[stratum]["fake"].append(self._features(fake))

    def raw_features(self):
        return _stack(self._feats, _FID_DIM)


### FVD over I3D clips ###

_FVD_DIM = 400
_I3D_INPUT = 224
_I3D_URL = "https://huggingface.co/flateon/FVD-I3D-torchscript/resolve/main/i3d_torchscript.pt"

# (clip length, stride). 16 is one generated block, so `f16` reads within-block continuity; 64 is
# four, so `f64` reads whether the rollout stays coherent across block boundaries -- the drift the
# CT paper's FVD_f16/FVD_f128 pair was built to expose. 128 does not fit: MR-RATE volumes are
# 143-200 slices at 1 mm, so a 128-slice clip would leave ~1 sample per case against 400 feature
# dimensions. The strides overlap by half so the covariance has enough samples to be an estimate.
CLIP_CONFIGS = {"f16": (16, 8), "f64": (64, 32)}


def load_i3d(device):
    """The torchscript I3D every PyTorch FVD is computed with, cached in the torch hub directory
    beside torchvision's own weights (so `TORCH_HOME` moves it). `MRFLOW_I3D_PATH` overrides the
    location for a node with no outbound route."""
    path = os.environ.get("MRFLOW_I3D_PATH") or os.path.join(
        torch.hub.get_dir(), "checkpoints", "i3d_torchscript.pt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.hub.download_url_to_file(_I3D_URL, path)
    return torch.jit.load(path).eval().to(device)


def _iter_clips(vol, clip_len, stride):
    """Windows of `clip_len` consecutive slices, every `stride` slices. A volume shorter than one
    clip yields nothing -- at 1 mm the shortest MR-RATE extent is 143 mm, so this only guards the
    degenerate rollouts that `run_shard` already counts as missing."""
    for start in range(0, vol.shape[0] - clip_len + 1, stride):
        yield vol[start:start + clip_len]


class ClipFVDAccumulator:
    """One I3D feature per clip, for each entry in `CLIP_CONFIGS`.

    Volume pairs arrive ONE AT A TIME and are released as soon as `add_pair` returns; only the
    400-d vectors accumulate. Real and generated are clipped identically, so a case contributes the
    same number of real and fake clips unless the rollout came out a different length -- which is
    exactly the error the fixed-length resize used to hide.
    """

    def __init__(self, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.i3d = load_i3d(device)
        self._feats = {name: _empty_sides() for name in CLIP_CONFIGS}

    @torch.no_grad()
    def _clip_features(self, vol, clip_len, stride):
        out = []
        for clip in _iter_clips(vol, clip_len, stride):
            x = _to_network_input(clip, _I3D_INPUT)          # (L, 3, 224, 224)
            x = x.permute(1, 0, 2, 3)[None].to(self.device)  # (1, 3, L, 224, 224)
            # StyleGAN-V's detector kwargs: the resize and the [-1, 1] rescale are already done,
            # and `return_features` takes the 400-d logits layer the TF reference reads as `Mean:0`.
            out.append(self.i3d(x, rescale=False, resize=False,
                                return_features=True).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, _FVD_DIM), np.float32)

    def add_pair(self, real, fake, stratum):
        for name, (clip_len, stride) in CLIP_CONFIGS.items():
            sides = self._feats[name][stratum]
            sides["real"].append(self._clip_features(real, clip_len, stride))
            sides["fake"].append(self._clip_features(fake, clip_len, stride))

    def raw_features(self):
        return {name: _stack(sides, _FVD_DIM) for name, sides in self._feats.items()}


### Inception Score ###

_IS_DIM = 1000
_IS_INPUT = 299
_IS_SPLITS = 10


class InceptionScoreAccumulator:
    """Inception-v3 class posteriors for the GENERATED volume's acquisition-plane slices.

    No-reference, so there is no `add_pair` and the ground truth never enters. ImageNet's 1,000
    classes do not describe an MR slice, so the posteriors are diffuse (measured 3.7 of a possible
    6.9 nats) and the score sits near 1.6-1.8 -- read it against another MR run, never against a
    natural-image number.
    """

    def __init__(self, device="auto", stride=4, batch_size=32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = tv_models.inception_v3(weights=weights,
                                            transform_input=False).eval().to(device)
        self._probs = []

    @torch.no_grad()
    def add(self, fake):
        kept = fake[:: self.stride]
        for chunk in np.split(kept, range(self.batch_size, len(kept), self.batch_size)):
            x = _to_network_input(chunk, _IS_INPUT).to(self.device)
            self._probs.append(torch.softmax(self.model(x), dim=1).float().cpu().numpy())

    def probs(self):
        return np.concatenate(self._probs) if self._probs else np.zeros((0, _IS_DIM), np.float32)


def inception_score(probs, splits=_IS_SPLITS):
    """`(mean, std)` of `exp(E_x KL(p(y|x) || p(y)))` over `splits` equal chunks.

    Salimans et al. 2016 as both reference ports compute it -- contiguous chunks, remainder rows
    dropped by the same integer division, each chunk forming its own marginal `p(y)`.

    The std is the spread between chunks, not an error bar on the mean: each chunk is a
    self-contained IS over a smaller sample, so it is biased low and the spread grows with the
    split count. It is also **not invariant to the shard layout** -- chunks are contiguous in the
    order shards were pooled, so `IS_std` is comparable only across runs with the same shard count.
    Measured: 0.069 at one shard against 0.029 at 32, while `IS_mean` moved 0.25%.
    """
    n = probs.shape[0]
    per_split = n // splits
    if per_split < 1:
        return float("nan"), float("nan")

    scores = []
    for k in range(splits):
        part = probs[k * per_split:(k + 1) * per_split].astype(np.float64)
        py = part.mean(axis=0)
        scores.append(np.exp(np.mean([stats.entropy(pyx, py) for pyx in part])))
    return float(np.mean(scores)), float(np.std(scores))


### Cross-shard pooling ###


def fid_pooled(raw_per_shard, strata=STRATA):
    """`(distance, n_real_rows, n_fake_rows)` over several shards' `SliceFIDAccumulator`
    features, merging the named strata. Concatenating before mean/covariance is associative, so
    this is what one process that had seen every pair itself would compute."""
    return _pooled(raw_per_shard, _FID_DIM, strata)


def fvd_pooled(raw_per_shard, name, strata=STRATA):
    """`(distance, n_real_clips, n_fake_clips)` for one `CLIP_CONFIGS` entry."""
    return _pooled([s[name] for s in raw_per_shard], _FVD_DIM, strata)


def _pooled(per_shard, dim, strata):
    empty = np.zeros((0, dim), np.float32)
    sides = {side: np.concatenate([s[stratum][side] for s in per_shard for stratum in strata]
                                  or [empty]) for side in ("real", "fake")}
    counts = (int(sides["real"].shape[0]), int(sides["fake"].shape[0]))
    # Both covariances must be estimable at all; whether they are estimable *well* is a matter of
    # rows against `dim`, which is why the counts are reported alongside every distance.
    if min(counts) < 2:
        return (float("nan"),) + counts
    return (_frechet(sides["fake"], sides["real"]),) + counts
