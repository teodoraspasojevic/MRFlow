"""FID, FVD and Inception Score, each by its field's reference protocol.

Every number here is meant to be quotable against the literature, so where a reference
implementation exists it is used rather than reimplemented, and where we deviate it is stated.
The three references, read from source rather than from memory:

  FID   github.com/mseitzer/pytorch-fid, tag v0.3.0. `evaluation/fid_inception.py` is that code
        verbatim. Backbone `pt_inception-2015-12-05` (the TF port, NOT torchvision's ImageNet
        Inception), 2048-d final pool; images as `[0, 1]` floats that the module itself resizes to
        299^2 (bilinear, `align_corners=False`, no antialias) and rescales to `[-1, 1]`; every
        image in the set is used; `eps*I` is added only when `sqrtm` returns non-finite.

  FVD   github.com/universome/stylegan-v, `src/metrics/frechet_video_distance.py`. Backbone the
        Kinetics-400 I3D torchscript (`i3d_torchscript.pt`), read at its 400-d pre-softmax layer
        with `rescale=True, resize=True, return_features=True` -- so the detector does its own
        `x/255*2-1` and its own bilinear resize to 224^2, and the caller hands it `[0, 255]`. Its
        dataset yields `load_n_consecutive=num_frames` with `discard_short_videos=True`: **one
        clip per video, no overlap**. Its Frechet distance adds no ridge at all.

  IS    github.com/toshas/torch-fidelity, `torch_fidelity/metric_isc.py`. Logits (not
        probabilities) shuffled with `np.random.RandomState(2020).permutation(N)`, cut into 10
        contiguous splits at `i*N//splits`, KL evaluated in float64.

Ours, and to be declared wherever these are quoted:

  geometry     both volumes are canonicalized into the 1 mm isotropic / 256^2 grid the model was
               *trained* in (below), rather than the reference being left in its released geometry
               with the generation resampled onto it. That is the distribution the model was asked
               to match. It also means these numbers are not comparable to a paper that scored
               native-geometry references.
  slice axis   FID and IS see slices of array axis 0 only -- the acquisition plane, the axis the
               model rolls out along, so the images scored are the images generated.
  intensity    `normalize01`, a 0.5/99.5-percentile window per volume, then a uint8 quantization,
               because both references consume uint8 images.
  `FVD_f64`    an extension: 64-frame clips read whether the rollout stays coherent across block
               boundaries (a block is 16 frames). Only `FVD_f16` is the standard protocol.
  strata       `thin`/`thick`, split on the ground truth's native slice spacing.

The geometry contract, in detail:

    ground truth   `canonicalize_gt`: in-plane resampled to 1 mm and cropped/padded to 256^2 with
                   preprocessing's own posterior shift, slice axis resampled to 1 mm at its native
                   extent (143-200 slices over MR-RATE). NOT stretched to a fixed slice count.
    generated      already exactly this by construction, so `canonicalize_generated` only checks.
    both           `normalize01` after the geometry, so the two are normalized on identical grids.

Two consequences worth knowing before reading a number:

  interpolated z   58% of MR-RATE is acquired above 1.5 mm slice spacing (median 4.05 mm), and
                   `preprocess_volume` trilinearly upsampled those to 1 mm for training. So for
                   thick-slice series the 1 mm reference is interpolated, and FVD there measures
                   agreement with interpolated data rather than with a real 1 mm acquisition.
                   MR-RATE contains no real 1 mm T2w at all to check against.
  no fixed length  neither side is padded or stretched to a common slice count, so a rollout that
                   stops early contributes fewer slices, and a rollout shorter than a clip drops
                   out of that FVD entirely -- which is why the real and generated sample counts
                   are both reported for every distance.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from echosyn.common.mrrate import AP_AXIS, _crop_pad, plane_order
from evaluation.fid_inception import (InceptionV3, calculate_frechet_distance,
                                      fid_inception_v3)

### Intensity ###


def normalize01(vol):
    """A 0.5/99.5-percentile window per volume, clipped to `[0, 1]`.

    Ours, applied identically to both sides, so absolute intensity scale never reaches a metric.
    Per volume rather than per slice: a slice of a real acquisition is a slice of a volume that was
    windowed as a whole, and re-windowing each slice would erase how much signal a slice carries.
    """
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def _quantize(vol):
    """`[0, 1]` float -> uint8. Both reference implementations consume uint8 images -- pytorch-fid
    reads PNGs, StyleGAN-V reads uint8 video frames -- so the feature extractors never see more
    precision than this, and neither should ours."""
    return np.clip(np.rint(vol * 255.0), 0, 255).astype(np.uint8)


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
    `antialias=` supports -- and it is needed here, because a native ground truth shrinks by up to
    2.7x on the way to 256^2 while a generated volume is never resized at all, and un-filtered
    decimation would fold that asymmetry into the real features as false structure. (The feature
    extractors' own resizes, further down, are un-filtered as their references specify -- by then
    both sides are 256^2 and see the identical operation.)
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
    return normalize01(vol)


def canonicalize_generated(produced):
    """The generated volume, `[0, 1]`. Already 1 mm isotropic and 256^2 by construction, so this
    only normalizes -- the crop/pad is a no-op guard for a config with a different `inplane_size`."""
    if produced.shape[1:] != (INPLANE, INPLANE):
        produced = _crop_pad(produced, INPLANE)
    return normalize01(produced)


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


def _frechet(feats_fake, feats_real):
    """pytorch-fid's own distance over two feature sets.

    `calculate_activation_statistics` is `np.mean` / `np.cov(rowvar=False)` on the stacked
    activations, and `calculate_frechet_distance` is vendored verbatim -- including that it
    regularizes only after `sqrtm` has already returned something non-finite. StyleGAN-V's FVD
    does not regularize at all, which is the same thing whenever sqrtm is finite.
    """
    mu_f, sigma_f = np.mean(feats_fake, axis=0), np.cov(feats_fake, rowvar=False)
    mu_r, sigma_r = np.mean(feats_real, axis=0), np.cov(feats_real, rowvar=False)
    return float(calculate_frechet_distance(mu_f, sigma_f, mu_r, sigma_r))


### FID: pytorch-fid, every slice of the acquisition plane ###

_FID_DIM = 2048


class SliceFIDAccumulator:
    """FID over slices of array axis 0 -- the acquisition plane, the axis the model rolls out
    along, so the images scored are the images generated.

    `InceptionV3` is pytorch-fid's, at its default block 3 (2048-d pool), and it does its own
    resize and `[-1, 1]` rescale, so a slice reaches it exactly as one of pytorch-fid's PNGs would:
    uint8 divided by 255, greyscale repeated to three channels. **Every slice is used**; no
    reference implementation subsamples its image set.
    """

    def __init__(self, device="auto", batch_size=64):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = InceptionV3().eval().to(device)
        self._feats = _empty_sides()

    @torch.no_grad()
    def _features(self, vol):
        out = []
        for chunk in np.split(_quantize(vol),
                              range(self.batch_size, len(vol), self.batch_size)):
            x = torch.from_numpy(chunk).to(self.device).float().div_(255.0)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            out.append(self.model(x)[0].squeeze(3).squeeze(2).float().cpu().numpy())
        return np.concatenate(out)

    def add_pair(self, real, fake, stratum):
        self._feats[stratum]["real"].append(self._features(real))
        self._feats[stratum]["fake"].append(self._features(fake))

    def raw_features(self):
        return _stack(self._feats, _FID_DIM)


### FVD: StyleGAN-V's I3D, one clip per volume ###

_FVD_DIM = 400
_I3D_URL = "https://huggingface.co/flateon/FVD-I3D-torchscript/resolve/main/i3d_torchscript.pt"

# Frames per clip. 16 is the standard FVD protocol and the length every published FVD uses; 64 is
# ours, four generated blocks, and reads whether the rollout stays coherent across block
# boundaries. One clip per volume either way -- see `_clip`.
CLIP_CONFIGS = {"f16": 16, "f64": 64}


def load_i3d(device):
    """The torchscript I3D StyleGAN-V's FVD is defined on, cached in the torch hub directory
    beside torchvision's own weights (so `TORCH_HOME` moves it). `MRFLOW_I3D_PATH` overrides the
    location for a node with no outbound route."""
    path = os.environ.get("MRFLOW_I3D_PATH") or os.path.join(
        torch.hub.get_dir(), "checkpoints", "i3d_torchscript.pt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.hub.download_url_to_file(_I3D_URL, path)
    return torch.jit.load(path).eval().to(device)


def _clip(vol, clip_len):
    """The one clip this volume contributes, or None if it is shorter than a clip.

    StyleGAN-V loads `num_frames` consecutive frames per video and discards videos too short to
    provide them; one video is one sample, and clips never overlap because there is only one. The
    offset is the single deviation: StyleGAN-V draws it at random, we center it, because these
    "videos" are anatomically ordered rather than arbitrary and a random offset would add sampling
    noise to a paired comparison without buying independence.
    """
    if vol.shape[0] < clip_len:
        return None
    start = (vol.shape[0] - clip_len) // 2
    return vol[start:start + clip_len]


class ClipFVDAccumulator:
    """One I3D feature per volume, for each entry in `CLIP_CONFIGS`.

    Volume pairs arrive ONE AT A TIME and are released as soon as `add_pair` returns; only the
    400-d vectors accumulate. A volume shorter than a clip contributes nothing on that side, so a
    rollout that stopped early lowers the generated count without being silently rescaled to fit --
    which is why both counts are reported with every distance.
    """

    def __init__(self, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.i3d = load_i3d(device)
        self._feats = {name: _empty_sides() for name in CLIP_CONFIGS}

    @torch.no_grad()
    def _clip_feature(self, vol, clip_len):
        clip = _clip(vol, clip_len)
        if clip is None:
            return np.zeros((0, _FVD_DIM), np.float32)
        # (1, 3, L, H, W) of [0, 255] floats: the detector's own kwargs do the rescale to [-1, 1]
        # and the bilinear resize to 224^2, which is how StyleGAN-V calls it.
        x = torch.from_numpy(_quantize(clip)).to(self.device).float()
        x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        feats = self.i3d(x, rescale=True, resize=True, return_features=True)
        return feats.float().cpu().numpy()

    def add_pair(self, real, fake, stratum):
        for name, clip_len in CLIP_CONFIGS.items():
            sides = self._feats[name][stratum]
            sides["real"].append(self._clip_feature(real, clip_len))
            sides["fake"].append(self._clip_feature(fake, clip_len))

    def raw_features(self):
        return {name: _stack(sides, _FVD_DIM) for name, sides in self._feats.items()}


### Inception Score: torch-fidelity's protocol, on the same TF-ported Inception ###

_IS_DIM = 1008
_IS_SPLITS = 10
_IS_RNG_SEED = 2020


class InceptionScoreAccumulator:
    """Inception logits for the GENERATED volume's acquisition-plane slices.

    No-reference, so there is no `add_pair` and the ground truth never enters. The network is
    `fid_inception_v3()` -- the same TF-ported Inception the FID runs on, taken all the way to its
    1008-way `fc` rather than stopping at the pool, so one backbone serves both metrics. The
    resize and `[-1, 1]` rescale mirror `InceptionV3.forward` exactly, since calling the bare
    network skips that wrapper.

    Read the value against another MR run and never against a natural-image number: ImageNet's
    1,000 classes do not describe an MR slice, so the posteriors are diffuse (measured 3.7 of a
    possible 6.9 nats) and the score sits near 1.6-1.8.
    """

    def __init__(self, device="auto", batch_size=64):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = fid_inception_v3().eval().to(device)
        self._logits = []

    @torch.no_grad()
    def add(self, fake):
        for chunk in np.split(_quantize(fake), range(self.batch_size, len(fake), self.batch_size)):
            x = torch.from_numpy(chunk).to(self.device).float().div_(255.0)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            self._logits.append(self.model(x * 2 - 1).float().cpu().numpy())

    def logits(self):
        return np.concatenate(self._logits) if self._logits else np.zeros((0, _IS_DIM), np.float32)


def inception_score(logits, splits=_IS_SPLITS, shuffle=True, rng_seed=_IS_RNG_SEED):
    """`(mean, std)` of `exp(E_x KL(p(y|x) || p(y)))`, as `torch_fidelity.metric_isc` computes it.

    Its three details, all reproduced: the rows are shuffled with a fixed-seed `RandomState`
    first; the splits are contiguous at `i*N//splits`, which uses every row rather than dropping a
    remainder; and the KL is evaluated in float64 from logits, not from pre-softmaxed
    probabilities.

    **The shuffle does not make this order-invariant**, and nothing does: a fixed permutation of a
    reordered set is still a different split assignment. What it buys is that each split is a
    random sample of the whole set rather than a contiguous block of two or three shards, so the
    ten values are comparable to each other. `IS_mean` moves very little under a reordering
    (measured 0.25% across a 1-vs-32 shard change); `IS_std` moves a lot, so quote it only between
    runs with the same shard count.

    The std is the spread between splits, not an error bar on the mean: each split is a
    self-contained IS over N/10 samples, so it is biased low and grows with the split count.
    """
    n = logits.shape[0]
    if n < splits:
        return float("nan"), float("nan")

    feature = torch.from_numpy(np.ascontiguousarray(logits))
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[torch.from_numpy(rng.permutation(n))]
    feature = feature.double()

    p, log_p = feature.softmax(dim=1), feature.log_softmax(dim=1)
    scores = []
    for i in range(splits):
        lo, hi = i * n // splits, (i + 1) * n // splits
        p_chunk, log_p_chunk = p[lo:hi], log_p[lo:hi]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = (p_chunk * (log_p_chunk - q_chunk.log())).sum(dim=1).mean().exp().item()
        scores.append(kl)
    return float(np.mean(scores)), float(np.std(scores))


### Cross-shard pooling ###


def fid_pooled(raw_per_shard, strata=STRATA):
    """`(distance, n_real, n_fake)` over several shards' `SliceFIDAccumulator` features, merging
    the named strata. Concatenating rows before mean/covariance is associative, so this is what one
    process that had seen every pair itself would compute."""
    return _pooled(raw_per_shard, _FID_DIM, strata)


def fvd_pooled(raw_per_shard, name, strata=STRATA):
    """`(distance, n_real, n_fake)` for one `CLIP_CONFIGS` entry."""
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
    try:
        return (_frechet(sides["fake"], sides["real"]),) + counts
    except ValueError as e:
        # pytorch-fid raises rather than returning a number when `sqrtm` of a badly rank-deficient
        # product comes back meaningfully complex -- reachable with a handful of samples against
        # 2048 or 400 dimensions, i.e. a nearly empty stratum. That must read as nan and let the
        # other strata through, not take the whole `--combine` down.
        print(f"[paper_metrics] Frechet distance undefined at n={counts}: {e}")
        return (float("nan"),) + counts
