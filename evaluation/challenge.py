"""Vendored port of the official VLM3D `mr-volume-generation` scoring container.

Source: github.com/forithmus/VLM3D-Dockers/tree/main/mr_challenges/mrgen_evaluation -- its
`modality_filter.py`, `metrics_basic.py` and `fid_2p5d.py` merged into one module, comments
translated from Turkish, math and control flow unchanged. Its `score.py` is not here: that file
only walks two on-disk directories, which does not apply when we generate in memory, so its
aggregation lives in `evaluation/__init__.py` instead.

Two additions are marked as such: `RunningMoments.array` / `finalize_pooled`, which let several
SLURM array tasks pool their slice features into one global Frechet distance, and `_matrix_sqrt`,
which drops a `sqrtm` kwarg scipy >= 1.17 removed. Neither changes a number.

FVD and Inception Score are scored by no MR container at all; they live in their own
section at the bottom, behind a header that says so. Nothing above that header may change
-- this file is what makes our numbers the leaderboard's.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from scipy import linalg, stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


### Modality scope ###

# Organizer decision: only T1w/T2w/FLAIR/SWI are scored. MRA/DWI/ADC are still released, just
# never scored. Lower-cased, as the official code keeps them.
ALLOWED_MODALITIES = frozenset({"t1w", "t2w", "flair", "swi"})


### MSE / PSNR / SSIM ###


def _normalize01(vol):
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def compute_basic_metrics(real, fake):
    """MSE/PSNR/SSIM for one (real, fake) volume pair.

    Both volumes are percentile-normalized here, so absolute intensity scale never reaches a
    metric, and a `fake` of a different shape is resampled onto `real`'s -- the official code's own
    fallback, and the reason the ground truth must reach this function in its released geometry.
    """
    if real.shape != fake.shape:
        from scipy.ndimage import zoom

        factors = [r / f for r, f in zip(real.shape, fake.shape)]
        fake = zoom(fake, factors, order=1)

    real_n = _normalize01(real)
    fake_n = _normalize01(fake)

    mse = float(np.mean((real_n - fake_n) ** 2))
    psnr = float(peak_signal_noise_ratio(real_n, fake_n, data_range=1.0))
    ssim = float(structural_similarity(real_n, fake_n, data_range=1.0))

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


### FID_2p5D ###

_FEATURE_DIM = 512
_INPUT_SIZE = 224

# Planes: axis=2 (XY, Z fixed), axis=1 (XZ, Y fixed), axis=0 (YZ, X fixed).
_AXIS_FOR_PLANE = {"XY": 2, "XZ": 1, "YZ": 0}


class SqueezeNetFeatureExtractor(nn.Module):
    """squeezenet1_1 with its classifier dropped, returning the global-average-pooled 512-d
    feature vector."""

    def __init__(self, device="cpu"):
        super().__init__()
        weights = tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        base = tv_models.squeezenet1_1(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()
        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


def _normalize_slice(sl):
    sl = sl.astype(np.float32)
    lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(sl)
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0)


def _slice_to_tensor(sl):
    sl = _normalize_slice(sl)
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False
    )
    t = t.repeat(1, 3, 1, 1)
    return t.squeeze(0)


def _iter_slices(volume, axis, stride=4):
    n = volume.shape[axis]
    for idx in range(0, n, stride):
        yield np.take(volume, idx, axis=axis)


class RunningMoments:
    """Accumulates slice features across many volumes, then computes mean/covariance in one pass
    at the end. Feature vectors are kept (float32, 512-d, a few hundred slices per volume, ~1000x
    smaller than the volume), never the volumes."""

    def __init__(self):
        self._chunks = []

    def add(self, feats):
        if feats.shape[0] > 0:
            self._chunks.append(feats)

    def array(self):
        """The concatenated `(N, 512)` feature matrix. Ours, for cross-shard pooling."""
        if not self._chunks:
            return np.zeros((0, _FEATURE_DIM), dtype=np.float32)
        return np.concatenate(self._chunks, axis=0)

    def finalize(self):
        all_feats = self.array()
        if all_feats.shape[0] == 0:
            return np.zeros(_FEATURE_DIM), np.eye(_FEATURE_DIM), 0
        mu = all_feats.mean(axis=0)
        sigma = np.cov(all_feats, rowvar=False)
        return mu, sigma, all_feats.shape[0]


class FIDAccumulator:
    """Takes volume pairs ONE AT A TIME, extracting and accumulating slice features immediately.
    The caller may release each pair as soon as `add_pair` returns."""

    def __init__(self, device="auto", stride=4, batch_size=32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SqueezeNetFeatureExtractor(device=device)
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        self.real_moments = {plane: RunningMoments() for plane in _AXIS_FOR_PLANE}
        self.fake_moments = {plane: RunningMoments() for plane in _AXIS_FOR_PLANE}

    @torch.no_grad()
    def _extract_volume_features(self, volume, axis):
        feats_list = []
        batch = []
        for sl in _iter_slices(volume, axis=axis, stride=self.stride):
            batch.append(_slice_to_tensor(sl))
            if len(batch) == self.batch_size:
                x = torch.stack(batch).to(self.device)
                feats_list.append(self.extractor(x).cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(self.device)
            feats_list.append(self.extractor(x).cpu().numpy())
        if not feats_list:
            return np.zeros((0, _FEATURE_DIM), dtype=np.float32)
        return np.concatenate(feats_list, axis=0)

    def add_pair(self, real_vol, fake_vol):
        for plane, axis in _AXIS_FOR_PLANE.items():
            self.real_moments[plane].add(self._extract_volume_features(real_vol, axis))
            self.fake_moments[plane].add(self._extract_volume_features(fake_vol, axis))

    def finalize(self):
        results = {}
        for plane in _AXIS_FOR_PLANE:
            mu_r, sigma_r, n_r = self.real_moments[plane].finalize()
            mu_f, sigma_f, n_f = self.fake_moments[plane].finalize()
            if n_r < 2 or n_f < 2:
                results[f"FID_2p5D_{plane}"] = float("nan")
                continue
            results[f"FID_2p5D_{plane}"] = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        valid_vals = [v for v in results.values() if not np.isnan(v)]
        results["FID_2p5D_Avg"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
        return results

    def raw_features(self):
        """`{plane: {"real": array, "fake": array}}`. Ours: what a SLURM array task hands to
        `finalize_pooled` so the per-plane distances are computed over every shard's slices at
        once, rather than averaged per shard."""
        return {plane: {"real": self.real_moments[plane].array(),
                        "fake": self.fake_moments[plane].array()}
                for plane in _AXIS_FOR_PLANE}


def finalize_pooled(raw_features_per_shard):
    """`FIDAccumulator.finalize()`, fed features assembled from several shards' `raw_features()`.
    Concatenating before mean/covariance is associative, so this is what one process that had seen
    every pair itself would compute."""
    empty = np.zeros((0, _FEATURE_DIM), dtype=np.float32)
    results = {}
    for plane in _AXIS_FOR_PLANE:
        real = np.concatenate([s[plane]["real"] for s in raw_features_per_shard] or [empty])
        fake = np.concatenate([s[plane]["fake"] for s in raw_features_per_shard] or [empty])
        if real.shape[0] < 2 or fake.shape[0] < 2:
            results[f"FID_2p5D_{plane}"] = float("nan")
            continue
        results[f"FID_2p5D_{plane}"] = frechet_distance(
            real.mean(axis=0), np.cov(real, rowvar=False),
            fake.mean(axis=0), np.cov(fake, rowvar=False))

    valid_vals = [v for v in results.values() if not np.isnan(v)]
    results["FID_2p5D_Avg"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
    return results


def _matrix_sqrt(a):
    """`scipy.linalg.sqrtm` without the `disp=` kwarg the official code passes -- scipy >= 1.17
    removed it, and this venv is on 1.18. Pre-1.17 `sqrtm` returns `(result, info)` when `disp`
    is given and just `result` otherwise, so dropping it is a version fix, not a math change."""
    result = linalg.sqrtm(a)
    return result[0] if isinstance(result, tuple) else result


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = _matrix_sqrt(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _matrix_sqrt((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


### FVD and Inception Score -- NOT part of any official MR container ###

# Everything below this line is ours. `mr-volume-generation` scores neither metric, so nothing
# here can make a number wrong on the leaderboard -- and nothing here may reach back above the
# line either.
#
# FVD *is* scored by the challenge's CT track, so it follows that container as closely as MR
# allows: `ct_challenges/ctgen_evaluation`'s `metrics3d._resize_vol` fixed-size trilinear resize,
# its `(B, C, D, 224, 224)` video in `[-1, 1]`, and its `FVD/fvd_pytorch_model.compute_fvd`
# arithmetic verbatim. Two things cannot carry over:
#
#   backbone   CT-Net windows Hounsfield units (`[-1000, 200]`) and classifies 18 CT findings.
#              MR has no HU scale, so features come instead from the I3D Kinetics-400 network FVD
#              is *defined* over -- the same network the CT container's own reference file,
#              `FVD/frechet_video_distance.py`, pulls from tf.hub, via the torchscript export
#              every PyTorch FVD uses.
#   intensity  the HU clip becomes `_normalize01`, the MR metric's own percentile normalization,
#              so FVD and MSE/PSNR/SSIM see the same two volumes.
#
# One deliberate departure: `evaluate_fvd.py` averages FVD over strata of `CHUNK = 4` pairs, which
# estimates a 400x400 covariance from 4 samples. We keep one feature vector per volume and compute
# a single distance over all of them, as `finalize_pooled` already does for FID.
#
# Inception Score appears nowhere in the challenge, so it is the canonical definition instead:
# `exp(E_x KL(p(y|x) || p(y)))` over ImageNet Inception-v3, split-averaged as Salimans et al. and
# the reference PyTorch port compute it, fed the slices FID_2p5D already looks at.

_FVD_FEATURE_DIM = 400
_FVD_TARGET_DHW = (201, 224, 224)  # metrics3d._resize_vol's default; 224 is I3D's own resolution
_I3D_URL = "https://huggingface.co/flateon/FVD-I3D-torchscript/resolve/main/i3d_torchscript.pt"

_IS_INPUT_SIZE = 299
_IS_NUM_CLASSES = 1000
_IS_SPLITS = 10


def _load_i3d(device):
    """The torchscript I3D every PyTorch FVD is computed with, cached in the torch hub directory
    beside torchvision's own weights (so `TORCH_HOME` moves it). `MRFLOW_I3D_PATH` overrides the
    location, for a node with no outbound route -- the CT container bakes CT-Net in the image for
    the same reason."""
    path = os.environ.get("MRFLOW_I3D_PATH") or os.path.join(
        torch.hub.get_dir(), "checkpoints", "i3d_torchscript.pt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(_I3D_URL, path)
    return torch.jit.load(path).eval().to(device)


def _volume_to_video(volume, device):
    """One volume as I3D's input: `(1, 3, 201, 224, 224)` in `[-1, 1]`.

    `metrics3d.fvd_pair`'s pipeline. Normalization is volume-wide rather than per slice -- unlike
    FID_2p5D, which sees each slice alone -- because intensity consistency *along* the stack is
    part of what a video metric measures, and it happens before the resize so the percentiles come
    from the volume as released. The fixed target size is also what makes a native-geometry ground
    truth and a 1 mm rollout comparable at all.
    """
    vol = torch.from_numpy(_normalize01(volume))[None, None].to(device)
    vol = torch.nn.functional.interpolate(vol, size=_FVD_TARGET_DHW, mode="trilinear",
                                          align_corners=False)
    return (vol * 2.0 - 1.0).repeat(1, 3, 1, 1, 1)


class FVDAccumulator:
    """Volume pairs ONE AT A TIME, like `FIDAccumulator`, keeping a single 400-d I3D feature
    vector per volume -- so a shard's whole FVD state is `(n_cases, 400)` twice over."""

    def __init__(self, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.i3d = _load_i3d(device)
        self._real = []
        self._fake = []

    @torch.no_grad()
    def _features(self, volume):
        video = _volume_to_video(volume, self.device)
        # StyleGAN-V's detector kwargs: the resize and rescale are already done above, and
        # `return_features` takes the 400-d logits layer the TF reference reads as `Mean:0`.
        feats = self.i3d(video, rescale=False, resize=False, return_features=True)
        return feats.float().cpu().numpy()

    def add_pair(self, real_vol, fake_vol):
        self._real.append(self._features(real_vol))
        self._fake.append(self._features(fake_vol))

    def raw_features(self):
        """`{"real": (N, 400), "fake": (N, 400)}` -- what a SLURM array task hands to
        `fvd_pooled`, so the distance is computed over every shard's volumes at once."""
        empty = np.zeros((0, _FVD_FEATURE_DIM), dtype=np.float32)
        return {"real": np.concatenate(self._real) if self._real else empty,
                "fake": np.concatenate(self._fake) if self._fake else empty}


def frechet_video_distance(feats_fake, feats_real):
    """`FVD/fvd_pytorch_model.compute_fvd`, unchanged apart from `_matrix_sqrt`.

    Not `frechet_distance` above: the two official files differ, and this is the one the FVD
    number is defined by. It always adds a trace-scaled `eps * I` ridge to both covariances, where
    the FID one adds its offset only after `sqrtm` has already returned something non-finite.
    """
    mu_gen, sigma_gen = feats_fake.mean(axis=0), np.cov(feats_fake, rowvar=False)
    mu_real, sigma_real = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)

    eps = 1e-6 * np.trace(sigma_gen + sigma_real) / sigma_gen.shape[0]
    identity = np.eye(sigma_gen.shape[0], dtype=sigma_gen.dtype)
    sigma_gen = sigma_gen + eps * identity
    sigma_real = sigma_real + eps * identity

    m = np.square(mu_gen - mu_real).sum()
    s = _matrix_sqrt(np.dot(sigma_gen, sigma_real))
    return float(np.real(m + np.trace(sigma_gen + sigma_real - s * 2)))


def fvd_pooled(raw_features_per_shard):
    """One FVD over several shards' `FVDAccumulator.raw_features()`. Needs `> 400` volumes to
    estimate its covariances honestly, the same caveat the 512-d FID carries."""
    empty = np.zeros((0, _FVD_FEATURE_DIM), dtype=np.float32)
    fake = np.concatenate([s["fake"] for s in raw_features_per_shard] or [empty])
    real = np.concatenate([s["real"] for s in raw_features_per_shard] or [empty])
    if fake.shape[0] < 2 or real.shape[0] < 2:
        return float("nan")
    return frechet_video_distance(fake, real)


def _slice_to_inception_input(sl):
    """`_slice_to_tensor` at Inception-v3's resolution and range: the same `_normalize_slice`
    FID_2p5D uses, then `[0, 1] -> [-1, 1]`, which is how the reference implementation feeds it."""
    t = torch.from_numpy(_normalize_slice(sl))[None, None]
    t = torch.nn.functional.interpolate(t, size=(_IS_INPUT_SIZE, _IS_INPUT_SIZE),
                                        mode="bilinear", align_corners=False)
    return (t * 2.0 - 1.0).repeat(1, 3, 1, 1).squeeze(0)


class InceptionScoreAccumulator:
    """Inception-v3 class posteriors for the GENERATED volumes' slices, at FID_2p5D's stride.

    No-reference, so there is no `add_pair` and the ground truth never enters. Slices are taken
    along array axis 0 -- the acquisition plane, the axis the model actually rolls out along, so
    the images scored are the images generated -- rather than all three of FID's planes, whose
    reprojected views the model never emits.
    """

    def __init__(self, device="auto", stride=4, batch_size=32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        # transform_input=False: the reference port's setting, since the input is already [-1, 1].
        self.model = tv_models.inception_v3(weights=weights, transform_input=False).eval().to(device)
        self._probs = []

    @torch.no_grad()
    def _predict(self, batch):
        x = torch.stack(batch).to(self.device)
        return torch.softmax(self.model(x), dim=1).float().cpu().numpy()

    def add(self, fake_vol):
        batch = []
        for sl in _iter_slices(fake_vol, axis=0, stride=self.stride):
            batch.append(_slice_to_inception_input(sl))
            if len(batch) == self.batch_size:
                self._probs.append(self._predict(batch))
                batch = []
        if batch:
            self._probs.append(self._predict(batch))

    def probs(self):
        """`(N_slices, 1000)`, one row per scored slice -- what a shard carries to
        `inception_score`, which needs the whole population to form its splits."""
        empty = np.zeros((0, _IS_NUM_CLASSES), dtype=np.float32)
        return np.concatenate(self._probs) if self._probs else empty


def inception_score(probs, splits=_IS_SPLITS):
    """`(mean, std)` of `exp(E_x KL(p(y|x) || p(y)))` over `splits` equal chunks of `probs`.

    Salimans et al. 2016 as the reference PyTorch port computes it, remainder rows dropped by the
    same integer division. The std is the spread across chunks, not an error bar on the mean.

    The one departure is float64: the reference sums 1000 log terms in whatever dtype the model
    emitted, which for us is float32. Worth ~1.4e-6 relative, and in the right direction.
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
