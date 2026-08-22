"""Vendored port of the official VLM3D `mr-volume-generation` scoring container.

Source: github.com/forithmus/VLM3D-Dockers/tree/main/mr_challenges/mrgen_evaluation -- its
`modality_filter.py`, `metrics_basic.py` and `fid_2p5d.py` merged into one module, comments
translated from Turkish, math and control flow unchanged. Its `score.py` is not here: that file
only walks two on-disk directories, which does not apply when we generate in memory, so its
aggregation lives in `evaluation/__init__.py` instead.

Two additions are marked as such: `RunningMoments.array` / `finalize_pooled`, which let several
SLURM array tasks pool their slice features into one global Frechet distance, and `_matrix_sqrt`,
which drops a `sqrtm` kwarg scipy >= 1.17 removed. Neither changes a number.

Everything else must stay byte-faithful -- this file is what makes our numbers the leaderboard's.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from scipy import linalg
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
