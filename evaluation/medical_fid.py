"""Frechet distances on medical-image backbones: `fid_3d_medicalnet` and `fid_2p5d_radimagenet_*`.

Both follow the protocol CCELLA (github.com/grabkeem/CCELLA, `scripts/evaluate_diffusion.py`) and
Alignment-to-Synthesis (arXiv:2506.00633) use, and which Report2CT (arXiv:2509.14780) describes for
its 2.5D numbers: a frozen medical backbone, one globally average-pooled feature vector per input,
and the ordinary FID mean/covariance arithmetic on top. The arithmetic is `paper_metrics.pooled`
-- pytorch-fid's, vendored -- so all three FIDs in this project are literally the same function on
different rows.

    fid_3d_medicalnet            one 2048-d vector per volume, from MedicalNet's 3D ResNet-50
                                 (`resnet_50_23dataset.pth`, the 23-dataset pretrain) at its
                                 `layer4` map, globally average-pooled -- the representation the
                                 task-specific `conv_seg` head sits on.
    fid_2p5d_radimagenet_{axial,coronal,sagittal}
                                 one 2048-d vector per **slice**, from the official RadImageNet
                                 ResNet-50 at its post-global-average-pool representation (the
                                 notebook's `children()[:9]`, before the classifier). The three
                                 anatomical planes are three **independent** distributions and
                                 three independent Frechet distances -- their rows are never
                                 pooled into one FID.
    fid_2p5d_radimagenet_mean    the unweighted arithmetic mean of those three distances.

"2.5D" here means the 2D encoder applied to all slices of three orthogonal planes. It does **not**
mean packing adjacent slices into RGB channels: a greyscale slice becomes three identical channels.

Both extractors are handed the pair `EvalAccumulator` already canonicalized -- the model's 1 mm
isotropic / 256^2 training grid, `normalize01`-windowed -- so real and generated differ in nothing
but content, and everything below is applied to the two sides by the same code path.

Decisions that the references do not pin down, made here once and stated:

  3D geometry      both volumes go through `to_canonical_axes` first, so every case reaches the
                   extractor in the same (S, R, A) anatomical frame. `canonicalize_*` leaves a
                   volume plane-first, which is a *different* axis order per acquisition plane, and
                   a 3D convolution is not orientation-equivariant -- without this the real
                   distribution would be a mixture of three orientations.
  3D input shape   a volume is then trilinearly resampled to a fixed `MEDICALNET_SHAPE` cube.
                   MedicalNet is fully convolutional and the global pool would swallow any size,
                   but the receptive field then covers a different fraction of the anatomy per
                   case, and generated volumes do not all have the same length. A fixed grid is
                   also what CCELLA feeds it, since its real and synthetic volumes are one shape by
                   construction. The cost is that the slice axis is *stretched*, so a rollout that
                   stopped early is rescaled rather than penalised for its length -- `FVD` and the
                   sample counts are where length is read.
  3D intensity     MedicalNet's own data pipeline (`datasets/brains18.py`,
                   `__itensity_normalize_one_volume__`) z-scores over the **nonzero** voxels, so
                   that is what the network's batch-norm statistics were estimated under and that
                   is what is done here. Its background fill -- standard normal noise in the zero
                   voxels -- is a training augmentation and is left out: it would make the metric
                   non-deterministic. CCELLA instead feeds `[0, 1]` volumes directly; we follow
                   MedicalNet's own pipeline.
  2.5D intensity   the **official** RadImageNet PyTorch preprocessing, from
                   `pytorch_example.ipynb`: `(image - 127.5) * 2 / 255` on an 8-bit `cv2.imread`,
                   i.e. `[0, 255] -> [-1, 1]`. Our common representation is already `[0, 1]`, so the
                   same mapping is `2x - 1` (`radimagenet_intensity`). Greyscale is replicated into
                   three identical channels and reversed to **BGR**, the convention `cv2.imread`
                   sets -- with identical channels the reversal is a no-op numerically, and it is
                   kept so the code says what the protocol is. **No ImageNet mean subtraction and
                   no torchvision normalization on top.** The `[0, 1]` window is per *volume*
                   (`normalize01`), never per slice: a per-slice min-max would erase how much
                   signal a slice carries and would behave differently on real and generated
                   artifacts. **This is not CCELLA's input.** CCELLA passes `[0, 1]` straight in
                   and documents no reason; we follow the upstream RadImageNet example instead, so
                   this number is *not* numerically interchangeable with a CCELLA RadImageNet FID.
  2.5D sampling    **every** slice of every plane -- `sample_every_k = 1`, `center_slices = false`,
                   `drop_empty = false`. That is the project-wide rule: use every slice unless the
                   extractor architecturally forbids it. A longer volume therefore contributes more
                   rows than a short one, which is not hidden: `n_fid_2p5d_radimagenet_*_real` /
                   `_fake` give the slice counts per plane and `n_scored_files` the study count, so
                   the weighting is visible in every run. The rows are never materialized --
                   `paper_metrics.Moments` streams the mean and covariance.

                   **The real and generated counts differ per plane, and that is the metric
                   working.** In the (S, R, A) frame a volume acquired in plane P contributes its
                   own T slices along P's axis and exactly 256 along the other two, so a plane's
                   real-minus-generated count is precisely the summed rollout-length error of the
                   volumes acquired in that plane. Measured on a 4-case smoke run: per-case length
                   errors of +8 (coronal), +26 and +10 (sagittal), -6 (axial) give plane deltas of
                   exactly +8, +36 and -6. **Do not equalize them.** Variable volume length is an
                   intended model output -- the stop token decides where a rollout ends, and `T` is
                   never touched in preprocessing for that reason -- so forcing the counts equal
                   would hide a real failure mode behind a tidier-looking table.
  2.5D geometry    slices are taken after undoing `plane_order`, i.e. in the repo's canonical
                   (S, R, A) axis order, so "axial" means the same cut for an axially and a
                   sagittally acquired series. Each slice is bilinearly resized to 224^2 --
                   un-filtered, like the resizes the other reference extractors do internally.
                   **The two out-of-plane views carry a length signal.** Their slices are
                   `T x 256`, and `T` differs between a ground truth and a rollout, so the resize
                   to a square stretches the two sides by slightly different factors. Nothing is
                   padded or cropped to hide that -- a model that systematically generates short
                   volumes should show it -- but it does mean those two planes are not purely a
                   content comparison. The in-plane one (the acquisition plane) is.

The weights are two downloads, cached in the torch hub directory beside the I3D and Inception ones
(`TORCH_HOME` moves them), each verified against the sha256 of the file in its official release and
loaded `strict=True`. Neither ever falls back to random or ImageNet weights: a hash mismatch or a
state-dict mismatch raises.
"""

import hashlib
import os
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from echosyn.common.mrrate import plane_order
from evaluation.medicalnet_resnet import resnet50 as medicalnet_resnet50
from evaluation.paper_metrics import _empty_sides

FEATURE_DIM = 2048


### Weights ###


def _cache_dir():
    return os.path.join(torch.hub.get_dir(), "checkpoints")


def _fetch(url, filename, sha256, member=None, env_var=None):
    """A Google-Drive-hosted checkpoint, cached and hash-checked.

    `member` pulls one file out of a downloaded zip -- both official releases ship several models
    in one archive. `env_var` names an override for a node with no outbound route, in the same
    spirit as `MRFLOW_I3D_PATH`.
    """
    override = os.environ.get(env_var) if env_var else None
    path = override or os.path.join(_cache_dir(), filename)
    if not os.path.exists(path):
        import gdown  # only needed on the first run; not imported at module load

        os.makedirs(_cache_dir(), exist_ok=True)
        download = path + (".zip" if member else ".part")
        gdown.download(url=url, output=download, quiet=False)
        if member:
            with zipfile.ZipFile(download) as archive, open(path, "wb") as out:
                out.write(archive.read(member))
            os.remove(download)
        else:
            os.rename(download, path)

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise RuntimeError(f"{path} is not the expected checkpoint: sha256 {digest.hexdigest()}, "
                           f"expected {sha256}. Delete it and let it re-download.")
    return path


# Tencent MedicalNet's 23-dataset ResNet-50 pretrain. The official release is one 2.8 GB archive
# (drive id 13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG) holding every depth; this is the single-file mirror
# torch.hub's `Warvito/MedicalNet-models` downloads, and its bytes are **identical** to that
# archive's `pretrain/resnet_50_23dataset.pth` -- verified by the sha256 below, which was taken
# from the official archive's member.
MEDICALNET = {
    "url": "https://drive.google.com/uc?export=download&id=1qXyw9S5f-6N1gKECDfMroRnPZfARbqOP",
    "filename": "resnet_50_23dataset.pth",
    "sha256": "ff48a62219073fb977fd3f4ddfb8dc1367f0ec156c8d6f6c37e205bd683a246e",
    "env_var": "MRFLOW_MEDICALNET_PATH",
}

# RadImageNet's own PyTorch release (README of github.com/BMEII-AI/RadImageNet), a zip of three
# converted backbones; only ResNet50 is kept. The state dict is a `nn.Sequential` of torchvision
# resnet50's children up to `layer4`, which is exactly what `_radimagenet_trunk` rebuilds.
RADIMAGENET = {
    "url": "https://drive.google.com/uc?export=download&id=1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR",
    "filename": "RadImageNet-ResNet50.pt",
    "sha256": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    "member": "RadImageNet_pytorch/ResNet50.pt",
    "env_var": "MRFLOW_RADIMAGENET_PATH",
}


def load_medicalnet(device):
    state = torch.load(_fetch(**MEDICALNET), map_location="cpu", weights_only=True)["state_dict"]
    model = medicalnet_resnet50()
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})  # strict
    return model.eval().to(device)


def load_radimagenet(device):
    state = torch.load(_fetch(**RADIMAGENET), map_location="cpu", weights_only=True)
    trunk = torch.nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:-2])
    trunk.load_state_dict({k[len("backbone."):]: v for k, v in state.items()})  # strict
    return trunk.eval().to(device)


### Geometry ###


def to_canonical_axes(vol, plane):
    """Undo `plane_order`: a plane-first volume back to the repo's canonical (S, R, A) axes.

    Both extractors start here, for the same reason: a plane name -- or a 3D convolution's sense of
    up -- must mean the same thing whichever way the series was acquired.
    """
    return vol.transpose(np.argsort(plane_order(plane)))


### fid_3d_medicalnet ###

MEDICALNET_SHAPE = (128, 128, 128)


def _medicalnet_input(vol):
    """A canonicalized `(T, H, W)` volume -> the `(1, 1, D, H, W)` tensor MedicalNet expects."""
    x = torch.from_numpy(np.ascontiguousarray(vol))[None, None]
    x = F.interpolate(x, size=MEDICALNET_SHAPE, mode="trilinear", align_corners=False)
    foreground = x[x > 0]
    if foreground.numel() < 2:
        return x
    # correction=0 so this is numpy's `pixels.std()`, which is what MedicalNet's own code calls.
    return (x - foreground.mean()) / (foreground.std(correction=0) + 1e-6)


class MedicalNet3dAccumulator:
    """One 2048-d feature per volume, in the (S, R, A) anatomical frame. Pairs arrive one at a time
    and may be released immediately."""

    def __init__(self, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = load_medicalnet(device)
        self._feats = _empty_sides(FEATURE_DIM)

    @torch.no_grad()
    def _features(self, vol):
        x = _medicalnet_input(vol).to(self.device)
        return self.model(x).mean((2, 3, 4)).float().cpu().numpy()

    def add_pair(self, real, fake, plane):
        self._feats["real"].add(self._features(to_canonical_axes(real, plane)))
        self._feats["fake"].add(self._features(to_canonical_axes(fake, plane)))

    def moments(self):
        return self._feats


### fid_2p5d_radimagenet_* ###

# Slicing axes of the repo's canonical (S, R, A) order: stepping along S gives axial cuts, along R
# sagittal ones, along A coronal ones. `PLANES` is also the reporting order.
PLANES = ("axial", "coronal", "sagittal")
_PLANE_AXIS = {"axial": 0, "sagittal": 1, "coronal": 2}

_RADIMAGENET_SIZE = 224


def radimagenet_intensity(x):
    """`[0, 1]` -> `[-1, 1]`, the official RadImageNet mapping.

    `pytorch_example.ipynb` reads an 8-bit image with `cv2.imread` and applies
    `(image - 127.5) * 2 / 255`, i.e. `[0, 255] -> [-1, 1]`. Our common representation is already
    `[0, 1]`, so the same mapping is `2x - 1`: 0 -> -1, 0.5 -> 0, 1 -> 1.

    That `[0, 1]` is `paper_metrics.normalize01`, spelled out there: a per-**volume** 0.5/99.5
    percentile window over every voxel (background included, numpy's linear interpolation), then
    `clip((v - lo) / (hi - lo), 0, 1)`. Per volume and never per slice.
    """
    return 2.0 * x - 1.0


def _plane_slices(vol, axis):
    """**Every** slice along `axis`, as a `(n, H, W)` array.

    No subsampling, no centre crop, no emptiness filter: `sample_every_k = 1`,
    `center_slices = false`, `drop_empty = false`. A longer volume therefore contributes more rows
    than a short one, which is why both the study count and the slice count are reported.
    """
    return vol.swapaxes(0, axis)


class RadImageNet2p5dAccumulator:
    """Per-plane slice features, three distributions per side."""

    def __init__(self, device="auto", batch_size=32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = load_radimagenet(device)
        self._feats = {p: _empty_sides(FEATURE_DIM) for p in PLANES}

    @torch.no_grad()
    def _add(self, moments, slices):
        """Every slice, in `batch_size` chunks straight into the streaming moments -- chunking is
        for memory only and changes no number."""
        for chunk in np.split(slices, range(self.batch_size, len(slices), self.batch_size)):
            x = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device).float().unsqueeze(1)
            x = radimagenet_intensity(x).repeat(1, 3, 1, 1)[:, [2, 1, 0]]
            x = F.interpolate(x, size=(_RADIMAGENET_SIZE,) * 2, mode="bilinear",
                              align_corners=False)
            moments.add(self.model(x).mean((2, 3)).float().cpu().numpy())

    def add_pair(self, real, fake, plane):
        real, fake = to_canonical_axes(real, plane), to_canonical_axes(fake, plane)
        for name, axis in _PLANE_AXIS.items():
            self._add(self._feats[name]["real"], _plane_slices(real, axis))
            self._add(self._feats[name]["fake"], _plane_slices(fake, axis))

    def moments(self):
        return self._feats


def signature():
    """What the cached features were computed with. `combine` refuses to pool shards whose
    signatures disagree, so a changed checkpoint or preprocessing can never be averaged into an
    older run's numbers."""
    return {
        "medicalnet_sha256": MEDICALNET["sha256"], "medicalnet_layer": "layer4/avgpool",
        "medicalnet_shape": MEDICALNET_SHAPE, "medicalnet_intensity": "zscore_nonzero",
        "geometry": "canonical_sra",
        "radimagenet_sha256": RADIMAGENET["sha256"], "radimagenet_layer": "layer4/avgpool",
        "radimagenet_size": _RADIMAGENET_SIZE,
        # v2: the official `2x-1` mapping. v1 fed [0, 1] straight in (CCELLA's habit) and its
        # numbers are not comparable with these.
        "radimagenet_intensity": "2x-1_bgr_v2",
        "slice_policy": "all_slices", "planes": PLANES,
    }
