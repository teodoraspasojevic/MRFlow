"""A 3D MR contrast classifier (T1w / T2w / SWI / FLAIR), for the modality-consistency metric.

The classifier sees exactly what the evaluation's metrics see: a volume already canonicalized by
`evaluation.paper_metrics` (`canonicalize_gt` for a ground truth, `canonicalize_generated` for a
rollout) -- 1 mm, 256^2, plane-first, `[0, 1]`. `classifier_input` is the only extra step, and it
belongs to the classifier alone: an area resize to `INPUT_SHAPE` and a uint8 quantization, so a
cached training input and an evaluation input are bit-identical for the same volume.

See `evaluation/modality/README.md` for splits, training and the metric.
"""

import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from echosyn.common.mrrate import MODALITY_ALIASES

# Class index order. Literal and fixed: it is baked into every checkpoint and every saved
# prediction, so it may never be reordered.
CLASSES = ("T1w", "T2w", "SWI", "FLAIR")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

INPUT_SHAPE = (64, 128, 128)  # (slices, H, W) after the classifier's own resize


def modality_index(modality):
    """Any spelling of one of the four contrasts -> class index. Raises on anything else (MRA,
    UNKNOWN, a typo), so an out-of-scope label can never be scored as a wrong prediction."""
    name = MODALITY_ALIASES.get(str(modality).strip().lower(), str(modality).strip())
    if name not in CLASS_TO_INDEX:
        raise KeyError(f"modality {modality!r} is not one of {CLASSES}")
    return CLASS_TO_INDEX[name]


def classifier_input(volume01, shape=INPUT_SHAPE):
    """A canonicalized `(T, H, W)` volume in `[0, 1]` -> uint8 `shape` array.

    Area-averaged, so decimation is anti-aliased; a volume shorter than `shape[0]` is repeated
    rather than interpolated. The slice count is thereby normalized away, which keeps a rollout's
    length from being a class cue.
    """
    x = torch.from_numpy(np.ascontiguousarray(volume01, dtype=np.float32))[None, None]
    x = F.interpolate(x, size=shape, mode="area")[0, 0].numpy()
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def to_tensor(inputs):
    """uint8 `(N, D, H, W)` (or one `(D, H, W)`) -> float `(N, 1, D, H, W)` in `[0, 1]`."""
    x = torch.as_tensor(np.asarray(inputs))
    if x.ndim == 3:
        x = x[None]
    return x[:, None].float() / 255.0


### Model ###


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm3d(cout)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.skip(x))


class ModalityNet3d(nn.Module):
    """A small 3D ResNet: stem, one residual block per stage, global average pool, linear head."""

    def __init__(self, widths=(32, 64, 128, 256), dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, widths[0], 5, 2, 2, bias=False),
                                  nn.BatchNorm3d(widths[0]), nn.ReLU(inplace=True))
        stages, cin = [], widths[0]
        for i, w in enumerate(widths):
            stages.append(_Block(cin, w, 1 if i == 0 else 2))
            cin = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.head(x.mean(dim=(2, 3, 4)))


### Checkpoints ###


def build_model(model_config):
    return ModalityNet3d(widths=tuple(model_config["widths"]), dropout=model_config["dropout"])


def save_checkpoint(path, model, meta, **training_state):
    """`meta` must say how to rebuild and read the model: its config, classes and input shape.
    `training_state` (optimizer, scheduler, epoch, ...) is only needed to resume."""
    torch.save({"model": model.state_dict(), "meta": meta, **training_state}, path)


def load_classifier(path, device="cpu"):
    """-> (model in eval mode, meta). Raises if the checkpoint's label mapping or input shape is
    not this code's, since a silently permuted class order would score every case wrong."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    if tuple(meta["classes"]) != CLASSES:
        raise ValueError(f"{path} was trained with classes {meta['classes']}, not {CLASSES}")
    if tuple(meta["input_shape"]) != INPUT_SHAPE:
        raise ValueError(f"{path} expects input {meta['input_shape']}, not {INPUT_SHAPE}")
    model = build_model(meta["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@torch.no_grad()
def predict_proba(model, inputs, batch_size=16, device=None):
    """uint8 `(N, D, H, W)` -> float64 `(N, NUM_CLASSES)` softmax probabilities, columns in
    `CLASSES` order. Batching does not change the result: the model is in eval mode."""
    device = device or next(model.parameters()).device
    out = []
    for start in range(0, len(inputs), batch_size):
        x = to_tensor(inputs[start:start + batch_size]).to(device)
        out.append(torch.softmax(model(x).float(), dim=1).double().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, NUM_CLASSES))
