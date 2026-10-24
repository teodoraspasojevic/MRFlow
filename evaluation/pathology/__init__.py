"""A 3D multi-label pathology classifier, for the pathology-consistency metric.

Labels are the MR-RATE **14 merged-group majority-vote** labels (the artifact
`baselines/ccella/labels.py` documents), one binary vector per *study*. Three BP groups have
exactly the member pathologies of a PP group, so their columns are identical by construction;
the classifier predicts the 11 distinct groups, `LABELS`, and each dropped BP column is named
alongside the PP label it duplicates.

The input is the modality classifier's: `evaluation.modality.classifier_input` over a volume
canonicalized by `evaluation.paper_metrics`. See `evaluation/pathology/README.md`.
"""

import csv
import os

import numpy as np
import torch

from evaluation.modality import INPUT_SHAPE, ModalityNet3d, file_sha256, to_tensor

# `mrrate_merged_labels.csv` header order, checked on read.
LABEL_COLUMNS = (
    "PP_Cerebrovascular", "PP_Neoplastic", "PP_Neurodegenerative", "PP_Spinal",
    "PP_Cystic_developmental", "PP_Infectious", "PP_Inflammatory", "PP_Unspecific_bucket",
    "BP_Atrophies", "BP_Contrast_enhancing_intracranial", "BP_Infectious_lesions",
    "BP_Edematous_lesions", "BP_Hemorrhagic_lesions", "BP_Cystic_lesions",
)
DUPLICATES = {"BP_Atrophies": "PP_Neurodegenerative",
              "BP_Contrast_enhancing_intracranial": "PP_Neoplastic",
              "BP_Infectious_lesions": "PP_Infectious"}

# Output order. Literal and fixed: baked into every checkpoint and saved prediction.
LABELS = tuple(c for c in LABEL_COLUMNS if c not in DUPLICATES)
NUM_LABELS = len(LABELS)

LABELS_CSV = "mrrate_merged_labels.csv"


def read_labels(labels_root):
    """`{study_uid: int8 array (NUM_LABELS,)}`. Raises on a header that is not `LABEL_COLUMNS`,
    a non-binary cell, a conflicting duplicate row, or a duplicate column that is not a duplicate."""
    path = os.path.join(labels_root, LABELS_CSV)
    keep = [LABEL_COLUMNS.index(name) for name in LABELS]
    table = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != ("study_uid",) + LABEL_COLUMNS:
            raise ValueError(f"{path} header is not {LABEL_COLUMNS}")
        for row in reader:
            if any(v not in ("0", "1") for v in row[1:]):
                raise ValueError(f"non-binary label in {path}")
            full = [int(v) for v in row[1:]]
            for bp, pp in DUPLICATES.items():
                if full[LABEL_COLUMNS.index(bp)] != full[LABEL_COLUMNS.index(pp)]:
                    raise ValueError(f"{bp} and {pp} differ for one study in {path}")
            vector = np.array([full[i] for i in keep], dtype=np.int8)
            if row[0] in table and not np.array_equal(table[row[0]], vector):
                raise ValueError(f"conflicting duplicate rows in {path}")
            table[row[0]] = vector
    return table


def labels_sha256(labels_root):
    return file_sha256(os.path.join(labels_root, LABELS_CSV))


def build_model(model_config):
    return ModalityNet3d(widths=tuple(model_config["widths"]), dropout=model_config["dropout"],
                         num_classes=NUM_LABELS)


def load_classifier(path, device="cpu"):
    """-> (model in eval mode, meta). Raises on another label order or input shape."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    if tuple(meta["labels"]) != LABELS:
        raise ValueError(f"{path} was trained with labels {meta['labels']}, not {LABELS}")
    if tuple(meta["input_shape"]) != INPUT_SHAPE:
        raise ValueError(f"{path} expects input {meta['input_shape']}, not {INPUT_SHAPE}")
    model = build_model(meta["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta


@torch.no_grad()
def predict_proba(model, inputs, batch_size=16, device=None):
    """uint8 `(N, D, H, W)` -> float64 `(N, NUM_LABELS)` sigmoid probabilities in `LABELS` order."""
    device = device or next(model.parameters()).device
    out = []
    for start in range(0, len(inputs), batch_size):
        x = to_tensor(inputs[start:start + batch_size]).to(device)
        out.append(torch.sigmoid(model(x).float()).double().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, NUM_LABELS))
