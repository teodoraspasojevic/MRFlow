"""The multi-label classification objective, and the metrics that report on it.

Upstream's auxiliary head is a two-way softmax trained with `FocalLoss(alpha=0.75,
use_softmax=True, reduction='none')` (`diff_model_train_all.py:685`). The 14 merged MR-RATE groups
are **not** mutually exclusive -- 41.4% of train studies carry `PP_Unspecific_bucket` and 18.7%
carry `PP_Neurodegenerative`, frequently together, and a study can hold up to 11 at once -- so a
softmax would force them onto a simplex and turn "which ones" into "which one". This module is
therefore a binary-cross-entropy objective instead.

**It is a drop-in for upstream's loss object.** `train_one_epoch` calls
`text_class_loss(class_pred, pirads_tensor)`, expects an unreduced `[B, C]` tensor, masks it with
`text_isnull` and reduces it as `sum(masked) / n_unmasked` -- a per-sample *sum over classes*, mean
over samples. Returning `reduction="none"` keeps every one of those lines working unchanged, and
keeps upstream's reduction convention rather than substituting a mean.

`BCEWithLogitsLoss` is used rather than a hand-rolled sigmoid-then-BCE because it is the
log-sum-exp-stable form, and because `binary_cross_entropy_with_logits` is on PyTorch's autocast
float32 list, so it stays in fp32 inside upstream's `autocast("cuda")` block.

`pos_weight` comes from the **train split only**, `neg / pos` per label, capped (default 20). The
cap matters: `PP_Spinal` sits at 2.05% and `PP_Inflammatory` at 3.07%, which ask for 47.8 and 31.6;
left uncapped, a handful of positives would dominate a term that is only meant to be auxiliary. The
computed values are written into the run's checkpoint and logged, so the weighting a checkpoint was
trained under is recoverable from the checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .labels import LABELS_14, NUM_LABELS


class MaskedMultiLabelBCE(nn.Module):
    """`BCEWithLogitsLoss(pos_weight, reduction="none")`, shaped for upstream's masking lines."""

    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is None:
            weight = None
        else:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            if weight.numel() != NUM_LABELS:
                raise ValueError(f"pos_weight has {weight.numel()} entries, expected {NUM_LABELS}")
        self.register_buffer("pos_weight", weight)

    def forward(self, logits, target):
        """`[B, 14]` unreduced loss. `logits` are raw -- never pre-sigmoided."""
        if logits.shape[-1] != NUM_LABELS:
            raise ValueError(f"expected {NUM_LABELS} logits, got {logits.shape[-1]}")
        weight = self.pos_weight
        return nn.functional.binary_cross_entropy_with_logits(
            logits.float(), target.float(),
            pos_weight=None if weight is None else weight.to(logits.device),
            reduction="none")


def build_class_loss(config, label_table, train_study_keys):
    """The loss, plus the `pos_weight` vector it was built with, for the checkpoint and the log."""
    from .labels import pos_weight

    weights, counts, total = pos_weight(label_table, train_study_keys,
                                        cap=config["model"]["pos_weight_cap"])
    return MaskedMultiLabelBCE(weights), {
        "pos_weight": weights,
        "n_positive": counts,
        "n_train_studies": total,
        "cap": config["model"]["pos_weight_cap"],
        "labels": list(LABELS_14),
    }


### Metrics ###########################################################################################

def pathology_metrics(logits, targets, mask):
    """Macro/micro AUROC and average precision, plus per-label values where they are defined.

    A label with no positives or no negatives in the evaluated subset has no AUROC and no AP; it is
    reported as `None` and left out of the macro average rather than being scored as 0.5 or 0, which
    would silently reward a subset that happens to be one-sided. `n_labels_scored` says how many
    survived, so a macro number is never read without knowing what it averaged.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    keep = mask.all(axis=1) if mask.ndim == 2 else mask.astype(bool)
    logits, targets = logits[keep], targets[keep]
    out = {"n_samples_scored": int(keep.sum()), "per_label": {}}
    if len(targets) == 0:
        return {**out, "auroc_macro": None, "auroc_micro": None,
                "ap_macro": None, "ap_micro": None, "n_labels_scored": 0}

    probs = 1.0 / (1.0 + np.exp(-logits))
    aurocs, aps = [], []
    for i, name in enumerate(LABELS_14):
        y, p = targets[:, i], probs[:, i]
        positives = int(y.sum())
        entry = {"n_positive": positives, "prevalence": float(y.mean())}
        if 0 < positives < len(y):
            entry["auroc"] = float(roc_auc_score(y, p))
            entry["ap"] = float(average_precision_score(y, p))
            aurocs.append(entry["auroc"])
            aps.append(entry["ap"])
        else:
            entry["auroc"] = entry["ap"] = None
        out["per_label"][name] = entry

    flat_y, flat_p = targets.ravel(), probs.ravel()
    micro_defined = 0 < flat_y.sum() < flat_y.size
    return {
        **out,
        "auroc_macro": float(np.mean(aurocs)) if aurocs else None,
        "auroc_micro": float(roc_auc_score(flat_y, flat_p)) if micro_defined else None,
        "ap_macro": float(np.mean(aps)) if aps else None,
        "ap_micro": float(average_precision_score(flat_y, flat_p)) if micro_defined else None,
        "n_labels_scored": len(aurocs),
    }
