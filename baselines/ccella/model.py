"""CCELLA with a 14-label pathology head and an explicit modality control. A subclass, nothing more.

`CCELLAMRRate` is the only model code in this baseline. It subclasses upstream's `CCELLA_LDM`
unchanged, calls its constructor with upstream's own arguments, and then replaces **three** things.
Nothing is reimplemented: the perceiver-resampler cascade, the six-block structure, the adapter
tokens handed to cross-attention, the MAISI U-Net and every one of its blocks are upstream's.

    1. the six `ClassAdaptor` output layers, 2 -> 14
       Upstream's `PerceiverResamplerCascade.__init__` constructs `ClassAdaptor()` with its
       defaults (`text_ldm.py:288`), so `num_classes` cannot be threaded in through the
       constructor. The fusion classifier *is* parameterised (`nn.Linear(layers*num_classes,
       num_classes)`, `text_ldm.py:961`), so passing `num_classes=14` already sizes it correctly
       and only the six heads need replacing.

    2. `unet.pirads_layer`, input width 2 -> 14
       `diffusion_model_unet_maisi_prostate.py:160` builds it as
       `self._create_embedding_module(2, time_embed_dim)` with a literal `2`. It is rebuilt here by
       calling that same method with 14, so the module is upstream's own construction, not a copy.

    3. `forward`, softmax -> sigmoid, plus `modality_id` passed through as `class_labels`
       The 14 groups are not mutually exclusive, so upstream's `torch.softmax(...)` would force
       them onto a simplex and make the conditioning vector say "which one" instead of "which
       ones". The `* 1e2` scale is kept exactly (see the note on it below), as is the position in
       the pipeline: the vector still goes in as `pirads=`, so it reaches `_get_input_embeddings`
       at the same point with the same concatenation order.

**The conditioning order is upstream's, unchanged.** `DiffusionModelUNetMaisi.forward` already does
`_get_time_and_class_embedding` (timestep embedding, then `emb += class_embedding(class_labels)`)
and then `_get_input_embeddings` (`cat(emb, pirads_layer(pirads))`, then
`cat(..., spacing_layer(spacing))`). Modality-added-to-timestep, then pathology, then spacing is
therefore what upstream already implements once `num_class_embeds` is set; this file supplies the
ids and never touches that method.

**On the `* 1e2` scale.** Upstream multiplies the predicted class vector by 100 before the U-Net,
matching the `* 1e2` its dataloader applies to `pirads` and `spacing`
(`diff_model_train_all.py:122`) so all three inputs to `_create_embedding_module` share a scale.
Sigmoid has the same `[0, 1]` range as softmax, so the factor transfers unchanged and is kept.
"""

from __future__ import annotations

import torch
from torch import nn

from .labels import NUM_LABELS
from .upstream import upstream_module

# The authoritative table, shared with MRFlow rather than restated. Ids may never move.
from echosyn.common.mrrate import MODALITY_TO_ID, NUM_MODALITY_CLASSES  # noqa: E402

CCELLA_LDM = upstream_module("scripts.text_ldm").CCELLA_LDM

CONDITION_SCALE = 1e2  # upstream's, see the module docstring


class CCELLAMRRate(CCELLA_LDM):
    """Upstream `CCELLA_LDM` with 14 non-exclusive pathology labels and a modality class input."""

    def __init__(self, *args, num_classes: int = NUM_LABELS,
                 num_class_embeds: int = NUM_MODALITY_CLASSES, **kwargs):
        if num_classes != NUM_LABELS:
            raise ValueError(f"num_classes must be {NUM_LABELS} (the 14 merged groups), "
                             f"got {num_classes}")
        if num_class_embeds != NUM_MODALITY_CLASSES:
            raise ValueError(f"num_class_embeds must be {NUM_MODALITY_CLASSES}, the size of "
                             f"echosyn.common.mrrate.MODALITY_TO_ID, got {num_class_embeds}")
        super().__init__(*args, num_classes=num_classes,
                         num_class_embeds=num_class_embeds, **kwargs)

        # (1) the six per-block heads. `hstack` of six 14-vectors is what `self.classifier`
        #     (already `Linear(6*14, 14)`) consumes, so the fusion width follows automatically.
        for block in self.ella.connector.cascade_blocks:
            block.fc2 = nn.Linear(block.fc2.in_features, num_classes)

        # (2) the U-Net's projection of that vector, rebuilt by upstream's own factory method.
        time_embed_dim = self.unet.block_out_channels[0] * 4
        self.unet.pirads_layer = self.unet._create_embedding_module(num_classes, time_embed_dim)

        self.num_classes = num_classes

    def forward(self, x, timesteps, spacing_tensor, text_encoding, modality_id=None):
        """Upstream's forward with sigmoid in place of softmax and modality as `class_labels`.

        `class_pred_pre` -- the raw logits -- is what comes back for the classification loss, and
        `class_pred` is what conditions the U-Net. They are the same tensor before and after an
        activation, so the diffusion gradient flows back through the pathology head: nothing is
        detached here, exactly as upstream does not detach.
        """
        if modality_id is None:
            raise ValueError("modality_id is required: the U-Net was built with "
                             "num_class_embeds set, so class_labels may not be None")
        context, class_hidden = self.ella(text_encoding, timesteps)
        class_pred_pre = self.classifier(class_hidden)
        class_pred = torch.sigmoid(class_pred_pre) * CONDITION_SCALE
        x = self.unet(
            x=x,
            context=context,
            timesteps=timesteps,
            class_labels=modality_id,
            pirads=class_pred,
            spacing_tensor=spacing_tensor,
        )
        return x, class_pred_pre


def build_model(config, device):
    """The model, via upstream's own `define_instance`, so the def JSON is read the way it is there."""
    from .config import upstream_namespace

    define_instance = upstream_module("scripts.utils").define_instance
    args = upstream_namespace(config)
    return define_instance(args, "diffusion_unet_def").to(device)


def modality_ids(names):
    """Modality strings -> a `long` tensor of ids, through MRFlow's own table."""
    from echosyn.common.mrrate import modality_to_id

    return torch.tensor([modality_to_id(n) for n in names], dtype=torch.long)


__all__ = ["CCELLAMRRate", "build_model", "modality_ids", "MODALITY_TO_ID", "CONDITION_SCALE"]
