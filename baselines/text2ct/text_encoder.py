"""Text2CT's frozen 3D-CLIP report encoder.

The encoder is `FrozenCLIP3D` from the upstream repo (`core/models/encoders/clip.py:171`): a CLIP
text tower whose position embeddings were widened to 512 tokens and which was trained
contrastively against a CTViT volume tower. `encode_text` pools, projects and **L2-normalizes**,
returning `[B, 1, 768]` -- one token, which is what `cross_attention_dim: 768` with a single
context token means downstream.

It is frozen here in the strongest sense: never constructed inside the training process at all
unless `data.text_mode = "online"`. Upstream caches report embeddings to `.npy`
(`scripts/save_embeddings_ctrate.py`) and the training script reads them, which is also what makes
the fine-tune affordable -- the tower is 3.1 GB, most of it the volume encoder this pipeline never
calls.

**Two environment shims live here and nowhere else.** The upstream package vendors its own copy of
transformers' CLIP tokenizer and processor, and those copies only construct under roughly
`transformers < 4.34`. Substituting the installed `transformers.CLIPTokenizer` is safe: both read
the same `openai/clip-vit-large-patch14` vocab and merges and produce the same ids (asserted in
`tests/test_finetune.py::test_tokenizer_shim_matches_upstream_vocabulary`), and the processor is
only used by `encode_vision`, which this pipeline never calls.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

CLIP_VERSION = "openai/clip-vit-large-patch14"
TEXT_EMBED_DIM = 768


def build_text_encoder(text2ct_root, clip_ckpt, device="cuda", max_length=512):
    """-> a frozen `FrozenCLIP3D` on `device`, with its weights loaded `strict=True`.

    Raises with an actionable message when the upstream tree or its dependencies are not importable
    -- that is an environment problem, never something to paper over with a random-weight fallback.
    """
    import transformers

    from .model import _add_upstream_to_path, _require, freeze

    _add_upstream_to_path(text2ct_root)
    try:
        from core.cfg_helper import model_cfg_bank
        from core.models.common.get_model import get_model
        import core.models.encoders.clip as clip_module
    except ImportError as error:
        raise ImportError(
            f"cannot import Text2CT's encoder from {text2ct_root!r}: {error}. It needs the upstream "
            f"clone and a venv with transformers 4.x (see baselines/text2ct/README.md); the MRFlow "
            f"venv ships transformers 5.x, where the vendored CLIP does not import."
        ) from error

    # See the module docstring: the vendored tokenizer/processor do not construct on a modern
    # transformers. The text tower itself is the vendored `CLIP3DModel` and is untouched.
    clip_module.CLIPTokenizer = transformers.CLIPTokenizer
    clip_module.CLIPProcessor = transformers.CLIPProcessor

    # `model_cfg_bank` resolves its yaml against a *relative* `configs/model`
    # (`core/cfg_helper.py:105`), so it only works from the upstream root. Point it at the clone
    # instead of chdir-ing, which would be a process-wide side effect.
    bank = model_cfg_bank()
    bank.cfg_dir = os.path.join(_add_upstream_to_path(text2ct_root), "configs", "model")
    cfg = bank("clip_3D")
    cfg.args["max_length"] = max_length
    encoder = get_model()(cfg)
    state = torch.load(_require(clip_ckpt, "3D-CLIP"), map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    return freeze(encoder.to(device), "text_encoder")


@torch.inference_mode()
def encode_reports(encoder, texts, batch_size=16):
    """-> `[N, 1, 768]` float32 on CPU. `inference_mode`, never `no_grad`: nothing is recorded."""
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        out.append(encoder(chunk, "encode_text").float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 1, TEXT_EMBED_DIM)


def null_context(batch_size, device, dtype=torch.float32):
    """The unconditional report branch.

    Zeros, because that is what Text2CT's own classifier-free guidance uses on **both** sides:
    training masks a fraction of samples with `cond_masked[mask_uncond] = 0`
    (`scripts/diff_model_train.py:335-337`) and sampling builds the unconditional branch as
    `torch.zeros_like(impression)` (`scripts/diff_model_infer.py:248`). The released UNet has
    therefore seen a zero context ~10% of the time and no other null exists. A learned null token
    would be a different model.
    """
    return torch.zeros(batch_size, 1, TEXT_EMBED_DIM, device=device, dtype=dtype)
