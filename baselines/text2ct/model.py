"""Building Text2CT's three components, and keeping two of them frozen.

The definitions are read out of the upstream repo's own `configs/config_rflow.json` -- never
restated here -- so this file cannot drift from the checkpoint it loads. What it adds is:

* strict weight loading with a readable error instead of a silent `strict=False` (upstream's
  `scripts/diff_model_train.py:165` loads the UNet with `strict=False`, which would swallow a real
  architecture mismatch);
* seeding the MR rows of the UNet's `class_embedding` so step 0 reproduces the pretrained model;
* the freezing contract for the VAE and the text encoder -- eval mode, `requires_grad=False`,
  `inference_mode` forwards, out of the optimizer, and checksummed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

from .mrrate_data import MODALITY_TO_ID, NUM_CLASS_EMBEDS, PRETRAINED_CT_CLASS_ID


def text2ct_namespace(text2ct_root, model_def="configs/config_rflow.json"):
    """The upstream model definition as the `Namespace` its own `define_instance` expects.

    Mirrors `scripts/diff_model_setting.py:51`'s `load_config`, minus the two configs that only
    carry training and inference settings -- those are this package's config file instead.
    """
    path = model_def if os.path.isabs(model_def) else os.path.join(text2ct_root, model_def)
    with open(path) as handle:
        model = json.load(handle)
    args = argparse.Namespace()
    for key, value in model.items():
        setattr(args, key, value)
    return args


def _define_instance(args, key, text2ct_root=None):
    """Upstream's own `scripts/utils.py:225`, imported rather than restated.

    Importing costs a `sys.path` entry and pulls in MONAI's transform stack, which is why it is done
    lazily. It is worth it: this is the function that turns `config_rflow.json` into the UNet, the
    VAE and the scheduler, so reusing it is what makes "the model is Text2CT's" literally true
    rather than a claim about a copy.
    """
    if text2ct_root:
        _add_upstream_to_path(text2ct_root)
    try:
        from scripts.utils import define_instance
    except ImportError as error:
        raise ImportError(
            f"cannot import Text2CT's scripts.utils: {error}. The upstream clone must be on "
            f"sys.path -- pass text2ct_root, or see baselines/text2ct/README.md for the venv."
        ) from error
    return define_instance(args, key)


def _add_upstream_to_path(text2ct_root):
    root = os.path.abspath(text2ct_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


### Weights ###


class WeightsMissing(FileNotFoundError):
    """A required checkpoint is not on disk."""


def _require(path, what):
    if not path or not os.path.exists(path):
        raise WeightsMissing(
            f"{what} checkpoint not found at {path!r}. Download the three files from "
            f"huggingface.co/dmolino/text2ct-weights into <text2ct_root>/models/.")
    return path


def _unwrap(checkpoint):
    """Upstream saves either a bare `state_dict` (the release) or a dict with `unet_state_dict`
    (its own training checkpoints). Handle both, and say which was found."""
    if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint:
        return checkpoint["unet_state_dict"], checkpoint
    return checkpoint, {}


def load_strict(module, state_dict, what):
    """Load with `strict=True` and a message naming the first few offending keys.

    Never widened to `strict=False`: the released UNet matches the config exactly (611 tensors),
    so any mismatch is a config divergence, not a version skew.
    """
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{what} checkpoint does not match the model definition.\n"
            f"  missing ({len(missing)}): {list(missing)[:6]}\n"
            f"  unexpected ({len(unexpected)}): {list(unexpected)[:6]}")
    return module


### The UNet, and its modality rows ###

MODALITY_INIT_MODES = ("ct_row", "zeros", "keep")


def seed_modality_embeddings(unet, class_ids, mode="ct_row"):
    """Give the MR class ids a starting vector, inside the UNet, before any training.

    Text2CT's UNet already carries MAISI's class-conditioning mechanism -- `num_class_embeds: 128`
    builds `nn.Embedding(128, time_embed_dim)` whose lookup is **added to the timestep embedding**
    (`monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py:319-323`), which is
    exactly the modality-conditioning hook this adaptation needs. So nothing is added to the
    architecture: the MR modalities take MAISI's own MR indices, and `class_embedding` trains with
    the rest of the UNet.

    What does need a decision is what those rows start at. Text2CT only ever trained id 1 (`ct`),
    so the MR rows hold an untrained init -- measured on the release, `||row_1|| = 2.18` against
    `0.07` for an unused row, i.e. near-zero but not zero.

    * `ct_row` (default) copies the trained CT row into every MR id. At step 0 the model is then
      bit-identical to the released model conditioned on CT, which is the strongest possible
      "loading the pretrained UNet preserves its original behavior".
    * `zeros` removes the class contribution entirely. Also stable, but it is *not* the pretrained
      function, since the pretrained one always added row 1.
    * `keep` leaves the checkpoint's own untrained rows.
    """
    if mode not in MODALITY_INIT_MODES:
        raise ValueError(f"modality_init must be one of {MODALITY_INIT_MODES}, got {mode!r}")
    if mode == "keep":
        return mode
    with torch.no_grad():
        source = unet.class_embedding.weight[PRETRAINED_CT_CLASS_ID].clone()
        for class_id in sorted(set(class_ids)):
            if class_id == PRETRAINED_CT_CLASS_ID:
                continue
            unet.class_embedding.weight[class_id] = source if mode == "ct_row" else 0.0
    return mode


def build_unet(text2ct_root, unet_ckpt, device, modality_init="ct_row",
               class_ids=tuple(MODALITY_TO_ID.values()), model_def="configs/config_rflow.json"):
    """The trainable component: Text2CT's 3D diffusion UNet with its released weights."""
    args = text2ct_namespace(text2ct_root, model_def)
    if args.diffusion_unet_def.get("num_class_embeds") != NUM_CLASS_EMBEDS:
        raise ValueError(
            f"diffusion_unet_def.num_class_embeds is "
            f"{args.diffusion_unet_def.get('num_class_embeds')}, but the modality vocabulary in "
            f"mrrate_data.py assumes {NUM_CLASS_EMBEDS}. Class ids would be out of range.")
    if max(class_ids) >= NUM_CLASS_EMBEDS:
        raise ValueError(f"modality id {max(class_ids)} >= num_class_embeds {NUM_CLASS_EMBEDS}")

    unet = _define_instance(args, "diffusion_unet_def", text2ct_root)
    state, extra = _unwrap(torch.load(_require(unet_ckpt, "UNet"), map_location="cpu",
                                      weights_only=False))
    load_strict(unet, state, "UNet")
    seed_modality_embeddings(unet, class_ids, modality_init)
    return unet.to(device), extra


_STOCK_MAISI_CONCAT = None


def set_fast_maisi_concat(enabled=True):
    """Stop (or restore) MONAI's MAISI autoencoder round-tripping its activations through host memory.

    `MaisiGroupNorm3D.forward` and `MaisiConvolution._concatenate_tensors` both branch on
    `max(size) < 500` (autoencoderkl_maisi.py:89 and :214) and, above it, concatenate their
    activations **on the CPU** one chunk at a time, with a `torch.cuda.empty_cache()` and a
    `gc.collect()` between chunks. It is a memory-saving heuristic for small GPUs, and at 512
    in-plane -- exactly Text2CT's grid -- it dominates everything else.

    Measured on one h200 encoding `(1, 1, 512, 512, 128)`:

        stock MONAI                    29.83 s   15.4 GiB
        group-norm concat on GPU        6.51 s   16.5 GiB
        both concats on GPU             0.55 s   16.5 GiB     <- 53.8x, +1.1 GiB

    and `z_mu`/`z_sigma` are **bit-identical** in all three (`max|diff| = 0.0`), because a
    concatenation is a concatenation. Without this, caching the train split is ~5,000 GPU-h instead
    of ~280. The 384-in-plane grid never crosses the threshold, which is why the effect is a cliff
    rather than a slope: 0.45 s at 384, 29.8 s at 512.

    Patching the library is deliberate and is the narrowest available fix -- the threshold is a
    literal inside two methods, with no constructor argument reaching it. It changes no weights and
    no arithmetic. `enabled=False` restores the stock methods, so the two paths can be compared in
    one process (`test_fast_concat_is_bit_identical_to_stock_monai`).
    """
    global _STOCK_MAISI_CONCAT
    from monai.apps.generation.maisi.networks import autoencoderkl_maisi as maisi

    if _STOCK_MAISI_CONCAT is None:
        _STOCK_MAISI_CONCAT = (maisi.MaisiGroupNorm3D._cat_inputs,
                               maisi.MaisiConvolution._concatenate_tensors)
    if not enabled:
        maisi.MaisiGroupNorm3D._cat_inputs, maisi.MaisiConvolution._concatenate_tensors = \
            _STOCK_MAISI_CONCAT
        return False

    def _cat_inputs(self, inputs):
        return torch.cat(inputs, dim=1)

    def _concatenate_tensors(self, outputs, split_size, padding):
        slices = [slice(None)] * 5
        for i in range(self.num_splits):
            slices[self.dim_split + 2] = (slice(None, split_size) if i == 0
                                          else slice(padding, padding + split_size))
            outputs[i] = outputs[i][tuple(slices)]
        return torch.cat(outputs, dim=self.dim_split + 2)

    maisi.MaisiGroupNorm3D._cat_inputs = _cat_inputs
    maisi.MaisiConvolution._concatenate_tensors = _concatenate_tensors
    return True


def build_vae(text2ct_root, vae_ckpt, device, model_def="configs/config_rflow.json",
              fast_concat=True):
    """The frozen volumetric VAE. Byte-identical to MAISI's `autoencoder.pt` and to the file the
    NVIDIA baseline uses, so nothing about it is CT-specific."""
    set_fast_maisi_concat(fast_concat)
    args = text2ct_namespace(text2ct_root, model_def)
    vae = _define_instance(args, "autoencoder_def", text2ct_root)
    load_strict(vae, torch.load(_require(vae_ckpt, "autoencoder"), map_location="cpu",
                                weights_only=True), "autoencoder")
    return freeze(vae.to(device), "vae")


def build_noise_scheduler(text2ct_root, model_def="configs/config_rflow.json"):
    """`RFlowScheduler` exactly as the release configures it: rectified flow, v-prediction,
    uniform timestep sampling with the resolution transform on."""
    return _define_instance(text2ct_namespace(text2ct_root, model_def), "noise_scheduler",
                            text2ct_root)


### Freezing ###


class FrozenGuard:
    """Bookkeeping for the components that must not move, and the assertions that prove they did not.

    Freezing is four separate things and all of them are needed: `eval()` (so dropout and any
    normalization running stats stop updating), `requires_grad_(False)` (so no graph is built),
    keeping the parameters out of the optimizer (so weight decay and momentum cannot touch them),
    and `inference_mode` around the forward (so nothing is recorded at all). The checksum is the
    belt-and-braces check that all four held.
    """

    def __init__(self):
        self.modules = {}
        self._checksums = {}

    def add(self, name, module):
        self.modules[name] = freeze(module, name)
        self._checksums[name] = module_checksum(module)
        return module

    def parameter_ids(self):
        return {id(p) for module in self.modules.values() for p in module.parameters()}

    def assert_eval_mode(self):
        for name, module in self.modules.items():
            training = [n for n, m in module.named_modules() if m.training]
            if training:
                raise AssertionError(f"frozen module {name!r} has submodules in train mode: "
                                     f"{training[:4]}")

    def assert_absent_from(self, optimizer):
        frozen = self.parameter_ids()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if id(param) in frozen:
                    raise AssertionError("a frozen parameter is in the optimizer")

    def assert_no_grads(self):
        for name, module in self.modules.items():
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    raise AssertionError(f"{name}.{param_name} has requires_grad=True")
                if param.grad is not None:
                    raise AssertionError(f"{name}.{param_name} received a gradient")

    def assert_unchanged(self):
        for name, module in self.modules.items():
            now = module_checksum(module)
            if now != self._checksums[name]:
                raise AssertionError(f"frozen module {name!r} changed: "
                                     f"{self._checksums[name]} -> {now}")

    def assert_all(self, optimizer=None):
        self.assert_eval_mode()
        self.assert_no_grads()
        self.assert_unchanged()
        if optimizer is not None:
            self.assert_absent_from(optimizer)


def freeze(module, name=""):
    """eval() + requires_grad_(False) for every parameter, and a `train()` that refuses to undo it.

    Overriding `train` matters: a frozen module nested inside something whose `.train()` is called
    -- by a wrapper, by DDP, by a future refactor -- would silently re-enable dropout and running
    statistics. Here it stays in eval whatever anyone calls.
    """
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    module.train = lambda mode=True, _m=module: _m  # already in eval; keep it there
    module._frozen_name = name
    return module


def module_checksum(module):
    """A cheap, order-stable digest of every parameter and buffer."""
    digest = hashlib.sha256()
    for name, tensor in sorted(list(module.state_dict().items())):
        digest.update(name.encode())
        digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()[:16]


def parameter_report(components):
    """`{name: module}` -> (lines, trainable, frozen). Printed at startup and after resume."""
    lines, trainable_total, frozen_total = [], 0, 0
    for name, module in components.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        trainable_total += trainable
        frozen_total += frozen
        lines.append(f"  {name:<14s} trainable {trainable:>12,}   frozen {frozen:>12,}")
    lines.append(f"  {'TOTAL':<14s} trainable {trainable_total:>12,}   "
                 f"frozen {frozen_total:>12,}")
    return lines, trainable_total, frozen_total
