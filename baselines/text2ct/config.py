"""Configuration for the MR-RATE fine-tune: defaults, YAML merge, validation.

Deliberately a plain nested dict rather than a schema class -- it is written verbatim into every
checkpoint, so it has to round-trip through JSON without a custom decoder.
"""

from __future__ import annotations

import copy
import json
import os

DEFAULTS = {
    # --- the upstream release ---
    "text2ct_root": None,                      # the pinned clone; required
    "model_def": "configs/config_rflow.json",  # read for the UNet, VAE and scheduler definitions
    "weights": {
        "unet": "models/unet_rflow_200ep.pt",
        "vae": "models/autoencoder_epoch273.pt",
        "clip": "models/CLIP3D_Finding_Impression_30ep.pt",
    },

    # --- MR-RATE ---
    "mrrate": {
        "raw_root": None,                      # required for prepare_data.py; unused when training
        "max_repeats": 1,                      # drop duplicate acquisitions of one contrast+plane
        "max_series_train": None,
        "max_series_val": None,
        "seed": 42,                            # list_series shuffle seed; must match MRFlow's
    },

    # The target grid. Fixed, because Text2CT's UNet is monolithic 3D, and chosen to reproduce the
    # released model's own latent shape -- 512 x 512 x 128 -> (4, 128, 128, 32), exactly what
    # `scripts/diff_model_create_training_data.py:181` logs for a CT. The spacing is not the CT's
    # 0.75/0.75/3.0: that would be a 384 mm field of view, ~40% air for a brain. 0.5/0.5/1.5 puts
    # the same latent shape over 256 x 256 x 192 mm. See the README's grid table.
    "volume": {
        "inplane_mm": 0.5,
        "slice_mm": 1.5,
        "inplane_size": 512,
        "num_slices": 128,
        "posterior_shift_mm": 15.0,
        "percentiles": [0.5, 99.5],
        "min_native_slices": 32,
    },

    "data": {
        "cache_root": None,                    # required
        "report_sections": ["findings", "impression"],
        # Off: the conditioning string is then exactly upstream's
        # "Findings: ... Impression: ...", and modality reaches the model only through the UNet's
        # class embedding. True additionally names the contrast in words -- an experiment, not the
        # design. Plane is not conditioned at all; see the README.
        "modality_prefix": False,
        "text_max_length": 512,
        "limit_train": None,
        "limit_val": 512,
        "num_workers": 8,
        "ignore_cache_meta": False,
        # Off: with cached embeddings the 3.1 GB tower is never resident, which is a stronger
        # guarantee than freezing it. Turn on to exercise the freezing assertions against it.
        "load_text_encoder": False,
    },

    "model": {
        "modality_init": "ct_row",             # ct_row | zeros | keep -- see model.seed_modality_embeddings
        # CT-derived latent scaling, kept under a fine-tune for the same reason MRFlow keeps FLUX's
        # factors: the trunk was trained with exactly this number.
        "scale_factor": 1.0287,
        "recompute_scale_factor": False,
        # Concatenate the MAISI autoencoder's activations on the GPU instead of through host
        # memory. Bit-identical output, 53.8x faster at this grid -- see model.set_fast_maisi_concat.
        "vae_fast_concat": True,
    },

    "train": {
        "max_steps": 60000,                    # optimizer updates, matched to MRFlow's fine-tune
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "optimizer": "adamw",                  # adamw | adam (upstream uses adam, no decay)
        "lr": 1.0e-4,
        "weight_decay": 1.0e-2,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "loss": "l1",                          # upstream's objective
        "scheduler": "cosine",                 # cosine | poly | constant
        "warmup_steps": 500,
        "min_lr": 1.0e-6,
        "poly_power": 2.0,                     # only for scheduler=poly (upstream's PolynomialLR)
        "precision": "bf16",                   # bf16 | fp16 | fp32
        "max_grad_norm": 1.0,
        "max_consecutive_skipped_steps": 25,
        "seed": 42,
        "log_every": 20,
        "val_every": 2000,
        "val_batches": 32,
        "checkpoint_every": 2000,
        "keep_checkpoints": 3,
        "num_workers": 8,
        "ddp_find_unused_parameters": True,    # as upstream; set False for a small speedup
    },

    # Classifier-free guidance. Report CFG is Text2CT's own and its defaults are the release's.
    "cfg": {
        "report_dropout_prob": 0.1,            # config_diff_model.json: conditional_free_guidance
        "guidance_scale": 5.0,                 # environment_diff_model_*.json, sampling only
        # Experimental, off: modality is required acquisition metadata, not optional semantics.
        "modality_dropout_prob": 0.0,
        "modality_guidance_scale": 1.0,
    },

    # Weights & Biases. `id` is fixed rather than generated, so a continuation job that resumes
    # from a checkpoint resumes the same run instead of starting a second one -- the same thing
    # MRFlow does by reading `wandb_args.id` back out of its saved config.
    "wandb": {
        "mode": "online",                      # online | offline | disabled
        "project": "MRFlow",                   # alongside MRFlow's own runs, which is the point
        "group": "baselines",
        "name": "text2ct_mrrate_ft",
        "id": "text2ct_mrrate_ft",
        "tags": ["text2ct", "mrrate", "baseline"],
    },

    "output_dir": None,                        # required
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in out:
            raise KeyError(f"unknown config key {key!r}. Known: {sorted(out)}")
        out[key] = _merge(out[key], value) if isinstance(out[key], dict) and \
            isinstance(value, dict) else value
    return out


def load_config(path=None, overrides=None):
    """DEFAULTS <- YAML file <- explicit overrides. Unknown keys raise."""
    layer = {}
    if path:
        import yaml
        with open(path) as handle:
            layer = yaml.safe_load(handle) or {}
    config = _merge(DEFAULTS, layer)
    return _merge(config, overrides or {})


def apply_dotted(config, assignments):
    """`--set train.lr=5e-5 data.limit_val=64` -> a config override dict."""
    for item in assignments or []:
        key, _, raw = item.partition("=")
        node, *rest = key.split(".")
        target, path = config, [node] + rest
        for part in path[:-1]:
            if part not in target:
                raise KeyError(f"unknown config key {key!r}")
            target = target[part]
        if path[-1] not in target:
            raise KeyError(f"unknown config key {key!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target[path[-1]] = value
    return config


def weight_paths(config):
    """The three checkpoints as absolute paths."""
    root = config["text2ct_root"]
    if not root:
        raise ValueError("config.text2ct_root is required (the pinned upstream clone)")
    return {name: path if os.path.isabs(path) else os.path.join(root, path)
            for name, path in config["weights"].items()}


def validate(config, require_data=True):
    """Fail on the mistakes that would otherwise surface hours in."""
    problems = []
    if not config["text2ct_root"]:
        problems.append("text2ct_root is required")
    if require_data and not config["data"]["cache_root"]:
        problems.append("data.cache_root is required")
    if not config["output_dir"]:
        problems.append("output_dir is required")

    volume = config["volume"]
    # The UNet has four levels, so its latent must be divisible by 8; the VAE compresses 4x per
    # axis. Every image axis therefore has to be a multiple of 32 -- verified empirically on the
    # same architecture in R2V-MR-Generation/data/geometry.py (UNET_SPATIAL_MULTIPLE).
    for axis, size in (("inplane_size", volume["inplane_size"]),
                       ("num_slices", volume["num_slices"])):
        if size % 32:
            problems.append(f"volume.{axis}={size} must be a multiple of 32")
    problems += check_input_problems(volume)

    train = config["train"]
    if train["precision"] not in ("bf16", "fp16", "fp32"):
        problems.append(f"train.precision={train['precision']!r} must be bf16, fp16 or fp32")
    if train["scheduler"] not in ("cosine", "poly", "constant"):
        problems.append(f"train.scheduler={train['scheduler']!r} must be cosine, poly or constant")
    if train["optimizer"] not in ("adamw", "adam"):
        problems.append(f"train.optimizer={train['optimizer']!r} must be adamw or adam")
    if train["loss"] not in ("l1", "l2"):
        problems.append(f"train.loss={train['loss']!r} must be l1 or l2")
    if train["gradient_accumulation_steps"] < 1 or train["micro_batch_size"] < 1:
        problems.append("micro_batch_size and gradient_accumulation_steps must be >= 1")

    if not 0.0 <= config["cfg"]["report_dropout_prob"] <= 1.0:
        problems.append("cfg.report_dropout_prob must be in [0, 1]")
    if config["model"]["modality_init"] not in ("ct_row", "zeros", "keep"):
        problems.append("model.modality_init must be ct_row, zeros or keep")
    if config["wandb"]["mode"] not in ("online", "offline", "disabled"):
        problems.append(f"wandb.mode={config['wandb']['mode']!r} must be online, offline or disabled")

    if problems:
        raise ValueError("invalid configuration:\n  " + "\n  ".join(problems))
    return config


def check_input_problems(volume):
    """`scripts/sample.py:25`'s `check_input`, restated so a grid the released sampler would reject
    fails at config time rather than at generation time.

    That validator is not called on the training path, so nothing forces a fine-tune to satisfy it.
    It is enforced anyway: a baseline that generates on a grid its own released interface refuses is
    not that baseline, and the constraint is cheap to honour.
    """
    size = (volume["inplane_size"], volume["inplane_size"], volume["num_slices"])
    spacing = (volume["inplane_mm"], volume["inplane_mm"], volume["slice_mm"])
    problems = []
    if size[0] not in (256, 384, 512):
        problems.append(f"volume.inplane_size={size[0]} is not in [256, 384, 512] "
                        f"(sample.py check_input)")
    if size[2] not in (128, 256, 384, 512, 640, 768):
        problems.append(f"volume.num_slices={size[2]} is not in [128, 256, 384, 512, 640, 768] "
                        f"(sample.py check_input)")
    if not 0.5 <= spacing[0] <= 3.0:
        problems.append(f"volume.inplane_mm={spacing[0]} is outside [0.5, 3.0] "
                        f"(sample.py check_input)")
    if not 0.5 <= spacing[2] <= 5.0:
        problems.append(f"volume.slice_mm={spacing[2]} is outside [0.5, 5.0] "
                        f"(sample.py check_input)")
    return problems


def grid_spacing(config):
    """The output grid's spacing in the volume's own (X, Y, Z) order -- what `spacing_tensor`
    describes, and what the saved NIfTI's affine has to carry."""
    volume = config["volume"]
    return (volume["inplane_mm"], volume["inplane_mm"], volume["slice_mm"])


def latent_shape(config, latent_channels=4, divisor=4):
    """The cached latent's shape. `divisor` is the VAE's compression, 4 per axis."""
    volume = config["volume"]
    return (latent_channels, volume["inplane_size"] // divisor, volume["inplane_size"] // divisor,
            volume["num_slices"] // divisor)


def effective_batch_size(config, world_size):
    train = config["train"]
    return train["micro_batch_size"] * train["gradient_accumulation_steps"] * world_size
