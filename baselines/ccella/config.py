"""One YAML for this adaptation, and the upstream `argparse.Namespace` built out of it.

Upstream CCELLA is configured by three JSON files (`*_env.json`, `*_config.json`, `*_def.json`)
merged into a flat `argparse.Namespace` by `scripts.diff_model_setting.load_config`, which
`scripts.utils.define_instance` then reads `_target_` blocks out of. **That mechanism is kept**:
`upstream_namespace` returns exactly such a namespace, so `define_instance(args, "noise_scheduler")`
and `define_instance(args, "diffusion_unet_def")` work unmodified. What changes is where the values
come from -- one YAML here instead of three JSONs there -- because this adaptation adds knobs
(checkpoint schedule, validation interval, W&B, cache root) that upstream has no field for.

The `_target_` blocks themselves live in `configs/ccella_def.json`, a copy of upstream's
`configs/CCELLA_def.json` with three values changed and nothing else. The diff is in the README.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_JSON = os.path.join(HERE, "configs", "ccella_def.json")


class ConfigError(ValueError):
    """A config that would train the wrong thing, caught before anything is allocated."""


# Every key a config may set, with its default. An unknown key raises rather than being ignored --
# a typo in a YAML is otherwise a silently different run.
DEFAULTS = {
    "upstream_root": "/hnvme/workspace/y100dc19-mrflow-final/baselines/upstream/CCELLA",
    "upstream_commit": "619dfbb6f5c20eaa6724a970485eb0a95c22146f",

    "mrrate": {
        "raw_root": "/hnvme/workspace/y100dc19-MR-Rate-raw",
        "labels_root": ("/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE/"
                        "contrastive-pretraining/scripts/eval_labels/splits_merged_majority"),
        "max_repeats": None,
        "max_series_train": None,
        "max_series_val": 2000,
        "max_series_test": None,
        "seed": 42,
    },

    # Upstream's own preprocessing constants. `percentile_lower/upper` and `clip` are
    # ScaleIntensityRangePercentilesd's arguments in `diff_model_create_training_data.create_transforms`;
    # the grid is the "already resampled to the desired (fixed) image size" precondition its README
    # states, which for MR-RATE has to be made explicit because the archives are native-space.
    "volume": {
        "grid": [256, 256, 224],      # RAS (L-R, P-A, I-S) voxels -- measured, see the README
        "spacing_mm": [1.0, 1.0, 1.0],
        "percentile_lower": 0.0,      # upstream: lower=0
        "percentile_upper": 99.5,     # upstream: upper=99.5
        "clip": True,
        # Applied AFTER the resample, so on the 1 mm grid it is also a threshold in mm. Matches
        # MRFlow's `mri.preprocess.min_slices`. Judging a volume on its NATIVE slice count instead
        # discards ~a third of MR-RATE, because a 26-slice 6 mm stack is 156 mm of anatomy.
        "min_extent_voxels": 32,
    },

    "text": {
        # "__zeros__" is a test-only stub encoder that needs no download and writes zeros. It is
        # part of the cache fingerprint, so a cache built with it can never be trained on by a
        # config naming a real encoder.
        "encoder": "google/flan-t5-xxl",
        # Where those weights actually live. `encoder` stays the portable identity that goes into
        # the cache fingerprint, so a cache does not become invalid by moving the files.
        "encoder_path": "/hnvme/workspace/y100dc19-mrflow-final/models/flan-t5-xxl",
        "hidden_size": 4096,          # FLAN-T5-XXL d_model; must equal diffusion_unet_def.input_dim
        "max_length": 512,
        "truncation_side": "left",
        "padding": "max_length",
        "sections": ["findings", "impression"],
    },

    "data": {
        "cache_root": "/hnvme/workspace/y100dc19-mrflow-final/baselines/ccella_cache",
        "num_workers": 8,
        "prefetch_factor": 2,
    },

    "model": {
        "autoencoder_path": ("/hnvme/workspace/y100dc19-mrflow-final/baselines/upstream/"
                             "Text2CT/models/autoencoder_epoch273.pt"),
        "latent_channels": 4,
        "pos_weight_cap": 20.0,
        "text_class_pred_weight": 1.0e-4,   # upstream's lambda, unchanged
        # null keeps the def JSON's value (upstream's `true`). Upstream raises when it is true and
        # no GPU is present, so the CPU tests set it false; a GPU run never touches this.
        "use_flash_attention": None,
    },

    "train": {
        "max_train_steps": 60000,
        "checkpoint_steps": [20000, 40000, 50000, 55000, 60000],
        "micro_batch_size": 4,
        "lr": 1.0e-4,
        "num_train_timesteps": 1000,
        "seed": 42,
        "log_every": 10,
        "output_dir": "/hnvme/workspace/y100dc19-mrflow-final/baselines/ccella_runs/ccella_mrrate",
    },

    "validation": {
        "every": 2500,
        "samples": 4,
        "subset": 256,
        "visualize": True,
        "inference_steps": 50,
        "fps": 16,
    },

    "wandb": {
        "project": "MRFlow",
        "group": "baselines",
        "name": "ccella_mrrate",
        "id": None,
        "mode": "online",
    },
}


def _merge(base, override, path=""):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        where = f"{path}.{key}" if path else key
        if key not in out:
            raise ConfigError(f"unknown config key {where!r}")
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = value
    return out


def load_config(path, overrides=None):
    """Read a YAML config, merge it onto `DEFAULTS`, apply dotted overrides, and validate."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    config = _merge(DEFAULTS, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node, *rest = key.split(".")
        target = config
        for part in [node] + rest[:-1]:
            if part not in target:
                raise ConfigError(f"unknown config key {key!r}")
            target = target[part]
        leaf = rest[-1] if rest else node
        if leaf not in target:
            raise ConfigError(f"unknown config key {key!r}")
        target[leaf] = yaml.safe_load(value)
    validate(config)
    return config


def validate(config):
    """Everything that must hold before a run is allowed to allocate anything."""
    grid = config["volume"]["grid"]
    if len(grid) != 3:
        raise ConfigError("volume.grid must be three voxel counts")
    # MAISI's autoencoder downsamples by 4, then the UNet's three downsample stages by 8 more.
    for size in grid:
        if size % 32:
            raise ConfigError(
                f"volume.grid entry {size} is not divisible by 32: the MAISI autoencoder "
                f"downsamples 4x and the UNet a further 8x, so a non-multiple loses voxels at a "
                f"stage boundary and the decoder returns a different shape than it was given")
    validate_checkpoint_steps(config["train"]["checkpoint_steps"],
                              config["train"]["max_train_steps"])
    every = config["validation"]["every"]
    if every <= 0:
        raise ConfigError("validation.every must be positive")
    if config["wandb"]["mode"] not in ("online", "offline", "disabled"):
        raise ConfigError(f"unknown wandb.mode {config['wandb']['mode']!r}")
    if config["model"]["pos_weight_cap"] <= 1:
        raise ConfigError("model.pos_weight_cap must exceed 1 to be a cap rather than a floor")
    with open(DEF_JSON) as handle:
        input_dim = json.load(handle)["diffusion_unet_def"]["input_dim"]
    if input_dim != config["text"]["hidden_size"]:
        raise ConfigError(
            f"text.hidden_size {config['text']['hidden_size']} does not match the adapter's "
            f"input_dim {input_dim} in {DEF_JSON}: the ELLA resampler projects the text encoder's "
            f"hidden width, so the two must agree or the first Linear sees the wrong shape")


# The two schedules this adaptation is specified for. A different `max_train_steps` is not guessed
# at -- the config has to say what it wants, because a schedule silently derived from a step count
# is a hyper-parameter nobody chose.
STANDARD_SCHEDULES = {
    60000: [20000, 40000, 50000, 55000, 60000],
    120000: [20000, 40000, 60000, 80000, 100000, 110000, 120000],
}


def validate_checkpoint_steps(steps, max_train_steps):
    """Strictly increasing, inside the run, and ending on the last step."""
    if not steps:
        raise ConfigError("train.checkpoint_steps is empty")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ConfigError(f"train.checkpoint_steps is not strictly increasing: {steps}")
    if any(s <= 0 for s in steps):
        raise ConfigError(f"train.checkpoint_steps must be positive: {steps}")
    if steps[-1] > max_train_steps:
        raise ConfigError(
            f"train.checkpoint_steps ends at {steps[-1]}, past max_train_steps {max_train_steps}")
    if steps[-1] != max_train_steps:
        raise ConfigError(
            f"train.checkpoint_steps must include the final step {max_train_steps}; got {steps[-1]}")
    standard = STANDARD_SCHEDULES.get(max_train_steps)
    if standard is None and steps != sorted(set(steps)):
        raise ConfigError("nonstandard max_train_steps needs an explicit, sorted schedule")
    return list(steps)


def upstream_namespace(config, split="train"):
    """The flat `argparse.Namespace` upstream's `define_instance` reads `_target_` blocks out of.

    Only the keys upstream actually consumes are set. Everything this adaptation adds lives in the
    YAML and is passed explicitly, so a namespace field never silently shadows a config value.
    """
    with open(DEF_JSON) as handle:
        model_def = json.load(handle)

    flash = config["model"]["use_flash_attention"]
    if flash is not None:
        model_def["diffusion_unet_def"]["use_flash_attention"] = bool(flash)

    args = argparse.Namespace()
    for key, value in model_def.items():
        setattr(args, key, value)

    # Upstream env/config fields that `load_unet_text` and `calculate_scale_factor` touch.
    args.existing_ckpt_filepath = None
    args.resume = False
    args.use_pretrained_unet = False
    args.pretrained_unet_path = None
    args.freeze_unet = False
    args.trained_autoencoder_path = config["model"]["autoencoder_path"]
    args.diffusion_unet_train = {
        "batch_size": config["train"]["micro_batch_size"],
        "lr": config["train"]["lr"],
        "imkey": "image",
        "num_workers": config["data"]["num_workers"],
        "cache_rate": 0.0,
    }
    args.diffusion_unet_inference = {
        "dim": list(config["volume"]["grid"]),
        "spacing": list(config["volume"]["spacing_mm"]),
        "num_inference_steps": config["validation"]["inference_steps"],
        "random_seed": config["train"]["seed"],
    }
    return args
