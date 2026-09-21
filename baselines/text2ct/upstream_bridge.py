"""The updater between this package and the unmodified upstream training code.

Two directions, and both exist to answer one question -- *is our training loop doing what
`scripts/diff_model_train.py` does?* -- with a run rather than an argument.

**Out**: `write_upstream_layout` re-emits a cache in the exact on-disk contract
`diff_model_train.py:508-521` derives (a latent `.nii.gz`, an info `.json`, a cond `.npy` per
series, plus the volume list), and `write_upstream_configs` writes the three JSONs its CLI takes. On
a few hundred series, `python scripts/diff_model_train.py --env_config <generated>` then runs
**verbatim**, with no patch, no import hook and no edit to the pinned clone.

**In**: `as_upstream_batch` renames our batch into the keys upstream's `train_one_epoch` reads, so
that function -- upstream's real code, not a transcription -- can be driven on our dataset and its
loss compared against `fine_tune.diffusion_loss` on the same tensors and the same RNG.

The out direction cannot be the production path: three files per series is 1.73M inodes over the
train split, against an 81,000 hard limit. It is a validation harness, sized for hundreds of series,
not the 575,328 the real run trains on.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import grid_spacing, weight_paths
from .mrrate_data import load_artifact, read_manifest

REPORT_ENCODER_MODEL = "xgem_3D"     # the suffix diff_model_train.py:515 builds the cond path with


def write_upstream_layout(config, split, out_root, limit=None):
    """Our bundled cache -> upstream's three-files-per-series contract. Returns the volume list path.

    The path arithmetic is upstream's, reproduced exactly:

        image_path = join(data_base_dir, "<id>.nii.gz")           resolve_path, :504
        str_img    = image_path.replace(data_base_dir, embedding_base_dir)          :510
        str_info   = str_img + ".json", then .replace("_emb", "")                   :513
        str_cond   = str_info.replace(".nii.gz.json", "_impression_xgem_3D.npy")    :515

    `data_base_dir` holds a **zero-byte placeholder** per series. Upstream only ever calls
    `os.path.exists` on it (`:511`) and never opens it -- the raw volume is not an input to training,
    only its path is. Writing the real volume there would be 34 MB per series for nothing.

    The latent goes out in upstream's own orientation, `(X, Y, Z, C)`
    (`diff_model_create_training_data.py:184`), so `LoadImaged` + `EnsureChannelFirstd` recover
    `(C, X, Y, Z)`. Its affine is unread by the training path -- spacing reaches the model from the
    info JSON -- but it is written correctly anyway.
    """
    import nibabel as nib

    if "_emb" in os.path.abspath(out_root):
        raise ValueError(f"out_root {out_root!r} contains '_emb', which diff_model_train.py:514 "
                         f"strips out of every derived path")
    data_dir = os.path.join(out_root, "dataset")
    embedding_dir = os.path.join(out_root, "embeddings")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    spacing = list(grid_spacing(config))
    affine = np.diag(spacing + [1.0])
    rows = read_manifest(config["data"]["cache_root"], split)[:limit]
    names = []
    for row in rows:
        name = row["sample_id"]
        latent = load_artifact(config["data"]["cache_root"], row, "latent_path")
        embedding = load_artifact(config["data"]["cache_root"], row, "embedding_path")

        open(os.path.join(data_dir, f"{name}.nii.gz"), "wb").close()      # existence-only stub
        nib.save(nib.Nifti1Image(np.float32(np.transpose(latent, (1, 2, 3, 0))), affine=affine),
                 os.path.join(embedding_dir, f"{name}.nii.gz"))
        with open(os.path.join(embedding_dir, f"{name}.nii.gz.json"), "w") as handle:
            json.dump({"spacing": spacing, "impression": ""}, handle)
        np.save(os.path.join(embedding_dir, f"{name}_impression_{REPORT_ENCODER_MODEL}.npy"),
                np.asarray(embedding, dtype=np.float32))
        names.append(f"{name}.nii.gz")

    list_path = os.path.join(out_root, f"{split}_data_volumes.json")
    with open(list_path, "w") as handle:
        json.dump({"training": [{"image": n} for n in names]}, handle, indent=1)
    return list_path


def write_upstream_configs(config, out_root, split="train", n_epochs=1, batch_size=1,
                           model_dir=None):
    """The three JSONs `scripts/diff_model_train.py` takes on the command line.

    `model_def` points at the **clone's own** `config_rflow.json` rather than a copy, so the model
    upstream builds here is the model this package builds. Only the environment and the training
    knobs are generated.
    """
    paths = weight_paths(config)
    model_dir = model_dir or os.path.join(out_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    env = {
        "data_base_dir": os.path.join(out_root, "dataset"),
        "embedding_base_dir": os.path.join(out_root, "embeddings"),
        "json_data_list": os.path.join(out_root, f"{split}_data_volumes.json"),
        "model_dir": model_dir,
        "model_filename": "unet_rflow_mrrate.pt",
        "output_dir": os.path.join(out_root, "predictions"),
        "output_prefix": "unet_rflow",
        "trained_autoencoder_path": paths["vae"],
        "existing_ckpt_filepath": paths["unet"],
        "use_cfg": True,
        "guidance_scale": config["cfg"]["guidance_scale"],
        "report_encoder_model": REPORT_ENCODER_MODEL,
    }
    train = {
        "diffusion_unet_train": {
            "batch_size": batch_size,
            "cache_rate": 0,
            "lr": config["train"]["lr"],
            "n_epochs": n_epochs,
            "n_epochs_total": n_epochs,
            "save_epoch_freq": max(1, n_epochs),
            "freq_to_print": 1,
            "conditional_free_guidance": config["cfg"]["report_dropout_prob"],
            "continue_training_from": None,
        },
        # Unused by training; kept so the file is a drop-in for config_diff_model.json.
        "diffusion_unet_inference": {
            "dim": [config["volume"]["inplane_size"], config["volume"]["inplane_size"],
                    config["volume"]["num_slices"]],
            "spacing": list(grid_spacing(config)),
            "top_region_index": [0, 1, 0, 0], "bottom_region_index": [0, 1, 0, 0],
            "random_seed": config["train"]["seed"], "num_inference_steps": 30, "modality": 1,
        },
    }
    written = {}
    for name, payload in (("environment.json", env), ("config_diff_model.json", train)):
        path = os.path.join(out_root, name)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1)
        written[name] = path
    written["config_rflow.json"] = os.path.join(config["text2ct_root"], config["model_def"])
    return written


def as_upstream_batch(batch):
    """Our batch dict -> the keys `diff_model_train.train_one_epoch` reads.

    Upstream's `image` is the **unscaled** latent (it multiplies by `scale_factor` itself, `:307`),
    its `spacing` is already `x 1e2` (its transform does that at `:111`), and its `cond` is
    `[B, 1, D]`, which it squeezes when 4-dimensional. Ours match all three, so this is a rename.
    """
    return {"image": batch["latent"], "spacing": batch["spacing"], "cond": batch["context"],
            "impression": [""] * len(batch["latent"])}


def upstream_train_one_epoch(text2ct_root, unet, batch, noise_scheduler, scale_factor, device,
                             report_dropout=0.0, amp=False):
    """Run **upstream's own** `train_one_epoch` over one batch and return its mean loss.

    The optimizer is created at `lr=0`, so upstream's `optimizer.step()` runs -- the code path is
    exercised in full -- without moving a weight, and the loss stays comparable to a second pass.
    `include_modality=True` means upstream uses its hardcoded `torch.ones(...)` (CT, class 1); a
    caller comparing against `fine_tune.diffusion_loss` must pass the same class id.
    """
    import logging

    import torch
    from torch.amp import GradScaler

    from .model import _add_upstream_to_path

    _add_upstream_to_path(text2ct_root)
    from scripts.diff_model_train import train_one_epoch

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=1, power=2.0)
    totals = train_one_epoch(
        0, unet, [as_upstream_batch(batch)], optimizer, scheduler, torch.nn.L1Loss(),
        GradScaler("cuda", enabled=amp), scale_factor, noise_scheduler, len(batch["latent"]),
        noise_scheduler.num_train_timesteps, device, logging.getLogger("upstream"), 0,
        amp=amp, freq_to_print=10 ** 9, conditional_free_guidance=report_dropout,
        ct_property_conditions=False, include_modality=True)
    return float(totals[0] / totals[1])
