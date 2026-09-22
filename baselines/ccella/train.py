#!/usr/bin/env python
"""Train CCELLA on MR-RATE. Upstream's training step, driven by global optimizer step.

    python -m baselines.ccella.train --config baselines/ccella/configs/mrrate_ccella.yaml
    torchrun --nproc_per_node 8 -m baselines.ccella.train --config <cfg>

**The step body is upstream's `train_one_epoch`**, called here rather than reimplemented: the noise
draw, the timestep draw, `noise_scheduler.add_noise`, the U-Net call, the L1 objective, the
classification term, `GradScaler`, the gradient clip and `lr_scheduler.step()` are all its lines.
The optimizer (`create_optimizer`), the LR schedule (`create_lr_scheduler`), the latent scale
factor (`calculate_scale_factor`), the noise scheduler (`define_instance`) and the model
(`load_unet_text`) are its functions too. Upstream is configured for a **single** AdamW group --
`multiple_lr` appears in none of its released configs, so `create_optimizer` is the path CCELLA was
published with, and it is the default here.

What this driver adds is everything that lives *outside* that step and that upstream has no field
for: a W&B run shaped like MRFlow's, validation on a fixed subset at a step interval, checkpoints
at an explicit list of global steps, and an exact-step resume. They reach inside the epoch through
the one `step_hook` the pinned patch adds; see `upstream.py` for why that hook could not be
avoided.

**Resume semantics.** `global_step` is the authority. A resumed run restores the model, optimizer,
scheduler, scaler, scale factor and RNG, then continues from the saved step: no optimizer update is
repeated and none is skipped. What is *not* restored is the position within an epoch's shuffle --
the partial epoch that was in flight restarts from its beginning, so a few samples may be seen
twice across the resume boundary. That is recorded in the checkpoint (`epoch`, `step_in_epoch`) and
logged, and it does not affect step accounting.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "baselines.ccella"

from .config import load_config, upstream_namespace
from .data import build_dataset, build_loader
from .labels import LABELS_14, read_label_table
from .loss import build_class_loss
from .store import cache_meta, check_cache_meta, file_sha256
from .text import tokenizer_settings
from .upstream import require_upstream, upstream_module


class NullWriter:
    """Stands in for upstream's `SummaryWriter`. Every scalar it is handed is logged to W&B by the
    step hook instead, so this swallows the duplicate rather than writing a second event file."""

    def add_scalar(self, *args, **kwargs):
        pass


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def expected_meta(config):
    """The fingerprint this configuration implies, for `check_cache_meta`."""
    from .labels import label_source_fingerprint

    return cache_meta(
        config,
        label_source_fingerprint(config["mrrate"]["labels_root"]),
        file_sha256(config["model"]["autoencoder_path"]),
        tokenizer_settings(config["text"]["encoder"], config["text"]["max_length"]),
    )


class Trainer:
    def __init__(self, config, args):
        require_upstream()
        self.config, self.args = config, args
        self.upstream = upstream_module("scripts.diff_model_train_all")

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.world_size > 1:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.is_main = self.rank == 0
        set_seed(config["train"]["seed"], self.rank)

        check_cache_meta(config["data"]["cache_root"], expected_meta(config))

        self.global_step = 0
        self.epoch = 0
        self.step_in_epoch = 0
        self.checkpoint_steps = list(config["train"]["checkpoint_steps"])
        self.max_steps = config["train"]["max_train_steps"]
        self.output_dir = config["train"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self._build()
        self._maybe_resume()
        self._init_wandb()

    # --- construction ---------------------------------------------------------------------------

    def _build(self):
        config, up = self.config, self.upstream
        ns = upstream_namespace(config)
        ns.diffusion_unet_train["n_epochs"] = 1  # unused: the schedule is driven by max_train_steps
        import logging
        self.logger = logging.getLogger("ccella")
        logging.basicConfig(level=logging.INFO if self.is_main else logging.WARNING)

        self.dataset = build_dataset(config, "train", self.world_size, self.rank,
                                     limit=self.args.limit)
        self.loader = build_loader(config, self.dataset, seed=config["train"]["seed"], epoch=0)

        self.model, _ = up.load_unet_text(ns, self.device, self.logger)
        self.optimizer = up.create_optimizer(self.model, config["train"]["lr"])
        self.lr_scheduler = up.create_lr_scheduler(self.optimizer, self.max_steps)
        self.noise_scheduler = up.define_instance(ns, "noise_scheduler")
        self.loss_pt = torch.nn.L1Loss()
        self.scaler = torch.amp.GradScaler("cuda")
        torch.set_float32_matmul_precision("highest")

        # Upstream's own estimator, on our loader: 1 / std(z) over the first batch, all-reduced.
        self.scale_factor = up.calculate_scale_factor(self.loader, self.device, self.logger, "image")

        labels = read_label_table(config["mrrate"]["labels_root"])
        train_uids = _train_study_uids(config)
        self.class_loss, self.class_weights = build_class_loss(config, labels, train_uids)
        self.class_loss = self.class_loss.to(self.device)
        if self.is_main:
            self.logger.info("pos_weight: %s", dict(zip(LABELS_14, [round(w, 3) for w in
                                                                    self.class_weights["pos_weight"]])))

    def _init_wandb(self):
        self.wandb = None
        if not self.is_main:
            return
        import wandb

        spec = self.config["wandb"]
        mode = "disabled" if self.args.no_wandb else os.environ.get("WANDB_MODE", spec["mode"])
        self.wandb = wandb
        wandb.init(project=spec["project"], group=spec["group"], name=spec["name"],
                   id=spec["id"] or spec["name"], resume="allow", mode=mode,
                   dir=self.output_dir, config=self.config)

    # --- the loop -------------------------------------------------------------------------------

    def run(self):
        self.t_last = time.time()
        while self.global_step < self.max_steps:
            self.loader = build_loader(self.config, self.dataset,
                                       seed=self.config["train"]["seed"], epoch=self.epoch)
            self.step_in_epoch = 0
            self.upstream.train_one_epoch(
                epoch=self.epoch,
                unet=self.model,
                train_loader=self.loader,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                loss_pt=self.loss_pt,
                scaler=self.scaler,
                scale_factor=self.scale_factor,
                noise_scheduler=self.noise_scheduler,
                num_images_per_batch=self.config["train"]["micro_batch_size"],
                num_train_timesteps=self.config["train"]["num_train_timesteps"],
                device=self.device,
                logger=self.logger,
                local_rank=self.rank,
                imkey="image",
                base_maisi=False,
                text_maisi=True,
                tensorboard_writer=NullWriter(),
                tfevents_dir=self.output_dir,
                text_class_weight=self.config["model"]["text_class_pred_weight"],
                text_class_loss=self.class_loss,
                step_hook=self._on_step,
            )
            self.epoch += 1
        if self.is_main and self.wandb is not None:
            self.wandb.finish()

    def _on_step(self, info):
        """Called by the patched `train_one_epoch` after every optimizer update.

        Returns False to break out of the epoch, which is how a checkpoint boundary and the end of
        training are reached mid-epoch.
        """
        self.global_step += 1
        self.step_in_epoch += 1

        if self.is_main and self.global_step % self.config["train"]["log_every"] == 0:
            now = time.time()
            elapsed = now - self.t_last
            self.t_last = now
            n = self.config["train"]["log_every"]
            payload = {
                "train/loss": float(info["loss"]),
                "train/diffusion_loss": float(info["diffusion_loss"]),
                "train/grad_norm": float(info["grad_norm"]),
                "train/step_time_s": elapsed / n,
                "train/samples_per_s": n * info["batch_size"] * self.world_size / max(elapsed, 1e-9),
                "train/labelled_fraction": 1.0 - info["n_masked"] / max(1, info["batch_size"]),
                "train/epoch": self.epoch,
            }
            if info["class_loss"] is not None:
                payload["train/class_loss"] = float(info["class_loss"])
            for i, lr in enumerate(info["lr"]):
                payload[f"train/lr_group{i}"] = lr
            self._log(payload)

        if self.global_step % self.config["validation"]["every"] == 0:
            self._validate()

        if self.global_step in self.checkpoint_steps:
            self.save_checkpoint()

        return self.global_step < self.max_steps

    def _log(self, payload):
        if self.is_main and self.wandb is not None:
            self.wandb.log(payload, step=self.global_step)

    def _validate(self):
        from .validate import run_validation

        try:
            metrics, media = run_validation(self)
        except Exception as exc:                      # validation is monitoring, never fatal
            self.logger.warning("validation failed at step %d: %s", self.global_step, exc)
            return
        if self.is_main:
            self._log(metrics)
            if media:
                self._log(media)

    # --- checkpointing --------------------------------------------------------------------------

    def _path(self, step):
        return os.path.join(self.output_dir, f"checkpoint-{step}.pt")

    def save_checkpoint(self):
        if not self.is_main:
            return
        path = self._path(self.global_step)
        if os.path.exists(path) and not self.args.overwrite_checkpoints:
            raise FileExistsError(
                f"{path} already exists. Refusing to overwrite a checkpoint: pass "
                f"--overwrite_checkpoints if that is really what you want.")
        model = self.model.module if hasattr(self.model, "module") else self.model
        payload = {
            "unet_state_dict": model.state_dict(),     # UNet + ELLA + heads + modality embedding
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "scale_factor": self.scale_factor,
            "rng": rng_state(),
            "config": self.config,
            "label_order": list(LABELS_14),
            "class_weights": self.class_weights,
            "upstream_commit": self.config["upstream_commit"],
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.logger.info("saved %s", path)

    def _maybe_resume(self):
        target = self.args.resume
        if not target:
            return
        path = target if target != "latest" else _latest_checkpoint(self.output_dir)
        if path is None:
            self.logger.info("no checkpoint to resume from; starting at step 0")
            return
        # CPU, always: the RNG states are ByteTensors and `torch.set_rng_state` rejects a CUDA one.
        # `load_state_dict` moves the weights onto the model's device by itself.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(payload["unet_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.lr_scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload["scaler"])
        self.global_step = payload["global_step"]
        self.epoch = payload["epoch"]
        # PolynomialLR stores `total_iters`, so the schedule is restored from the checkpoint. If the
        # config now asks for a different run length, the restored schedule silently wins -- which
        # at best decays to a different endpoint and at worst pins the lr at zero.
        saved_max = payload.get("config", {}).get("train", {}).get("max_train_steps")
        if saved_max is not None and saved_max != self.max_steps:
            self.logger.warning(
                "checkpoint was written under max_train_steps=%s but this config says %s. The LR "
                "schedule was restored from the checkpoint and still decays over %s steps.",
                saved_max, self.max_steps, saved_max)
        self.scale_factor = torch.as_tensor(payload["scale_factor"]).to(self.device)
        set_rng_state(payload["rng"])
        if payload["label_order"] != list(LABELS_14):
            raise ValueError("checkpoint label order differs from labels.LABELS_14")
        self.logger.info("resumed %s at global step %d (epoch %d)",
                         path, self.global_step, self.epoch)


def _latest_checkpoint(output_dir):
    steps = []
    for name in os.listdir(output_dir) if os.path.isdir(output_dir) else []:
        if name.startswith("checkpoint-") and name.endswith(".pt"):
            steps.append(int(name[len("checkpoint-"):-len(".pt")]))
    return os.path.join(output_dir, f"checkpoint-{max(steps)}.pt") if steps else None


def _train_study_uids(config):
    """Train-split study uids, for `pos_weight`. Read from the parquet, never from the cache, so
    the weighting does not depend on how much of the split happens to be preprocessed yet."""
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.join(config["mrrate"]["raw_root"], "studies.parquet"),
                          columns=["study_uid", "split", "has_report"])
    return [table["study_uid"][i].as_py() for i in range(table.num_rows)
            if table["split"][i].as_py() == "train" and table["has_report"][i].as_py()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="cap the training rows (smoke only)")
    parser.add_argument("--overwrite_checkpoints", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    Trainer(config, args).run()


if __name__ == "__main__":
    main()
