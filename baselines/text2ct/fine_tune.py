#!/usr/bin/env python
"""Fine-tune Text2CT's diffusion UNet on MR-RATE. The VAE and the 3D-CLIP encoder stay frozen.

    # single GPU
    python -m baselines.text2ct.fine_tune --config baselines/text2ct/configs/mrrate_finetune.yaml

    # 8 GPUs over 2 nodes (torchrun per node)
    torchrun --nnodes 2 --nproc_per_node 4 --node_rank $SLURM_NODEID \
        --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 \
        -m baselines.text2ct.fine_tune --config baselines/text2ct/configs/mrrate_finetune.yaml

    # bounded self-check: builds everything, trains a few steps, asserts the freezing contract
    python -m baselines.text2ct.fine_tune --config <cfg> --smoke --steps 3

**What is trained.** Only `unet.parameters()`, which includes `class_embedding` -- the modality
condition is MAISI's own class-embedding hook, added to the timestep embedding inside the UNet, so
it is a UNet parameter by construction rather than a bolted-on module. The VAE and the report
encoder never enter the optimizer, never leave eval mode, and are checksummed before and after
training so a violation cannot be silent.

**The objective is upstream's, unchanged**: rectified flow with v-prediction, `z_t` from
`RFlowScheduler.add_noise`, target `z_0 - eps`, L1 loss (`scripts/diff_model_train.py:327-380`).
Report classifier-free guidance is also upstream's -- 10% of samples train with a zero context, and
the sampler contrasts a zero-context branch against the conditional one. Modality is supplied in
**both** branches.

**What this script adds over `scripts/diff_model_train.py`**: gradient accumulation, gradient
clipping and a non-finite-gradient skip, a DistributedSampler (upstream partitions the file list
once, so shuffling never crosses ranks), a validation loss on a held-out split, step-based
checkpointing with a working resume (upstream's `load_training_state` reads an undefined
`checkpoint` name), mixed precision that can be bf16, and the freezing assertions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

if __package__ in (None, ""):                  # allow `python baselines/text2ct/fine_tune.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "baselines.text2ct"

from .config import (apply_dotted, effective_batch_size, grid_spacing, latent_shape,
                     load_config, validate, weight_paths)
from .model import (FrozenGuard, build_noise_scheduler, build_unet, build_vae, module_checksum,
                    parameter_report)
from .mrrate_data import (MODALITY_TO_ID, Text2CTLatentDataset, cache_meta, check_cache_meta,
                          modality_vocabulary)
from .text_encoder import TEXT_EMBED_DIM, null_context


### Process group ###


def setup_distributed():
    """-> (rank, world_size, local_rank, device). Works unlaunched, under torchrun, and under srun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            # The modulo only ever bites when a debug run oversubscribes one GPU; under torchrun in
            # production LOCAL_RANK is always below the visible device count, so it is the identity.
            local_rank %= torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        return dist.get_rank(), dist.get_world_size(), local_rank, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message):
    if is_main():
        print(message, flush=True)


def seed_everything(seed, rank=0):
    """Per-rank seeds: identical seeds across ranks would draw the same noise on every GPU, which
    silently shrinks the effective batch's noise diversity."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


### Tracking ###


class Tracker:
    """Rank-0 W&B logging, and a no-op everywhere else.

    The run `id` is fixed in the config, so a continuation job that resumes from a checkpoint
    resumes the **same** W&B run rather than opening a second one -- which is what makes a 60,000-
    step fine-tune spread over several 24 h jobs read as one curve. `resume="allow"` creates it on
    the first job and attaches on every later one.

    Compute nodes have no direct outbound route, so `mode: online` needs the site proxy exported in
    the job script; `offline` writes to `<output_dir>/wandb` for a later `wandb sync`, and
    `disabled` turns the whole thing into a no-op without touching any call site.
    """

    def __init__(self, config, world_size, enabled=True):
        self.run = None
        settings = config["wandb"]
        if not enabled or not is_main() or settings["mode"] == "disabled":
            return
        import wandb

        self.run = wandb.init(
            project=settings["project"], group=settings["group"], name=settings["name"],
            id=settings["id"], tags=list(settings["tags"]), mode=settings["mode"],
            resume="allow", dir=config["output_dir"],
            config={**config, "world_size": world_size,
                    "effective_batch_size": effective_batch_size(config, world_size)})

    def log(self, values, step):
        if self.run is not None:
            self.run.log(values, step=step)

    def summary(self, values):
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self):
        if self.run is not None:
            self.run.finish()


### Optimizer and schedule ###


def build_optimizer(unet, train):
    params = [p for p in unet.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("the UNet has no trainable parameters")
    if train["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=train["lr"], betas=tuple(train["betas"]),
                                eps=train["eps"])
    return torch.optim.AdamW(params, lr=train["lr"], betas=tuple(train["betas"]),
                             eps=train["eps"], weight_decay=train["weight_decay"])


def lr_at(step, train):
    """The schedule as a pure function of the step, so resume needs no scheduler state to be right.

    `cosine` mirrors MRFlow's fine-tune shape (linear warmup, cosine to `min_lr`); `poly` is
    upstream's `PolynomialLR(power=2.0)`, which starts at the full lr with no warmup -- appropriate
    when training from MAISI, less so when fine-tuning an already-converged UNet.
    """
    warmup, total, base, floor = (train["warmup_steps"], train["max_steps"], train["lr"],
                                  train["min_lr"])
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if train["scheduler"] == "constant":
        return base
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if train["scheduler"] == "poly":
        return floor + (base - floor) * (1.0 - progress) ** train["poly_power"]
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


### One training step's forward ###


def diffusion_loss(unet, batch, noise_scheduler, config, device, generator=None):
    """The upstream objective, with report (and optionally modality) dropout applied per sample.

    `latent` arrives unscaled; `* scale_factor` is upstream's `images = images * scale_factor`
    (`scripts/diff_model_train.py:307`).
    """
    scale = config["model"]["scale_factor"]
    latent = batch["latent"].to(device, non_blocking=True) * scale
    context = batch["context"].to(device, non_blocking=True)
    class_labels = batch["class_label"].to(device, non_blocking=True)
    spacing = batch["spacing"].to(device, non_blocking=True)
    batch_size = latent.shape[0]

    noise = torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=generator)
    timesteps = noise_scheduler.sample_timesteps(latent)
    noisy = noise_scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)

    # Report dropout: zero the context, which is the only null this model has.
    drop = config["cfg"]["report_dropout_prob"]
    if drop > 0:
        mask = torch.rand(batch_size, device=device, generator=generator) < drop
        context = torch.where(mask[:, None, None], torch.zeros_like(context), context)
    # Modality dropout is off by default; when on it swaps in the CFG_NULL id, never a zero tensor
    # (the class embedding is a lookup, so there is no "zero" index to reach for).
    mod_drop = config["cfg"]["modality_dropout_prob"]
    if mod_drop > 0:
        mask = torch.rand(batch_size, device=device, generator=generator) < mod_drop
        class_labels = torch.where(mask, torch.full_like(class_labels,
                                                         MODALITY_TO_ID["CFG_NULL"]), class_labels)

    predicted = unet(x=noisy, timesteps=timesteps, context=context, class_labels=class_labels,
                     spacing_tensor=spacing)
    target = latent - noise                    # RFlowScheduler is v-prediction only
    loss_fn = torch.nn.functional.l1_loss if config["train"]["loss"] == "l1" else \
        torch.nn.functional.mse_loss
    return loss_fn(predicted.float(), target.float())


### Checkpoints ###


def checkpoint_name(step):
    return f"checkpoint-{step:07d}.pt"


def save_checkpoint(path, unet, optimizer, scaler, step, epoch, config, meta, scale_factor,
                    num_train_timesteps):
    """Written with upstream's own key names, so `scripts/diff_model_infer.py:57` loads it as-is.

    Everything needed to resume is here **and** everything needed to interpret the weights later:
    the modality vocabulary (a class id is meaningless without it) and the cache fingerprint the
    run trained against.
    """
    module = unet.module if isinstance(unet, DistributedDataParallel) else unet
    payload = {
        "unet_state_dict": module.state_dict(),
        "scale_factor": scale_factor,
        "num_train_timesteps": num_train_timesteps,
        "epoch": epoch,
        "global_step": step,
        "optimizer": optimizer.state_dict(),
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "modality_vocabulary": modality_vocabulary(),
        "config": config,
        "cache_meta": meta,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_checkpoints(directory, keep):
    files = sorted(f for f in os.listdir(directory)
                   if f.startswith("checkpoint-") and f.endswith(".pt"))
    for name in files[:-keep] if keep > 0 else []:
        os.remove(os.path.join(directory, name))


def latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None
    files = sorted(f for f in os.listdir(directory)
                   if f.startswith("checkpoint-") and f.endswith(".pt"))
    return os.path.join(directory, files[-1]) if files else None


def load_checkpoint(path, unet, optimizer, scaler, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    module = unet.module if isinstance(unet, DistributedDataParallel) else unet
    missing, unexpected = module.load_state_dict(payload["unet_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint {path} does not match the model: missing {missing[:4]}, "
                           f"unexpected {unexpected[:4]}")
    if payload.get("modality_vocabulary", {}).get("modality_to_id") != MODALITY_TO_ID:
        raise RuntimeError(
            f"checkpoint {path} was trained with a different modality vocabulary:\n"
            f"  checkpoint: {payload.get('modality_vocabulary', {}).get('modality_to_id')}\n"
            f"  current:    {MODALITY_TO_ID}\nClass ids may never move.")
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("grad_scaler"):
        scaler.load_state_dict(payload["grad_scaler"])
    return payload.get("global_step", 0), payload.get("epoch", 0)


### Validation ###


@torch.no_grad()
def validation_loss(unet, loader, noise_scheduler, config, device, max_batches, seed=1234):
    """Same objective on the held-out split, with noise and timesteps drawn from a fixed generator
    so the number is comparable across steps rather than a fresh sample each time."""
    if loader is None:
        return float("nan")
    was_training = unet.training
    unet.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        total += float(diffusion_loss(unet, batch, noise_scheduler, config, device, generator))
        count += 1
    if was_training:
        unet.train()
    if count == 0:
        return float("nan")
    value = torch.tensor([total, count], device=device)
    if dist.is_initialized():
        dist.all_reduce(value)
    return float(value[0] / value[1])


### Data ###


def build_loaders(config, rank, world_size, smoke=False):
    """-> (train_loader, val_loader, train_dataset). In smoke mode without a cache, synthetic."""
    train_cfg, data_cfg = config["train"], config["data"]
    if smoke and not (data_cfg["cache_root"] and
                      os.path.isdir(os.path.join(data_cfg["cache_root"], "manifest"))):
        log("no cache found -- smoke test runs on synthetic latents")
        dataset = SyntheticLatentDataset(config, n=max(8, train_cfg["micro_batch_size"] * 4))
        val_dataset = SyntheticLatentDataset(config, n=train_cfg["micro_batch_size"] * 2, seed=7)
    else:
        dataset = Text2CTLatentDataset(data_cfg["cache_root"], "train", grid_spacing(config),
                                       limit=data_cfg["limit_train"])
        try:
            val_dataset = Text2CTLatentDataset(data_cfg["cache_root"], "val", grid_spacing(config),
                                               limit=data_cfg["limit_val"])
        except RuntimeError:
            log("no val manifest -- validation loss disabled")
            val_dataset = None
        log(f"train: {len(dataset)} series  {dataset.counts()}")

    workers = 0 if smoke else train_cfg["num_workers"]
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(dataset, batch_size=train_cfg["micro_batch_size"],
                              sampler=sampler, shuffle=sampler is None, drop_last=True,
                              num_workers=workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=workers > 0)
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=False) if world_size > 1 else None
        val_loader = DataLoader(val_dataset, batch_size=train_cfg["micro_batch_size"],
                                sampler=val_sampler, shuffle=False, num_workers=0,
                                pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, dataset


class SyntheticLatentDataset(torch.utils.data.Dataset):
    """Stand-in for the cache, so the pipeline is testable without 750 GB of latents on disk.

    Shapes, dtypes and value ranges match `Text2CTLatentDataset` exactly: a real VAE latent is
    roughly unit-scaled (hence `scale_factor ~ 1`), and a real context is an L2-normalized 768-d
    vector, so both are drawn that way rather than as arbitrary noise.
    """

    def __init__(self, config, n=16, seed=0, latent_channels=4):
        self.shape = latent_shape(config, latent_channels)
        self.spacing = np.array(grid_spacing(config), dtype=np.float32) * 1e2
        self.ids = sorted(set(MODALITY_TO_ID.values()) - {MODALITY_TO_ID["CFG_NULL"]})
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(self.seed * 100003 + idx)
        context = torch.randn(1, TEXT_EMBED_DIM, generator=gen)
        return {
            "latent": torch.randn(self.shape, generator=gen),
            "context": context / context.norm(),
            "class_label": torch.tensor(self.ids[idx % len(self.ids)], dtype=torch.long),
            "spacing": torch.from_numpy(self.spacing.copy()),
            "sample_id": f"synthetic-{idx:04d}",
        }


### Sampling check ###


@torch.no_grad()
def cfg_sample_step(unet, config, device, noise_scheduler, steps=2, batch_size=1):
    """One short classifier-free-guided rollout, purely as a shape-and-finiteness check.

    Reproduces `scripts/diff_model_infer.py:239-262` exactly: an unconditional branch with a zero
    context and the **same** modality and spacing, a conditional branch, and
    `uncond + s * (cond - uncond)`. When `cfg.modality_guidance_scale != 1` a third branch is added
    at `CFG_NULL` -- off by default, and an experiment rather than the method.
    """
    latent = torch.randn((batch_size,) + latent_shape(config), device=device)
    context = torch.randn(batch_size, 1, TEXT_EMBED_DIM, device=device)
    context = context / context.norm(dim=-1, keepdim=True)
    class_labels = torch.full((batch_size,), MODALITY_TO_ID["T1w"], device=device,
                              dtype=torch.long)
    spacing = torch.tensor(grid_spacing(config), device=device).float().mul(1e2)\
        .repeat(batch_size, 1)

    noise_scheduler.set_timesteps(num_inference_steps=steps,
                                 input_img_size_numel=torch.prod(torch.tensor(latent.shape[2:])))
    timesteps = noise_scheduler.timesteps
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
    guidance = config["cfg"]["guidance_scale"]
    modality_guidance = config["cfg"]["modality_guidance_scale"]

    for t, next_t in zip(timesteps, next_timesteps):
        kwargs = {"timesteps": torch.Tensor((t,)).to(device).expand(batch_size),
                  "spacing_tensor": spacing, "class_labels": class_labels}
        v_uncond = unet(x=latent, context=null_context(batch_size, device, latent.dtype), **kwargs)
        v_cond = unet(x=latent, context=context, **kwargs)
        v = v_uncond + guidance * (v_cond - v_uncond)
        if modality_guidance != 1.0:
            null_modality = dict(kwargs, class_labels=torch.full_like(class_labels,
                                                                      MODALITY_TO_ID["CFG_NULL"]))
            v_base = unet(x=latent, context=null_context(batch_size, device, latent.dtype),
                          **null_modality)
            v = v_base + modality_guidance * (v_uncond - v_base) + guidance * (v_cond - v_uncond)
        latent, _ = noise_scheduler.step(v, t, latent, next_t)
    return latent


### Main ###


def train(config, smoke=False, smoke_steps=3, device_override=None, resume="latest"):
    rank, world_size, local_rank, device = setup_distributed()
    if device_override:
        device = torch.device(device_override)
    train_cfg = config["train"]
    seed_everything(train_cfg["seed"], rank)
    torch.set_float32_matmul_precision("highest")   # as upstream

    paths = weight_paths(config)
    output_dir = config["output_dir"]
    if is_main():
        os.makedirs(output_dir, exist_ok=True)

    meta = None
    if config["data"]["cache_root"] and \
            os.path.isdir(os.path.join(config["data"]["cache_root"], "manifest")):
        meta = cache_meta(config, paths["vae"], paths["clip"])
        problems = check_cache_meta(config["data"]["cache_root"], meta,
                                    strict=not config["data"]["ignore_cache_meta"])
        if problems:
            log("WARNING: cache metadata mismatch ignored:\n" + "\n".join(problems))

    # --- models ---
    unet, released = build_unet(config["text2ct_root"], paths["unet"], device,
                                modality_init=config["model"]["modality_init"],
                                class_ids=tuple(MODALITY_TO_ID.values()),
                                model_def=config["model_def"])
    noise_scheduler = build_noise_scheduler(config["text2ct_root"], config["model_def"])
    scale_factor = float(config["model"]["scale_factor"])
    if "scale_factor" in released and float(released["scale_factor"]) != scale_factor:
        log(f"note: {paths['unet']} carries scale_factor {float(released['scale_factor'])}, "
            f"config says {scale_factor}; the config wins")
    if config["model"]["recompute_scale_factor"]:
        raise NotImplementedError(
            "model.recompute_scale_factor is refused. Upstream recomputes 1/std(z) from the first "
            "batch, which is right for a scratch run and wrong here: the released UNet was trained "
            "with 1.0287 and a fine-tune inherits it -- the same call MRFlow makes about FLUX's "
            "scaling_factor. Set model.scale_factor explicitly if you really mean to change it.")

    # The VAE is loaded whenever it is on disk, even though cached latents mean it is never called:
    # it is what makes the freezing report and the frozen-checksum assertion cover a real module
    # rather than an empty set. The text encoder is 3.1 GB and stays off by default -- not being
    # resident at all is a stronger guarantee than being frozen.
    guard = FrozenGuard()
    components = {"unet": unet}
    if os.path.exists(paths["vae"]):
        components["vae"] = guard.add("vae", build_vae(
            config["text2ct_root"], paths["vae"], device, config["model_def"],
            fast_concat=config["model"]["vae_fast_concat"]))
    else:
        log(f"note: {paths['vae']} not found -- the VAE is not resident, so nothing to freeze")
    if config["data"]["load_text_encoder"]:
        from .text_encoder import build_text_encoder
        components["text_encoder"] = guard.add(
            "text_encoder", build_text_encoder(config["text2ct_root"], paths["clip"], device,
                                               config["data"]["text_max_length"]))

    lines, trainable, frozen = parameter_report(components)
    log("parameters by component:\n" + "\n".join(lines))
    log(f"modality vocabulary: {MODALITY_TO_ID} (init={config['model']['modality_init']})")
    log(f"scale_factor {scale_factor}  |  effective batch "
        f"{effective_batch_size(config, world_size)} "
        f"({train_cfg['micro_batch_size']} x {train_cfg['gradient_accumulation_steps']} x "
        f"{world_size})")

    train_loader, val_loader, dataset = build_loaders(config, rank, world_size, smoke)
    optimizer = build_optimizer(unet, train_cfg)
    guard.assert_absent_from(optimizer)

    use_amp = train_cfg["precision"] != "fp32" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if train_cfg["precision"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    if world_size > 1:
        unet = DistributedDataParallel(
            unet, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=train_cfg["ddp_find_unused_parameters"])

    global_step, epoch = 0, 0
    resume_path = latest_checkpoint(output_dir) if resume == "latest" else resume
    if resume_path and os.path.exists(resume_path):
        global_step, epoch = load_checkpoint(resume_path, unet, optimizer, scaler, device)
        log(f"resumed from {resume_path} at step {global_step}, epoch {epoch}")

    tracker = Tracker(config, world_size, enabled=not smoke)
    tracker.summary({"trainable_parameters": trainable, "frozen_parameters": frozen,
                     "train_series": len(dataset), "scale_factor": scale_factor,
                     "modality_vocabulary": str(MODALITY_TO_ID)})

    max_steps = smoke_steps if smoke else train_cfg["max_steps"]
    accumulation = train_cfg["gradient_accumulation_steps"]
    unet.train()
    guard.assert_eval_mode()
    before = {name: module_checksum(m) for name, m in guard.modules.items()}

    log(f"training to {max_steps} optimizer steps "
        f"({len(train_loader)} micro-batches per rank per epoch)")
    skipped, consecutive_skipped, started = 0, 0, time.time()
    grad_seen = False
    while global_step < max_steps:
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for micro, batch in enumerate(train_loader):
            accumulating = (micro + 1) % accumulation != 0
            sync = unet.no_sync() if (accumulating and isinstance(unet, DistributedDataParallel)) \
                else _nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = diffusion_loss(unet, batch, noise_scheduler, config, device)
                scaler.scale(loss / accumulation).backward()
            if accumulating:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (unet.module if isinstance(unet, DistributedDataParallel) else unet).parameters(),
                train_cfg["max_grad_norm"])
            if not grad_seen:
                grad_seen = _assert_unet_gradients(unet)
            if torch.isfinite(grad_norm):
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(global_step, train_cfg)
                scaler.step(optimizer)
                scaler.update()
                consecutive_skipped = 0
            else:
                # bf16 runs without a GradScaler, so nothing else would drop a bad update, and one
                # nan reaches every weight through clip_grad_norm_ (1.0/nan is nan).
                skipped += 1
                consecutive_skipped += 1
                scaler.update()
                if consecutive_skipped > train_cfg["max_consecutive_skipped_steps"]:
                    raise RuntimeError(f"{consecutive_skipped} consecutive non-finite gradients")
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % train_cfg["log_every"] == 0 or global_step == 1:
                rate = (time.time() - started) / max(1, global_step)
                memory = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0
                log(f"step {global_step}/{max_steps}  loss {float(loss):.4f}  "
                    f"lr {optimizer.param_groups[0]['lr']:.3e}  |grad| {float(grad_norm):.3f}  "
                    f"skipped {skipped}  {rate:.2f}s/step  mem {memory:.1f} GiB")
                tracker.log({"train/loss": float(loss),
                             "train/lr": optimizer.param_groups[0]["lr"],
                             "train/grad_norm": float(grad_norm),
                             "train/skipped_steps": skipped,
                             "train/seconds_per_step": rate,
                             "train/gpu_memory_gib": memory,
                             "train/epoch": epoch}, step=global_step)
            if train_cfg["val_every"] and global_step % train_cfg["val_every"] == 0:
                value = validation_loss(unet, val_loader, noise_scheduler, config, device,
                                        train_cfg["val_batches"])
                log(f"step {global_step}  val_loss {value:.4f}")
                tracker.log({"val/loss": value}, step=global_step)
            if is_main() and train_cfg["checkpoint_every"] and \
                    global_step % train_cfg["checkpoint_every"] == 0:
                path = os.path.join(output_dir, checkpoint_name(global_step))
                save_checkpoint(path, unet, optimizer, scaler, global_step, epoch, config, meta,
                                scale_factor, noise_scheduler.num_train_timesteps)
                prune_checkpoints(output_dir, train_cfg["keep_checkpoints"])
                log(f"saved {path}")
            if global_step >= max_steps:
                break
        epoch += 1

    guard.assert_all(optimizer)
    for name, digest in before.items():
        if module_checksum(guard.modules[name]) != digest:
            raise AssertionError(f"frozen module {name!r} changed during training")

    if is_main():
        path = os.path.join(output_dir, checkpoint_name(global_step))
        if not os.path.exists(path):        # the periodic save may already have written this step
            save_checkpoint(path, unet, optimizer, scaler, global_step, epoch, config, meta,
                            scale_factor, noise_scheduler.num_train_timesteps)
            log(f"saved {path}")
        with open(os.path.join(output_dir, "config.json"), "w") as handle:
            json.dump(config, handle, indent=1)

    if smoke:
        value = validation_loss(unet, val_loader, noise_scheduler, config, device, 2)
        sample = cfg_sample_step(unet, config, device,
                                 build_noise_scheduler(config["text2ct_root"],
                                                       config["model_def"]))
        log(f"smoke: val_loss {value:.4f}  cfg sample {tuple(sample.shape)} "
            f"finite={bool(torch.isfinite(sample).all())}")
        log(f"smoke: frozen components verified unchanged ({sorted(guard.modules) or 'none loaded'})")

    tracker.summary({"final_step": global_step, "skipped_steps_total": skipped})
    tracker.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    return {"global_step": global_step, "skipped_steps": skipped,
            "trainable_parameters": trainable, "frozen_parameters": frozen}


def _assert_unet_gradients(unet):
    """At least one trainable parameter has a finite, nonzero gradient, and none is non-finite."""
    module = unet.module if isinstance(unet, DistributedDataParallel) else unet
    nonzero = False
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return True                        # the caller's skip path handles it
        nonzero = nonzero or bool(param.grad.abs().sum() > 0)
    if not nonzero:
        raise AssertionError("no UNet parameter received a nonzero gradient")
    return True


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", help="YAML; defaults come from baselines/text2ct/config.py")
    parser.add_argument("--output_dir", help="overrides config.output_dir")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted config overrides, e.g. train.lr=5e-5")
    parser.add_argument("--smoke", action="store_true",
                        help="bounded self-check: a few steps, the freezing assertions, one "
                             "CFG sampling pass. Falls back to synthetic latents with no cache.")
    parser.add_argument("--steps", type=int, default=3, help="smoke steps")
    parser.add_argument("--device", help="force a device, e.g. cpu")
    parser.add_argument("--no_wandb", action="store_true",
                        help="shorthand for --set wandb.mode=\"disabled\"")
    parser.add_argument("--resume", default="latest",
                        help="'latest' (default), 'none', or a checkpoint path")
    args = parser.parse_args(argv)

    config = apply_dotted(load_config(args.config), args.set)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.no_wandb:
        config["wandb"]["mode"] = "disabled"
    validate(config, require_data=not args.smoke)
    return train(config, smoke=args.smoke, smoke_steps=args.steps, device_override=args.device,
                 resume=None if args.resume == "none" else args.resume)


if __name__ == "__main__":
    main()
