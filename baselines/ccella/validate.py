"""Validation: loss, pathology metrics, and a preview video in MRFlow's own style.

Two things happen, both on a **fixed** subset (`data.fixed_validation_subset` -- sorted by
`sample_id` and strided, so it is the same rows on every rank, every run and across a resume):

1. *Scoring.* Each sample is noised at a deterministic timestep and the same forward upstream's
   training step makes is run under `no_grad`. That gives the validation diffusion loss, the
   validation classification loss, and the logits that `loss.pathology_metrics` turns into macro and
   micro AUROC / average precision plus a per-label table. The noise and the timesteps come from a
   generator seeded on the sample index, so the number moves only when the model does.

2. *Previewing.* A handful of volumes are sampled from noise and decoded, and the reference and the
   generation are put side by side as one video down the axial axis -- `label_frames` and
   `wandb.Video` from `echosyn.common`, the same helpers `lvfm/train.py::log_validation` uses, so a
   CCELLA run and an MRFlow run look alike in W&B. The caption carries modality, plane, spacing,
   the predicted pathology probabilities and the ground-truth labels. **It never carries report
   text, a patient identifier or a study identifier** -- the manifest stores only hashed keys, and
   the caption is built from the manifest.

The sampling loop mirrors `evaluate_diffusion.run_inference_textclass` (noise, `set_timesteps`,
`unet` then `noise_scheduler.step` per timestep, then `ReconModelRaw`). It is written out here
rather than called because that function calls the U-Net without `modality_id`, and modality is an
explicit control in this adaptation. `ReconModelRaw` itself is imported from upstream, so the
`z / scale_factor` decode is not restated.

Validation is monitoring: `Trainer._validate` catches everything it raises. It must never be able
to end a training run.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import CcellaDataset, fixed_validation_subset
from .labels import LABELS_14
from .loss import pathology_metrics


@torch.no_grad()
def score_subset(trainer, rows):
    """Validation losses and the pathology logits, on a deterministic noising of `rows`."""
    config = trainer.config
    dataset = CcellaDataset(config["data"]["cache_root"], rows,
                            config["model"]["latent_channels"])
    model = trainer.model
    model.eval()

    logits, targets, masks = [], [], []
    diffusion_total, class_total, n_batches, n_labelled = 0.0, 0.0, 0, 0
    batch_size = config["train"]["micro_batch_size"]

    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([b["image"] for b in batch]).to(trainer.device) * trainer.scale_factor
        text = torch.stack([b["text"] for b in batch]).to(trainer.device)
        spacing = torch.stack([b["spacing"] for b in batch]).to(trainer.device)
        pirads = torch.stack([b["pirads"] for b in batch]).to(trainer.device)
        isnull = torch.stack([b["text_isnull"] for b in batch]).to(trainer.device)
        modality = torch.stack([b["modality_id"] for b in batch]).to(trainer.device)

        generator = torch.Generator(device="cpu").manual_seed(1_000_003 + start)
        noise = torch.randn(images.shape, generator=generator).to(trainer.device)
        timesteps = torch.randint(0, config["train"]["num_train_timesteps"], (images.shape[0],),
                                  generator=generator).to(trainer.device).long()
        noisy = trainer.noise_scheduler.add_noise(original_samples=images, noise=noise,
                                                  timesteps=timesteps)

        with torch.amp.autocast("cuda", enabled=trainer.device.type == "cuda"):
            noise_pred, class_pred = model(x=noisy, timesteps=timesteps, spacing_tensor=spacing,
                                           text_encoding=text, modality_id=modality)
            diffusion = trainer.loss_pt(noise_pred.float(), noise.float())
            per_element = trainer.class_loss(class_pred, pirads)

        keep = (1 - isnull).unsqueeze(1)
        if keep.sum() > 0:
            class_total += float((per_element * keep).sum() / keep.sum().clamp(min=1))
        diffusion_total += float(diffusion)
        n_batches += 1
        n_labelled += int((1 - isnull).sum())

        logits.append(class_pred.float().cpu().numpy())
        targets.append(pirads.cpu().numpy())
        masks.append(np.repeat((1 - isnull).cpu().numpy()[:, None], len(LABELS_14), axis=1))

    model.train()
    metrics = pathology_metrics(np.concatenate(logits), np.concatenate(targets),
                                np.concatenate(masks))
    metrics["diffusion_loss"] = diffusion_total / max(1, n_batches)
    metrics["class_loss"] = class_total / max(1, n_batches)
    metrics["total_loss"] = (metrics["diffusion_loss"]
                             + config["model"]["text_class_pred_weight"] * metrics["class_loss"])
    metrics["n_labelled"] = n_labelled
    return metrics


@torch.no_grad()
def sample_volumes(trainer, rows, autoencoder):
    """Sample `len(rows)` volumes from noise. Mirrors `evaluate_diffusion.run_inference_textclass`."""
    from .upstream import upstream_module

    ReconModelRaw = upstream_module("scripts.utils").ReconModelRaw
    config = trainer.config
    dataset = CcellaDataset(config["data"]["cache_root"], rows, config["model"]["latent_channels"])
    batch = [dataset[i] for i in range(len(dataset))]

    text = torch.stack([b["text"] for b in batch]).to(trainer.device)
    spacing = torch.stack([b["spacing"] for b in batch]).to(trainer.device)
    modality = torch.stack([b["modality_id"] for b in batch]).to(trainer.device)
    reference = torch.stack([b["image"] for b in batch]).to(trainer.device)

    generator = torch.Generator(device="cpu").manual_seed(config["train"]["seed"])
    shape = (len(batch), config["model"]["latent_channels"]) + tuple(reference.shape[2:])
    image = torch.randn(shape, generator=generator).to(trainer.device)

    scheduler = trainer.noise_scheduler
    scheduler.set_timesteps(num_inference_steps=config["validation"]["inference_steps"])
    trainer.model.eval()
    class_pred = None
    with torch.amp.autocast("cuda", enabled=trainer.device.type == "cuda"):
        for t in scheduler.timesteps:
            timesteps = torch.full((len(batch),), float(t), device=trainer.device)
            model_output, class_pred = trainer.model(
                x=image, timesteps=timesteps, spacing_tensor=spacing,
                text_encoding=text, modality_id=modality)
            image, _ = scheduler.step(model_output, t, image)
        recon = ReconModelRaw(autoencoder=autoencoder).to(trainer.device)
        generated = torch.clip(recon(image, scale_factor=trainer.scale_factor), 0.0, 1.0)
        truth = torch.clip(recon(reference * trainer.scale_factor,
                                 scale_factor=trainer.scale_factor), 0.0, 1.0)
    trainer.model.train()
    scheduler.set_timesteps(num_inference_steps=config["train"]["num_train_timesteps"])
    return truth, generated, torch.sigmoid(class_pred.float()).cpu().numpy()


def _to_frames(volume):
    """`(1, X, Y, Z)` in [0, 1] -> `(C, T, H, W)` uint8, stepping down the axial (Z) axis."""
    from einops import rearrange

    array = (volume[0].permute(2, 0, 1) * 255).clamp(0, 255).to(torch.uint8).cpu()
    array = array.unsqueeze(0).repeat(3, 1, 1, 1)            # grayscale -> RGB, as MRFlow logs
    return rearrange(array, "c t h w -> c t h w")


def preview_video(trainer, rows, autoencoder):
    """A `wandb.Video` of reference | generated, labelled, plus a non-identifying caption."""
    import wandb
    from einops import rearrange

    from echosyn.common import label_frames

    truth, generated, probs = sample_volumes(trainer, rows, autoencoder)
    ref = torch.stack([_to_frames(truth[i]) for i in range(truth.shape[0])])
    gen = torch.stack([_to_frames(generated[i]) for i in range(generated.shape[0])])
    frames = torch.cat([label_frames(ref, "ground truth"), label_frames(gen, "generated")], dim=3)
    frames = rearrange(frames, "b c t h w -> t c h (b w)").numpy()

    captions = []
    for row, prob in zip(rows, probs):
        positives = [n for n, v in zip(LABELS_14, row["labels"]) if v == "1"]
        top = sorted(zip(LABELS_14, prob), key=lambda kv: -kv[1])[:3]
        captions.append(
            f"{row['modality']}/{row['plane']} spacing {row['spacing_mm'].replace(';', '/')} | "
            f"true: {','.join(positives) or 'none'} | "
            f"pred: {', '.join(f'{n}={p:.2f}' for n, p in top)}")
    return {"validation": wandb.Video(frames, caption=" || ".join(captions),
                                      fps=trainer.config["validation"]["fps"], format="mp4")}


def run_validation(trainer):
    """`(scalar metrics, media)` for one validation pass. Called only from `Trainer._validate`."""
    config = trainer.config
    rows = fixed_validation_subset(config, "val")
    metrics = score_subset(trainer, rows)

    payload = {
        "val/total_loss": metrics["total_loss"],
        "val/diffusion_loss": metrics["diffusion_loss"],
        "val/class_loss": metrics["class_loss"],
        "val/n_samples_scored": metrics["n_samples_scored"],
        "val/n_labels_scored": metrics["n_labels_scored"],
    }
    for key in ("auroc_macro", "auroc_micro", "ap_macro", "ap_micro"):
        if metrics[key] is not None:
            payload[f"val/{key}"] = metrics[key]
    for name, entry in metrics["per_label"].items():
        if entry["auroc"] is not None:
            payload[f"val/auroc/{name}"] = entry["auroc"]
            payload[f"val/ap/{name}"] = entry["ap"]
        payload[f"val/prevalence/{name}"] = entry["prevalence"]

    media = None
    if config["validation"]["visualize"] and trainer.is_main:
        from .volume import build_autoencoder

        autoencoder = build_autoencoder(config, trainer.device)
        try:
            media = preview_video(trainer, rows[:config["validation"]["samples"]], autoencoder)
        finally:
            del autoencoder
            torch.cuda.empty_cache()
    return payload, media
