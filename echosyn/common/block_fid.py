"""Block-level FID and FVD for training validation.

A fixed set of val block pairs: each target block is generated from noise by fixed-step Euler,
conditioned on its real previous block, report, modality and plane -- one teacher-forced step of
the rollout -- and the generated blocks are compared as a distribution with the real target blocks.
Same extractors and intensity handling as evaluation/paper_metrics.py (pytorch-fid's Inception
over every slice, StyleGAN-V's I3D over each 16-frame block, per-block 0.5/99.5 window, uint8).

What it is not: the evaluation's FID/FVD. Those score full rollouts against whole volumes, so
they include drift and stopping; this scores the next-block distribution only, on 128 blocks
where FVD needs thousands of clips for an absolute value. It is for comparing checkpoints of one
run -- same blocks, same noise every time -- not for quoting.
"""

import numpy as np
import torch

from echosyn.common import sample_latents, scale_latents, to_uint8_frames, unscale_latents
from echosyn.common.report_alignment import _text
from evaluation.paper_metrics import (ClipFVDAccumulator, Fid2dInceptionAccumulator, Moments,
                                      normalize01, pooled)

_EXTRACTORS = {}
_REAL = {}  # the real side never changes within a run, so its features are computed once


def _extractors(device):
    if device not in _EXTRACTORS:
        _EXTRACTORS[device] = (Fid2dInceptionAccumulator(device), ClipFVDAccumulator(device))
    return _EXTRACTORS[device]


def _decode(vae, latents, vae_scaling, config):
    """Scaled latents [B, C, T, h, w] -> per-block `[0, 1]` volumes [B, T, H, W], as numpy."""
    B = latents.shape[0]
    frames = unscale_latents(latents.float(), vae_scaling).transpose(1, 2).flatten(0, 1)
    decoded = torch.cat([vae.decode(c).sample for c in frames.split(32)])
    vols = to_uint8_frames(decoded, config)[:, 0].view(B, -1, *decoded.shape[-2:])
    return [normalize01(v.cpu().numpy().astype(np.float32)) for v in vols]


def _features(vols, device):
    fid, fvd = _extractors(device)
    inception, i3d = Moments(2048), Moments(400)
    for v in vols:
        inception.add(fid._features(v))
        i3d.add(fvd._clip_feature(v, v.shape[0]))
    return inception, i3d


@torch.no_grad()
def block_fid_fvd(denoiser, vae, dataset, config, vae_scaling, blocks=128, steps=50,
                  batch_size=16, seed=0):
    """`{fid_block, fvd_block, n_blocks}` for the denoiser as it is now."""
    device = next(denoiser.parameters()).device
    dtype = next(denoiser.parameters()).dtype
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:blocks]
    gen = torch.Generator(device=device).manual_seed(seed)
    key = (seed, len(indices))
    reals, fakes = [], []
    for start in range(0, len(indices), batch_size):
        items = [dataset[i] for i in indices[start:start + batch_size].tolist()]
        B = len(items)
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed + start)
            cond = scale_latents(sample_latents(config, torch.stack(
                [x["image"] for x in items]).to(device)), vae_scaling)
            real = scale_latents(sample_latents(config, torch.stack(
                [x["video"] for x in items]).to(device)), vae_scaling)
        texts = [_text(x) for x in items]
        kwargs = {k: torch.stack([x[k] for x in texts]).to(device) for k in texts[0]}
        kwargs["encoder_hidden_states"] = kwargs["encoder_hidden_states"].to(dtype)
        modality_id = torch.stack([x["modality_id"] for x in items]).to(device)
        plane_id = torch.stack([x["plane_id"] for x in items]).to(device)

        # Euler from t=1 (noise) to t=0 (data); the velocity is dz/dt, so each step subtracts it.
        z = torch.randn(real.shape, device=device, generator=gen)
        for k in range(steps):
            t = torch.full((B,), 1 - k / steps, device=device, dtype=dtype)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                v = denoiser(z.to(dtype), t, **kwargs, cond_image=cond.to(dtype),
                             modality_id=modality_id, plane_id=plane_id).sample
            z = z - v.float() / steps

        fakes += _decode(vae, z, vae_scaling, config)
        if key not in _REAL:
            reals += _decode(vae, real, vae_scaling, config)

    if key not in _REAL:
        _REAL[key] = _features(reals, device)
    real_inc, real_i3d = _REAL[key]
    fake_inc, fake_i3d = _features(fakes, device)
    return {
        "fid_block": pooled({"real": real_inc, "fake": fake_inc})[0],
        "fvd_block": pooled({"real": real_i3d, "fake": fake_i3d})[0],
        "n_blocks": len(fakes),
    }
