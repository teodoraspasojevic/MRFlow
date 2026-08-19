"""Zero-shot flow-matching loss of the CTFlow checkpoint on MR data, per normalization.

CTFlow was trained on CT volumes in [0, 1]. This measures which MR setup its prior transfers to,
with no training -- just forward passes. Each range is tried twice: once with FLUX's own
scaling_factor / shift_factor, once standardized on the measured latent moments.

Settled on 2026-08-19 in favour of [0, 1], which the pipeline now uses; kept because it is the check
to re-run whenever the VAE, the init checkpoint, or the preprocessing changes.

Read the relative column: it is the MSE divided by the target velocity's own magnitude, so 1.0 is
the "predict zero" baseline and anything above 1.0 means the checkpoint is worse than useless as an
initialization.

    python tools/ctflow_transfer_check.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml --n 20
"""

import argparse
import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from echosyn.common import (get_noise, get_vae_scaler, instantiate,
                            instantiate_class_from_config, load_init_weights, scale_latents)
from echosyn.common.mrrate import (_unit, build_text_encoder, encode_conditioning, encode_volume,
                                   list_series, preprocess_volume, read_member, read_report)

# name -> (a, b), mapping a [0, 1] volume into the range under test as volume * a + b.
RANGES = {"[0, 1]": (1.0, 0.0), "[-1, 1]": (2.0, -1.0)}
TIMESTEPS = torch.linspace(0.05, 0.95, 10)


def flow_loss(denoiser, cond, target, embedding, noise):
    """Flow-matching MSE over TIMESTEPS, mirroring lvfm/train.py, absolute and relative."""
    offset = 1e-5
    u = (1 - offset) * noise - target
    losses = []
    for t in TIMESTEPS.to(target.device):
        z_t = (1 - t) * target + (offset + (1 - offset) * t) * noise
        v = denoiser(z_t, timestep=t, cond_image=cond, encoder_hidden_states=embedding).sample
        losses.append(F.mse_loss(v, u).item())
    mse = sum(losses) / len(losses)
    return {"mse": mse, "relative": mse / u.pow(2).mean().item(),
            "mean": target.mean().item(), "std": target.std().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=20, help="Series to test.")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    mri = config.mri
    block = config.globals.target_nframes
    preprocess_args = OmegaConf.to_container(mri.preprocess, resolve=True)
    device = "cuda"

    vae = instantiate(config.vae).eval().to(device, torch.float32)
    vae_scaling = get_vae_scaler(config, "cpu")
    tokenizer, text_encoder = build_text_encoder(mri.text_checkpoint, device)
    denoiser = instantiate_class_from_config(config.denoiser).eval()
    load_init_weights(config, denoiser, logging.getLogger(__name__))
    denoiser = denoiser.to(device)

    # Encode once and keep the latents, so every variant below scores the exact same volumes.
    series = list_series(mri.raw_root, args.split, mri.max_repeats, None, config.seed)[:args.n]
    latents = {name: [] for name in RANGES}
    embeddings = []
    with torch.no_grad():
        for entry in series:
            volume, spacing = preprocess_volume(read_member(entry["archive"], entry["member"]),
                                                entry["plane"], **preprocess_args)
            embeddings.append(_unit(encode_conditioning(
                tokenizer, text_encoder, read_report(entry["archive"], entry["study_uid"]),
                entry["modality"], entry["plane"], spacing, mri.marker_weight,
                mri.text_max_length,
            )).unsqueeze(0))
            for name, (a, b) in RANGES.items():
                z = encode_volume(vae, volume * a + b, mri.vae_batch_size, torch.float32)
                latents[name].append(scale_latents(z[None], vae_scaling))

    # One noise draw per volume, shared by every variant, so all comparisons are paired.
    torch.manual_seed(config.seed)
    noises = [get_noise(z[:, :, block:2 * block], noise_offset=config.get("noise_offset", 0.0))
              for z in latents[next(iter(RANGES))]]

    variants = {}
    for name, zs in latents.items():
        variants[f"{name} FLUX"] = (zs, 0.0, 1.0)
        variants[f"{name} standardized"] = (
            zs, sum(z.mean().item() for z in zs) / len(zs),
            sum(z.std().item() for z in zs) / len(zs))

    print(f"\n{'variant':>24} {'mean':>8} {'std':>8} {'flow MSE':>10} {'relative':>10}")
    with torch.no_grad():
        for name, (zs, shift, scale) in variants.items():
            runs = []
            for z, embedding, noise in zip(zs, embeddings, noises):
                z = ((z - shift) / scale).to(device)
                runs.append(flow_loss(denoiser, z[:, :, :block], z[:, :, block:2 * block],
                                      embedding.to(device), noise.to(device)))
            m = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
            print(f"{name:>24} {m['mean']:8.4f} {m['std']:8.4f} "
                  f"{m['mse']:10.4f} {m['relative']:10.4f}")


if __name__ == "__main__":
    main()
