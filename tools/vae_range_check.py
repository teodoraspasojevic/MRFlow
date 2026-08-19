"""VAE reconstruction quality with volumes normalized to [-1, 1] vs [0, 1].

Both are affine remaps of the same clipped data, so this round-trips the same slices through both
and scores them back in the pipeline's own [0, 1] units -- scoring each in its own units would
flatter whichever has the smaller dynamic range.

    python tools/vae_range_check.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml --n 20
"""

import argparse

import torch
from omegaconf import OmegaConf

from echosyn.common import instantiate, paired_image_metrics, to_uint8_frames
from echosyn.common.mrrate import list_series, preprocess_volume, read_member

# name -> (a, b), mapping a [0, 1] volume into the range under test as volume * a + b.
RANGES = {"[0, 1]": (1.0, 0.0), "[-1, 1]": (2.0, -1.0)}


def roundtrip(vae, slices, batch_size=16):
    """[T, S, S] -> VAE reconstruction, same shape and range."""
    out = []
    for chunk in slices.split(batch_size):
        rgb = chunk[:, None].repeat(1, 3, 1, 1).to(vae.device, vae.dtype)
        latent = vae.encode(rgb).latent_dist.mean
        out.append(vae.decode(latent).sample[:, 0].float().cpu())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=20, help="Series to test.")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    mri = config.mri
    preprocess_args = OmegaConf.to_container(mri.preprocess, resolve=True)
    vae = instantiate(config.vae).eval().to("cuda", torch.float32)

    scores = {name: [] for name in RANGES}
    series = list_series(mri.raw_root, args.split, mri.max_repeats, None, config.seed)[:args.n]
    with torch.no_grad():
        for entry in series:
            volume, _ = preprocess_volume(read_member(entry["archive"], entry["member"]),
                                          entry["plane"], **preprocess_args)
            reference = to_uint8_frames(volume[None], config)
            for name, (a, b) in RANGES.items():
                recon = (roundtrip(vae, volume[0] * a + b) - b) / a
                scores[name].append(
                    paired_image_metrics(to_uint8_frames(recon[None, None], config), reference))

    print()
    for name, runs in scores.items():
        psnr = sum(r["psnr"] for r in runs) / len(runs)
        ssim = sum(r["ssim"] for r in runs) / len(runs)
        print(f"{name:>8}   PSNR {psnr:6.2f}   SSIM {ssim:.4f}   ({len(runs)} volumes)")


if __name__ == "__main__":
    main()
