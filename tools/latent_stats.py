"""Per-channel mean and std of the preprocessed MR latents.

FLUX's `scaling_factor` / `shift_factor` were fit on natural images. This says whether they
actually standardize brain MRI latents, which is what the flow-matching objective assumes.

    python tools/latent_stats.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml --limit 200
"""

import argparse
import glob
import os

import torch
from omegaconf import OmegaConf

from echosyn.common import get_vae_scaler


def latent_moments(paths):
    """Per-channel (mean, std) over every latent pixel in `paths`."""
    total, total_sq, count = 0.0, 0.0, 0
    for path in paths:
        x = torch.load(path, map_location="cpu").float().flatten(1)
        total = total + x.sum(1).double()
        total_sq = total_sq + x.pow(2).sum(1).double()
        count += x.shape[1]
    mean = total / count
    return mean, (total_sq / count - mean.pow(2)).sqrt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=200, help="Latent files to read.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    pattern = os.path.join(config.mri.dataset_root, "latents", "*", "*.pt")
    paths = sorted(glob.glob(pattern))[:args.limit]
    if not paths:
        raise SystemExit(f"No latents at {pattern} -- run lvfm/preprocess_mrrate.py first.")

    mean, std = latent_moments(paths)
    scaler = get_vae_scaler(config, "cpu")
    shift, scale = scaler["mean"].item(), scaler["std"].item()

    print(f"\n{len(paths)} volumes, {len(mean)} channels\n")
    print(f"{'ch':>3} {'mean':>9} {'std':>9} |{'scaled mean':>13} {'scaled std':>12}")
    for c in range(len(mean)):
        print(f"{c:>3} {mean[c]:9.4f} {std[c]:9.4f} |"
              f"{(mean[c] - shift) * scale:13.4f} {std[c] * scale:12.4f}")

    pooled_std = (std.pow(2) + mean.pow(2)).mean().sub(mean.mean().pow(2)).sqrt()
    print(f"\nFLUX config:          shift={shift:.4f}  scale={scale:.4f}")
    print(f"Standardizing needs:  shift={mean.mean():.4f}  scale={1 / pooled_std:.4f}")
    print("Scaled std far from 1.0 means the flow objective mixes latents and noise at "
          "mismatched scale.")


if __name__ == "__main__":
    main()
