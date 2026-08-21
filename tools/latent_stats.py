"""Per-channel mean and std of the preprocessed MR latents.

FLUX's `scaling_factor` / `shift_factor` were fit on natural images. This says whether they
actually standardize brain MRI latents, which is what the flow-matching objective assumes:
`train.py` calls `scale_latents` on every block before interpolating it with noise, so a scaled
std far from 1.0 means data and noise are mixed at mismatched scale.

Diagnostic only. Do NOT copy the measured moments into the config while `init_from` points at a
CTFlow checkpoint -- the trunk was trained with exactly 0.3611/0.1159, and substituting measured
statistics made zero-shot transfer worse (0.106 -> 0.116, tools/ctflow_transfer_check.py).

    python tools/latent_stats.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
        --split train --limit 200
"""

import argparse
import random

from omegaconf import OmegaConf

from echosyn.common import get_vae_scaler
from echosyn.common.mrrate import load_artifact, read_manifest


def latent_moments(root, rows):
    """Per-channel (mean, std) over every latent pixel in `rows`."""
    total, total_sq, count = 0.0, 0.0, 0
    for row in rows:
        x = load_artifact(root, row, "latent_path").float().flatten(1)
        total = total + x.sum(1).double()
        total_sq = total_sq + x.pow(2).sum(1).double()
        count += x.shape[1]
    mean = total / count
    return mean, (total_sq / count - mean.pow(2)).sqrt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=200, help="Volumes to read.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    root = config.mri.dataset_root
    rows = read_manifest(root, args.split)
    if not rows:
        raise SystemExit(f"No {args.split!r} rows in {root}/manifest -- run preprocessing first.")
    # Sampled, not the first N: shard manifests are read in order, so a prefix is one shard.
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]

    mean, std = latent_moments(root, rows)
    scaler = get_vae_scaler(config, "cpu")
    shift, scale = scaler["mean"].item(), scaler["std"].item()

    print(f"\n{len(rows)} {args.split} volumes, {len(mean)} channels\n")
    print(f"{'ch':>3} {'mean':>9} {'std':>9} |{'scaled mean':>13} {'scaled std':>12}")
    for c in range(len(mean)):
        print(f"{c:>3} {mean[c]:9.4f} {std[c]:9.4f} |"
              f"{(mean[c] - shift) * scale:13.4f} {std[c] * scale:12.4f}")

    pooled_std = (std.pow(2) + mean.pow(2)).mean().sub(mean.mean().pow(2)).sqrt()
    print(f"\nFLUX config:          shift={shift:.4f}  scale={scale:.4f}")
    print(f"Standardizing needs:  shift={mean.mean():.4f}  scale={1 / pooled_std:.4f}")
    print("Diagnostic only -- see the module docstring before changing the config.")


if __name__ == "__main__":
    main()
