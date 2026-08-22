"""Roll out MRFlow over an MR-RATE split and score it with the official VLM3D challenge metrics.

    python evaluation/main.py --config <experiment>/config.yaml \
        --ckpt <experiment>/checkpoint-N/denoiser_ema --split val --limit 32

Every case is derived from the raw MR-RATE archives in one read, so no preprocessing pass is
needed and `test` works the same as `val`:

    ground truth   `load_native_volume` -- RAS-reoriented, plane-first, otherwise the released
                   volume. NOT resampled, normalized or cropped: the official metric normalizes
                   both volumes itself and resamples the *generated* one onto this shape, which is
                   exactly what the leaderboard does to a submission.
    cases          `list_series`, the same deterministic, deduplicated series list preprocessing
                   uses -- so `n_total_files` counts eligible MR-RATE series rather than the files
                   in the platform's ground-truth directory.
    conditioning   CXR-BERT over the study's report plus the series' acquisition markers -- the
                   same `encode_conditioning` call preprocessing makes, so the embedding the model
                   sees here is the one it trained against.
    generation     one `REGIMES` entry. The challenge is report-to-volume, so `full-body` is the
                   default; the others exist to diagnose it.

Long rollouts, so this shards like `preprocess_mrrate.py`: each SLURM array task takes an
interleaved slice of the split and writes its own `shard-NNNN.pt`, then one `--combine` pass pools
them -- FID included, since the per-plane distances are computed over every shard's slice features
at once rather than averaged per shard.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from echosyn.common import *
from echosyn.common.mrrate import (build_text_encoder, encode_conditioning, encode_volume,
                                   list_series, load_native_volume, preprocess_volume,
                                   read_member, read_report, sample_id)
from auto_regressive_generate import LatentAutoregressiveGenerator
from evaluation import METRIC_KEYS, ChallengeAccumulator, combine, comparison_frames


### Inference regimes ###

# Each takes the generator, the case's conditioning embedding, the block budget and a zero-argument
# `gt_latent` that VAE-encodes the case's preprocessed ground truth on demand -- so a regime that
# needs no ground truth never pays for one. Adding a regime is adding an entry here.


def full_body(generator, embedding, max_blocks, gt_latent):
    """Report-to-volume: seed from the black boundary token, roll out until the white one."""
    return generator.generate(embedding, max_blocks=max_blocks)


def gt_head(generator, embedding, max_blocks, gt_latent):
    """As above, but seeded with the volume's own first block instead of the black token."""
    first_block = scale_latents(gt_latent()[:, :, :generator.block_size], generator.vae_scaling)
    return generator.generate(embedding, max_blocks=max_blocks - 1, gt_first_block=first_block)


REGIMES = {"full-body": full_body, "gt-head": gt_head}


def gt_latent(generator, config, nii_bytes, entry):
    """The case's ground truth in latent space, for the regimes that seed from it. Encoded through
    the same preprocess-then-VAE path training used, so a seed block is in-distribution."""
    volume, _ = preprocess_volume(nii_bytes, entry["plane"],
                                  **OmegaConf.to_container(config.mri.preprocess, resolve=True))
    latent = encode_volume(generator.vae, volume, config.mri.vae_batch_size)
    return latent.unsqueeze(0).float().to(generator.device)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MRFlow on the VLM3D challenge metrics")
    parser.add_argument("--config", type=str, required=True,
                        help="The config saved into the experiment dir, next to the checkpoints.")
    parser.add_argument("--ckpt", type=str, help="Path to denoiser_ema. Not needed for --combine.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--regime", type=str, default="full-body", choices=list(REGIMES))
    parser.add_argument("--out", type=str, default=None,
                        help="Results dir. Default: <output_dir>/eval/<regime>-<split>.")
    parser.add_argument("--shard", type=int, default=0, help="This task's index.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of tasks.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N cases of this shard.")
    parser.add_argument("--max_blocks", type=int, default=None,
                        help="Rollout budget. Default: max_slices / target_nframes.")
    parser.add_argument("--examples", type=int, default=2,
                        help="Ground-truth-vs-generated mp4s this shard keeps, for W&B.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run an existing shard.")
    parser.add_argument("--combine", action="store_true",
                        help="Pool the shards already in --out into metrics.json and log to W&B.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")

    args = parser.parse_args()
    if not args.combine and not args.ckpt:
        parser.error("--ckpt is required unless --combine")
    return args


def build_generator(config, ckpt, device):
    """The same generator `auto_regressive_generate/main.py` builds, from the same config."""
    denoiser = instantiate_class_from_config(config.denoiser)
    denoiser = denoiser.from_pretrained(ckpt).to(device).eval()
    vae = instantiate(config.vae).eval().to(device)
    return LatentAutoregressiveGenerator(
        denoiser=denoiser, vae=vae, device=device,
        vae_scaling=get_vae_scaler(config, device), config=config,
        block_size=config.globals.target_nframes,
    )


def run_shard(config, args, out, device):
    """Generate and score this shard's slice of the split; write its state for `--combine`."""
    mri = config.mri
    series = list_series(mri.raw_root, args.split, mri.max_repeats,
                         mri.get(f"max_series_{args.split}"), config.seed)
    series = series[args.shard::args.num_shards][:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(series)} {args.split} cases, "
          f"regime {args.regime}, max_blocks {args.max_blocks}")

    generator = build_generator(config, args.ckpt, device)
    tokenizer, text_encoder = build_text_encoder(mri.text_checkpoint, device)
    accumulator = ChallengeAccumulator(device=device)
    generate = REGIMES[args.regime]
    examples_left = args.examples
    if examples_left:
        os.makedirs(os.path.join(out, "examples"), exist_ok=True)

    for entry in tqdm(series, disable=None):
        case_id = sample_id(entry["study_uid"], entry["series_id"])
        bucket = f"{entry['modality']}__{entry['plane']}"
        if not ChallengeAccumulator.is_scored(entry["modality"]):
            accumulator.add_missing(case_id, bucket, entry["modality"])
            continue

        # Seeded off the case rather than its position, so a rerun -- or the same case under a
        # different shard count -- draws the same noise and scores the same. Without this, two
        # evaluations of one checkpoint differ by sampling noise alone.
        torch.manual_seed(config.seed + int(case_id, 16) % 2 ** 31)
        try:
            nii_bytes = read_member(entry["archive"], entry["member"])
            real, spacing = load_native_volume(nii_bytes, entry["plane"])
            embedding = encode_conditioning(
                tokenizer, text_encoder, read_report(entry["archive"], entry["study_uid"]),
                entry["modality"], entry["plane"], spacing, mri.marker_weight, mri.text_max_length,
            )
            embedding = (embedding / (embedding.norm(p=2) + 1e-6)).unsqueeze(0).to(device)

            latent = generate(generator, embedding, args.max_blocks,
                              lambda: gt_latent(generator, config, nii_bytes, entry))
            if latent.shape[2] == 0:
                raise RuntimeError("every generated slice was a stop frame")
            produced = generator.decode_latent(latent)[0, 0].numpy().astype(np.float32)
        except Exception as e:
            # One unreadable series or collapsed rollout must not lose the shard; the official
            # scoring counts it as a missing output, which is the same penalty the platform applies.
            print(f"[shard {args.shard}] {case_id} failed: {type(e).__name__}: {e}")
            accumulator.add_missing(case_id, bucket, entry["modality"])
            continue

        accumulator.add(case_id, bucket, entry["modality"], real, produced)

        if examples_left > 0:
            examples_left -= 1
            save_as_mp4(torch.from_numpy(comparison_frames(real, produced)),
                        os.path.join(out, "examples", f"{bucket}-{case_id}.mp4"),
                        fps=config.globals.target_fps)

    torch.save(accumulator.state(), os.path.join(out, f"shard-{args.shard:04d}.pt"))
    print(f"[shard {args.shard}] {accumulator.n_total} cases, {accumulator.n_missing} missing")


def print_metrics(metrics, out):
    print(f"\n{'=' * 60}")
    for key in METRIC_KEYS:
        print(f"  {key:34s} {metrics[key]}")
    print(f"{'=' * 60}\nfull results -> {out}/metrics.json")


def log_wandb(config, args, metrics, out):
    """The metrics table, the run summary and the scalars, plus whatever example mp4s the shards
    kept. Table row order is METRIC_KEYS -- FID average first, then PSNR/SSIM/MSE, then the
    per-plane FIDs -- which is the order the R2V-MR-Generation baseline logs, so the two models'
    runs read side by side."""
    run = wandb.init(
        project=config.wandb_args.project,
        name=f"eval-{args.regime}-{args.split}-{config.wandb_args.name}",
        group=config.wandb_args.group,
        mode="disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "online"),
        config={"regime": args.regime, "split": args.split, "ckpt": args.ckpt,
                "max_blocks": args.max_blocks},
    )
    examples = sorted(glob(os.path.join(out, "examples", "*.mp4")))
    run.log({
        "challenge_metrics": wandb.Table(columns=["metric", "value"],
                                         data=[[k, metrics[k]] for k in METRIC_KEYS]),
        **metrics,
        **{f"examples/{os.path.basename(p)[:-4]}": wandb.Video(
            p, caption="ground truth | generated") for p in examples},
    })
    run.summary.update(metrics)
    run.finish()
    print(f"W&B: {len(METRIC_KEYS)} metric rows, {len(examples)} example videos")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or os.path.join(config.output_dir, "eval", f"{args.regime}-{args.split}")
    os.makedirs(out, exist_ok=True)
    if args.max_blocks is None:
        args.max_blocks = config.mri.preprocess.max_slices // config.globals.target_nframes

    if not args.combine:
        shard_path = os.path.join(out, f"shard-{args.shard:04d}.pt")
        if os.path.exists(shard_path) and not args.overwrite:
            print(f"{shard_path} already written; pass --overwrite to redo it")
        else:
            run_shard(config, args, out, device)
        if args.num_shards > 1:
            return  # the --combine pass pools every shard, including this one

    shards = sorted(glob(os.path.join(out, "shard-*.pt")))
    states = [torch.load(p, weights_only=False) for p in shards]
    if not states:
        raise SystemExit(f"no shard-*.pt in {out}")
    result = combine(states)
    result.update({"regime": args.regime, "split": args.split, "ckpt": args.ckpt})
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print_metrics(result["metrics"], out)
    log_wandb(config, args, result["metrics"], out)


if __name__ == "__main__":
    main()
