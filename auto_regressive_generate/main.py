"""
Auto-regressive inference script for CT volume generation.

Usage:
    python auto_regressive_generate/main.py \
        --config /path/to/experiment/config.yaml \
        --ckpt   /path/to/checkpoint/denoiser_ema \
        --embedding /path/to/ct_embedding.pt \
        --output /path/to/output_dir \
        --type full-body

Inference modes:
    full-body   Generate the entire volume from scratch (default).
    gt-head     Use the ground-truth first block, then roll out auto-regressively.
    block-wise  Teacher-forcing: condition each step on the GT block.
"""

import argparse
import os

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from echosyn.common import *
from echosyn.common.mrrate import modality_to_id, plane_to_id
from auto_regressive_generate import LatentAutoregressiveGenerator


def save_video_as_frames(video: torch.Tensor, output_dir: str, n_frames: int = 201):
    """
    Save a video tensor [C, T, H, W] as PNG frames.

    Args:
        video: Tensor of shape [C, T, H, W], pixel values in [0, 255].
        output_dir: Directory to save frames.
        n_frames: Total frames to save (pad last frame if needed).
    """
    os.makedirs(output_dir, exist_ok=True)

    video = video.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
    T = video.shape[0]

    if T >= n_frames:
        video = video[:n_frames]
    else:
        pad = video[-1:].repeat(n_frames - T, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    if video.dtype == torch.uint8:
        video = video.float() / 255.0
    else:
        video = video.clamp(0, 1)

    for i, frame in enumerate(video):
        pil_image = transforms.ToPILImage()(frame)
        pil_image.save(os.path.join(output_dir, f"frame_{i:03d}.png"), format="PNG", optimize=True)

    print(f"Saved {n_frames} frames to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-regressive CT volume generation")
    parser.add_argument("--embedding", type=str, required=True, help="Path to CT embedding .pt file")
    parser.add_argument("--gt-latent", type=str, default=None, help="Path to GT latent .pt file (gt-head / block-wise modes)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (denoiser_ema)")
    parser.add_argument("--output", type=str, default="output_frames", help="Output directory for generated frames")
    parser.add_argument("--modality", type=str, required=True, help="T1w | T2w | FLAIR | SWI | MRA")
    parser.add_argument("--plane", type=str, required=True, help="AXIAL | SAGITTAL | CORONAL")
    parser.add_argument("--modality-cfg-scale", type=float, default=None, help="Overrides config.guidance")
    parser.add_argument("--report-cfg-scale", type=float, default=None, help="Overrides config.guidance")
    parser.add_argument(
        "--type",
        type=str,
        default="full-body",
        choices=["full-body", "gt-head", "block-wise"],
        help="Inference type",
    )
    parser.add_argument(
        "--max-blocks", type=int, default=2,
        help="Blocks to generate (full-body/gt-head) or cap at (block-wise, which otherwise "
             "generates every block the loaded --gt-latent has). Was hardcoded to 20 (full-body) "
             "/ 19 (gt-head) / unbounded (block-wise); defaults low here since this is the debug "
             "script -- pass a bigger value for a real full-length rollout.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Repeat the single loaded --embedding/--modality/--plane this many times to exercise "
             "generate()'s batched path (real per-case batching needs distinct embeddings, which "
             "this single-embedding debug script doesn't take -- see submission/predict.py for that).",
    )
    parser.add_argument(
        "--trim", action=argparse.BooleanOptionalAction, default=True,
        help="Trim stop frames from the generated output (generator.trim / generate_next_block's "
             "own trim= in block-wise mode). --no-trim keeps every generated frame, including the "
             "blank tail past the stop point.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load denoiser
    denoiser = instantiate_class_from_config(config.denoiser)
    denoiser = denoiser.from_pretrained(args.ckpt).to(device).eval()

    # Load VAE
    vae = instantiate(config.vae).eval().to(device)
    vae_scaling = get_vae_scaler(config, device)

    # Load embedding
    prompt_embedding = torch.load(args.embedding, map_location=device)
    prompt_embedding = prompt_embedding.unsqueeze(0)
    prompt_embedding = prompt_embedding / (prompt_embedding.norm(p=2) + 1e-6)

    # Class labels. Resolved through the shared tables, so an id means the same thing here as it
    # did in training, and the checkpoint's own mapping is checked against them.
    check_label_mapping(config)
    modality_id = torch.tensor([modality_to_id(args.modality)], device=device)
    plane_id = torch.tensor([plane_to_id(args.plane)], device=device)

    # Repeat the single loaded case --batch-size times so this debug script can exercise
    # generate()'s real batched path -- every sample in the "batch" is the same case/noise seed
    # source, not distinct real cases (see the --batch-size help string).
    prompt_embedding = prompt_embedding.repeat(args.batch_size, 1, 1)
    modality_id = modality_id.repeat(args.batch_size)
    plane_id = plane_id.repeat(args.batch_size)

    # `is not None`, not `or`: a scale of 0.0 is falsy but meaningful -- it drops that guidance term
    # entirely (s_rep=0 generates from modality alone), and `or` silently fell back to the config.
    guidance = config.get("guidance", {})
    modality_cfg_scale = (args.modality_cfg_scale if args.modality_cfg_scale is not None
                          else guidance.get("modality_cfg_scale", 1.0))
    report_cfg_scale = (args.report_cfg_scale if args.report_cfg_scale is not None
                        else guidance.get("report_cfg_scale", 1.0))

    # Init generator
    generator = LatentAutoregressiveGenerator(
        denoiser=denoiser,
        vae=vae,
        device=device,
        vae_scaling=vae_scaling,
        config=config,
        modality_cfg_scale=modality_cfg_scale,
        report_cfg_scale=report_cfg_scale,
    )
    generator.trim = args.trim

    # Run inference
    if args.type == "full-body":
        result_latent, lengths = generator.generate(
            prompt_embeds=prompt_embedding,
            modality_id=modality_id,
            plane_id=plane_id,
            max_blocks=args.max_blocks,
        )

    elif args.type == "gt-head":
        assert args.gt_latent is not None, "Must provide --gt-latent for gt-head mode"
        gt_latent = torch.load(args.gt_latent, map_location=device)
        if gt_latent.dim() == 4:
            gt_latent = gt_latent.unsqueeze(0)
        gt_latent = gt_latent.to(device)
        block_size = generator.block_size
        # Jiayi's code would add black and white tokens to the gt encoded volumes.
        # That's why she had  the line: gt_first_block = gt_latent[:, :, block_size:2 * block_size, :, :]
        # My preprocessing code did not add black and white tokens, so I just take the first block_size frames!
        gt_first_block = gt_latent[:, :, :block_size, :, :]
        gt_first_block = sample_latents(config, gt_first_block)
        gt_first_block = scale_latents(gt_first_block, vae_scaling)
        gt_first_block = gt_first_block.repeat(args.batch_size, 1, 1, 1, 1)
        result_latent, lengths = generator.generate(
            prompt_embeds=prompt_embedding,
            modality_id=modality_id,
            plane_id=plane_id,
            max_blocks=args.max_blocks,
            gt_first_block=gt_first_block,
        )

    elif args.type == "block-wise":
        assert args.gt_latent is not None, "Must provide --gt-latent for block-wise mode"
        gt_latent = torch.load(args.gt_latent, map_location=device)
        if gt_latent.dim() == 4:
            gt_latent = gt_latent.unsqueeze(0)
        gt_latent = gt_latent.repeat(args.batch_size, 1, 1, 1, 1)
        B, _, T_total, H, W = gt_latent.shape  # dim 1 is 2 * latent_channels, sampled per block below
        block_size = generator.block_size
        num_blocks = min(T_total // block_size - 1, args.max_blocks)

        blocks = []
        for i in range(num_blocks):
            gt_block = gt_latent[:, :, i * block_size:(i + 1) * block_size, :, :]
            gt_block = sample_latents(config, gt_block)
            gt_block = scale_latents(gt_block, vae_scaling)
            next_block, _lengths = generator.generate_next_block(gt_block, prompt_embedding,
                                                                  modality_id, plane_id, trim=args.trim)
            if next_block.shape[2] == 0:
                print(f"[Block-wise] Stop at block {i} (all frames trimmed).")
                break
            blocks.append(next_block)
            if generator.is_stop_frame(next_block).all():
                print(f"[Block-wise] Stop block detected at block {i}")
                break

        if not blocks:
            print("No valid block generated!")
            return
        result_latent = torch.cat(blocks, dim=2)
        # generate_next_block's own trim=args.trim already cut each block as it was produced;
        # this is a second, final pass over the concatenated sequence for lengths consistent
        # with the other two modes (both of which trim once, at the end, via generate()).
        if args.trim:
            result_latent, lengths = generator.trim_stop_frames(result_latent)
        else:
            lengths = [result_latent.shape[2]] * result_latent.shape[0]

    # Decode and save -- one sample at a time (decode_latent's trim_blank_tail is per-sample).
    for i in range(args.batch_size):
        sample_latent = result_latent[i:i + 1, :, :lengths[i]]
        out_dir = args.output if args.batch_size == 1 else f"{args.output}_sample{i}"
        decoded = generator.decode_latent(sample_latent, trim_blank=args.trim)
        save_video_as_frames(decoded[0], out_dir)


if __name__ == "__main__":
    main()
