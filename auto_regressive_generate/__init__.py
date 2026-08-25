import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint

from echosyn.common import *
from echosyn.common.mrrate import CFG_NULL_MODALITY_ID


class LatentAutoregressiveGenerator:
    """
    Auto-regressive generator for latent video volumes.

    Generates a sequence of latent blocks conditioned on:
    - A text/CT embedding (prompt_embeds)
    - The previously generated block (cond_image)
    - Modality and plane class ids

    Uses flow matching (Euler ODE) for denoising.
    """

    def __init__(
        self,
        denoiser,
        vae,
        device,
        vae_scaling,
        config,
        block_size: int = 16,
        overlap: int = 8,
        eps: float = 0.1,
        modality_cfg_scale: float = 1.0,
        report_cfg_scale: float = 1.0,
    ):
        self.denoiser = denoiser
        self.vae = vae
        self.device = device
        self.vae_scaling = vae_scaling
        self.config = config
        self.block_size = block_size
        self.overlap = overlap
        self.eps = eps
        self.modality_cfg_scale = modality_cfg_scale
        self.report_cfg_scale = report_cfg_scale
        self.dtype = torch.float32
        self.trim = True

        res = self.config.globals.resolution
        self.zero_latent = self.encode_image(torch.full((1, 3, res, res), float(self.config.black_value)))

        one_latent = self.encode_image(torch.full((1, 3, res, res), float(self.config.white_value)))
        one_latent = sample_latents(self.config, one_latent)
        one_latent = scale_latents(one_latent, vae_scaling) # white token is scaled now
        self.one_latent = one_latent

    def encode_image(self, img):
        with torch.no_grad():
            img = img.to(self.device).to(self.dtype)
            img = self.vae.encode(img).latent_dist.sample()
        return img

    def decode_latent(self, latents, max_batch_size=64):
        with torch.no_grad():
            latents = unscale_latents(latents, self.vae_scaling)
            b, c, t, h, w = latents.shape
            latents = rearrange(latents, "b c t h w -> (b t) c h w")

            decoded_chunks = []
            for chunk in latents.split(max_batch_size, dim=0):
                decoded = self.vae.decode(chunk.float()).sample
                decoded_chunks.append(decoded)

            latents = torch.cat(decoded_chunks, dim=0)
            latents = to_uint8_frames(latents, self.config).cpu()
            latents = rearrange(latents, "(b t) c h w -> b c t h w", b=b)
            print("Decoded latents shape:", latents.shape)

        return latents

    def velocity(self, t, y, prompt_embeds, cond_image, modality_id, plane_id):
        """One velocity for the ODE right-hand side, guided over modality and report.

        Three branches, batched into a single denoiser call:

            BASE      modality CFG_NULL, null report
            MODALITY  real modality,     null report
            FULL      real modality,     real report

            v = v_base + s_mod * (v_modality - v_base) + s_rep * (v_full - v_modality)

        Everything non-semantic is the same tensor repeated -- the noisy latent, the timestep, the
        previous block and the plane -- so each difference isolates one condition. Plane is never
        nulled: it is geometry, not semantics.

        At both scales exactly 1 the sum collapses to v_full, so the single conditional call is
        taken instead of paying for three.
        """
        s_mod, s_rep = self.modality_cfg_scale, self.report_cfg_scale
        if s_mod == 1.0 and s_rep == 1.0:
            return self.denoiser(
                y, t, encoder_hidden_states=prompt_embeds, cond_image=cond_image,
                modality_id=modality_id, plane_id=plane_id,
            ).sample

        K = 3
        null_modality = torch.full_like(modality_id, CFG_NULL_MODALITY_ID)
        drop = torch.tensor([True, True, False], device=y.device)
        v = self.denoiser(
            y.repeat(K, 1, 1, 1, 1), t,
            encoder_hidden_states=prompt_embeds.repeat(K, 1, 1),
            cond_image=cond_image.repeat(K, 1, 1, 1, 1),
            modality_id=torch.cat([null_modality, modality_id, modality_id]),
            plane_id=plane_id.repeat(K),
            force_drop_ids=drop.repeat_interleave(modality_id.shape[0]),
        ).sample

        v_base, v_modality, v_full = v.chunk(K)
        return v_base + s_mod * (v_modality - v_base) + s_rep * (v_full - v_modality)

    def is_stop_frame(self, latent: torch.Tensor):
        """Check if the last block is all-white (stop signal)."""
        last_frame = latent[:, :, 0, :, :]
        target = self.one_latent.expand_as(last_frame)
        if_stop = torch.mean((torch.abs(last_frame - target) < self.eps).float())
        return if_stop > 0.9

    def trim_stop_frames(self, latent: torch.Tensor):
        """Remove trailing all-white frames from the end of the volume."""
        B, C, T, H, W = latent.shape
        target = self.one_latent.expand(B, C, H, W)

        keep_until = T
        for i in reversed(range(T)):
            frame = latent[:, :, i, :, :]
            if torch.mean((torch.abs(frame - target) < self.eps).float()) > 0.9:
                keep_until -= 1
            else:
                break

        return latent[:, :, :keep_until, :, :]

    def generate(self, prompt_embeds, modality_id, plane_id, max_blocks=30, gt_first_block=None):
        """
        Generate a full latent volume auto-regressively.

        Args:
            prompt_embeds: Text/CT embedding [B, 1, D].
            modality_id: Modality class ids [B], dtype long.
            plane_id: Plane class ids [B], dtype long.
            max_blocks: Maximum number of blocks to generate.
            gt_first_block: Optional ground-truth first block for gt-head inference mode.

        Returns:
            Tensor of shape [B, C, T, H, W].
        """
        self.denoiser.eval()

        B = 1
        C = self.config.globals.latent_channels
        H = W = self.config.globals.latent_res
        T = self.block_size

        if gt_first_block is None:
            init_block = self.zero_latent.unsqueeze(0).permute(0, 2, 1, 3, 4)
            init_block = init_block.repeat(B, 1, self.block_size, 1, 1)
            init_block = scale_latents(init_block, self.vae_scaling)
        else:
            init_block = gt_first_block

        blocks = [init_block]
        cur_step = 0
        block_times = []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            while cur_step < max_blocks:
                cur_latent = torch.cat(blocks, dim=2) # concatenate over T dimension

                z = torch.randn((B, C, T, H, W), device=self.device, dtype=self.dtype)

                cond_image_latent = cur_latent[:, :, -self.block_size:, :, :]
                cond_image_latent = sample_latents(self.config, cond_image_latent)

                def rhs(t, y):
                    return self.velocity(t, y, prompt_embeds, cond_image_latent,
                                         modality_id, plane_id)

                timesteps = torch.linspace(1.0, 0.0, steps=201, device=self.device, dtype=self.dtype)
                start_time = time.time()
                new_block = odeint(
                    rhs,
                    z,
                    timesteps,
                    atol=1e-5,
                    rtol=1e-5,
                    adjoint_params=self.denoiser.parameters(),
                    method="euler",
                )[-1]
                elapsed = time.time() - start_time
                block_times.append(elapsed)

                blocks.append(new_block)
                cur_step += 1

                if self.overlap > 0:
                    last_overlap = new_block[:, :, -self.overlap:, :, :] # takes the last `overlap` frames of the new block
                    blocks[-1] = new_block[:, :, :-self.overlap, :, :] # takes the first `overlap` frames of the new block
                    blocks.append(last_overlap) # final result that the first and last 8 slices are added as separate chuncks to `blocks`

                if self.is_stop_frame(new_block):
                    print("[Generator] Stop block detected. Ending generation.")
                    break

        full_latent = torch.cat(blocks, dim=2)
        if gt_first_block is None and full_latent.shape[2] > 16:
            full_latent = full_latent[:, :, 16:]  # remove the zero-padded init block
        if self.trim:
            full_latent = self.trim_stop_frames(full_latent)

        if block_times:
            avg_time = sum(block_times) / len(block_times)
            total_time = sum(block_times)
            print(
                f"[Timing] Avg block time: {avg_time:.3f}s | "
                f"Total: {total_time:.3f}s | Blocks: {len(block_times)}"
            )

        return full_latent

    def generate_next_block(self, prev_latent, prompt_embeds, modality_id, plane_id):
        """
        Generate a single next block (used in block-wise inference mode).

        Args:
            prev_latent: Previous block latent [B, C, T, H, W].
            prompt_embeds: Text/CT embedding [B, 1, D].
            modality_id: Modality class ids [B], dtype long.
            plane_id: Plane class ids [B], dtype long.

        Returns:
            Next block latent [B, C, T', H, W] with stop frames trimmed.
        """
        B = 1
        C = self.config.globals.latent_channels
        H = W = self.config.globals.latent_res
        T = self.block_size

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            z = torch.randn((B, C, T, H, W), device=self.device, dtype=self.dtype)

            def rhs(t, y):
                return self.velocity(t, y, prompt_embeds, prev_latent, modality_id, plane_id)

            # Same euler/201 schedule as `generate`, so block-wise numbers stay comparable to a
            # full rollout -- an adaptive solver's step count depends on the guided field.
            timesteps = torch.linspace(1.0, 0.0, steps=201, device=self.device, dtype=self.dtype)
            start_time = time.time()
            new_block = odeint(
                rhs,
                z,
                timesteps,
                atol=1e-5,
                rtol=1e-5,
                adjoint_params=self.denoiser.parameters(),
                method="euler",
            )[-1]
            elapsed = time.time() - start_time
            print(f"[Timing] Single block generation time: {elapsed:.3f}s")

        new_block = self.trim_stop_frames(new_block)
        return new_block
