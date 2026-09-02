import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint

from echosyn.common import *
from echosyn.common.mrrate import CFG_NULL_MODALITY_ID


def safe_compile(model: torch.nn.Module, name: str = "model") -> torch.nn.Module:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    try:
        print(f"[compile] {name} starting...", flush=True)
        start = time.time()
        compiled = torch.compile(model, dynamic=True)
        print(f"[compile] {name} wrapped in {time.time() - start:.1f}s "
              f"(actual compilation happens lazily, on first forward call)")
        return compiled
    except Exception as exc:  # pragma: no cover -- environment-dependent (Triton/Inductor availability)
        print(f"[compile] falling back to eager for {name}: {exc}")
        return model


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
        modality_cfg_scale: float = 1.0,
        report_cfg_scale: float = 1.0,
        ode_steps: int = 201,
        use_bf16: bool = False,
        use_compile: bool = False,
    ):
        self.denoiser = denoiser
        self.vae = vae
        self.device = device
        self.vae_scaling = vae_scaling
        self.config = config
        self.block_size = block_size
        self.modality_cfg_scale = modality_cfg_scale
        self.report_cfg_scale = report_cfg_scale
        self.ode_steps = ode_steps
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.trim = True

        if use_compile:
            self.denoiser = safe_compile(self.denoiser, "denoiser")

        res = self.config.globals.resolution
        self.zero_latent = self.encode_image(torch.full((1, 3, res, res), float(self.config.black_value)))

    def encode_image(self, img):
        with torch.no_grad():
            img = img.to(self.device).float()
            img = self.vae.encode(img).latent_dist.sample()
        return img

    def trim_blank_tail(self, decoded: torch.Tensor, var_thresh: float = 0.005) -> torch.Tensor:
        """Look for near white slices from the back of the one input volume.

        Returns:
            `decoded`, its T axis cut at the first near-uniform frame (or unchanged if none found).
        """
        T = decoded.shape[1]
        for t in range(T):
            if decoded[0, t].float().var() < var_thresh:
                return decoded[:, :max(1, t)]
        return decoded

    def decode_latent(self, latents, max_batch_size=64, trim_blank=True):
        with torch.no_grad():
            latents = unscale_latents(latents, self.vae_scaling)
            b, c, t, h, w = latents.shape
            latents = rearrange(latents, "b c t h w -> (b t) c h w")

            decoded_chunks = []
            for chunk in latents.split(max_batch_size, dim=0):
                decoded = self.vae.decode(chunk.float()).sample
                decoded_chunks.append(decoded)

            latents = torch.cat(decoded_chunks, dim=0)
            latents = rearrange(latents, "(b t) c h w -> b c t h w", b=b)

            if trim_blank:
                if b != 1:
                    raise NotImplementedError(
                        "decode_latent(trim_blank=True) is per-sample -- call it once per "
                        "sample for B > 1, exactly as every current caller already does."
                    )
                latents = self.trim_blank_tail(latents[0]).unsqueeze(0)

            latents = to_uint8_frames(latents, self.config).cpu()
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

    def is_stop_frame(self, latent: torch.Tensor) -> torch.Tensor:
        """Check if the last frame of each sample in the latent tensor is near-uniform (low variance -> blank/padding)."""
        last_frame = latent[:, :, -1, :, :]
        return last_frame.float().var(dim=(1, 2, 3)) < 0.05

    def trim_stop_frames(self, latent: torch.Tensor):
        """Remove trailing near-uniform (blank/padding) frames from the end of the volume, per
        sample -- same variance test as `is_stop_frame`, applied per frame instead of just the
        block's last one, so trimming stays consistent with what stopped generation in the
        first place.

        Returns:
            `(latent, lengths)`: `latent` is `[B, C, T', H, W]`; `lengths` is a `[B]` list of each
            sample's own valid frame count. For `B == 1` this is exactly the previous behaviour --
            `latent[:, :, :lengths[0]]` -- just returned alongside its own length instead of bare.
        """
        B, C, T, H, W = latent.shape

        frame_var = latent.float().var(dim=(1, 3, 4))  # [B, T]
        is_stop = frame_var < 0.05  # [B, T]

        lengths = []
        for b in range(B):
            length = T
            for i in reversed(range(T)):
                if is_stop[b, i]:
                    length -= 1
                else:
                    break
            lengths.append(length)

        keep_until = max(lengths) if lengths else T
        return latent[:, :, :keep_until, :, :], lengths

    def generate_next_block(self, prev_latent, prompt_embeds, modality_id, plane_id, trim=True):
        """
        Generate a single next block. Used both directly (block-wise inference mode) and as
        `generate()`'s own per-step building block for a full rollout.

        Returns:
            `(latent, lengths)`. If `trim`: latent is `[B, C, T', H, W]` with stop frames trimmed
            to the longest sample in the batch, and `lengths` is that batch's own per-sample valid
            frame count (see `trim_stop_frames`). If not `trim`: latent is the raw `[B, C, T, H, W]`
            block and `lengths` is `None`.
        """
        B = prompt_embeds.shape[0]
        C = self.config.globals.latent_channels
        H = W = self.config.globals.latent_res
        T = self.block_size

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            z = torch.randn((B, C, T, H, W), device=self.device, dtype=self.dtype)

            def rhs(t, y):
                return self.velocity(t, y, prompt_embeds, prev_latent, modality_id, plane_id)

            timesteps = torch.linspace(1.0, 0.0, steps=self.ode_steps, device=self.device, dtype=torch.float32)
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

        if not trim:
            return new_block, None
        return self.trim_stop_frames(new_block)

    def generate(self, prompt_embeds, modality_id, plane_id, max_blocks=30, gt_first_block=None):
        """
        Generate a full latent volume auto-regressively, for a batch of `B` independent volumes
        (`B == 1` is the original single-case usage, unchanged in behaviour). Every block after the
        first is produced by `generate_next_block(..., trim=False)` -- the same per-step logic
        block-wise mode uses, reused rather than duplicated -- so there is exactly one place the
        Euler integration for a single block is implemented.

        Volumes that reach their own stop frame early still ride along (untrimmed) for the
        remaining iterations the batch's slowest member needs; trimming to each sample's own
        length happens once, at the end, via `trim_stop_frames`.

        Returns:
            `(latent, lengths)`: latent is `[B, C, T, H, W]`, T padded to the longest sample in
            the batch; lengths is a `[B]` list of each sample's own valid frame count (see
            `trim_stop_frames`). At B == 1 this is exactly the previous return, just no longer
            unwrapped from the tuple for you -- see the callers in auto_regressive_generate/main.py
            and evaluation/main.py for the one-line adjustment.
        """
        self.denoiser.eval()

        B = prompt_embeds.shape[0]

        if gt_first_block is None:
            init_block = self.zero_latent.unsqueeze(0).permute(0, 2, 1, 3, 4)
            init_block = init_block.repeat(B, 1, self.block_size, 1, 1)
            init_block = scale_latents(init_block, self.vae_scaling)
            init_block = init_block.to(self.dtype)
        else:
            init_block = gt_first_block

        blocks = [init_block]
        cur_step = 0
        stopped_at = [None] * B
        block_times = []

        while cur_step < max_blocks:
            cond_image_latent = sample_latents(self.config, blocks[-1])

            start_time = time.time()
            new_block, _ = self.generate_next_block(cond_image_latent, prompt_embeds,
                                                     modality_id, plane_id, trim=False) # trim set to False here so trim doesn't happen in the middle of the roll-out
            block_times.append(time.time() - start_time)

            blocks.append(new_block)
            cur_step += 1

            stop_mask = self.is_stop_frame(new_block)  # [B]
            for i in range(B):
                if stopped_at[i] is None and bool(stop_mask[i]):
                    stopped_at[i] = cur_step
            if all(s is not None for s in stopped_at):
                print("[Generator] Stop block detected for every sample. Ending generation.")
                break

        full_latent = torch.cat(blocks, dim=2)
        if gt_first_block is None and full_latent.shape[2] > 16:
            full_latent = full_latent[:, :, 16:]  # remove the zero-padded init block

        lengths = [full_latent.shape[2]] * B
        if self.trim:
            full_latent, lengths = self.trim_stop_frames(full_latent)

        if block_times:
            avg_time = sum(block_times) / len(block_times)
            total_time = sum(block_times)
            print(
                f"[Timing] batch={B} avg block time: {avg_time:.3f}s | "
                f"Total: {total_time:.3f}s | Blocks: {len(block_times)}"
            )

        return full_latent, lengths
