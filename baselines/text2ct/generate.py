"""Generate the frozen population with a fine-tuned Text2CT checkpoint, one NIfTI per case.

    python -m baselines.text2ct.generate \
        --checkpoint $WS/baselines/runs/text2ct_mrrate_ft/checkpoint-0060000.pt \
        --cases baselines/cases-val-n100.json \
        --prompts $WS/baselines/prompts-val-n100.json \
        --out $WS/baselines/runs/text2ct_val_step60000/nifti \
        --shard 0 --num_shards 16

Writes `<out>/<case_id>.nii.gz` and nothing else, exactly as `baselines/nvidia_r2v/generate.py`
does. `baselines.common.ingest` turns those into the evaluation cache; this script never sees a
ground truth, never scores, and never imports MRFlow's evaluation.

**The sampling loop is upstream's, not a reinterpretation of it.** `scripts/diff_model_infer.py`'s
`run_inference` is reproduced call for call -- `RFlowScheduler.set_timesteps` with
`input_img_size_numel`, the `(t, next_t)` pairing, the zero-context unconditional branch,
`uncond + s * (cond - uncond)`, then `ReconModel` under upstream's own `SlidingWindowInferer` --
and `ReconModel`/`dynamic_infer` are imported from the pinned clone rather than restated.

**Four things this script must get right; each is a silent failure, never an error:**

  the report string    must be byte-identical to what the fine-tune trained on, or the frozen
                       encoder is being asked a different question than it was tuned against. It is
                       built by the same `format_report` with the settings read back out of the
                       checkpoint's own config, not from a default.
  the modality id      comes from the checkpoint's recorded `modality_vocabulary`. A class id is
                       meaningless without the table it was trained under, so a mismatch raises.
  no HU mapping        `diff_model_demo.py:194` maps the decoder's [0, 1] to [-1000, 1000] HU. That
                       is a CT convention; MR has no Hounsfield scale and the fine-tune's targets
                       were [0, 1]. Writing HU here would be a silent 2000x offset that still
                       produces plausible metrics. This is the analogue of nvidia_r2v's
                       `postprocess=False`.
  the geometry         `save_generated_nifti` inverts `preprocess_volume` exactly, so `ingest.py`'s
                       `read_canonical` recovers the orientation the model generated in. Verified
                       round-trip, and at 0.985 correlation against MRFlow's own grid on a real
                       series.

**Seeding is off the case id**, as `evaluation/main.py` and the R2V driver both do, so a rerun at a
different `--num_shards` draws the same noise and reproduces the run.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "baselines.text2ct"

from .config import grid_spacing, latent_shape, load_config, weight_paths
from .model import _add_upstream_to_path, build_noise_scheduler, build_unet, build_vae
from .mrrate_data import MODALITY_TO_ID, format_report, modality_to_id, save_generated_nifti
from .text_encoder import build_text_encoder, encode_reports, null_context

# Modalities the evaluation scores. `EvalAccumulator.is_scored` is the authority and `ingest.py`
# calls it; this copy only decides which cases are worth GPU time, so drift costs minutes, never a
# wrong number.
SCORED_MODALITIES = ("t1w", "t2w", "flair", "swi")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a frozen population with a fine-tuned "
                                                 "Text2CT checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint-NNNNNNN.pt from fine_tune.py")
    parser.add_argument("--cases", required=True, help="baselines/cases-*.json")
    parser.add_argument("--prompts", required=True, help="the matching prompts-*.json")
    parser.add_argument("--out", required=True, help="Directory for <case_id>.nii.gz")
    parser.add_argument("--config", default=None,
                        help="Overrides the config stored in the checkpoint. Rarely right.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N cases of this shard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Upstream's own default (config_diff_model.json).")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Report CFG. Defaults to the checkpoint's cfg.guidance_scale (5.0, "
                             "upstream's own). 1.0 is the plain conditional prediction.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate cases whose NIfTI already exists. Off, so a requeued "
                             "array task resumes instead of redoing its shard.")
    return parser.parse_args()


def load_checkpoint_config(path, override=None):
    """The config the checkpoint was trained under, and its recorded modality vocabulary.

    Read from the checkpoint rather than from a config file on disk: the grid, the report sections
    and the class ids are properties of these weights, and a file can have moved on since.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = load_config(override) if override else payload["config"]
    vocabulary = payload.get("modality_vocabulary", {}).get("modality_to_id")
    if vocabulary != MODALITY_TO_ID:
        raise SystemExit(
            f"{path} was trained with a different modality vocabulary:\n"
            f"  checkpoint: {vocabulary}\n  current:    {MODALITY_TO_ID}\n"
            f"A class id means nothing without the table it was trained under.")
    return config, payload["unet_state_dict"], float(payload["scale_factor"]), payload


class Text2CTSampler:
    """The released model with fine-tuned UNet weights, driven as `diff_model_infer.py` drives it."""

    def __init__(self, config, state_dict, scale_factor, device, num_inference_steps,
                 guidance_scale):
        self.config, self.device = config, torch.device(device)
        self.scale_factor = scale_factor
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        paths = weight_paths(config)

        self.unet, _ = build_unet(config["text2ct_root"], paths["unet"], self.device,
                                  modality_init="keep", model_def=config["model_def"])
        missing, unexpected = self.unet.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise SystemExit(f"fine-tuned weights do not match the model definition: "
                             f"missing {list(missing)[:4]}, unexpected {list(unexpected)[:4]}")
        self.unet.eval()
        self.vae = build_vae(config["text2ct_root"], paths["vae"], self.device,
                             config["model_def"], fast_concat=config["model"]["vae_fast_concat"])
        self.encoder = build_text_encoder(config["text2ct_root"], paths["clip"], self.device,
                                          config["data"]["text_max_length"])

        _add_upstream_to_path(config["text2ct_root"])
        from monai.inferers.inferer import SlidingWindowInferer

        from scripts.sample import ReconModel
        from scripts.utils import dynamic_infer

        self._dynamic_infer = dynamic_infer
        self.recon = ReconModel(autoencoder=self.vae, scale_factor=scale_factor).to(self.device)
        # Upstream's own decode settings (diff_model_infer.py); at this latent size `dynamic_infer`
        # takes the sliding-window branch, which is what the released model does too.
        self.inferer = SlidingWindowInferer(roi_size=[80, 80, 80], sw_batch_size=1, progress=False,
                                            mode="gaussian", overlap=0.4, sw_device=self.device,
                                            device=self.device)
        self.spacing = torch.tensor(grid_spacing(config), device=self.device).float().mul(1e2)
        self.latent_shape = latent_shape(config)

    @torch.inference_mode()
    def generate(self, text, modality, seed):
        """-> `(X, Y, Z)` float32 in the decoder's own range. No HU mapping; see the module docstring."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
        context = encode_reports(self.encoder, [text]).to(self.device)
        labels = torch.tensor([modality_to_id(modality)], device=self.device, dtype=torch.long)

        scheduler = build_noise_scheduler(self.config["text2ct_root"], self.config["model_def"])
        scheduler.set_timesteps(num_inference_steps=self.num_inference_steps,
                                input_img_size_numel=torch.prod(torch.tensor(latent.shape[2:])))
        timesteps = scheduler.timesteps
        next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))

        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            for t, next_t in zip(timesteps, next_timesteps):
                kwargs = {"timesteps": torch.Tensor((t,)).to(self.device),
                          "spacing_tensor": self.spacing[None], "class_labels": labels}
                cond = self.unet(x=latent, context=context, **kwargs)
                if self.guidance_scale != 1.0:
                    uncond = self.unet(x=latent, context=null_context(1, self.device), **kwargs)
                    cond = uncond + self.guidance_scale * (cond - uncond)
                latent, _ = scheduler.step(cond, t, latent, next_t)
            volume = self._dynamic_infer(self.inferer, self.recon, latent)
        return volume.squeeze().float().cpu().numpy()


def main():
    args = parse_args()
    config, state_dict, scale_factor, payload = load_checkpoint_config(args.checkpoint, args.config)
    guidance = args.guidance_scale if args.guidance_scale is not None \
        else config["cfg"]["guidance_scale"]

    from baselines.common.cases import load_cases
    from baselines.common.prompts import load_prompts

    cases = load_cases(args.cases)[args.shard::args.num_shards][:args.limit]
    prompts = load_prompts(args.prompts)
    os.makedirs(args.out, exist_ok=True)

    volume_cfg, data_cfg = config["volume"], config["data"]
    print(f"[shard {args.shard}/{args.num_shards}] {len(cases)} cases, "
          f"{args.num_inference_steps} steps, report cfg {guidance}, seed {args.seed}")
    print(f"  checkpoint: {args.checkpoint} (step {payload.get('global_step')}, "
          f"epoch {payload.get('epoch')}, scale_factor {scale_factor})")
    print(f"  grid: {volume_cfg['inplane_size']}^2 x {volume_cfg['num_slices']} at "
          f"{grid_spacing(config)} mm | sections {list(data_cfg['report_sections'])} "
          f"| modality_prefix {data_cfg['modality_prefix']}")

    sampler = Text2CTSampler(config, state_dict, scale_factor, args.device,
                             args.num_inference_steps, guidance)

    done = skipped = failed = 0
    for case in cases:
        target = os.path.join(args.out, f"{case['case_id']}.nii.gz")
        if case["modality"].lower() not in SCORED_MODALITIES:
            skipped += 1
            continue
        if os.path.exists(target) and not args.overwrite:
            skipped += 1
            continue
        try:
            # The training string, rebuilt from the checkpoint's own settings. Plane is absent by
            # design -- this adaptation does not condition on it.
            text = format_report(prompts[case["case_id"]], data_cfg["report_sections"],
                                 case["modality"], data_cfg["modality_prefix"])
            volume = sampler.generate(
                text, case["modality"],
                # Off the case id, not its position, so a rerun at a different --num_shards is the
                # same volume. Matches evaluation/main.py and the R2V driver.
                seed=args.seed + int(case["case_id"], 16) % 2 ** 31)
            save_generated_nifti(volume, case["plane"], target, volume_cfg["inplane_mm"],
                                 volume_cfg["slice_mm"])
            done += 1
        except Exception as error:                 # one bad case must not lose the shard
            print(f"  {case['case_id']} failed: {type(error).__name__}: {error}")
            failed += 1

    print(f"[shard {args.shard}] {done} generated, {skipped} skipped, {failed} failed -> {args.out}")
    with open(os.path.join(args.out, f"_generate-{args.shard:04d}.json"), "w") as handle:
        json.dump({"shard": args.shard, "num_shards": args.num_shards,
                   "checkpoint": args.checkpoint, "step": payload.get("global_step"),
                   "epoch": payload.get("epoch"), "seed": args.seed,
                   "guidance_scale": guidance, "scale_factor": scale_factor,
                   "num_inference_steps": args.num_inference_steps,
                   "grid": [volume_cfg["inplane_size"], volume_cfg["inplane_size"],
                            volume_cfg["num_slices"]], "spacing_mm": list(grid_spacing(config)),
                   "generated": done, "skipped": skipped, "failed": failed}, handle, indent=2)


if __name__ == "__main__":
    main()
