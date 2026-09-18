"""Generate the frozen population with the **released** NV-Generate-MR-Brain -- no adapter, no
report, no cross-attention. The conditioning floor.

    python -m baselines.nvidia_r2v.generate_unconditional \
        --cases baselines/cases-test-n100.json \
        --out $WS/baselines/runs/nvidia_stock/nifti --shard 0 --num_shards 16

**This row does not perform the task.** It is a modality-conditioned generator, included so the
report-alignment metrics have a measured zero-point: what HLIP reads when a generator provably never
saw the report. Expect its FID/FVD to be *good* -- an unconditional model is not penalised for
ignoring the report, and conditioning pulls samples off the mode -- so those columns are not
comparable and must be marked as such.

**The sampling loop is NVIDIA's own, unmodified.** `models/nvidia.run_inference` is copied verbatim
from NV-Generate-CTMR in the upstream clone, and its `unet_inputs` carry only `x`, `timesteps`,
`spacing_tensor` and `class_labels` (the modality code) -- there is no `context` key, so the report
cannot enter even in principle. `load_autoencoder_and_unet` is NVIDIA's own `load_models`. Nothing
here re-implements either.

**Everything else is held identical to the conditioned arms** so the comparison is case-for-case:
the same frozen population, the same `GeometryPolicy(per_modality_plane)` grid and spacing, the same
seed-from-case-id (so a rerun at any `--num_shards` reproduces the volumes), the same
`SCORED_MODALITIES` filter, and the same `[0, 1]` output convention -- `run_inference` returns int16
`[0, 1000]` for MR, divided back here exactly as `cli.evaluate`'s generation task does.
"""

import argparse
import json
import os

from baselines.nvidia_r2v.generate import SCORED_MODALITIES

# NVIDIA's own modality class codes for rflow-mr-brain, from `cli/evaluate.py`. Ids may not move.
MODALITY_CODE = {"T1w": 9, "T2w": 10, "FLAIR": 11, "SWI": 20}
# `run_inference` emits int16 [0, 1000] for MR; the percentile space the rest of the pipeline uses
# is [0, 1]. The same constant and the same reason as `cli.evaluate`.
INTENSITY_SCALE = 1000.0


def build(args):
    """-> a `generate(case, dim_xyz, spacing_xyz, seed) -> volume` closure, via NVIDIA's loaders."""
    import numpy as np
    import torch

    from mrrate_r2v.models.nvidia import (DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG,
                                          DEFAULT_NETWORK_CONFIG, load_autoencoder_and_unet,
                                          prepare_tensors, run_inference, set_random_seed)

    autoencoder, unet, scale_factor, cfg = load_autoencoder_and_unet(
        DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG, args.network_config or DEFAULT_NETWORK_CONFIG,
        args.device, autoencoder_checkpoint_override=args.vae_checkpoint,
        unet_checkpoint_override=args.base_checkpoint)
    cfg.cfg_guidance_scale = cfg.diffusion_unet_inference["cfg_guidance_scale"]
    # The latent divisor NVIDIA derives from the network config rather than a literal.
    n_levels = max(1, len(cfg.diffusion_unet_def["num_channels"])
                   if isinstance(cfg.diffusion_unet_def["num_channels"], list)
                   else len(cfg.diffusion_unet_def["attention_levels"]))
    divisor = 2 ** (n_levels - 2)
    top_region, bottom_region, _spacing, _modality = prepare_tensors(cfg, args.device)

    import logging
    log = logging.getLogger("generate_unconditional")

    def generate(case, dim_xyz, spacing_xyz, seed):
        spacing_tensor = torch.from_numpy(
            np.array(spacing_xyz, dtype=float) * 1e2)[None].half().to(args.device)
        modality_tensor = MODALITY_CODE[case["modality"]] * torch.ones(
            (1,), dtype=torch.long).to(args.device)
        set_random_seed(seed)
        with torch.no_grad():
            raw = run_inference(cfg, args.device, autoencoder, unet, scale_factor, top_region,
                                bottom_region, spacing_tensor, modality_tensor, tuple(dim_xyz),
                                divisor, log)
        return raw.astype(np.float32) / INTENSITY_SCALE

    return generate, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base_checkpoint", default="/hnvme/workspace/y100dc19-nvidia-mri-brain/"
                                                     "models/diff_unet_3d_rflow-mr-brain_v0.pt")
    parser.add_argument("--vae_checkpoint", default="/hnvme/workspace/y100dc19-nvidia-mri-brain/"
                                                    "models/autoencoder_v1.pt")
    parser.add_argument("--network_config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from mrrate_r2v.data.geometry import GeometryPolicy, dhw_to_xyz
    from mrrate_r2v.sampling import save_volume

    from baselines.common.cases import load_cases

    cases = load_cases(args.cases)[args.shard::args.num_shards][:args.limit]
    os.makedirs(args.out, exist_ok=True)
    geometry = GeometryPolicy(mode="per_modality_plane")

    print(f"[shard {args.shard}/{args.num_shards}] stock NV-Generate-MR-Brain (report-blind), "
          f"{len(cases)} cases, seed {args.seed}")
    generate, cfg = build(args)
    print(f"  {cfg.diffusion_unet_inference['num_inference_steps']} steps, "
          f"modality cfg {cfg.cfg_guidance_scale}")

    done = skipped = failed = 0
    for case in cases:
        target = os.path.join(args.out, f"{case['case_id']}.nii.gz")
        if case["modality"].lower() not in SCORED_MODALITIES:
            skipped += 1
            continue
        if os.path.exists(target) and not args.overwrite:
            skipped += 1
            continue

        spec = geometry.resolve(case["modality"], case["plane"])
        dim_xyz = [int(v) for v in dhw_to_xyz(spec.target_shape)]
        spacing_xyz = [float(v) for v in dhw_to_xyz(spec.target_spacing)]
        try:
            volume = generate(case, dim_xyz, spacing_xyz,
                              args.seed + int(case["case_id"], 16) % 2 ** 31)
            save_volume(volume, spacing_xyz, target)
            done += 1
        except Exception as e:                 # one bad case must not lose the shard
            print(f"  {case['case_id']} failed: {type(e).__name__}: {e}")
            failed += 1

    print(f"[shard {args.shard}] {done} generated, {skipped} skipped, {failed} failed -> {args.out}")
    with open(os.path.join(args.out, f"_generate-{args.shard:04d}.json"), "w") as handle:
        json.dump({"shard": args.shard, "num_shards": args.num_shards, "arm": "stock",
                   "base_checkpoint": args.base_checkpoint, "seed": args.seed,
                   "report_conditioned": False,
                   "done": done, "skipped": skipped, "failed": failed}, handle, indent=1)


if __name__ == "__main__":
    main()
