"""Roll out MRFlow over an MR-RATE split and score it with the official VLM3D challenge metrics.

    python evaluation/main.py --config <experiment>/config.yaml \
        --ckpt <experiment>/checkpoint-N/denoiser_ema --split val --limit 32

Every case is derived from the raw MR-RATE archives in one read, so no preprocessing pass is
needed and `test` works the same as `val`:

    ground truth   `load_native_volume` -- RAS-reoriented, plane-first, otherwise the released
                   volume. NOT resampled, normalized or cropped: the official metric normalizes
                   both volumes itself and resamples the *generated* one onto this shape, which is
                   exactly what the leaderboard does to a submission.
    cases          `list_series` at `max_repeats=1` -- eligible MR-RATE series, one acquisition
                   per contrast and plane, so `n_total_files` counts series rather than the files
                   in the platform's ground-truth directory. Deduplicating here and not from
                   `mri.max_repeats` is deliberate; see `run_shard`.
    conditioning   the config's own `mri.conditioning`, built through the same
                   `build_conditioner` factory preprocessing uses, so the embedding the model sees
                   here is the one it trained against -- whichever configuration that was.
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
from echosyn.common.mrrate import (build_conditioner, encode_volume, list_series,
                                   load_native_volume, modality_to_id, plane_order, plane_to_id,
                                   preprocess_volume, read_member, read_report, sample_id)
from auto_regressive_generate import LatentAutoregressiveGenerator
from evaluation import METRIC_KEYS, ChallengeAccumulator, combine, comparison_frames


### Inference regimes ###

# Each takes the generator, the case's conditioning embedding, the block budget and a zero-argument
# `gt_latent` that VAE-encodes the case's preprocessed ground truth on demand -- so a regime that
# needs no ground truth never pays for one. Adding a regime is adding an entry here.


def full_body(generator, embedding, labels, max_blocks, gt_latent):
    """Report-to-volume: seed from the black boundary token, roll out until the white one."""
    return generator.generate(embedding, *labels, max_blocks=max_blocks)


def gt_head(generator, embedding, labels, max_blocks, gt_latent):
    """As above, but seeded with the volume's own first block instead of the black token."""
    # gt_latent() carries both posterior parameters; scaling it whole would scale the stds too.
    first_block = sample_latents(generator.config, gt_latent()[:, :, :generator.block_size])
    first_block = scale_latents(first_block, generator.vae_scaling)
    return generator.generate(embedding, *labels, max_blocks=max_blocks - 1,
                              gt_first_block=first_block)


REGIMES = {"full-body": full_body, "gt-head": gt_head}


def gt_latent(generator, config, nii_bytes, entry):
    """The case's ground truth in latent space, for the regimes that seed from it. Encoded through
    the same preprocess-then-VAE path training used, so a seed block is in-distribution."""
    volume, _ = preprocess_volume(nii_bytes, entry["plane"],
                                  **OmegaConf.to_container(config.mri.preprocess, resolve=True))
    latent = encode_volume(generator.vae, volume, config.mri.vae_batch_size)
    return latent.unsqueeze(0).float().to(generator.device)


### Case selection ###


def select_cases(series, n_per_bucket):
    """The first `n_per_bucket` cases of every (modality, plane) bucket, each bucket ordered by
    `(study_uid, series_id)`.

    `list_series` shuffles on a fixed seed, so its order is reproducible but not balanced: an
    interleaved slice of it is whatever the split's modality mix happens to be. This instead
    matches `R2V-MR-Generation`'s `select_eval_cases` -- ordered by a property of the data rather
    than of the parquet rows, so no RNG is involved at all, and every prefix is bucket-balanced.
    On MR-RATE's test split 10 of the 12 buckets hold a scored modality, so 100 per bucket is
    1,000 scored cases -- measured to be R2V's set exactly, case for case, and its full-split
    population likewise (29,016 scored series both sides).

    Out-of-scope modalities (MRA) are capped like any other bucket rather than dropped, so
    `n_excluded_out_of_scope_modality` still reports them. They cost nothing -- `is_scored` skips
    them before the rollout.
    """
    buckets = {}
    for entry in series:
        buckets.setdefault((entry["modality"], entry["plane"]), []).append(entry)

    selected = []
    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda e: (str(e["study_uid"]), str(e["series_id"])))
        selected.extend(ordered[:n_per_bucket])
    return selected


def save_case(root, bucket, case_id, real, produced, report, entry, spacing):
    """One case's ground truth, generated volume and report, for offline inspection.

    Both volumes are written in the plane-first axis order the metric compares them in, with a
    diagonal affine built from `plane_order`-permuted spacing -- `read_canonical` returns spacing
    in `(S, R, A)` while `load_native_volume` permutes only the array, so the two have to be
    realigned here or a viewer shows the wrong aspect ratio. The generated volume is 1 mm isotropic
    by construction.

    **The shapes differ on purpose.** `compute_basic_metrics` zooms the generated volume onto the
    ground truth's shape rather than resampling the reference, so what is written here is what each
    side actually was, not a registered pair.
    """
    import nibabel as nib

    directory = os.path.join(root, f"{bucket}-{case_id}")
    os.makedirs(directory, exist_ok=True)
    gt_spacing = [spacing[i] for i in plane_order(entry["plane"])]
    nib.save(nib.Nifti1Image(real, np.diag(gt_spacing + [1.0])),
             os.path.join(directory, "ground_truth.nii.gz"))
    nib.save(nib.Nifti1Image(produced, np.eye(4)),
             os.path.join(directory, "generated.nii.gz"))
    with open(os.path.join(directory, "case.json"), "w") as handle:
        json.dump({"case_id": case_id, "bucket": bucket, "modality": entry["modality"],
                   "plane": entry["plane"], "study_uid": entry["study_uid"],
                   "series_id": entry["series_id"],
                   "gt_shape": list(real.shape), "gt_spacing_mm": gt_spacing,
                   "generated_shape": list(produced.shape), "generated_spacing_mm": [1.0, 1.0, 1.0],
                   "report": report}, handle, indent=2)


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
    parser.add_argument("--n_per_bucket", type=int, default=None,
                        help="Cases per (modality, plane) bucket, applied before sharding. "
                             "100 reproduces the R2V-MR-Generation population (1,000 scored "
                             "cases). Default: the whole split.")
    parser.add_argument("--max_blocks", type=int, default=None,
                        help="Rollout budget. Default: max_slices / target_nframes.")
    parser.add_argument("--examples", type=int, default=2,
                        help="Ground-truth-vs-generated mp4s this shard keeps, for W&B.")
    parser.add_argument("--save_volumes", type=int, default=0,
                        help="Cases of this shard to write to <out>/volumes/ as ground_truth."
                             "nii.gz + generated.nii.gz + case.json (report included). Off by "
                             "default: a native-geometry volume is ~150 MB before compression.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run an existing shard.")
    parser.add_argument("--combine", action="store_true",
                        help="Pool the shards already in --out into metrics.json and log to W&B.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False,
                        help="Run the denoiser under bfloat16 autocast (the VAE stays fp32). ~3.2x "
                             "faster per block, and the precision training itself ran in -- but it "
                             "does not reproduce an fp32 sample: measured, one denoiser call "
                             "differs by 1.9e-2 relative and a rollout lands 23.6 dB from its fp32 "
                             "counterpart. Off, so scores stay comparable with everything already "
                             "measured; turn it on only for both sides of a comparison.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="torch.compile the denoiser, with static shapes. Unlike --bf16 this "
                             "leaves the sampler alone (measured 1.91e-2 relative velocity "
                             "difference with bf16 on, against 1.90e-2 for bf16 alone -- i.e. "
                             "compile contributes nothing to the drift). The first two or three "
                             "blocks of a run are still compiling, so pass --no-compile for a "
                             "handful of cases.")
    parser.add_argument("--ode_steps", type=int, default=201,
                        help="Fixed Euler steps per block. 201 is what every number in this "
                             "project was produced with, so leave it there to stay comparable; "
                             "lowering it is the cheapest speed lever, at a cost this flag exists "
                             "to measure. Exposed here only -- the debug script and the "
                             "submission container both stay on the 201-step default.")
    parser.add_argument("--modality_cfg_scale", type=float, default=None,
                        help="Overrides config.guidance.modality_cfg_scale.")
    parser.add_argument("--report_cfg_scale", type=float, default=None,
                        help="Overrides config.guidance.report_cfg_scale.")

    args = parser.parse_args()
    if not args.combine and not args.ckpt:
        parser.error("--ckpt is required unless --combine")
    return args


def build_generator(config, ckpt, device, args):
    """The same generator `auto_regressive_generate/main.py` builds, from the same config, plus the
    three speed switches -- `--compile`, which does not change the sample, and `--bf16` and
    `--ode_steps`, which do. Everything else about the rollout is fixed: fp32 unless asked, one
    case at a time. At their defaults (no bf16, 201 steps) a run is the same sampler every number
    in this project was produced with; the debug script and the submission get no such knobs."""
    denoiser = instantiate_class_from_config(config.denoiser)
    denoiser = denoiser.from_pretrained(ckpt).to(device).eval()
    vae = instantiate(config.vae).eval().to(device)
    return LatentAutoregressiveGenerator(
        denoiser=denoiser, vae=vae, device=device,
        vae_scaling=get_vae_scaler(config, device), config=config,
        block_size=config.globals.target_nframes,
        modality_cfg_scale=config.guidance.modality_cfg_scale,
        report_cfg_scale=config.guidance.report_cfg_scale,
        use_bf16=args.bf16, use_compile=args.compile, ode_steps=args.ode_steps,
    )


def run_shard(config, args, out, device):
    """Generate and score this shard's slice of the split; write its state for `--combine`."""
    mri = config.mri
    series = list_series(mri.raw_root, args.split, 1,
                         mri.get(f"max_series_{args.split}"), config.seed)
    # Before the shard slice, so every task carves its cases out of the same selected population.
    if args.n_per_bucket:
        series = select_cases(series, args.n_per_bucket)
    series = series[args.shard::args.num_shards][:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(series)} {args.split} cases, "
          f"regime {args.regime}, max_blocks {args.max_blocks}, "
          f"n_per_bucket {args.n_per_bucket}, bf16 {args.bf16}, compile {args.compile}, "
          f"ode_steps {args.ode_steps}")

    generator = build_generator(config, args.ckpt, device, args)
    conditioner = build_conditioner(mri, device)
    accumulator = ChallengeAccumulator(device=device)
    generate = REGIMES[args.regime]
    examples_left = args.examples
    if examples_left:
        os.makedirs(os.path.join(out, "examples"), exist_ok=True)
    volumes_left = args.save_volumes

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
            report = read_report(entry["archive"], entry["study_uid"])
            embedding = conditioner.encode(report, entry["modality"], entry["plane"])
            embedding = (embedding / (embedding.norm(p=2) + 1e-6)).unsqueeze(0).to(device)

            labels = (torch.tensor([modality_to_id(entry["modality"])], device=device),
                      torch.tensor([plane_to_id(entry["plane"])], device=device))
            latent = generate(generator, embedding, labels, args.max_blocks,
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

        accumulator.add(case_id, bucket, entry["modality"], real, produced, spacing,
                        entry["plane"])

        if volumes_left > 0:
            volumes_left -= 1
            save_case(os.path.join(out, "volumes"), bucket, case_id, real, produced, report,
                      entry, spacing)

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
    kept. Table row order is METRIC_KEYS -- the headline block first (both FVDs, both FIDs, IS,
    then PSNR/MSE/SSIM), then the strata splits, the per-plane FIDs and the counts.

    The guidance scales go in the run *name* as well as the config. A cfg sweep is several runs
    over one checkpoint that differ in nothing else, so without them in the label the run table is
    a column of identical names -- and at 1.0/1.0 the sampler takes its single-conditional
    short-circuit, i.e. no guidance at all, which is worth being able to see at a glance."""
    guidance = config.guidance
    run = wandb.init(
        project=config.wandb_args.project,
        name=f"eval-{args.regime}-{args.split}"
             f"-mod{guidance.modality_cfg_scale:g}-rep{guidance.report_cfg_scale:g}"
             f"-{config.wandb_args.name}",
        group=config.wandb_args.group,
        mode="disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "online"),
        config={"regime": args.regime, "split": args.split, "ckpt": args.ckpt,
                "max_blocks": args.max_blocks, "n_per_bucket": args.n_per_bucket,
                "modality_cfg_scale": guidance.modality_cfg_scale,
                "report_cfg_scale": guidance.report_cfg_scale},
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

    # CLI guidance scales override the config, so a sweep is one sbatch argument.
    check_label_mapping(config)
    for name in ("modality_cfg_scale", "report_cfg_scale"):
        if getattr(args, name) is not None:
            config.guidance[name] = getattr(args, name)

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
