"""Generate the frozen population with a trained R2V adapter, one NIfTI per case.

    python -m baselines.nvidia_r2v.generate --arm A \
        --cases baselines/cases-test-n100.json \
        --prompts $WS/baselines/prompts-test-n100.json \
        --out $WS/baselines/runs/nvidia_r2v_armA_cfg7/nifti \
        --shard 0 --num_shards 16

Writes `<out>/<case_id>.nii.gz` and nothing else. `baselines.common.ingest` turns those into the
evaluation cache; this script never sees a ground truth, never scores, and never imports MRFlow's
evaluation.

**The model is driven through `mrrate_r2v`'s own loader, unmodified.** `build_sampler` reads the
adapter checkpoint first -- it names the text encoder, the context width and the adapter geometry --
so nothing is re-specified here and a mismatch raises instead of generating nonsense. The same
entry point `cli.evaluate` and the challenge container use.

**Three things this script must get right, and each is a silent failure if it does not:**

  postprocess=False   `postprocess_mr` maps the decoder's ~[0, 1] to int16 [0, 1000]. Both are
                      valid outputs; `cli.evaluate` scores the former, so that is what is written
                      here. A 1000x offset produces plausible metrics, never an error.
  the trained format  A/B/C were trained on `findings_impression_meta`, whose
                      `[MODALITY]/[PLANE]/[SPACING]` prefix is part of what the model learned.
                      D/E never see a joined string and take three separate section tokens instead.
                      Which of the two applies is read off the embedder (`needs_sections`), not
                      hardcoded per arm.
  the geometry bucket `(modality, plane)` picks the output grid, and the spacing in the text, the
                      numeric `spacing_tensor` and the decoded grid must all agree. All three come
                      from one `GeometryPolicy.resolve` call below.

**Seeding deviates from `cli.evaluate`, deliberately.** That used `args.seed + case.index`, which
ties the noise draw to a case's position in the list -- so the same case drawn under a different
shard count is a different volume. Here it is derived from the case id, as `evaluation/main.py`
does, so a rerun at any `--num_shards` reproduces the run. The cost is that these volumes are not
bit-identical to the 2026-08 sweep; that sweep only produced challenge-family numbers, which are
being recomputed anyway.
"""

import argparse
import json
import os
from types import SimpleNamespace

# Modalities the evaluation scores. `EvalAccumulator.is_scored` is the authority and `ingest.py`
# calls it; this copy only decides which cases are worth GPU time, so drift costs minutes, never a
# wrong number.
SCORED_MODALITIES = ("t1w", "t2w", "flair", "swi")

# Per arm: the run directory, and the report guidance scale that arm was tuned to. Both were
# selected on FID_2p5D over the same 1,000 test cases -- see README.md. Everything else
# (30 steps, modality guidance 10.0) is shared, and is NVIDIA's own default.
ARMS = {
    "A": {"run": "r2v_final_A_cxr_bert_cls", "checkpoint": "adapter_last.pt",
          "report_guidance_scale": 7.0, "report_format": "findings_impression_meta"},
    "E": {"run": "r2v_final_E_report2ct_style_meta", "checkpoint": "adapter_last.pt",
          "report_guidance_scale": 3.0, "report_format": None},
}

WORKSPACE = os.environ.get("R2V_WORKSPACE", "/hnvme/workspace/y100dc19-nvidia-mri-brain")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a frozen population with an R2V adapter.")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--cases", required=True, help="baselines/cases-*.json")
    parser.add_argument("--prompts", required=True, help="the matching prompts-*.json")
    parser.add_argument("--out", required=True, help="Directory for <case_id>.nii.gz")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N cases of this shard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30, help="NVIDIA's own default.")
    parser.add_argument("--modality_guidance_scale", type=float, default=10.0,
                        help="NVIDIA's own default; the cfg sweep never moved it off 10.")
    parser.add_argument("--report_guidance_scale", type=float, default=None,
                        help="Overrides the arm's tuned value. Only for a sweep.")
    parser.add_argument("--adapter", type=str, default=None, help="Overrides the arm's checkpoint.")
    parser.add_argument("--base_checkpoint", type=str,
                        default=f"{WORKSPACE}/models/diff_unet_3d_rflow-mr-brain_v0.pt")
    parser.add_argument("--vae_checkpoint", type=str,
                        default=f"{WORKSPACE}/models/autoencoder_v1.pt")
    parser.add_argument("--device", default="cuda",
                        help="cuda for any real run. cpu builds the sampler and loads every "
                             "checkpoint, which is enough to validate wiring on a login node, but "
                             "the sliding-window decode makes generation impractically slow.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate cases whose NIfTI already exists. Off, so a requeued "
                             "array task resumes instead of redoing its shard.")
    return parser.parse_args()


def build(args, arm):
    """`mrrate_r2v`'s own sampler, built exactly as `cli.evaluate` builds it."""
    from mrrate_r2v.cli.generate_r2v import build_sampler, trained_report_formats

    sampler, embedder, payload = build_sampler(SimpleNamespace(
        base_checkpoint=args.base_checkpoint, vae_checkpoint=args.vae_checkpoint,
        adapter=args.adapter, network_config=None,
        text_encoder=None, text_checkpoint=None, max_report_tokens=None,
        device=args.device, latent_only=False,
        report_guidance_scale=args.report_guidance_scale,
        modality_guidance_scale=args.modality_guidance_scale,
        num_inference_steps=args.num_inference_steps, seed=args.seed,
        batched_guidance=True, allow_base_mismatch=False,
    ))
    needs_sections = bool(getattr(embedder, "needs_sections", False))

    # **The checkpoint's format is a spec, not a name.** Arm A records
    # `'findings_impression_meta,impression_findings_meta'` -- it was trained on a uniform draw
    # between the two orders, so either is in distribution and an exact string match against one of
    # them is wrong. `trained_report_formats` is the package's own parser for that spec.
    #
    # A sectioned arm records no format at all and never sees a joined string, so there is nothing
    # to match: what matters for it is that the embedder wants sections, which is checked instead.
    trained = trained_report_formats(payload)
    if needs_sections:
        if arm["report_format"] is not None:
            raise SystemExit(f"arm {args.arm} is configured with a joined report format but this "
                             f"checkpoint's encoder wants sections -- wrong adapter for this arm")
    elif arm["report_format"] not in trained:
        raise SystemExit(f"arm {args.arm} generates with report_format={arm['report_format']!r}, "
                         f"which this checkpoint was not trained on (trained: {trained}) -- wrong "
                         f"adapter for this arm")
    if args.device == "cpu":
        _enable_cpu_decode(sampler)
    return sampler, needs_sections, payload


def _enable_cpu_decode(sampler):
    """Make the MAISI decoder runnable on CPU. **Wiring checks only -- never a result.**

    NVIDIA's own `config_network_rflow.json` sets `"norm_float16": true`, so MAISI's
    `MaisiGroupNorm3D` emits float16. On CUDA that is harmless: `ReportToVolumeSampler.decode` runs
    under `autocast`, so the convolution that consumes it runs in half too. On CPU `autocast` is
    disabled by design (`amp = self.config.autocast and device.type == "cuda"`), the conv weights
    stay float32, and the decode dies with

        RuntimeError: Input type (c10::Half) and bias type (float) should be the same

    inside `monai/apps/generation/maisi/networks/autoencoderkl_maisi.py`. Measured: turning the flag
    off on the 26 norm modules is the whole fix -- the decode then completes and returns a sensible
    float32 volume. `sample_latent` is unaffected either way.

    Only ever applied under `--device cpu`, which exists so the checkpoint loading, the conditioning
    and the geometry can be validated on a login node. A CPU decode is not numerically identical to
    the GPU one and is far too slow for a real run.
    """
    patched = 0
    for module in sampler.autoencoder.modules():
        if getattr(module, "norm_float16", False):
            module.norm_float16 = False
            patched += 1
    print(f"  [cpu] norm_float16 disabled on {patched} MAISI norm modules -- wiring check only, "
          f"not numerically the GPU path")


def conditioning_for(prompt, case, spacing_xyz, needs_sections, report_format):
    """`(report_text, report_sections)` for one case, composed the way this arm was trained.

    A sectioned arm never sees the joined string, so its acquisition markers go in their own
    `acquisition` token; a joined arm carries the same markers as a text prefix. Both are built by
    `mrrate_r2v.textenc.formats`, so the string produced here is byte-identical to the one the
    Dataset produced during training.
    """
    from mrrate_r2v.data.reports import ReportRecord
    from mrrate_r2v.textenc.formats import format_report, with_acquisition_section

    record = ReportRecord(raw=prompt["report"], clinical_information=prompt["clinical_information"],
                          technique=prompt["technique"], findings=prompt["findings"],
                          impression=prompt["impression"])
    if needs_sections:
        sections = with_acquisition_section(
            {"findings": record.findings, "impression": record.impression},
            case["modality"], case["plane"], spacing_xyz)
        return record.compose(("findings", "impression")), sections
    return format_report(record, report_format, modality=case["modality"], plane=case["plane"],
                         spacing_mm_xyz=spacing_xyz), None


def main():
    args = parse_args()
    arm = ARMS[args.arm]
    if args.adapter is None:
        args.adapter = os.path.join(WORKSPACE, "runs", arm["run"], arm["checkpoint"])
    if args.report_guidance_scale is None:
        args.report_guidance_scale = arm["report_guidance_scale"]
    for path in (args.adapter, args.base_checkpoint, args.vae_checkpoint):
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint: {path}\nThe NVIDIA workspace expires "
                             f"2026-10-13; set R2V_WORKSPACE or --base_checkpoint/--vae_checkpoint "
                             f"to wherever these were staged.")

    from mrrate_r2v.data.geometry import GeometryPolicy, dhw_to_xyz
    from mrrate_r2v.sampling import save_volume

    from baselines.common.cases import load_cases
    from baselines.common.prompts import load_prompts

    cases = load_cases(args.cases)[args.shard::args.num_shards][:args.limit]
    prompts = load_prompts(args.prompts)
    os.makedirs(args.out, exist_ok=True)
    geometry = GeometryPolicy(mode="per_modality_plane")

    print(f"[shard {args.shard}/{args.num_shards}] arm {args.arm}, {len(cases)} cases, "
          f"report_cfg {args.report_guidance_scale}, modality_cfg {args.modality_guidance_scale}, "
          f"{args.num_inference_steps} steps, seed {args.seed}")
    print(f"  adapter: {args.adapter}")

    sampler, needs_sections, _payload = build(args, arm)
    print(f"  conditioning encodes sections separately: {needs_sections}")

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
        report_text, sections = conditioning_for(prompts[case["case_id"]], case, spacing_xyz,
                                                 needs_sections, arm["report_format"])
        try:
            volume = sampler.generate(
                report_text, dim_xyz, spacing_xyz,
                # Off the case id, not its position: a rerun at a different --num_shards draws the
                # same noise. `cli.evaluate` used seed + index and does not have that property.
                seed=args.seed + int(case["case_id"], 16) % 2 ** 31,
                modality=case["modality"], report_sections=sections,
                # The decoder's own ~[0, 1], which is what cli.evaluate scores. postprocess_mr's
                # int16 [0, 1000] would be a silent 1000x offset.
                postprocess=False,
            )
            save_volume(volume, spacing_xyz, target)
            done += 1
        except Exception as e:                 # one bad case must not lose the shard
            print(f"  {case['case_id']} failed: {type(e).__name__}: {e}")
            failed += 1

    print(f"[shard {args.shard}] {done} generated, {skipped} skipped, {failed} failed -> {args.out}")
    with open(os.path.join(args.out, f"_generate-{args.shard:04d}.json"), "w") as handle:
        json.dump({"shard": args.shard, "num_shards": args.num_shards, "arm": args.arm,
                   "adapter": args.adapter, "seed": args.seed,
                   "report_guidance_scale": args.report_guidance_scale,
                   "modality_guidance_scale": args.modality_guidance_scale,
                   "num_inference_steps": args.num_inference_steps,
                   "generated": done, "skipped": skipped, "failed": failed}, handle, indent=2)


if __name__ == "__main__":
    main()
