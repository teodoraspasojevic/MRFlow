# NV-Generate-MR-Brain + report adapter (the challenge submission)

Our own VLM3D `mr-volume-generation` entry: NVIDIA's frozen `NV-Generate-MR-Brain` rflow UNet and
autoencoder, plus a trained cross-attention adapter carrying the report. The strongest of the three
baselines and the only one already trained on MR-RATE brain MRI, so it is the one that has to be
beaten rather than merely cleared.

    upstream   https://github.com/teodoraspasojevic/R2V-MR-Generation  @ ad5dca1 (2026-09-08)
    clone      $WS/baselines/upstream/R2V-MR-Generation
    working    ../../R2V-MR-Generation  (same commit; the sibling checkout used during the challenge)

## Arms to run, and at which guidance

Five adapters were trained, differing only in how the report becomes a conditioning tensor. Two are
worth scoring here:

| arm | conditioning | best report cfg | why this one |
|---|---|---|---|
| **A** | CXR-BERT CLS, `(B, 1, 768)` | **7.0** | best FID of the five, and the exact conditioning form MRFlow uses — the honest head-to-head |
| **E** | sectioned fusion + acquisition token, `(B, 3, 2560)` | **3.0** | the richest conditioning of the five; the strongest report-conditioning story on the other side |

Best cfg per arm was selected on FID_2p5D over 1,000 test cases: A=7, B=4, C=4, D=4, E=3. Fixed for
both: 30 inference steps, modality guidance 10.0, seed 42, `posterior_shift_mm=15`, percentile
normalizer, `report_format=findings_impression_meta` (A) / sectioned findings+impression+acquisition
(E). Change any of these and it is no longer the arm that was measured.

## Checkpoints

    adapters   /hnvme/workspace/y100dc19-nvidia-mri-brain/runs/r2v_final_{A_cxr_bert_cls,E_report2ct_style_meta}/adapter_last.pt
    backup     ../../MyModels/NVIDIA-R2V-MR/          (md5-verified copy; adapters only)
    base       $WS_nvidia/models/{autoencoder_v1.pt,diff_unet_3d_rflow-mr-brain_v0.pt}
    encoders   $WS_nvidia/pretrained/{BiomedVLP-CXR-BERT-specialized,MedEmbed-large-v0.1,Bio_ClinicalBERT}

**`/hnvme/workspace/y100dc19-nvidia-mri-brain` expires 2026-10-13.** The adapters are backed up;
the base weights, the two `.sif` containers and `pretrained/` are **not**, and arm E needs two of
those encoders. Stage them into `y100dc19-mrflow-final` (expires 2026-11-24) before that date or
this baseline stops being runnable.

## What already exists, and what it is not

The challenge sweep scored both arms on **exactly this population** — verified in
`evaluation/main.py:select_cases`, which reproduces R2V's `select_eval_cases` case for case at
`n_per_bucket=100`. Results are in W&B `mri-report2volume-miccai/mr-rate-r2v-eval`, group
`sweep_cfg` (the on-disk `$WS/cache/r2v/results` was lost when that workspace was cleaned):

| | A @ cfg 7 | E @ cfg 3 |
|---|---|---|
| FID_2p5D_Avg | 337.66 | 398.77 |
| SSIM_mean | 0.4401 | 0.4279 |
| PSNR_mean | 13.382 | 13.534 |

Those are the **challenge family only** — squeezenet 2.5D FID and the paired MSE/PSNR/SSIM. None of
FVD, `fid_2d_inception`, `fid_3d_medicalnet`, `fid_2p5d_radimagenet_*`, IS or HLIP exists for this
model, which is the entire reason for re-generating rather than quoting. (The `fvd`/`fid_2p5d` in
each `train_summary.json` are training-time validation on a different population and are not
comparable to anything here.)

Two facts to decide on before fixing the configuration, both measured on these same 1,000 cases:

- arm A's `adapter_step3000` scores FID **325.8** against `adapter_last`'s 337.7, and dropping
  modality guidance 10→7 scores **315.3**. The submitted configuration was not the best one
  measured. Defensible either way; "what we submitted" is the cleaner claim.
- SSIM and FID rank the arms oppositely, and every arm's SSIM collapses at cfg 7 (B 0.28, E 0.29).
  A wins on FID while being the weakest conditioning mechanism — one token makes the
  cross-attention softmax degenerate, so the report can only add a per-channel bias.

## Running it

```bash
# once, if the population/prompts are not exported yet
python -m baselines.common.cases   --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
    --split test --n_per_bucket 100 --out baselines/cases-test-n100.json
python -m baselines.common.prompts --cases baselines/cases-test-n100.json \
    --out $WS/baselines/prompts-test-n100.json

# generate (16 tasks, well under an hour each)
sbatch --array=0-15 baselines/nvidia_r2v/run_shards.sh A
sbatch --array=0-15 baselines/nvidia_r2v/run_shards.sh E

# ingest + score, once the array is done
baselines/score.sh nvidia_r2v_armA_cfg7 "nvidia_r2v armA cfg7 @ad5dca1"
baselines/score.sh nvidia_r2v_armE_cfg3 "nvidia_r2v armE cfg3 @ad5dca1"
```

`generate.py` writes `$WS/baselines/runs/<tag>/nifti/<case_id>.nii.gz` and nothing else. It skips a
case whose file already exists, so a requeued or preempted array task resumes rather than redoing
its shard. MRA cases are skipped outright — the evaluation excludes that modality, so generating it
is wasted GPU; `ingest.py` still records them as excluded.

**No separate venv.** Verified 2026-09-16: the whole `mrrate_r2v` stack — monai 1.6.0, transformers
5.14.1, the MAISI UNet, all three text encoders — imports in the MRFlow workspace venv, so
`run_shards.sh` uses it. The clone is put on `PYTHONPATH` through a `mrrate_r2v` symlink, because
`R2V-MR-Generation` carries a dash and is not an importable package name.

## What the driver gets right, and why each one matters

Verified by running the whole chain on CPU (`--device cpu`): base weights 435/435, 0 unexpected
keys, 0 shape mismatches, all shared tensors bit-equal, 74 adapter tensors, divisor 4,
`scale_factor` 0.970450 — then one generation at `--num_inference_steps 1`, ingested to a
`(175, 256, 256)` fp16 `[0, 1]` cache entry. 160 slices at 1.094 mm becoming 175 at 1 mm is the
resample landing where it should.

**`--device cpu` needs one shim, and it is not free.** NVIDIA's `config_network_rflow.json` sets
`"norm_float16": true`, so MAISI's `MaisiGroupNorm3D` emits float16. Under CUDA the decode runs in
`autocast` and the next convolution matches; on CPU `autocast` is disabled by design and the decode
dies with `Input type (c10::Half) and bias type (float) should be the same` inside MONAI. Measured:
clearing that flag on the 26 norm modules is the entire fix. `_enable_cpu_decode` does exactly that
and only under `--device cpu`, so a CPU run validates wiring but is **not** numerically the GPU
path.

- **`postprocess=False`.** `postprocess_mr` maps the decoder's ~[0, 1] to int16 [0, 1000]. Both are
  valid model outputs; `cli.evaluate` scores the former. A 1000x offset yields plausible metrics
  and never an error, which is why this is stated rather than left to a default.
- **The trained format is a spec, not a name.** Arm A records
  `'findings_impression_meta,impression_findings_meta'` — it trained on a uniform draw between the
  two orders. An exact-string guard against one of them rejects the arm, so the check goes through
  `trained_report_formats`. Arm E records no format at all and is checked on `needs_sections`
  instead, because a sectioned arm never sees a joined string.
- **Geometry comes from one `GeometryPolicy.resolve` call**, so the `[SPACING]` marker in the text,
  the numeric `spacing_tensor` and the decoded grid cannot disagree.
- **Seeding is off the case id, not its position.** `cli.evaluate` used `seed + case.index`, which
  makes the same case a different volume under a different shard count. This matches
  `evaluation/main.py`'s convention instead, so a rerun at any `--num_shards` reproduces the run.
  The cost: these volumes are not bit-identical to the 2026-08 sweep — which only produced
  challenge-family numbers, and those are being recomputed anyway.
- **`posterior_shift_mm` stays at ingest's default of 0.** This model trained at shift 15 and its
  output already carries that framing; re-shifting would move it off its own centre.

## Before the next run

`/hnvme/workspace/y100dc19-nvidia-mri-brain` **expires 2026-10-13**. `run_shards.sh` preflights
every path and fails before the queue takes a GPU, but once the workspace is gone the baseline is
unrunnable until the base weights and `pretrained/` are restaged. Set `R2V_WORKSPACE` after moving
them.
