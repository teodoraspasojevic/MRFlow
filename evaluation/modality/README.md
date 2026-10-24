# Modality consistency

Is a generated volume the contrast it was asked for? A 3D classifier, trained on MR-RATE, predicts
T1w / T2w / SWI / FLAIR for

- each **generated** volume, scored against the modality it was **conditioned on**, and
- the **ground truth** of the same case, scored against its **dataset label** — the empirical
  ceiling. A generated score is read against it, never against 100%.

In MR-RATE the two targets are the same field: a rollout is conditioned on its series' own
`modality`. `load_run` checks that each manifest's `conditioning_uid` hashes to the recorded
study/modality/plane, which is what ties the recorded modality to the conditioning.

| file | what |
|---|---|
| [`__init__.py`](__init__.py) | label mapping, `classifier_input`, the model, checkpoint save/load, `predict_proba` |
| [`data.py`](data.py) | split manifest, input caching (`prepare`), `ModalityDataset` |
| [`train.py`](train.py) | training, val-only checkpoint selection, resume |
| [`evaluate.py`](evaluate.py) | held-out report for a checkpoint |
| [`metrics.py`](metrics.py) | accuracy / balanced accuracy / P-R-F1 / confusion / AUROC / NLL / ECE, bootstrap CIs |
| [`consistency.py`](consistency.py) | the metric: `ModalityConsistency`, paired summary, CLI over an evaluation run |
| [`config.yaml`](config.yaml) | the one config; real Helma paths |
| [`slurm/`](slurm) | `prepare.sh` (CPU), `train.sh` (h200), `consistency.sh` (CPU) |
| [`tests/`](tests) | `pytest evaluation/modality/tests`, synthetic fixtures, no data or GPU needed |

## Commands

```bash
# 1. splits (login node, ~1 min) and cached inputs (8 CPU tasks, ~8 min)
python -m evaluation.modality.data splits
sbatch --array=0-7 evaluation/modality/slurm/prepare.sh

# 2. train, select on val, then report val and the held-out test split (~35 min, 1 h200)
sbatch evaluation/modality/slurm/train.sh
#    = python -m evaluation.modality.train
#      python -m evaluation.modality.evaluate --ckpt $CKPT --split test

# 3. modality consistency of an existing evaluation run: generated and ground truth together
sbatch evaluation/modality/slurm/consistency.sh <eval out dir>
#    = python -m evaluation.modality.consistency --run <eval out dir> --ckpt $CKPT \
#          --cases baselines/cases-test-n100.json

# or inside a normal evaluation, adding the modality_* keys to metrics.json
python evaluation/main.py ... --modality_ckpt $CKPT
```

`CKPT=/hnvme/workspace/y100dc19-mrflow-final/modality_classifier/runs/modality_net3d/best.pt`.
`slurm/consistency.sh` reads it from `$MODALITY_CKPT`, defaulting to that path. Both
`--modality_ckpt` and the CLI are opt-in: without them every existing command and output is
unchanged, and the `modality_*` keys read nan.

## Label mapping

`CLASSES = ("T1w", "T2w", "SWI", "FLAIR")`, indices 0–3, literal and fixed. Stored in every
checkpoint; `load_classifier` refuses a checkpoint with another order. Labels are normalized through
`mrrate.MODALITY_ALIASES` (`t1`, `swan`, lowercase, ...), and anything else — MRA, UNKNOWN, the
CFG null — raises rather than being scored. MRA cases of the evaluation population stay
out-of-scope, as for every other metric.

## Data and splits

- **Source.** The MR-RATE series index, through `list_series(max_repeats=1)` — the evaluation's own
  eligibility filter (no derived or localizer series, report present, one acquisition per
  contrast and plane per study). The label is the index's `modality` column.
- **Splits.** MR-RATE's official study-level `train` / `val` / `test`, which are patient-disjoint
  (checked: 0 of 83,425 patients appear in two splits). `train` trains, `val` selects the
  checkpoint, `test` is the held-out report.
- **The evaluation population is held out entirely.** Every patient with a case in
  `baselines/cases-test-n100.json` is removed from the classifier's test split, so the 1,000
  consistency cases take no part in training, selection or the held-out numbers. They are MR-RATE
  test cases, so they were never in `train` or `val` either. There is no overlap to document.
- **Per split,** series are sorted by `(study_uid, series_id)`, shuffled on `f"{seed}-{split}"`,
  and capped per class: 5,000 / 500 / 1,000 per class for train / val / test (26,000 series).
  `check_splits` raises on a duplicated case id, a patient in two splits, or an evaluation patient
  anywhere.
- **Imbalance** (train is 211k T1w against 62k SWI) is handled by that per-class cap, which makes
  every split balanced by construction. The loss also supports inverse-frequency class weights
  (`train.class_weighting`), which are exactly 1 on these splits.
- The manifest is `<data_root>/splits.csv`. A copy goes into the run directory and its sha256 into
  every checkpoint; resuming on a different manifest raises.

## Input path

```
MR-RATE series -> read_member -> load_native_volume -> canonicalize_gt    -> classifier_input
cached rollout -> np.load                           -> canonicalize_generated -> classifier_input
```

Everything up to `classifier_input` is the evaluation's existing code, called unchanged. These are
the same arrays `EvalAccumulator.add` builds for every other metric: 1 mm, 256², plane-first,
0.5/99.5-percentile `[0, 1]`. `classifier_input` belongs to the classifier alone. It area-resizes
to `64 x 128 x 128` (which also normalizes away the slice count, so rollout length is not a class
cue) and quantizes to uint8, so a cached training input and an evaluation input are bit-identical
for the same volume. The training inputs are that output, one zip per prepare task (the `/hnvme`
inode quota rules out a file per series). A series that fails is recorded with its error, never
dropped.

Training augmentation (on by default, train split only): random flips on all three axes and a
gamma in `[e^-0.2, e^0.2]`. There is no test-time augmentation.

## Model and training

A small 3D ResNet (`ModalityNet3d`): a 5³ stride-2 stem, one residual block per stage at widths
32/64/128/256 (stride 2 after the first), global average pool, dropout 0.2, linear head. 3.6 M
parameters. AdamW (lr 1e-3, wd 1e-4), one warmup epoch then cosine, batch 16, 20 epochs, bf16
autocast, cross-entropy.

**Checkpoint selection uses val only:** `best.pt` is the epoch with the highest val balanced
accuracy, ties going to lower val NLL. `last.pt` is rewritten every epoch with the optimizer and
scheduler. Re-running resumes from it, and because every epoch reseeds torch and the loader from
`seed + epoch`, a resumed run ends bit-identical to an unbroken one (tested). Checkpoint `meta`
records the config, classes, input shape, splits sha256, git commit, class weights, counts and the
selected epoch's val metrics.

## Metrics

From [`metrics.py`](metrics.py). A missing or failed case has no prediction, and **every key says
which denominator it uses**:

- **strict** (no suffix, primary): every expected case. A failure counts as wrong, gets
  probability 0 for its expected class, and lands in the confusion matrix's extra `failed` column.
- **`_evaluated`**: only the cases that produced a prediction. The probability-based metrics
  (NLL, ECE, AUROC) are always this kind.

| metric | definition |
|---|---|
| accuracy | top-1 correct / n |
| balanced accuracy | mean per-class recall over classes with support |
| precision / recall / F1 | per class; macro = unweighted mean over classes with support. Precision of a never-predicted class is 0 |
| confusion matrix | `counts[true][predicted]`, columns `CLASSES + ["failed"]` |
| AUROC | one-vs-rest per class (Mann–Whitney, ties = ½), macro = mean of the defined ones |
| NLL | mean `-log p(expected)` |
| ECE | top-1 confidence, 15 equal-width bins, weighted by bin size |
| mean p(expected) | mean probability given to the expected class |
| 95% CI | percentile bootstrap over cases, 2,000 resamples, fixed seed |

The consistency metric adds, for the paired generated / ground-truth sets:

| key | definition |
|---|---|
| `modality_{gen,gt}_accuracy`, `_balanced_accuracy`, `_macro_f1`, `_mean_p_expected`, `_recall_<class>` | strict, per side |
| `modality_accuracy_gap`, `modality_balanced_accuracy_gap` | GT − generated |
| `..._gap_ci_low` / `_high` | **paired** bootstrap of GT − generated: the same resampled case indices for both sides, 2,000 resamples, seed 0 |
| `modality_accuracy_normalized` | generated / GT; defined when GT > 0 |
| `modality_balanced_accuracy_normalized` | chance-corrected, (generated − ¼) / (GT − ¼); defined when GT balanced accuracy > ¼ |
| `n_modality_cases` | in-scope cases expected (1,000 for the frozen population) |
| `n_modality_{gen,gt}_{evaluated,missing,failed}` | missing = no volume (generation missing or file absent); failed = a volume that could not be read or classified |

Generated and ground-truth records are keyed by case id. Duplicates raise, and `load_run --cases`
asserts that the run's case set and each case's modality, plane, study and source member equal the
frozen population's, and that each cached file is named after its case.

Inside `evaluation/main.py`, the ground truth of a case whose generation failed is still read and
classified, so the ceiling keeps every case in both paths.

## Outputs

`evaluate.py` → `<ckpt dir>/eval-<split>/`: `predictions.csv` (case id, patient, study, series,
plane, archive/member, expected, predicted, `p_<class>`, confidence, correct, status, error),
`metrics.json`, `report.md`.

`consistency.py` → `<run>/modality_consistency/`: `cases.csv` (one row per case: case id, bucket,
study, source member, generated file, then `gen_*` and `gt_*` prediction fields),
`modality_consistency.json` (the scalar `metrics` above plus the full per-side summaries, CIs and
per-class breakdown), `report.md`.

## Results

Measured 2026-09-23. The checkpoint is `runs/modality_net3d/best.pt`, epoch 18 of 20, chosen on
val (balanced accuracy 1.000, 2,000 series).

**Held-out test split** (4,000 series, 1,000 per class, evaluation patients excluded, 0 failures):
accuracy **0.9985** [95% CI 0.9972, 0.9995], balanced accuracy 0.9985, macro F1 0.9985, NLL
0.0031, ECE 0.0014, macro AUROC 1.000. Per-class recall: T1w 0.998, T2w 0.997, SWI 1.000,
FLAIR 0.999. The 6 errors are 2 T1w→FLAIR, 3 T2w→FLAIR and 1 FLAIR→T2w. By plane: axial 0.9996,
coronal 0.9985, sagittal 0.9958.

**The frozen 1,000-case population** (`baselines/cases-test-n100.json`), strict (a missing
generation counts as wrong). The ground truth is the same in every row: accuracy 0.9960
[0.992, 0.999], balanced accuracy 0.9967, from 1 T2w→FLAIR and 3 FLAIR→T2w. It came out
identical in all seven independent runs.

| run | gen acc | gen bal. acc | gap GT − gen [paired 95% CI] | normalized | missing |
|---|---|---|---|---|---|
| MRFlow CT-init ckpt-55000, cfg 1/1 | 0.966 | 0.972 | 0.030 [0.018, 0.042] | 0.970 | 0 |
| MRFlow scratch (sectioned) ckpt-100000, mod7/rep10 | 0.919 | 0.933 | 0.077 [0.061, 0.095] | 0.923 | 37 |
| MRFlow scratch ckpt-100000, mod7/rep7 | 0.933 | 0.944 | 0.063 [0.048, 0.080] | 0.937 | 48 |
| NVIDIA R2V arm A cfg7 | 0.947 | 0.944 | 0.049 [0.035, 0.064] | 0.951 | 0 |
| NVIDIA R2V arm E cfg4 | 0.998 | 0.998 | −0.002 [−0.007, 0.003] | 1.002 | 0 |
| NVIDIA stock (modality-only) | 0.899 | 0.916 | 0.097 [0.079, 0.116] | 0.903 | 0 |
| Text2CT MR-RATE ft step56000 cfg10 | 0.934 | 0.942 | 0.062 [0.047, 0.077] | 0.938 | 0 |

Normalized = generated accuracy / GT accuracy. On the cases that produced a volume, the two
scratch runs score 0.954 (963) and 0.980 (952). Their strict score is lower mainly because of the
missing volumes, not because of wrong contrasts. The weak classes differ by model: FLAIR→T1w for
MRFlow CT-init (17 of 300), T2w→T1w for the NVIDIA stock model (95 of 300), T2w→FLAIR for
Text2CT (37 of 300). Per-case predictions are in each run's `modality_consistency/cases.csv`.

**Limits.** The classifier is near-perfect on real data, so the ceiling sits at 0.996 and the
metric can only separate models by how far below that they fall. A generated volume that is
correctly classified is not thereby realistic, only recognisably that contrast. The classifier
has never seen generated data, so its errors on generations include distribution shift as well as
genuinely wrong contrasts.
