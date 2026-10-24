# Pathology consistency

Does a generated volume show the pathology of the report it was generated from? A 3D multi-label
classifier, trained on MR-RATE, predicts 11 pathology groups for

- each **generated** volume, and
- the **ground truth** of the same case,

both scored against the **same** target: the study's labels, which were extracted from the very
report the generator was conditioned on. The ground truth is the empirical ceiling, i.e. how much
of a report's pathology a classifier can read off one real series of that study. A generated score
is read against it, never against AUROC 1.

This is the modality-consistency metric's design ([`../modality`](../modality/README.md)) applied
to a harder target, and it reuses that package wherever the two agree: the input path, the input
cache, the split rules, the model family, run loading and alignment checks.

| file | what |
|---|---|
| [`__init__.py`](__init__.py) | the label table and its 11 kept columns, the model, checkpoint loading, `predict_proba` |
| [`data.py`](data.py) | per-study split manifest, input caching (via `modality.data.prepare_shard`), `PathologyDataset` |
| [`train.py`](train.py) | BCE training, val-only checkpoint and threshold selection, resume |
| [`evaluate.py`](evaluate.py) | held-out report for a checkpoint |
| [`metrics.py`](metrics.py) | AUROC / AUPRC / thresholded P-R-F1 / Brier / ECE, study-clustered bootstrap |
| [`consistency.py`](consistency.py) | the metric: `PathologyConsistency`, paired summary, CLI over an evaluation run |
| [`config.yaml`](config.yaml) | the one config; real Helma paths |
| [`slurm/`](slurm) | `prepare.sh` (CPU), `train.sh` (h200), `consistency.sh` (CPU) |
| [`tests/`](tests) | `pytest evaluation/pathology/tests`, synthetic fixtures, no data or GPU needed |

## Commands

```bash
# 1. splits (login node, ~2 min) and cached inputs (32 CPU tasks, ~40 min, ~171 GB, 64 files)
python -m evaluation.pathology.data splits
sbatch --array=0-31 evaluation/pathology/slurm/prepare.sh

# 2. train, select on val, then report val and the held-out test split (1 h200)
sbatch evaluation/pathology/slurm/train.sh

# 3. pathology consistency of an existing evaluation run
sbatch evaluation/pathology/slurm/consistency.sh <eval out dir>
#    = python -m evaluation.pathology.consistency --run <eval out dir> --ckpt $CKPT \
#          --cases baselines/cases-test-n100.json

# or inside a normal evaluation, adding the pathology_* keys to metrics.json
python evaluation/main.py ... --pathology_ckpt $CKPT
```

`CKPT=/hnvme/workspace/y100dc19-mrflow-final/pathology_classifier/runs/pathology_net3d/best.pt`,
which `slurm/consistency.sh` reads from `$PATHOLOGY_CKPT`. The metric is opt-in: without the flag
every existing output is unchanged and the `pathology_*` keys read nan.

## Labels

The **14 merged-group majority-vote labels**
(`MR-RATE/contrastive-pretraining/scripts/eval_labels/splits_merged_majority/mrrate_merged_labels.csv`),
the artifact `baselines/ccella/labels.py` documents: three LLMs label each report over MR-RATE's
37 pathologies, a pathology is positive on a strict majority, and a group is the OR of its members.
It is **not** the single-model 37-label `labels.json` shipped in the study archives.

Three BP groups have exactly a PP group's member pathologies, so their columns are identical by
construction (`BP_Atrophies` = `PP_Neurodegenerative`, `BP_Contrast_enhancing_intracranial` =
`PP_Neoplastic`, `BP_Infectious_lesions` = `PP_Infectious`). Predicting both would count each of
those findings twice in every macro mean. The classifier therefore predicts the **11 distinct
groups**, `LABELS`, in the CSV's header order:

`PP_Cerebrovascular, PP_Neoplastic, PP_Neurodegenerative, PP_Spinal, PP_Cystic_developmental,
PP_Infectious, PP_Inflammatory, PP_Unspecific_bucket, BP_Edematous_lesions,
BP_Hemorrhagic_lesions, BP_Cystic_lesions`

`read_labels` raises if the header is not the 14 columns in that order, a cell is not 0/1, a
dropped BP column differs from its PP twin, or a study has two conflicting rows. The table's sha256
goes into every checkpoint, and the metric refuses to run against a different table.

## Data and splits

- **Splits** are MR-RATE's official patient-disjoint ones; the label artifact's own `splits.csv`
  agrees with them for 100% of studies and patients. `train` trains, `val` selects the checkpoint
  and the thresholds, `test` is the held-out report.
- **The evaluation population is held out entirely**, as for the modality classifier: every
  patient with a case in `baselines/cases-test-n100.json` is removed from the classifier's test
  split, so no part of classifier training, selection or held-out reporting touches those cases.
- **Sampling is per study**, because a label vector belongs to a study rather than a series.
  Eligible series are `list_series(max_repeats=1)` ones of the four in-scope contrasts, restricted
  to studies the table labels (it misses 0.3%). Each study contributes up to 2 series to train and
  exactly 1 to val and test, drawn on `f"{seed}-{split}"`. That gives 163,309 train series over
  81,881 studies, 3,331 val and 4,536 test studies.
- **Imbalance** is per label (prevalence 1.4%–42%): BCE with `pos_weight = neg/pos` from the
  train split, capped at 10.

## Input path, model and training

The same input as the modality classifier, bit for bit:

```
MR-RATE series -> read_member -> load_native_volume -> canonicalize_gt        -> classifier_input
cached rollout -> np.load                           -> canonicalize_generated -> classifier_input
```

The model is the same `ModalityNet3d` 3D ResNet with an 11-way sigmoid head. It is wider
(48/96/192/384, 8.1 M parameters) and uses dropout 0.3. Training is the same recipe: AdamW
lr 1e-3, one warmup epoch then cosine, batch 16, 20 epochs, bf16, flips + gamma augmentation, no
test-time augmentation, and per-epoch reseeding, so a resumed run ends bit-identical to an
unbroken one (tested).

**Selection uses val only.** `best.pt` is the epoch with the highest val macro AUROC (ties to
lower BCE). Its meta stores what the metrics must never re-derive elsewhere:

- `thresholds`: per label, the F1-optimal threshold on val at that epoch;
- `prior`: the train prevalence, used for failed cases (below);
- `labels_sha256`, `splits_sha256`, config, git commit, counts.

## Metrics

From [`metrics.py`](metrics.py). A case whose volume is missing or unreadable has no prediction.

- **strict** (no suffix, primary): every expected case. A failed case gets `prior` for every label
  — what a model that never saw an image would output. It ranks at chance and is thresholded like
  any other case, so failures pull every metric down instead of vanishing.
- **`_evaluated`**: only the cases that produced a prediction.

| metric | definition |
|---|---|
| AUROC | per label, Mann–Whitney (ties = ½) |
| AUPRC | per label, step-sum average precision (sklearn's `average_precision_score`, ties included); chance = prevalence |
| precision / recall / specificity / F1 | per label, at the val-chosen threshold |
| Brier, ECE | per label; ECE over 10 equal-width bins |
| macro_* | unweighted mean over the **scored** labels: ≥ `min_positives` (10) positives *and* negatives in the population being scored. The rest are reported per label, marked unscored |
| exact_match, hamming_accuracy, mean_bce | over all 11 labels, thresholded |
| 95% CI | percentile bootstrap, 2,000 resamples, resampling **studies**: in the 1,000-case population a study contributes ~4 cases and one label vector, so a per-case bootstrap would understate the uncertainty |

The consistency metric adds, for the paired generated / ground-truth sets (same cases, same scored
labels, same thresholds):

| key | definition |
|---|---|
| `pathology_{gen,gt}_macro_{auroc,auprc,f1}`, `_auroc_<label>` | strict, per side |
| `pathology_macro_{auroc,auprc}_gap`, `pathology_macro_f1_gap` | GT − generated |
| `..._gap_ci_low` / `_high` | paired study bootstrap of GT − generated |
| `pathology_macro_auroc_normalized` | (gen − 0.5) / (GT − 0.5), defined when GT > 0.5 |
| `pathology_macro_auprc_normalized` | (gen − p) / (GT − p), p = mean prevalence of the scored labels, defined when GT > p |
| `pathology_gen_gt_prob_correlation` | per scored label, Pearson r between generated and GT probabilities over cases with both, then the mean |
| `n_pathology_{cases,studies,scored_labels,unlabelled}` | population; a study without labels is counted, never scored |
| `n_pathology_{gen,gt}_{evaluated,missing,failed}` | as for the modality metric |

The report also breaks macro AUROC down by contrast (T1w / T2w / SWI / FLAIR), each with its own
scored-label set. Alignment is the modality metric's `load_run`: case ids unique, the case set and
identity fields equal to the frozen population, cached files named after their case,
`conditioning_uid` consistent.

## Outputs

`evaluate.py` → `<ckpt dir>/eval-<split>/`: `predictions.csv` (ids, patient, study, series,
contrast, plane, source member, status, error, then `target_<label>`, `p_<label>`,
`pred_<label>`), `metrics.json`, `report.md`.

`consistency.py` → `<run>/pathology_consistency/`: `cases.csv` (one row per case: ids, study,
source member, generated file, `target_*`, then `gen_status`/`gen_error`/`gen_p_*` and the same
for `gt`), `pathology_consistency.json` (thresholds, the scalar `metrics`, full per-side
summaries, CIs, per-label and per-contrast breakdowns), `report.md`.

## Results

RESULTS_PLACEHOLDER
