"""The 14 merged-group MR-RATE pathology labels, and the one place they are read.

**The authoritative local source** is the 3-model majority-vote artifact produced by
`MR-RATE/contrastive-pretraining/scripts/eval_labels/build_merged_group_labels.py --source majority`:

    <MRRATE_LABELS_ROOT>/mrrate_merged_labels.csv    study_uid + 14 binary group columns
    <MRRATE_LABELS_ROOT>/splits.csv                  batch_id,patient_uid,study_uid,split
    <MRRATE_LABELS_ROOT>/group_definitions.json      group -> member pathologies (documentation)

Three models (Claude Opus 4.7, GPT-5.5, Nemotron-3 Super 120B) each label a study over MR-RATE's
own 37 SNOMED/RadLex pathologies; a pathology is positive when a strict majority of the *present*
votes agree; a group is the logical OR of its member pathologies. Two of the 37 (`Empty sella
syndrome`, `Hyperostosis of skull`) belong to no group and are deliberately excluded upstream.

**This is not the 37-label schema.** `<study>/labels.json` inside the MR-RATE archives and
`pathology_labels/mrrate_labels.csv` in the dataset release are the single-model (Qwen3.5-35B) 37
categories. They are a different artifact with a different vote rule and must not be substituted or
remapped -- `LABEL_SOURCE_ID` below is part of the cache fingerprint so a swap cannot go unnoticed.

**Ordering is the CSV header order and may never move**, for the same reason `MODALITY_TO_ID` may
not: a classifier head's output channel is meaningless without it, and a checkpoint outlives any
one run. `read_label_table` asserts the header against `LABELS_14` rather than trusting it.

**Missingness is per study, not per label.** The upstream builder resolves every vote into a 0/1
before writing -- a pathology no model voted on becomes 0, and a study no model saw is *skipped*
entirely. So the only missingness visible downstream is a study absent from the CSV, and the mask
this module returns is therefore all-ones or all-zeros for a given study. It is shaped per label
anyway, so a future label source with genuine per-label uncertainty needs no change here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

# The 14 columns, in `mrrate_merged_labels.csv` header order: 8 Pathophysiologie + 6 Bildphänotyp.
LABELS_14 = (
    "PP_Cerebrovascular",
    "PP_Neoplastic",
    "PP_Neurodegenerative",
    "PP_Spinal",
    "PP_Cystic_developmental",
    "PP_Infectious",
    "PP_Inflammatory",
    "PP_Unspecific_bucket",
    "BP_Atrophies",
    "BP_Contrast_enhancing_intracranial",
    "BP_Infectious_lesions",
    "BP_Edematous_lesions",
    "BP_Hemorrhagic_lesions",
    "BP_Cystic_lesions",
)
NUM_LABELS = len(LABELS_14)

# Goes into the cache fingerprint. Bump it if the label artifact is ever regenerated.
LABEL_SOURCE_ID = "mrrate-merged-majority-3model-14group"

LABELS_CSV_NAME = "mrrate_merged_labels.csv"
SPLITS_CSV_NAME = "splits.csv"

# Three PP/BP pairs share their member pathology lists exactly, so their columns are identical by
# construction, not by coincidence in the data. Reported by the audit and expected in any metric.
DUPLICATE_COLUMN_PAIRS = (
    ("PP_Neurodegenerative", "BP_Atrophies"),
    ("PP_Neoplastic", "BP_Contrast_enhancing_intracranial"),
    ("PP_Infectious", "BP_Infectious_lesions"),
)


class LabelSourceError(RuntimeError):
    """The label artifact is missing, or its header is not the 14 columns in the fixed order."""


def read_label_table(labels_root):
    """`{study_uid: (0/1,) * 14}` from `mrrate_merged_labels.csv`.

    Raises rather than tolerating a header that is not exactly `LABELS_14` in order, a non-binary
    value, or a duplicate `study_uid` whose rows disagree. A duplicate row that agrees with itself
    is collapsed, because that is a harmless re-emission rather than a conflict.
    """
    path = os.path.join(labels_root, LABELS_CSV_NAME)
    if not os.path.exists(path):
        raise LabelSourceError(f"no label table at {path}")

    table = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if tuple(header) != ("study_uid",) + LABELS_14:
            raise LabelSourceError(
                f"{path} header is not the expected 14 columns in order.\n"
                f"  expected: {('study_uid',) + LABELS_14}\n  found:    {tuple(header)}"
            )
        for row in reader:
            uid, values = row[0], row[1:]
            if any(v not in ("0", "1") for v in values):
                raise LabelSourceError(f"non-binary value in {path} for one study: {values}")
            vector = tuple(int(v) for v in values)
            if uid in table and table[uid] != vector:
                raise LabelSourceError(f"conflicting duplicate rows in {path} for one study")
            table[uid] = vector
    return table


def read_label_splits(labels_root):
    """`{study_uid: split}` from the label artifact's own `splits.csv`."""
    path = os.path.join(labels_root, SPLITS_CSV_NAME)
    if not os.path.exists(path):
        raise LabelSourceError(f"no split table at {path}")
    with open(path, newline="") as handle:
        return {row["study_uid"]: row["split"] for row in csv.DictReader(handle)}


def label_vector(table, study_uid):
    """`(labels, mask)` for one study, both length 14.

    A study the table does not cover gets an all-zero label vector and an **all-zero mask**. The
    zeros are never seen by the classification loss -- the mask is what the loss multiplies by -- so
    the sample still trains the diffusion objective. Absence is never a negative.
    """
    vector = table.get(study_uid)
    if vector is None:
        return (0,) * NUM_LABELS, (0,) * NUM_LABELS
    return vector, (1,) * NUM_LABELS


def pos_weight(table, study_uids, cap=20.0):
    """`BCEWithLogitsLoss(pos_weight=...)` from the **train split only**, `(neg / pos)` per label.

    Capped at `cap` because a label at 0.2% prevalence would otherwise ask for a weight near 500 and
    a single positive would dominate the gradient. A label with no positives at all gets `cap` too,
    which is inert: with no positive sample the term never fires.
    """
    counts = [0] * NUM_LABELS
    total = 0
    for uid in study_uids:
        vector = table.get(uid)
        if vector is None:
            continue
        total += 1
        for i, v in enumerate(vector):
            counts[i] += v
    weights = []
    for pos in counts:
        neg = total - pos
        weights.append(min(cap, neg / pos) if pos else cap)
    return weights, counts, total


def group_definitions(labels_root):
    """`{group: [member pathologies]}` -- documentation only, never read by the model."""
    path = os.path.join(labels_root, "group_definitions.json")
    with open(path) as handle:
        return json.load(handle)


def label_source_fingerprint(labels_root):
    """A digest of the label artifact, for the cache fingerprint.

    The sha256 of the label CSV plus the ordering itself, so regenerating the labels or reordering
    the columns both invalidate a cache built against the old ones.
    """
    digest = hashlib.sha256()
    digest.update(LABEL_SOURCE_ID.encode())
    digest.update("\n".join(LABELS_14).encode())
    with open(os.path.join(labels_root, LABELS_CSV_NAME), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
