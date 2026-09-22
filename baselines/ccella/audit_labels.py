#!/usr/bin/env python
"""Audit the 14-label MR-RATE annotations against the population CCELLA would train on.

    python -m baselines.ccella.audit_labels --out baselines/ccella/label_audit.json

Answers, per split, the questions that decide whether a multi-label auxiliary head is trainable at
all: how many studies carry labels, how many series resolve to exactly one study report and one
14-vector, which labels are constant, and what the prevalence is. Everything it writes is an
aggregate -- **no study identifiers, no report text, no patient metadata** -- so the output is safe
to commit next to the code.

It reads the parquet indices and the label CSV only. No volume is opened, no GPU is used, and it
runs on a login node in about a minute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "baselines.ccella"

from .labels import (DUPLICATE_COLUMN_PAIRS, LABEL_SOURCE_ID, LABELS_14, NUM_LABELS,
                     label_source_fingerprint, pos_weight, read_label_splits, read_label_table)

DEFAULT_LABELS_ROOT = ("/home/hpc/y100dc/y100dc19/VLM3D-MRI-R2V-MICCAI-26/MR-RATE/"
                       "contrastive-pretraining/scripts/eval_labels/splits_merged_majority")
DEFAULT_RAW_ROOT = "/hnvme/workspace/y100dc19-MR-Rate-raw"
SPLITS = ("train", "val", "test")


def read_population(raw_root):
    """The intended CCELLA population, by `echosyn.common.mrrate.list_series`' own rules.

    Series that are derived, localizers, image-less, or whose study has no report are dropped --
    the same filter MRFlow applies -- but `max_repeats`/`max_series` are **not**, because the audit
    describes the whole eligible pool rather than one run's sample of it.
    """
    import pyarrow.parquet as pq

    studies = pq.read_table(os.path.join(raw_root, "studies.parquet"),
                            columns=["study_uid", "split", "has_report", "has_labels"])
    study_split, with_report, with_37 = {}, set(), set()
    for i in range(studies.num_rows):
        uid = studies["study_uid"][i].as_py()
        study_split[uid] = studies["split"][i].as_py()
        if studies["has_report"][i].as_py():
            with_report.add(uid)
        if studies["has_labels"][i].as_py():
            with_37.add(uid)

    table = pq.read_table(
        os.path.join(raw_root, "series.parquet"),
        columns=["study_uid", "split", "modality", "plane", "repeat", "is_derived",
                 "is_localizer", "image_present"])
    col = {name: table.column(name) for name in table.column_names}
    series = []
    for i in range(table.num_rows):
        if not col["image_present"][i].as_py():
            continue
        if col["is_derived"][i].as_py() or col["is_localizer"][i].as_py():
            continue
        uid = col["study_uid"][i].as_py()
        if uid not in with_report:
            continue
        series.append({
            "study_uid": uid,
            "split": col["split"][i].as_py(),
            "modality": col["modality"][i].as_py() or "UNKNOWN",
            "plane": col["plane"][i].as_py() or "UNKNOWN",
            "repeat": col["repeat"][i].as_py() or 0,
        })
    return study_split, with_report, with_37, series


def audit(labels_root, raw_root):
    table = read_label_table(labels_root)
    label_splits = read_label_splits(labels_root)
    study_split, with_report, with_37, series = read_population(raw_root)

    report = {
        "label_source": {
            "id": LABEL_SOURCE_ID,
            "root": labels_root,
            "sha256": label_source_fingerprint(labels_root),
            "n_labels": NUM_LABELS,
            "label_order": list(LABELS_14),
            "duplicate_column_pairs": [list(p) for p in DUPLICATE_COLUMN_PAIRS],
            "n_rows": len(table),
            "values": "binary 0/1, fully resolved upstream -- no NaN, no uncertain, no masked cell",
            "missingness": "per study only: a study absent from the CSV has no label vector",
            "join_key": "study_uid",
        },
        "raw_root": raw_root,
        "splits": {},
        "population": {},
        "consistency": {},
    }

    # --- split-level coverage -------------------------------------------------------------------
    for split in SPLITS:
        studies_in_split = {u for u, s in study_split.items() if s == split}
        reported = studies_in_split & with_report
        labelled = {u for u in reported if u in table}
        vectors = [table[u] for u in labelled]

        positives = [0] * NUM_LABELS
        for vector in vectors:
            for i, v in enumerate(vector):
                positives[i] += v
        per_study = Counter(sum(v) for v in vectors)

        report["splits"][split] = {
            "n_studies_total": len(studies_in_split),
            "n_studies_with_report": len(reported),
            "n_studies_with_all_14_labels": len(labelled),
            "pct_studies_with_all_14_labels": round(100 * len(labelled) / max(1, len(reported)), 3),
            "n_studies_with_partial_labels": 0,   # schema has no partial state; see label_source
            "n_studies_with_no_labels": len(reported) - len(labelled),
            "n_studies_with_37_label_json": len(studies_in_split & with_37),
            "label_positives": {
                name: {
                    "n_positive": positives[i],
                    "prevalence_pct": round(100 * positives[i] / max(1, len(labelled)), 4),
                }
                for i, name in enumerate(LABELS_14)
            },
            "positive_labels_per_study": {str(k): per_study[k] for k in sorted(per_study)},
            "n_studies_all_negative": per_study.get(0, 0),
            "pct_studies_all_negative": round(100 * per_study.get(0, 0) / max(1, len(labelled)), 3),
        }

    # --- constant / near-constant labels, on train only -----------------------------------------
    train_labelled = [u for u, s in study_split.items()
                      if s == "train" and u in with_report and u in table]
    weights, counts, total = pos_weight(table, train_labelled)
    report["train_pos_weight"] = {
        "cap": 20.0,
        "n_train_studies": total,
        "per_label": {name: {"n_positive": counts[i],
                             "prevalence_pct": round(100 * counts[i] / max(1, total), 4),
                             "pos_weight": round(weights[i], 4),
                             "capped": bool(counts[i] and (total - counts[i]) / counts[i] > 20.0)}
                      for i, name in enumerate(LABELS_14)},
    }
    report["constant_or_near_constant_labels"] = [
        name for i, name in enumerate(LABELS_14)
        if counts[i] == 0 or counts[i] == total or 100 * counts[i] / max(1, total) < 0.5
    ]

    # --- series-level resolution ----------------------------------------------------------------
    for split in SPLITS:
        in_split = [s for s in series if s["split"] == split]
        resolved = [s for s in in_split if s["study_uid"] in table]
        buckets = Counter((s["modality"], s["plane"]) for s in in_split)
        report["population"][split] = {
            "n_eligible_series": len(in_split),
            "n_series_resolving_to_a_14_vector": len(resolved),
            "pct_series_resolving": round(100 * len(resolved) / max(1, len(in_split)), 3),
            "n_series_without_labels": len(in_split) - len(resolved),
            "n_distinct_studies": len({s["study_uid"] for s in in_split}),
            "series_per_study_max": max(Counter(s["study_uid"] for s in in_split).values(),
                                        default=0),
            "n_buckets_modality_plane": len(buckets),
        }

    # --- cross-source consistency ---------------------------------------------------------------
    label_only = set(table) - set(study_split)
    split_disagree = sum(1 for u, s in label_splits.items()
                         if u in study_split and study_split[u] != s)
    report["consistency"] = {
        "n_study_uids_in_labels_not_in_mrrate_index": len(label_only),
        "n_study_uids_in_label_splits_disagreeing_with_mrrate_split": split_disagree,
        "n_duplicate_study_uid_rows_in_label_csv": 0,  # read_label_table raises on a conflict
        "duplicate_columns_identical_in_data": {
            f"{a}=={b}": all(table[u][LABELS_14.index(a)] == table[u][LABELS_14.index(b)]
                             for u in table)
            for a, b in DUPLICATE_COLUMN_PAIRS
        },
        "n_studies_with_report_but_no_labels": sum(
            report["splits"][s]["n_studies_with_no_labels"] for s in SPLITS),
        "n_studies_with_labels_but_no_report": len(
            {u for u in table if u in study_split and u not in with_report}),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit the 14-label MR-RATE annotations.")
    parser.add_argument("--labels_root", default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "label_audit.json"))
    args = parser.parse_args()

    report = audit(args.labels_root, args.raw_root)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)

    src = report["label_source"]
    print(f"label source {src['id']}  sha256 {src['sha256'][:16]}...  rows {src['n_rows']:,}")
    for split in SPLITS:
        s, p = report["splits"][split], report["population"][split]
        print(f"{split:5s} studies w/ report {s['n_studies_with_report']:6,}  "
              f"labelled {s['n_studies_with_all_14_labels']:6,} "
              f"({s['pct_studies_with_all_14_labels']:.2f}%)  "
              f"all-negative {s['pct_studies_all_negative']:.1f}%  |  "
              f"series {p['n_eligible_series']:7,}  resolving {p['pct_series_resolving']:.2f}%")
    print(f"near-constant labels: {report['constant_or_near_constant_labels'] or 'none'}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
