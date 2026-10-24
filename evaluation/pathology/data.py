"""Splits, cached inputs and the dataset for the pathology classifier.

    python -m evaluation.pathology.data splits  --config evaluation/pathology/config.yaml
    python -m evaluation.pathology.data prepare --config evaluation/pathology/config.yaml \
        --shard 0 --num_shards 16

The same rules as `evaluation.modality.data`, and its code where it applies: MR-RATE's official
patient-disjoint splits (train trains, val selects, test is the held-out report), every patient
of the evaluation population removed from test, `list_series(max_repeats=1)` eligibility, inputs
cached through the evaluation's own ground-truth path by `prepare_shard`.

What differs is the sampling, because the label belongs to a study rather than a series: each
labelled study contributes `per_study[split]` of its T1w/T2w/SWI/FLAIR series, drawn on the config
seed. Studies the label table does not cover are left out (0.3% of MR-RATE).
"""

import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from echosyn.common.mrrate import list_series, sample_id
from evaluation.modality import CLASSES, to_tensor
from evaluation.modality.data import (SPLIT_FIELDS, SPLITS, ModalityDataset, augment,
                                      check_splits, load_eval_cases, patient_of, prepare_shard,
                                      read_csv, write_csv)
from evaluation.pathology import read_labels


def build_splits(raw_root, labels, per_study, eval_cases, seed, series_by_split=None,
                 patients=None):
    """-> split rows. `series_by_split` / `patients` default to MR-RATE's parquet indices."""
    patients = patients if patients is not None else patient_of(raw_root)
    held_out = {patients[c["study_uid"]] for c in eval_cases}

    rows = []
    for split in SPLITS:
        series = (series_by_split[split] if series_by_split is not None
                  else list_series(raw_root, split, 1))
        series = sorted((e for e in series if e["modality"] in CLASSES and e["study_uid"] in labels),
                        key=lambda e: (e["study_uid"], e["series_id"]))
        if split == "test":
            series = [e for e in series if patients[e["study_uid"]] not in held_out]
        random.Random(f"{seed}-{split}").shuffle(series)

        taken = {}
        for e in series:
            if taken.get(e["study_uid"], 0) < per_study[split]:
                taken[e["study_uid"]] = taken.get(e["study_uid"], 0) + 1
                rows.append({"split": split, "case_id": sample_id(e["study_uid"], e["series_id"]),
                             "patient_uid": patients[e["study_uid"]],
                             **{k: e[k] for k in SPLIT_FIELDS[3:]}})
    check_splits(rows, held_out)
    return rows


class PathologyDataset(ModalityDataset):
    """`ModalityDataset` with the study's label vector as the target."""

    def __init__(self, data_root, split, labels, augment=False):
        super().__init__(data_root, split, augment)
        self.labels = np.stack([labels[r["study_uid"]] for r in self.rows_ok]).astype(np.float32)

    def __getitem__(self, i):
        x = to_tensor(self.load(i))[0]
        if self.augment:
            x = augment(x)
        return x, torch.from_numpy(self.labels[i])


def main():
    parser = argparse.ArgumentParser(description="Build splits / prepare classifier inputs.")
    parser.add_argument("command", choices=["splits", "prepare"])
    parser.add_argument("--config", default="evaluation/pathology/config.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    path = os.path.join(config.data_root, "splits.csv")

    if args.command == "splits":
        labels = read_labels(config.labels_root)
        rows = build_splits(config.raw_root, labels, OmegaConf.to_container(config.per_study),
                            load_eval_cases(config.eval_cases), config.seed)
        write_csv(path, rows, SPLIT_FIELDS)
        for split in SPLITS:
            chosen = [r for r in rows if r["split"] == split]
            prevalence = np.stack([labels[r["study_uid"]] for r in chosen]).mean(0)
            print(f"{split:5s} {len(chosen)} series, {len({r['study_uid'] for r in chosen})} "
                  f"studies, prevalence {np.round(prevalence, 3).tolist()}")
        print(f"{len(rows)} rows -> {path}")
    else:
        rows = read_csv(path)[args.shard::args.num_shards]
        prepare_shard(rows, config.data_root, args.shard, args.workers)


if __name__ == "__main__":
    main()
