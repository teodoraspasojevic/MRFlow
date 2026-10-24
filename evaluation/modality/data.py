"""Splits, cached classifier inputs and the training dataset.

    python -m evaluation.modality.data splits  --config evaluation/modality/config.yaml
    python -m evaluation.modality.data prepare --config evaluation/modality/config.yaml \
        --shard 0 --num_shards 16

**Splits** are MR-RATE's official study-level ones, which are patient-disjoint: `train` trains,
`val` selects the checkpoint, `test` is the held-out report. Every patient with a case in the
evaluation population (`eval_cases`, the 1,000 generated/ground-truth pairs) is removed from the
classifier's `test` split, so the modality-consistency cases are never used for training, model
selection or the held-out numbers. Within a split, series come from `list_series(max_repeats=1)` --
the evaluation's own eligibility filter -- sorted, shuffled on the config seed, and capped per class.

**Inputs** are computed through the evaluation's ground-truth path unchanged -- `read_member`,
`load_native_volume`, `canonicalize_gt` -- then `classifier_input`, and stored as uint8 `.npy`
members of one zip per shard (the `/hnvme` inode quota rules out a file per series). A series that
fails is recorded with its error in the shard's CSV, never dropped silently.
"""

import argparse
import csv
import io
import os
import random
import zipfile
from multiprocessing import Pool

import numpy as np
import torch
from omegaconf import OmegaConf

from echosyn.common.mrrate import (list_series, load_native_volume, read_bundled, read_member,
                                   sample_id)
from evaluation.modality import CLASSES, classifier_input, modality_index, to_tensor
from evaluation.paper_metrics import canonicalize_gt

SPLITS = ("train", "val", "test")
SPLIT_FIELDS = ("split", "case_id", "patient_uid", "study_uid", "series_id", "modality", "plane",
                "archive", "member")
STATUS_FIELDS = ("case_id", "status", "zip", "error")


### Splits ###


def patient_of(raw_root):
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.join(raw_root, "studies.parquet"),
                          columns=["study_uid", "patient_uid"]).to_pydict()
    return dict(zip(table["study_uid"], table["patient_uid"]))


def load_eval_cases(path):
    import json

    with open(path) as handle:
        return json.load(handle)["cases"]


def build_splits(raw_root, per_class, eval_cases, seed, series_by_split=None, patients=None):
    """-> split rows. `series_by_split` / `patients` default to reading MR-RATE's parquet indices;
    tests pass small synthetic ones."""
    patients = patients if patients is not None else patient_of(raw_root)
    held_out = {patients[c["study_uid"]] for c in eval_cases}

    rows = []
    for split in SPLITS:
        series = (series_by_split[split] if series_by_split is not None
                  else list_series(raw_root, split, 1))
        series = sorted((e for e in series if e["modality"] in CLASSES),
                        key=lambda e: (e["study_uid"], e["series_id"]))
        if split == "test":
            series = [e for e in series if patients[e["study_uid"]] not in held_out]
        random.Random(f"{seed}-{split}").shuffle(series)

        taken = {name: 0 for name in CLASSES}
        for e in series:
            if taken[e["modality"]] < per_class[split]:
                taken[e["modality"]] += 1
                rows.append({"split": split, "case_id": sample_id(e["study_uid"], e["series_id"]),
                             "patient_uid": patients[e["study_uid"]],
                             **{k: e[k] for k in SPLIT_FIELDS[3:]}})
    check_splits(rows, held_out)
    return rows


def check_splits(rows, held_out_patients=()):
    """Raises on a duplicated case, a patient in two splits, or an evaluation patient anywhere."""
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in the split manifest")
    split_of = {}
    for r in rows:
        if split_of.setdefault(r["patient_uid"], r["split"]) != r["split"]:
            raise ValueError(f"patient {r['patient_uid']} is in both {split_of[r['patient_uid']]} "
                             f"and {r['split']}")
    leaked = set(split_of) & set(held_out_patients)
    if leaked:
        raise ValueError(f"{len(leaked)} evaluation-population patients are in the classifier's "
                         f"splits")


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


### Inputs ###


def gt_input(archive, member, plane):
    """One MR-RATE series -> its uint8 classifier input, through the evaluation's GT path."""
    real, spacing = load_native_volume(read_member(archive, member), plane)
    return classifier_input(canonicalize_gt(real, spacing, plane))


def _prepare_one(row):
    try:
        return row["case_id"], gt_input(row["archive"], row["member"], row["plane"]), ""
    except Exception as e:
        return row["case_id"], None, f"{type(e).__name__}: {e}"


def prepare_shard(rows, data_root, shard, workers):
    """Write `inputs/shard-NNNN.zip` and its status CSV. The CSV is written last, so a killed task
    leaves no status file and the dataset refuses to load rather than training on part of a shard."""
    name = f"shard-{shard:04d}"
    zip_rel = os.path.join("inputs", f"{name}.zip")
    zip_path = os.path.join(data_root, zip_rel)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    tmp = f"{zip_path}.tmp.{os.getpid()}"

    rows = sorted(rows, key=lambda r: r["archive"])  # keep each worker's tar handles warm
    status = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf, Pool(workers) as pool:
        for i, (case_id, array, error) in enumerate(pool.imap(_prepare_one, rows, chunksize=4)):
            if array is None:
                print(f"[{name}] {case_id} failed: {error}")
                status.append({"case_id": case_id, "status": "failed", "zip": "", "error": error})
                continue
            buf = io.BytesIO()
            np.save(buf, array)
            zf.writestr(f"{case_id}.npy", buf.getvalue())
            status.append({"case_id": case_id, "status": "ok", "zip": zip_rel, "error": ""})
            if (i + 1) % 500 == 0:
                print(f"[{name}] {i + 1}/{len(rows)}")
    os.replace(tmp, zip_path)
    write_csv(os.path.join(data_root, "inputs", f"{name}.csv"), status, STATUS_FIELDS)
    print(f"[{name}] {sum(s['status'] == 'ok' for s in status)} ok, "
          f"{sum(s['status'] == 'failed' for s in status)} failed")


def read_status(data_root):
    from glob import glob

    return {r["case_id"]: r for path in sorted(glob(os.path.join(data_root, "inputs", "*.csv")))
            for r in read_csv(path)}


### Dataset ###


class ModalityDataset(torch.utils.data.Dataset):
    """One split's cached inputs as `(float [1, D, H, W] in [0, 1], class index)`.

    `rows` keeps every split row, failed ones included, with their status -- evaluation reports
    them. `self.rows_ok` is what is served. Raises if any row has no status at all, which means a
    prepare task never finished.
    """

    def __init__(self, data_root, split, augment=False):
        self.root, self.augment = data_root, augment
        status = read_status(data_root)
        rows = [r for r in read_csv(os.path.join(data_root, "splits.csv")) if r["split"] == split]
        missing = [r["case_id"] for r in rows if r["case_id"] not in status]
        if missing:
            raise RuntimeError(f"{len(missing)} {split} cases were never prepared (e.g. "
                               f"{missing[0]}): a prepare task did not finish")
        self.rows = [{**r, **status[r["case_id"]]} for r in rows]
        self.rows_ok = [r for r in self.rows if r["status"] == "ok"]
        self.labels = np.array([modality_index(r["modality"]) for r in self.rows_ok])
        print(f"[{split}] {len(self.rows_ok)} inputs, "
              f"{len(self.rows) - len(self.rows_ok)} failed in prepare")

    def __len__(self):
        return len(self.rows_ok)

    def load(self, i):
        row = self.rows_ok[i]
        return np.load(io.BytesIO(read_bundled(os.path.join(self.root, row["zip"]),
                                               f"{row['case_id']}.npy")))

    def __getitem__(self, i):
        x = to_tensor(self.load(i))[0]
        if self.augment:
            x = augment(x)
        return x, int(self.labels[i])


def augment(x):
    """Flips on all three axes and a mild gamma. Uses torch's RNG, which the DataLoader seeds per
    worker, so an epoch is reproducible from the training seed."""
    for axis in (1, 2, 3):
        if torch.rand(()) < 0.5:
            x = x.flip(axis)
    gamma = torch.exp(torch.empty(()).uniform_(-0.2, 0.2))
    return x.clamp(0, 1) ** gamma


def main():
    parser = argparse.ArgumentParser(description="Build splits / prepare classifier inputs.")
    parser.add_argument("command", choices=["splits", "prepare"])
    parser.add_argument("--config", default="evaluation/modality/config.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    path = os.path.join(config.data_root, "splits.csv")

    if args.command == "splits":
        rows = build_splits(config.raw_root, OmegaConf.to_container(config.splits.per_class),
                            load_eval_cases(config.eval_cases), config.seed)
        write_csv(path, rows, SPLIT_FIELDS)
        for split in SPLITS:
            counts = {c: sum(r["split"] == split and r["modality"] == c for r in rows)
                      for c in CLASSES}
            print(f"{split:5s} {counts}")
        print(f"{len(rows)} rows -> {path}")
    else:
        rows = read_csv(path)[args.shard::args.num_shards]
        prepare_shard(rows, config.data_root, args.shard, args.workers)


if __name__ == "__main__":
    main()
