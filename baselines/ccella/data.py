"""The zip-backed dataset. The only part of upstream's data path that is replaced.

Upstream builds a MONAI `CacheDataset` over `LoadImaged(keys=[imkey, "text"])` on per-file paths
(`diff_model_train_all.prepare_data_text_class`). That is the one thing the inode limit forbids, so
this module reads the same arrays out of the shard zips instead. **The batch it yields is the batch
upstream's `train_one_epoch` already reads** -- same keys, same dtypes, same scaling:

    image        float32 (C, X, Y, Z)   the MAISI latent, cast from the stored fp16
    spacing      float32 (3,)           native acquisition spacing in mm, * 1e2 as upstream does
    text         float32 (512, H)       FLAN-T5 last_hidden_state, cast from the stored fp16
    text_isnull  float32 ()             1 when this sample's classification target is unavailable
    pirads       float32 (14,)          the 14 merged-group labels, 0/1
    modality_id  int64   ()             index into echosyn.common.mrrate.MODALITY_TO_ID

Two of those need saying out loud.

**`text_isnull` is the classification mask.** Upstream reads it in exactly one place -- to zero the
classification term for samples it cannot supervise (`train_one_epoch`, the `isnull` lines) -- and
nowhere else. Its meaning here is the same ("this sample has no classification target") with a
different cause: MR-RATE series always have a report, because `list_series` drops report-less
studies, but ~0.2% of studies have no row in the label table. Those samples keep training the
diffusion objective and contribute nothing to the classification loss. **A missing label is never a
negative.**

**`pirads` is NOT multiplied by 1e2, and that is a required deviation.** Upstream's transform scales
it alongside `spacing` (`diff_model_train_all.py:122`), which is harmless for a softmax
cross-entropy target -- it just multiplies the term by 100 -- but a binary-cross-entropy target must
lie in `[0, 1]`. `spacing` keeps the `* 1e2`, because that is a model *input* whose projection was
designed against that scale. See the README for why upstream's `text_class_pred_weight` of 1e-4 is
nonetheless kept unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .labels import NUM_LABELS
from .store import ZipReader, read_manifest

CONDITION_SCALE = 1e2  # upstream's Lambdad scale on `spacing`


class CcellaDataset(Dataset):
    """One manifest row -> one upstream training sample.

    **Nothing is opened in the constructor.** `ZipReader` opens lazily and re-opens after a fork, so
    a DataLoader worker never inherits the parent's file offsets -- the failure mode there is
    silently interleaved bytes rather than an error.
    """

    def __init__(self, cache_root, rows, latent_channels=4):
        self.cache_root = cache_root
        self.rows = list(rows)
        self.latent_channels = latent_channels
        self.reader = ZipReader(cache_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        latent = self.reader.read_array(row["latent_zip"], row["latent_member"])
        if latent.shape[0] != self.latent_channels:
            raise ValueError(f"{row['latent_member']} has {latent.shape[0]} channels, "
                             f"expected {self.latent_channels}")
        text = self.reader.read_array(row["report_zip"], row["report_member"])

        labels = np.array([int(c) for c in row["labels"]], dtype=np.float32)
        mask = np.array([int(c) for c in row["label_mask"]], dtype=np.float32)
        if labels.size != NUM_LABELS or mask.size != NUM_LABELS:
            raise ValueError(f"manifest row has {labels.size} labels, expected {NUM_LABELS}")
        spacing = np.array([float(v) for v in row["spacing_mm"].split(";")], dtype=np.float32)

        return {
            "image": torch.from_numpy(latent.astype(np.float32)),
            "text": torch.from_numpy(text.astype(np.float32)),
            "spacing": torch.from_numpy(spacing * CONDITION_SCALE),
            "pirads": torch.from_numpy(labels),
            # mask.max() == 0 exactly when the study has no label row; see the module docstring.
            "text_isnull": torch.tensor(0.0 if mask.max() > 0 else 1.0),
            "modality_id": torch.tensor(int(row["modality_id"]), dtype=torch.long),
        }


def partition_rows(rows, world_size, rank):
    """Upstream's own DDP split, so the sharding is not a second implementation.

    `monai.data.partition_dataset(shuffle=True, even_divisible=True)` is what
    `diff_model_train` calls before building its loader; the seed is MONAI's default, so every rank
    derives the same partitioning from the same list.
    """
    if world_size <= 1:
        return list(rows)
    from monai.data import partition_dataset

    return partition_dataset(data=list(rows), shuffle=True, num_partitions=world_size,
                             even_divisible=True)[rank]


def build_dataset(config, split, world_size=1, rank=0, limit=None):
    """Manifest -> dataset for one rank."""
    rows = read_manifest(config["data"]["cache_root"], split)
    if limit:
        rows = rows[:limit]
    rows = partition_rows(rows, world_size, rank)
    return CcellaDataset(config["data"]["cache_root"], rows,
                         config["model"]["latent_channels"])


def build_loader(config, dataset, shuffle=True, batch_size=None, seed=None, epoch=0):
    """A DataLoader whose shuffle is reproducible per epoch.

    `cache_rate` never appears: nothing is held in RAM, every sample is a seek into a zip. The
    generator is seeded from `(seed, epoch)` so a given epoch replays identically after a resume.
    """
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed if seed is not None else 0) * 1_000_003 + int(epoch))
    workers = config["data"]["num_workers"]
    return DataLoader(
        dataset,
        batch_size=batch_size or config["train"]["micro_batch_size"],
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=False,
        prefetch_factor=config["data"]["prefetch_factor"] if workers else None,
    )


def fixed_validation_subset(config, split="val", size=None, seed=None):
    """A deterministic validation subset: the same rows on every rank, run and resume.

    Sorted by `sample_id` then strided, rather than sampled with an RNG, so the subset does not
    depend on library RNG behaviour and is stable if the manifest gains shards.
    """
    rows = read_manifest(config["data"]["cache_root"], split)
    rows.sort(key=lambda r: r["sample_id"])
    size = size or config["validation"]["subset"]
    if len(rows) <= size:
        return rows
    stride = len(rows) / size
    return [rows[int(i * stride)] for i in range(size)]
