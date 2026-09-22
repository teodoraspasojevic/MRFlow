#!/usr/bin/env python
"""MR-RATE -> the zip-sharded latent and report cache `train.py` reads.

    python -m baselines.ccella.prepare_data --config <cfg> --split val   --shard 0 --num_shards 4
    python -m baselines.ccella.prepare_data --config <cfg> --split train --shard $SLURM_ARRAY_TASK_ID \
        --num_shards 64

One pass per series writes the volume latent; the report embedding is written once per **study**,
the first time one of its series is reached, because a report is a study-level artifact and a
FLAN-T5-XXL hidden state is 4 MB. Both frozen models -- the MAISI autoencoder and the text encoder
-- are loaded once per task.

This replaces upstream's two scripts (`diff_model_create_training_data.py` +
`gen_json_maisi_merged.py`). The *transformations* are upstream's, in `volume.py` and `text.py`;
what is replaced is the reader (tars, not a directory tree) and the writer (zip shards, not one
file per artifact). Upstream's `gen_json_maisi_merged.py` could not have been reused in any case:
as released it indexes a list with a string key (`train_dict['text']` inside a loop over `entry`,
twice) and raises immediately.

**Shard 0 writes `cache_meta.json`**, the fingerprint `train.py` checks before it allocates
anything. Let task 0 finish before training.

Resume: a shard whose manifest already exists is skipped unless `--overwrite`. The manifest is
written only after both archives are sealed, so a killed task leaves nothing a later run reads as
complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "baselines.ccella"

from .config import load_config
from .labels import label_source_fingerprint, label_vector, read_label_table
from .store import (MANIFEST_DIR, ShardStore, cache_meta, file_sha256, sample_key, study_key,
                    write_cache_meta, write_manifest)
from .text import format_report, tokenizer_settings
from .volume import VolumeUnusable, build_autoencoder, encode_volume, preprocess_volume

ZEROS_ENCODER = "__zeros__"


class _ZeroEncoder:
    """A text encoder that needs no download, for smoke tests only.

    Its name goes into the cache fingerprint, so a cache built with it is refused by any config
    naming a real encoder -- it cannot be mistaken for a real cache.
    """

    def __init__(self, hidden_size, max_length):
        self.hidden_size, self.max_length = hidden_size, max_length

    def __call__(self, text):
        return np.zeros((self.max_length, self.hidden_size), dtype=np.float16)


def build_encoder(config, device):
    """`(callable(text) -> [L, H] fp16, tokenizer_settings dict)`."""
    spec = config["text"]
    if spec["encoder"] == ZEROS_ENCODER:
        return _ZeroEncoder(spec["hidden_size"], spec["max_length"]), tokenizer_settings(
            ZEROS_ENCODER, spec["max_length"])

    from .text import build_text_encoder, encode_report

    tokenizer, model = build_text_encoder(device, spec.get("encoder_path") or spec["encoder"])
    if model.config.d_model != spec["hidden_size"]:
        raise ValueError(f"{spec['encoder']} has hidden size {model.config.d_model}, config says "
                         f"{spec['hidden_size']}")
    return (lambda text: encode_report(tokenizer, model, text, spec["max_length"]),
            tokenizer_settings(spec["encoder"], spec["max_length"]))


def shard_slice(items, shard, num_shards):
    """Contiguous slice. `list_series` already shuffled deterministically, so every task is mixed."""
    per = (len(items) + num_shards - 1) // num_shards
    return items[shard * per:(shard + 1) * per]


def prepare_shard(config, split, shard, num_shards, limit=None, overwrite=False):
    from echosyn.common.mrrate import read_member, read_report

    from .upstream import require_upstream
    require_upstream()

    cache_root = config["data"]["cache_root"]
    manifest_path = os.path.join(cache_root, MANIFEST_DIR, f"{split}-{shard:04d}.csv")
    if os.path.exists(manifest_path) and not overwrite:
        print(f"{manifest_path} exists; nothing to do (use --overwrite to redo this shard)")
        return manifest_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_label_table(config["mrrate"]["labels_root"])

    series = _list_series(config, split)
    series = shard_slice(series, shard, num_shards)
    if limit:
        series = series[:limit]
    print(f"{split} shard {shard}/{num_shards}: {len(series)} series on {device}")

    autoencoder = build_autoencoder(config, device)
    encode_text, tok_settings = build_encoder(config, device)

    store = ShardStore(cache_root, split, shard)
    rows, skipped, masked, reports = [], 0, 0, {}
    started = time.time()
    try:
        for index, item in enumerate(series):
            try:
                blob = read_member(item["archive"], item["member"])
                volume, spacing = preprocess_volume(blob, config)
            except (VolumeUnusable, IOError, KeyError) as exc:
                skipped += 1
                print(f"  skip {index}: {type(exc).__name__}: {exc}")
                continue

            skey = study_key(item["study_uid"])
            if not store.has_report(skey):
                report = read_report(item["archive"], item["study_uid"])
                text = format_report(report, tuple(config["text"]["sections"]))
                store.save_report(skey, encode_text(text))
                reports[skey] = bool(text)
            report_member = f"{skey}.npy"

            key = sample_key(item["study_uid"], item["series_id"])
            latent_member = store.save_latent(key, encode_volume(autoencoder, volume, device))

            vector, mask = label_vector(labels, item["study_uid"])
            masked += 0 if max(mask) else 1
            rows.append({
                "sample_id": key,
                "split": split,
                "study_key": skey,
                "series_key": key,
                "latent_zip": store.latent_name,
                "latent_member": latent_member,
                "report_zip": store.report_name,
                "report_member": report_member,
                "modality": item["modality"],
                "modality_id": _modality_id(item["modality"]),
                "plane": item["plane"],
                "spacing_mm": ";".join(f"{v:.6f}" for v in spacing),
                "grid": ";".join(str(v) for v in config["volume"]["grid"]),
                "labels": "".join(str(v) for v in vector),
                "label_mask": "".join(str(v) for v in mask),
            })
            if (index + 1) % 100 == 0:
                rate = (index + 1) / (time.time() - started)
                print(f"  {index + 1}/{len(series)}  {rate:.2f} series/s  "
                      f"{store.n_reports} reports  {skipped} skipped", flush=True)
        store.close()
    except BaseException:
        store.abort()
        raise

    write_manifest(cache_root, split, shard, rows)
    print(f"{len(rows)} series, {store.n_reports} reports, {skipped} skipped, "
          f"{masked} without labels -> {manifest_path}")

    if shard == 0:
        meta = cache_meta(config, label_source_fingerprint(config["mrrate"]["labels_root"]),
                          file_sha256(config["model"]["autoencoder_path"]), tok_settings)
        print("cache_meta ->", write_cache_meta(cache_root, meta))
    return manifest_path


def _modality_id(name):
    from echosyn.common.mrrate import modality_to_id

    return modality_to_id(name)


def _list_series(config, split):
    from echosyn.common.mrrate import list_series

    spec = config["mrrate"]
    return list_series(spec["raw_root"], split, max_repeats=spec["max_repeats"],
                       max_series=spec[f"max_series_{split}"], seed=spec["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="dotted config overrides, e.g. data.cache_root=/tmp/x")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    prepare_shard(config, args.split, args.shard, args.num_shards, args.limit, args.overwrite)


if __name__ == "__main__":
    main()
