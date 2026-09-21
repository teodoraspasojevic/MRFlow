#!/usr/bin/env python
"""MR-RATE -> the cached VAE latents and report embeddings `fine_tune.py` trains on.

    python -m baselines.text2ct.prepare_data --config <cfg> --split val --shard 0 --num_shards 4
    python -m baselines.text2ct.prepare_data --config <cfg> --split train --shard $SLURM_ARRAY_TASK_ID \
        --num_shards 64

One pass per series writes both artifacts, because a volume is one *series* while a report is one
*study*: many series share a report, and encoding per series is what lets the two live in one
manifest row. This mirrors upstream's two scripts
(`scripts/diff_model_create_training_data.py` + `scripts/save_embeddings_ctrate.py`) merged into
one, which also means the frozen VAE and the frozen text encoder are loaded once.

**Artifacts go into one zip per shard.** Two files per series over the 575k-series train split is
1.15M inodes, against this account's 102,400-inode hard limit on `/hnvme`. `ZIP_STORED` keeps
random access (the central directory records every member's offset) at ~1 file per shard.

Task 0 also writes `cache_meta.json`, the fingerprint `fine_tune.py` checks before training: the
target grid, the percentiles, the report sections, and the sha256 of both frozen checkpoints. Change
any of them and the cached arrays mean something else, so a stale cache is refused rather than
silently trained on.
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
    __package__ = "baselines.text2ct"

from .config import apply_dotted, load_config, validate, weight_paths
from .model import build_vae
from .mrrate_data import (CACHE_META_NAME, LatentStore, ReportMissing, VolumeUnusable,
                          cache_meta, format_report, list_series, modality_name, plane_name,
                          preprocess_volume, read_member, read_report, sample_id, write_manifest)
from .text_encoder import build_text_encoder, encode_reports


@torch.inference_mode()
def encode_volume(vae, array, device):
    """`(X, Y, Z)` in [0, 1] -> `(C, X/4, Y/4, Z/4)` fp16.

    `encode_stage_2_inputs` is what upstream calls
    (`scripts/diff_model_create_training_data.py:181`); MAISI's autoencoder samples the posterior
    inside it, so one stored latent is one draw, exactly as upstream's cache holds.
    """
    tensor = torch.from_numpy(array)[None, None].to(device)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        latent = vae.encode_stage_2_inputs(tensor)
    return latent[0].float().cpu().numpy().astype(np.float16)


def prepare(config, split, shard, num_shards, limit=None, overwrite=False, bundle=True,
            device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = weight_paths(config)
    cache_root = config["data"]["cache_root"]
    volume_cfg, data_cfg, mri = config["volume"], config["data"], config["mrrate"]

    manifest_path = os.path.join(cache_root, "manifest", f"{split}-{shard:04d}.csv")
    if os.path.exists(manifest_path) and not overwrite:
        print(f"{manifest_path} exists -- pass --overwrite to redo this shard")
        return manifest_path

    series = list_series(mri["raw_root"], split, mri["max_repeats"],
                         mri.get(f"max_series_{split}"), mri["seed"])
    mine = series[shard::num_shards][:limit]
    print(f"[shard {shard}/{num_shards}] {len(mine)} of {len(series)} {split} series -> "
          f"{cache_root}", flush=True)

    vae = build_vae(config["text2ct_root"], paths["vae"], device, config["model_def"],
                    fast_concat=config["model"]["vae_fast_concat"])
    encoder = build_text_encoder(config["text2ct_root"], paths["clip"], device,
                                 data_cfg["text_max_length"])

    store = LatentStore(cache_root, shard, split, bundle=bundle)
    reports, rows, failures = {}, [], {}
    started = time.time()
    for i, entry in enumerate(mine):
        name = sample_id(entry["study_uid"], entry["series_id"])
        try:
            if entry["study_uid"] not in reports:
                reports[entry["study_uid"]] = read_report(entry["archive"], entry["study_uid"])
            text = format_report(reports[entry["study_uid"]], data_cfg["report_sections"],
                                 entry["modality"], data_cfg["modality_prefix"])
            array, spacing, native_slices = preprocess_volume(
                read_member(entry["archive"], entry["member"]), entry["plane"],
                volume_cfg["inplane_mm"], volume_cfg["slice_mm"], volume_cfg["inplane_size"],
                volume_cfg["num_slices"], volume_cfg["posterior_shift_mm"],
                volume_cfg["min_native_slices"], tuple(volume_cfg["percentiles"]))
            latent = encode_volume(vae, array, device)
            embedding = encode_reports(encoder, [text])[0].numpy().astype(np.float32)
        except (VolumeUnusable, ReportMissing, KeyError, OSError, ValueError) as error:
            failures[type(error).__name__] = failures.get(type(error).__name__, 0) + 1
            continue

        rows.append({
            "sample_id": name, "split": split,
            "modality": modality_name(entry["modality"]), "plane": plane_name(entry["plane"]),
            "n_native_slices": native_slices, "native_spacing": ",".join(f"{v:.4f}" for v in spacing),
            "latent_path": store.save(f"{name}.latent.npy", latent),
            "embedding_path": store.save(f"{name}.text.npy", embedding),
            "zip": store.zip_name,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(mine)}  {len(rows)} kept  "
                  f"{(time.time() - started) / (i + 1):.2f}s/series", flush=True)

    store.close()
    write_manifest(manifest_path, rows)
    print(f"[shard {shard}] {len(rows)} kept, dropped {failures or 0} -> {manifest_path}")

    if shard == 0:
        with open(os.path.join(cache_root, CACHE_META_NAME), "w") as handle:
            json.dump(cache_meta(config, paths["vae"], paths["clip"]), handle, indent=1)
        print(f"[shard 0] wrote {CACHE_META_NAME}")
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="stop after this many series (debugging)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--loose", action="store_true",
                        help="one file per artifact instead of one zip per shard. Only for a tiny "
                             "cache -- the full split would blow the inode quota.")
    parser.add_argument("--device")
    parser.add_argument("--upstream_layout", metavar="DIR",
                        help="after building the cache, re-emit it in the on-disk contract "
                             "scripts/diff_model_train.py expects, plus its three config JSONs, so "
                             "the unmodified upstream trainer can run on it. Three files per "
                             "series -- a validation harness for hundreds, never the full split.")
    args = parser.parse_args(argv)

    config = validate(apply_dotted(load_config(args.config), args.set))
    if not config["mrrate"]["raw_root"]:
        raise SystemExit("config.mrrate.raw_root is required to build the cache")
    manifest = prepare(config, args.split, args.shard, args.num_shards, args.limit, args.overwrite,
                       bundle=not args.loose, device=args.device)
    if args.upstream_layout:
        from .upstream_bridge import write_upstream_configs, write_upstream_layout

        listing = write_upstream_layout(config, args.split, args.upstream_layout, args.limit)
        written = write_upstream_configs(config, args.upstream_layout, args.split)
        print(f"upstream layout -> {listing}")
        for name, path in written.items():
            print(f"  {name}: {path}")
        print(f"\n  cd {config['text2ct_root']} && python scripts/diff_model_train.py \\\n"
              f"      --env_config {written['environment.json']} \\\n"
              f"      --model_config {written['config_diff_model.json']} \\\n"
              f"      --model_def {written['config_rflow.json']} --num_gpus 1 --no_amp")
    return manifest


if __name__ == "__main__":
    main()
