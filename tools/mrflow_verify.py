"""Check that a preprocessed MR-RATE split is complete before training on it.

`preprocess_mrrate.py` runs as an independent SLURM array and nothing downstream notices a task
that never finished: `MRRateLatentBlockDataset` globs `manifest/*.csv` and raises only if it finds
*zero* rows, so 63 of 64 shards would train quietly on 98% of the data. Per-shard skip counts are
printed to stdout and then thrown away, and `store.close()` runs before `write_manifest`, so a job
killed between them leaves a bundle no manifest points at.

This recomputes the expected series list -- deterministic, because `list_series` shuffles on a fixed
seed and then truncates -- and compares it against what is actually on disk.

    python tools/mrflow_verify.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
        --split train --num_shards 64

Reads parquet indices, manifest CSVs and zip central directories only: no GPU, no volume decoding.
Exits non-zero if anything is missing, so it can gate the training submission.
"""

import argparse
import csv
import os
import zipfile
from collections import Counter

from omegaconf import OmegaConf

from echosyn.common.mrrate import list_series, sample_id


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a preprocessed MR-RATE split")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_shards", type=int, required=True,
                        help="Array size used to preprocess. Required: a missing final shard is "
                             "invisible if inferred from the manifests that exist.")
    parser.add_argument("--list_missing", type=int, default=0,
                        help="Print up to N missing sample ids.")
    return parser.parse_args()


def listing(root, container):
    """Every artifact path available in one container -- a zip bundle, or the loose tree."""
    if container:
        return set(zipfile.ZipFile(os.path.join(root, container)).namelist())
    out = set()
    for kind in ("latents", "embeddings"):
        base = os.path.join(root, kind)
        for sub in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if os.path.isdir(os.path.join(base, sub)):
                out |= {f"{kind}/{sub}/{f}" for f in os.listdir(os.path.join(base, sub))}
    return out


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    mri, root = config.mri, config.mri.dataset_root
    min_slices = 2 * config.globals.target_nframes

    # Which series each shard was responsible for. Same seed and same stride as preprocessing.
    series = list_series(mri.raw_root, args.split, mri.max_repeats,
                         mri.get(f"max_series_{args.split}"), config.seed)
    owner = {sample_id(e["study_uid"], e["series_id"]): shard
             for shard in range(args.num_shards)
             for e in series[shard::args.num_shards]}
    print(f"{args.split}: {len(owner)} series expected across {args.num_shards} shards")

    listings, found, problems = {}, set(), []
    print(f"\n{'shard':>5}  {'rows':>7}  {'ok':>7}  status")
    for shard in range(args.num_shards):
        manifest = os.path.join(root, "manifest", f"{args.split}-{shard:04d}.csv")
        if not os.path.exists(manifest):
            problems.append(f"shard {shard}: no manifest -- task never finished")
            print(f"{shard:>5}  {'-':>7}  {'-':>7}  MISSING MANIFEST")
            continue

        with open(manifest, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

        ok, notes = 0, Counter()
        for row in rows:
            container = row.get("zip") or ""
            if container not in listings:
                listings[container] = listing(root, container)
            have = listings[container]
            if row["latent_path"] not in have or row["embedding_path"] not in have:
                notes["artifact absent"] += 1
            elif int(row["n_slices"]) < min_slices:
                notes["too few slices"] += 1          # dataset has no short-volume fallback
            elif owner.get(row["sample_id"]) != shard:
                notes["wrong shard"] += 1             # manifest and series list disagree
            else:
                ok += 1
                found.add(row["sample_id"])
        status = "ok" if ok == len(rows) else ", ".join(f"{n} {k}" for k, n in notes.items())
        print(f"{shard:>5}  {len(rows):>7}  {ok:>7}  {status}")
        if notes:
            problems.append(f"shard {shard}: {dict(notes)}")

    # Bundles and temp files nothing points at: a job killed between close() and write_manifest.
    referenced = set(listings) - {""}
    bundles = {f"latents/{f}" for f in os.listdir(os.path.join(root, "latents"))
               if f.endswith(".zip")} if os.path.isdir(os.path.join(root, "latents")) else set()
    for orphan in sorted(bundles - referenced):
        problems.append(f"{orphan}: bundle no manifest references")
    for name in sorted(os.listdir(os.path.join(root, "latents")) if bundles else []):
        if ".tmp." in name:
            problems.append(f"latents/{name}: leftover temp file from a killed task")

    for name in ("black", "white"):
        if not os.path.exists(os.path.join(root, "boundary", f"{name}.pt")):
            problems.append(f"boundary/{name}.pt missing -- shard 0 never finished")

    missing = set(owner) - found
    print(f"\nencoded {len(found)}/{len(owner)} expected series ({100*len(found)/max(len(owner),1):.2f}%)")
    if missing:
        by_shard = Counter(owner[s] for s in missing)
        print(f"missing {len(missing)}, worst shards: {by_shard.most_common(5)}")
        for sid in sorted(missing)[:args.list_missing]:
            print(f"  missing {sid} (shard {owner[sid]})")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems[:40]:
            print(f"  - {p}")
        print("\nFAIL")
        raise SystemExit(1)
    print("\nOK -- every expected series is encoded and readable")


if __name__ == "__main__":
    main()
