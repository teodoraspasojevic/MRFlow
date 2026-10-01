#!/usr/bin/env python
"""Gate between preprocessing and training: is the cache a complete, non-overlapping partition?

    python -m baselines.ccella.verify_cache --config <cfg> --split train --num_shards 64

Exits non-zero on anything that would make a training run quietly wrong. It needs no GPU, decodes
no volume, and reads only manifests and zip central directories, so it runs on a login node in
under a minute.

**Why this exists.** Nothing else notices a shard that never finished or one that finished with the
wrong contents. `read_manifest` globs `manifest/<split>-*.csv` and concatenates whatever it finds,
so 45 of 64 shards trains quietly on 68% of the data, and a shard written under a different
`num_shards` overlaps its neighbours without any error at all. Both have happened here: a 34-element
recovery array re-partitioned shard 12 into a 1/34 slice that duplicated 13,621 series against
shards 22 and 23, and it was invisible until the manifests were compared against each other.

What it checks:

  * `cache_meta.json` exists and its `num_shards` matches the one given;
  * every shard id in `[0, num_shards)` has a manifest;
  * no `sample_id` appears twice across the split;
  * the union of `sample_id`s is **exactly** the set `list_series` + `shard_slice` imply -- nothing
    missing, nothing extra, and each id in the shard it belongs to;
  * every artifact a manifest row references exists in the zip it names;
  * labels and masks are 14 characters, and `modality_id` is in range.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from collections import Counter, defaultdict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "baselines.ccella"

from .config import load_config
from .labels import NUM_LABELS
from .store import CACHE_META_NAME, MANIFEST_DIR, read_cache_meta, sample_key


def expected_partition(config, split, num_shards):
    """`{shard: {sample_id}}` the cache should hold, from the same two functions preprocessing uses."""
    from echosyn.common.mrrate import list_series

    from .prepare_data import shard_slice

    spec = config["mrrate"]
    series = list_series(spec["raw_root"], split, max_repeats=spec["max_repeats"],
                         max_series=spec[f"max_series_{split}"], seed=spec["seed"])
    return {shard: {sample_key(i["study_uid"], i["series_id"])
                    for i in shard_slice(series, shard, num_shards)}
            for shard in range(num_shards)}


def verify(config, split, num_shards, check_members=True):
    cache_root = config["data"]["cache_root"]
    problems, notes = [], []

    meta = read_cache_meta(cache_root)
    if meta is None:
        problems.append(f"{CACHE_META_NAME} is missing: shard 0 never finished")
    else:
        built_with = meta.get("num_shards", {}).get(split)
        if built_with not in (None, num_shards):
            problems.append(f"{CACHE_META_NAME} says the {split} split was built with "
                            f"num_shards={built_with}, asked {num_shards}")
        elif built_with is None:
            notes.append(f"{CACHE_META_NAME} records no num_shards for {split} "
                         f"(cache predates the stamp); trusting the {num_shards} given")

    found = {}
    for shard in range(num_shards):
        path = os.path.join(cache_root, MANIFEST_DIR, f"{split}-{shard:04d}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as handle:
            found[shard] = list(csv.DictReader(handle))

    absent = [s for s in range(num_shards) if s not in found]
    if absent:
        problems.append(f"{len(absent)} of {num_shards} shards have no manifest: {absent}")

    ids = [r["sample_id"] for rows in found.values() for r in rows]
    duplicated = [i for i, c in Counter(ids).items() if c > 1]
    if duplicated:
        owners = defaultdict(set)
        for shard, rows in found.items():
            for r in rows:
                if r["sample_id"] in set(duplicated):
                    owners[r["sample_id"]].add(shard)
        shards = sorted({s for v in owners.values() for s in v})
        problems.append(f"{len(duplicated)} sample_ids appear in more than one shard "
                        f"(shards {shards}) -- these shards were written under different "
                        f"num_shards and overlap")

    expected = expected_partition(config, split, num_shards)
    for shard, rows in sorted(found.items()):
        have, want = {r["sample_id"] for r in rows}, expected[shard]
        if have - want:
            problems.append(f"shard {shard}: {len(have - want)} rows do not belong to it")
        missing = want - have
        if missing:
            notes.append(f"shard {shard}: {len(missing)} of {len(want)} series missing "
                         f"(skipped as unusable, or the task died mid-shard)")

    total_expected = sum(len(v) for v in expected.values())
    notes.insert(0, f"{len(set(ids)):,} distinct series across {len(found)}/{num_shards} shards "
                    f"(the split holds {total_expected:,})")

    # rows must point at artifacts that are actually there
    if check_members and found:
        cache = {}
        bad = 0
        for shard, rows in sorted(found.items()):
            for r in rows:
                for zip_key, member_key in (("latent_zip", "latent_member"),
                                            ("report_zip", "report_member")):
                    name = r[zip_key]
                    if name not in cache:
                        path = os.path.join(cache_root, name)
                        cache[name] = (set(zipfile.ZipFile(path).namelist())
                                       if os.path.exists(path) else None)
                    names = cache[name]
                    if names is None:
                        problems.append(f"shard {shard}: missing archive {name}")
                        cache[name] = set()
                    elif r[member_key] not in names:
                        bad += 1
        if bad:
            problems.append(f"{bad} manifest rows reference a member that is not in its archive")
        notes.append(f"{len(cache)} archives opened, all members present"
                     if not bad else f"{len(cache)} archives opened")

    for shard, rows in sorted(found.items()):
        for r in rows[:0] or rows:
            if len(r["labels"]) != NUM_LABELS or len(r["label_mask"]) != NUM_LABELS:
                problems.append(f"shard {shard}: a row has a label vector that is not {NUM_LABELS} wide")
                break
            if not 0 <= int(r["modality_id"]) <= 6:
                problems.append(f"shard {shard}: modality_id out of range")
                break

    masked = sum(1 for rows in found.values() for r in rows if set(r["label_mask"]) == {"0"})
    notes.append(f"{masked:,} series have no label row and are masked out of the classification loss")
    return problems, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--num_shards", type=int, required=True,
                        help="the constant the cache was built with, not the array size")
    parser.add_argument("--skip_members", action="store_true",
                        help="skip opening every archive (faster, weaker)")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    problems, notes = verify(config, args.split, args.num_shards, not args.skip_members)
    for note in notes:
        print(f"  {note}")
    if problems:
        print(f"\nFAILED -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")
        raise SystemExit(1)
    print(f"\nOK -- {args.split} cache is a complete, non-overlapping partition")


if __name__ == "__main__":
    main()
