"""Encode MR-RATE volumes and reports into the latent blocks lvfm/train.py trains on.

Volume and report are done in one pass per series, because the conditioning text embeds that
series' own modality and plane -- so there is one embedding per series, not per study, and loading
the VAE and the text encoder twice would buy nothing.

Written to run as a SLURM array: each task takes an interleaved slice of the series list and
writes its own manifest CSV, which the dataset globs. Shard 0 also writes the two boundary latents.

    python lvfm/preprocess_mrrate.py --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
        --split train --shard $SLURM_ARRAY_TASK_ID --num_shards 64 [--zip]

`--zip` bundles a shard's artifacts into one zip rather than two .pt files per series, which takes
a full split from ~244k files to one per shard. Only worth it under an inode quota -- see
LatentStore. Either layout trains unchanged; the manifest records which one was written.

`--embeddings_only` re-encodes just the report embeddings, under `mri.text_encoder`, into
`mri.embedding_root`, leaving the latents alone. Point the dataset's `embedding_root` at the same
directory and it joins the two trees on `sample_id`.
"""

import argparse
import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from echosyn.common import instantiate
from echosyn.common.mrrate import (EMBEDDING_MANIFEST_FIELDS, MANIFEST_FIELDS, LatentStore,
                                   VolumeTooShort, build_conditioner, encode_boundary,
                                   encode_hlip_null, encode_neurobert_null, encode_volume,
                                   list_series, load_neurobert, preprocess_volume, read_member,
                                   read_report, sample_id, write_manifest)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MR-RATE into latent blocks")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--shard", type=int, default=0, help="This task's index.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of tasks.")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode existing samples.")
    parser.add_argument("--zip", action="store_true",
                        help="Bundle this shard into one zip instead of two files per series.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N series (smoke test).")
    parser.add_argument("--embeddings_only", action="store_true",
                        help="Re-encode only the report embeddings, into mri.embedding_root.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    mri = config.mri
    root = mri.dataset_root
    if args.embeddings_only:
        # Its own directory: both trees write <root>/manifest/<split>-<shard>.csv, so sharing one
        # would overwrite the latent manifests.
        root = mri.embedding_root
        if os.path.realpath(root) == os.path.realpath(mri.dataset_root):
            raise SystemExit("mri.embedding_root must differ from mri.dataset_root")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    preprocess_args = OmegaConf.to_container(mri.preprocess, resolve=True)

    vae = None if args.embeddings_only else instantiate(config.vae).eval().to(device, dtype)
    encode_text = build_conditioner(mri, device)

    # The two learned sequence boundaries. Written once, by shard 0, through the same encode path
    # as real volumes; both training and inference read them from here.
    if args.shard == 0 and not args.embeddings_only:
        os.makedirs(os.path.join(root, "boundary"), exist_ok=True)
        for name, value in (("black", config.black_value), ("white", config.white_value)):
            path = os.path.join(root, "boundary", f"{name}.pt")
            if args.overwrite or not os.path.exists(path):
                torch.save(encode_boundary(vae, value, mri.preprocess.inplane_size), path)
                print(f"Wrote {path}")

    # The null report for token-sequence conditioning, also once, by shard 0. train.py loads it
    # into the denoiser's learned null at the start of a fresh run.
    if args.shard == 0 and mri.get("text_encoder") in ("hlip_tokens", "neurobert_tokens"):
        path = os.path.join(root, "null_text.pt")
        if args.overwrite or not os.path.exists(path):
            if mri.text_encoder == "neurobert_tokens":
                null = encode_neurobert_null(*load_neurobert(device), mri.text_max_length)
            else:
                from evaluation.hlip_metrics import load_hlip
                null = encode_hlip_null(*load_hlip(device))
            torch.save(null, path)
            print(f"Wrote {path}")

    series = list_series(mri.raw_root, args.split, mri.max_repeats,
                         mri.get(f"max_series_{args.split}"), config.seed)
    if args.embeddings_only:
        # Whole archives per task, in archive order: reading a report means scanning its tar's
        # headers, so each tar should be opened by one task, once. The dataset joins on sample_id,
        # so which task holds a series does not matter.
        archives = sorted({e["archive"] for e in series})[args.shard::args.num_shards]
        mine = set(archives)
        series = sorted((e for e in series if e["archive"] in mine), key=lambda e: e["archive"])
    else:
        series = series[args.shard::args.num_shards]
    if args.limit:
        series = series[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(series)} {args.split} series")

    store = LatentStore(root, f"{args.split}-{args.shard:04d}", bundle=args.zip,
                        bundle_dir="embeddings" if args.embeddings_only else "latents")
    # Token sequences carry no acquisition markers, so they are stored once per study and every
    # series row points at that member: ~4x smaller than one copy per series.
    per_study = mri.get("text_encoder") in ("hlip_tokens", "neurobert_tokens")
    rows, skipped, cache = [], {}, {}
    truncated = 0
    for entry in tqdm(series, disable=None):
        sid = sample_id(entry["study_uid"], entry["series_id"])
        latent_path = store.member("latents", sid)
        embed_path = store.member("embeddings", entry["study_uid"] if per_study else sid)
        try:
            if args.embeddings_only and per_study:
                if entry["study_uid"] not in cache:
                    cache[entry["study_uid"]] = True
                    artifact = encode_text(read_report(entry["archive"], entry["study_uid"]),
                                           None, None)
                    if "neurobert" in artifact:
                        truncated += artifact["neurobert"]["n_tokens"] > mri.text_max_length
                    store.write(embed_path, artifact)
                rows.append({"sample_id": sid, "split": args.split,
                             "embedding_path": embed_path, "zip": store.zip_path or ""})
                continue
            if args.embeddings_only:
                # Every listed series, including ones the latent pass dropped as too short; the
                # dataset joins from the latent manifest, which decides what exists.
                if args.overwrite or not store.done(embed_path):
                    # A study's series share its report, so each (study, modality, plane) is
                    # encoded once.
                    key = (entry["study_uid"], entry["modality"], entry["plane"])
                    if key not in cache:
                        cache[key] = encode_text(read_report(entry["archive"], entry["study_uid"]),
                                                 entry["modality"], entry["plane"])
                    store.write(embed_path, cache[key])
                rows.append({"sample_id": sid, "split": args.split,
                             "embedding_path": embed_path, "zip": store.zip_path or ""})
                continue
            # Both artifacts are produced together -- the conditioning text carries the volume's
            # native spacing, which only the volume pass knows -- so an incomplete pair is redone.
            if not args.overwrite and store.done(latent_path, embed_path):
                latent = store.read(latent_path)
            else:
                volume, _ = preprocess_volume(
                    read_member(entry["archive"], entry["member"]), entry["plane"],
                    **preprocess_args,
                )
                latent = encode_volume(vae, volume, mri.vae_batch_size)
                store.write(latent_path, latent)
                store.write(embed_path, encode_text(
                    read_report(entry["archive"], entry["study_uid"]),
                    entry["modality"], entry["plane"]))
        except VolumeTooShort:
            skipped["too_short"] = skipped.get("too_short", 0) + 1
            continue
        except Exception as e:
            # A handful of MR-RATE series have unreadable headers or truncated members. One bad
            # series must not kill a shard of thousands, but the reasons are counted and printed.
            reason = type(e).__name__
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        rows.append({
            "sample_id": sid,
            "split": args.split,
            "modality": entry["modality"],
            "plane": entry["plane"],
            "n_slices": latent.shape[1],
            "latent_path": latent_path,
            "embedding_path": embed_path,
            "zip": store.zip_path or "",
        })

    store.close()
    write_manifest(os.path.join(root, "manifest", f"{args.split}-{args.shard:04d}.csv"), rows,
                   EMBEDDING_MANIFEST_FIELDS if args.embeddings_only else MANIFEST_FIELDS)
    print(f"[shard {args.shard}] wrote {len(rows)} rows; skipped {sum(skipped.values())} {skipped}")
    if mri.get("text_encoder") == "neurobert_tokens" and args.embeddings_only:
        print(f"[shard {args.shard}] {truncated} of {len(cache)} reports truncated at "
              f"{mri.text_max_length} tokens")


if __name__ == "__main__":
    main()
