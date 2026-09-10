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

`--embeddings_only` re-encodes just the report embeddings, into `mri.embedding_root`, leaving the
latents alone. That is how the conditioning changes without paying to re-encode the volumes: the
report is the only input, so the train split is ~20 GPU-hours at the sectioned conditioning's
measured 123 ms/series, against ~290 for the volume pass. Point the dataset's `embedding_root` at
the same directory and it joins the two trees on `sample_id`.
"""

import argparse
import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from echosyn.common import instantiate
from echosyn.common.mrrate import (EMBEDDING_MANIFEST_FIELDS, MANIFEST_FIELDS, LatentStore,
                                   VolumeTooShort, build_conditioner, encode_boundary,
                                   encode_volume, list_series, preprocess_volume, read_member,
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
    # Set a root to one of the two optional report embeddings
    if args.embeddings_only and not mri.get("embedding_root"):
        raise SystemExit(
            f"--embeddings_only needs mri.embedding_root, and {args.config} has none. That key is "
            f"where this conditioning's embeddings go, while the latents under mri.dataset_root "
            f"are read as they are. Point it at a new directory and set the dataset stanza's "
            f"embedding_root to match."
        )
    root = mri.embedding_root if args.embeddings_only else mri.dataset_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    preprocess_args = OmegaConf.to_container(mri.preprocess, resolve=True)

    vae = None if args.embeddings_only else instantiate(config.vae).eval().to(device, dtype)
    conditioner = build_conditioner(mri, device)

    # The two learned sequence boundaries. Written once, by shard 0, through the same encode path
    # as real volumes; both training and inference read them from here.
    if args.shard == 0 and not args.embeddings_only:
        os.makedirs(os.path.join(root, "boundary"), exist_ok=True)
        for name, value in (("black", config.black_value), ("white", config.white_value)):
            path = os.path.join(root, "boundary", f"{name}.pt")
            if args.overwrite or not os.path.exists(path):
                torch.save(encode_boundary(vae, value, mri.preprocess.inplane_size), path)
                print(f"Wrote {path}")

    series = list_series(mri.raw_root, args.split, mri.max_repeats,
                         mri.get(f"max_series_{args.split}"), config.seed)
    series = series[args.shard::args.num_shards]
    if args.limit:
        series = series[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(series)} {args.split} series")

    store = LatentStore(root, f"{args.split}-{args.shard:04d}", bundle=args.zip,
                        bundle_dir="embeddings" if args.embeddings_only else "latents")

    def embedding_for(entry):
        return conditioner.encode(read_report(entry["archive"], entry["study_uid"]),
                                  entry["modality"], entry["plane"])

    rows, skipped = [], {}
    for entry in tqdm(series, disable=None):
        sid = sample_id(entry["study_uid"], entry["series_id"])
        embed_path = store.member("embeddings", sid)
        try:
            if args.embeddings_only:
                # No volume is decoded, so a series the latent pass dropped as too short is still
                # embedded here. Harmless: the dataset joins from the latent manifest, which stays
                # the authority on which series exist.
                if args.overwrite or not store.done(embed_path):
                    store.write(embed_path, embedding_for(entry))
                rows.append({"sample_id": sid, "split": args.split,
                             "embedding_path": embed_path, "zip": store.zip_path or ""})
                continue
            # Both artifacts are produced together -- the conditioning text carries the volume's
            # native spacing, which only the volume pass knows -- so an incomplete pair is redone.
            latent_path = store.member("latents", sid)
            if not args.overwrite and store.done(latent_path, embed_path):
                latent = store.read(latent_path)
            else:
                volume, _ = preprocess_volume(
                    read_member(entry["archive"], entry["member"]), entry["plane"],
                    **preprocess_args,
                )
                latent = encode_volume(vae, volume, mri.vae_batch_size)
                store.write(latent_path, latent)
                store.write(embed_path, embedding_for(entry))
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


if __name__ == "__main__":
    main()
