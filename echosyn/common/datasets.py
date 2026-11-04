import os
import random

import torch
from torch.utils.data import Dataset

from echosyn.common.mrrate import (label_counts, load_artifact, modality_to_id, plane_to_id,
                                   read_manifest, text_condition)


class LatentBlockDataset(Dataset):
    """
    Dataset for auto-regressive block-wise training.

    Loads pre-encoded latent volumes and text embeddings.
    Returns consecutive block pairs (current block as condition, next block as target).
    """

    def __init__(self, root_dir, embedding_dir, block_size=16):
        self.root_dir = root_dir
        self.embedding_dir = embedding_dir
        self.block_size = block_size

        self.file_paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt")
        ])

        self.embedding_paths = [
            os.path.join(embedding_dir, os.path.basename(f))
            for f in self.file_paths
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        latent_path = self.file_paths[idx]
        embed_path = self.embedding_paths[idx]

        latent = torch.load(latent_path, map_location="cpu")   # [C, T, H, W]
        embedding = torch.load(embed_path, map_location="cpu") # [N, D]
        embedding = embedding[0].unsqueeze(0)
        embedding = embedding / (embedding.norm(p=2) + 1e-6)

        C, T, H, W = latent.shape
        max_start = T - 2 * self.block_size

        # 50% chance to sample from the start (bias toward beginning of volume)
        if random.random() < 0.5:
            t = 0
        else:
            t = random.randint(0, max_start)

        block_curr = latent[:, t:t + self.block_size]
        block_next = latent[:, t + self.block_size:t + 2 * self.block_size]

        return {
            "image": block_curr,     # condition: [C, T, H, W]
            "video": block_next,     # target:    [C, T, H, W]
            "embedding": embedding,  # text embedding: [1, D]
        }


class MRRateLatentBlockDataset(Dataset):
    """
    Dataset for auto-regressive block-wise training on MR-RATE.

    Returns the same three keys as LatentBlockDataset, so the training loop is unchanged. What
    differs is that the two sequence boundaries are sampled explicitly, because the rollout needs
    all three cases trained rather than just the interior:

        start     black boundary block  -> first block          (inference seeds with black)
        interior  block [t, t+B)        -> block [t+B, t+2B)
        end       last block            -> white boundary block (the learned stop signal)

    Without `start` the black-seeded first inference step is off-distribution; without `end` the
    stop token is never emitted and every volume runs to max_blocks. `p_start=0.3` matches the
    released CTFlow checkpoint's own run, whose name records `black_rate_0.3`.

    Preprocessing keeps only volumes with at least 2 * block_size slices, so every row can produce
    any of the three kinds and there is no short-volume special case.

    Latents and boundary tokens are stored as the posterior's (mean, std) on the channel axis, so
    everything returned is 2C wide and train.py samples it. Slicing is on time, so it is unaffected.

    Modality and plane are resolved to integer class ids at construction, so an unmapped label
    fails here rather than 40 minutes into a run.

    `text_sections`, when set, reads token-sequence artifacts (`encode_hlip_tokens`,
    `encode_neurobert_tokens`): `embedding` is then the sections' token sequences padded to
    `max_text_tokens`, plus `embedding_mask` and `pooled`. `global_condition: none` leaves `pooled`
    out, for a denoiser with `pooled_channels: 0`.

    `embedding_root`, when set, is a second tree holding a different conditioning's report
    embeddings (`lvfm/preprocess_mrrate.py --embeddings_only`); its rows are joined to the latent
    manifest on `sample_id`. Unset, the embeddings stored next to the latents are used.
    """

    def __init__(self, root, split, block_size=16, p_start=0.3, p_end=0.2,
                 deterministic=False, seed=42, embedding_root=None, text_sections=None,
                 max_text_tokens=None, global_condition="mean_pool"):
        self.root = root
        self.text_sections = list(text_sections) if text_sections else None
        self.max_text_tokens = max_text_tokens
        assert global_condition in ("mean_pool", "none"), global_condition
        self.global_condition = global_condition
        self.block_size = block_size
        self.p_start = p_start
        self.p_end = p_end
        self.deterministic = deterministic
        self.seed = seed

        self.rows = read_manifest(root, split)
        if not self.rows:
            raise RuntimeError(f"no {split!r} rows found in {root}/manifest/*.csv")

        # Resolve every label up front: the whole manifest is validated once, in the main process.
        self.labels = [(modality_to_id(r["modality"]), plane_to_id(r["plane"])) for r in self.rows]

        self.embedding_root = embedding_root or root
        self.embedding_rows = self.rows
        if embedding_root:
            by_id = {r["sample_id"]: r for r in read_manifest(embedding_root, split)}
            absent = [r["sample_id"] for r in self.rows if r["sample_id"] not in by_id]
            if absent:
                raise RuntimeError(f"{len(absent)} of {len(self.rows)} {split!r} series have no "
                                   f"embedding under {embedding_root} (first: {absent[:3]})")
            self.embedding_rows = [by_id[r["sample_id"]] for r in self.rows]
            print(f"[{split}] embeddings from {embedding_root}")
        modality, plane, pair = label_counts(self.rows)
        print(f"[{split}] {len(self.rows)} series")
        print(f"[{split}] modality: {dict(sorted(modality.items()))}")
        print(f"[{split}] plane:    {dict(sorted(plane.items()))}")
        print(f"[{split}] pairs:    "
              f"{ {f'{m}/{p}': n for (m, p), n in sorted(pair.items())} }")

        black, white = (
            torch.load(os.path.join(root, "boundary", f"{name}.pt"), map_location="cpu")
            for name in ("black", "white")
        )
        # (2C, s, s) -> (2C, B, s, s), matching what the generator does for its seed block.
        # Repeat before sampling, so every frame gets its own draw.
        self.black = black.unsqueeze(1).repeat(1, block_size, 1, 1)
        self.white = white.unsqueeze(1).repeat(1, block_size, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        # __getitem__ runs in dataloader workers, so a deterministic draw has to come from
        # (seed, idx) rather than global RNG state, which varies with worker count.
        rng = random.Random(f"{self.seed}-{idx}") if self.deterministic else random

        latent = load_artifact(self.root, row, "latent_path")
        # A trimmed dataset version (lvfm/trim_mrrate.py) reads the latents it was built from and
        # records the retained window; manifests without these columns use every slice.
        if row.get("trim_stop"):
            if latent.shape[1] != int(row["source_slices"]):
                raise RuntimeError(f"{row['sample_id']}: latent has {latent.shape[1]} slices, the "
                                   f"trim bounds were computed on {row['source_slices']}")
            latent = latent[:, int(row["trim_start"]):int(row["trim_stop"])]
        b, T = self.block_size, latent.shape[1]

        roll = rng.random()
        if roll < self.p_start:
            block_curr, block_next = self.black, latent[:, :b]
        elif roll < self.p_start + self.p_end:
            block_curr, block_next = latent[:, T - b:], self.white
        else:
            t = rng.randint(0, T - 2 * b)
            block_curr, block_next = latent[:, t:t + b], latent[:, t + b:t + 2 * b]

        embedding = load_artifact(self.embedding_root, self.embedding_rows[idx], "embedding_path")
        text = {}
        if self.text_sections:
            # Token-sequence conditioning: padded tokens and their mask for cross-attention, the
            # pooled vector for the timestep embedding. Hidden states are not normalized.
            embedding, mask, pooled = text_condition(embedding, self.text_sections,
                                                     self.max_text_tokens)
            text = {"embedding_mask": mask}
            if self.global_condition != "none":
                text["pooled"] = pooled
        else:
            embedding = embedding / (embedding.norm(p=2) + 1e-6)

        modality_id, plane_id = self.labels[idx]

        return {
            **text,
            # Posterior parameters, not latents -- train.py samples these before scaling.
            "image": block_curr.float(),   # condition: [2C, T, H, W]
            "video": block_next.float(),   # target:    [2C, T, H, W]
            "embedding": embedding,        # text embedding: [1, D]
            # Scalars, so the default collator stacks them into [B] class-label tensors.
            "modality_id": torch.tensor(modality_id, dtype=torch.long),
            "plane_id": torch.tensor(plane_id, dtype=torch.long),
        }


def instantiate_dataset(configs, split=None):
    datasets = []
    for cfg in configs:
        if not cfg.get("active", False):
            continue
        name = cfg.name
        params = dict(cfg.params)

        if name == "LatentBlock":
            dataset = LatentBlockDataset(**params)
        elif name == "MRRateLatentBlock":
            dataset = MRRateLatentBlockDataset(**params)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(datasets)
