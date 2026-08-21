import os
import random

import torch
from torch.utils.data import Dataset

from echosyn.common.mrrate import load_artifact, read_manifest


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
    """

    def __init__(self, root, split, block_size=16, p_start=0.3, p_end=0.2,
                 deterministic=False, seed=42):
        self.root = root
        self.block_size = block_size
        self.p_start = p_start
        self.p_end = p_end
        self.deterministic = deterministic
        self.seed = seed

        self.rows = read_manifest(root, split)
        if not self.rows:
            raise RuntimeError(f"no {split!r} rows found in {root}/manifest/*.csv")

        black, white = (
            torch.load(os.path.join(root, "boundary", f"{name}.pt"), map_location="cpu")
            for name in ("black", "white")
        )
        # (C, s, s) -> (C, B, s, s) by repeating along the block axis, which is what the generator
        # does for its seed block -- so a trained start example and an inference seed are identical.
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
        b, T = self.block_size, latent.shape[1]

        roll = rng.random()
        if roll < self.p_start:
            block_curr, block_next = self.black, latent[:, :b]
        elif roll < self.p_start + self.p_end:
            block_curr, block_next = latent[:, T - b:], self.white
        else:
            t = rng.randint(0, T - 2 * b)
            block_curr, block_next = latent[:, t:t + b], latent[:, t + b:t + 2 * b]

        embedding = load_artifact(self.root, row, "embedding_path")
        embedding = embedding / (embedding.norm(p=2) + 1e-6)

        return {
            "image": block_curr.float(),   # condition: [C, T, H, W]
            "video": block_next.float(),   # target:    [C, T, H, W]
            "embedding": embedding,        # text embedding: [1, D]
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
