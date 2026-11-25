# Adapted from CTFlow (https://github.com/WongJiayi/CTFlow); modified for MRFlow.
import os
import random

import torch
from torch.utils.data import Dataset

from mrflow.mri import label_counts, load_artifact, modality_to_id, plane_to_id, read_manifest
from mrflow.text import text_condition


class MRRateLatentBlockDataset(Dataset):
    """Pairs of consecutive latent blocks from MR-RATE: `image` is the condition, `video` the target.

    Three kinds of pair are sampled: start (black block -> first block, p_start), end (last
    block -> white block, p_end, the stop signal) and interior. Latents are stored as the VAE
    posterior's (mean, std), so blocks are 2C wide and are sampled in train.py.
    `embedding_root` optionally points at a separate tree of report embeddings.
    """

    def __init__(self, root, split, block_size=16, p_start=0.3, p_end=0.2,
                 deterministic=False, seed=42, embedding_root=None, text_sections=("neurobert",),
                 max_text_tokens=512, global_condition="mean_pool"):
        self.root = root
        self.text_sections = list(text_sections)
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
        # Optional trim columns keep only slices [trim_start, trim_stop) of the stored latent, e.g. to
        # drop blank end slices; manifests without these columns use every slice.
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
        # NeuroBERT report tokens, padded, and their mask for cross-attention; the pooled vector
        # for the timestep embedding. Hidden states are not normalized.
        embedding, mask, pooled = text_condition(embedding, self.text_sections,
                                                 self.max_text_tokens)
        text = {"embedding_mask": mask}
        if self.global_condition != "none":
            text["pooled"] = pooled

        modality_id, plane_id = self.labels[idx]

        return {
            **text,
            # Posterior parameters, not latents -- train.py samples these before scaling.
            "image": block_curr.float(),   # condition: [2C, T, H, W]
            "video": block_next.float(),   # target:    [2C, T, H, W]
            "embedding": embedding,        # report tokens: [max_text_tokens, D]
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

        if name == "MRRateLatentBlock":
            dataset = MRRateLatentBlockDataset(**params)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(datasets)
