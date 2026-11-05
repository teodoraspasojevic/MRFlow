"""Report-volume alignment of a checkpoint, measured with its own flow-matching loss.

Each query is a val block pair from a different study. Its target block is scored under K
candidate reports -- its own and K-1 from other studies -- with the same noise and timesteps for
every candidate, and the report with the lowest loss is the model's retrieval. A model that ignores
the report ranks its own at chance (1/K); one that uses it ranks it first. The same pass scores the
null report, so the gap to it says how much the report lowers the loss at all.

Cheap enough for validation (no sampling, no VAE) and run on fixed queries, noise and candidates,
so the numbers of two checkpoints are comparable. Within a run only: the scale depends on the text
encoder. The external judge is still the HLIP study-level metric in evaluation/.

    python -m echosyn.common.report_alignment --config <run>/config.yaml --ckpt <run>/checkpoint-N/denoiser_ema
"""

import random

import torch

from echosyn.common import sample_latents, scale_latents


def _text(item):
    """The denoiser's text kwargs for one dataset item."""
    out = {"encoder_hidden_states": item["embedding"]}
    if "embedding_mask" in item:
        out["mask"] = item["embedding_mask"]
    if "pooled" in item:
        out["pooled"] = item["pooled"]
    return out


def pick_queries(dataset, n, seed=0):
    """`n` dataset indices from distinct studies with distinct, non-empty reports."""
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    picked, studies, reports = [], set(), []
    for i in order:
        study = dataset.embedding_rows[i]["embedding_path"]
        if study in studies:
            continue
        item = dataset[i]
        if "embedding_mask" in item and item["embedding_mask"].sum() <= 2:
            continue  # no findings or impression: just <s></s>
        if any(torch.equal(item["embedding"], r) for r in reports):
            continue
        studies.add(study)
        reports.append(item["embedding"])
        picked.append(i)
        if len(picked) == n:
            break
    return picked


@torch.no_grad()
def report_alignment(denoiser, dataset, config, vae_scaling, queries=64, candidates=16,
                     timesteps=8, batch_size=32, seed=0):
    """`{report_r1, report_r5, report_mrr, report_mean_rank, report_chance, report_loss,
    report_gap_shuffled, report_gap_null}` for one checkpoint.

    Ranks treat ties at their expected value under a random tie-break, so an untrained model whose
    output ignores the text scores exactly chance rather than a hit. The gaps are relative:
    (loss under another / the null report - loss under the true one) / loss under the true one.
    """
    device = next(denoiser.parameters()).device
    dtype = next(denoiser.parameters()).dtype
    items = [dataset[i] for i in pick_queries(dataset, queries, seed)]
    n, K = len(items), min(candidates, len(items))
    rng = random.Random(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    t = (torch.arange(timesteps, device=device) + 0.5) / timesteps  # stratified, fixed

    ranks, true_loss, other_loss, null_loss = [], [], [], []
    for q, item in enumerate(items):
        others = rng.sample([j for j in range(n) if j != q], K - 1)
        cands = [q] + others  # candidate 0 is the true report
        # One fixed draw of the latents and noise per query, shared by every candidate.
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed + q)
            cond = scale_latents(sample_latents(config, item["image"][None].to(device)), vae_scaling)
            x0 = scale_latents(sample_latents(config, item["video"][None].to(device)), vae_scaling)
        x1 = torch.randn((timesteps, *x0.shape[1:]), device=device, generator=gen)
        tt = t.view(-1, 1, 1, 1, 1)
        z_t = (1 - tt) * x0 + (1e-5 + (1 - 1e-5) * tt) * x1
        u = (1 - 1e-5) * x1 - x0

        # Rows: (candidate report c, timestep j) for c in cands, plus the null report last.
        rows = [(c, j) for c in cands + [None] for j in range(timesteps)]
        losses = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            texts = [_text(items[c if c is not None else q]) for c, _ in chunk]
            js = torch.tensor([j for _, j in chunk], device=device)
            kwargs = {k: torch.stack([x[k] for x in texts]).to(device) for k in texts[0]}
            kwargs["encoder_hidden_states"] = kwargs["encoder_hidden_states"].to(dtype)
            v = denoiser(
                z_t[js].to(dtype), t[js].to(dtype), **kwargs,
                cond_image=cond.expand(len(chunk), -1, -1, -1, -1).to(dtype),
                modality_id=item["modality_id"].repeat(len(chunk)).to(device),
                plane_id=item["plane_id"].repeat(len(chunk)).to(device),
                force_drop_ids=torch.tensor([c is None for c, _ in chunk], device=device),
            ).sample
            losses.append((v.float() - u[js]).pow(2).flatten(1).mean(1))
        loss = torch.cat(losses).view(K + 1, timesteps).mean(1)  # [K candidates + null]

        cand, own = loss[:K], loss[0]
        lower = (cand < own).sum().item()
        ties = (cand == own).sum().item() - 1
        ranks.append((lower, ties))
        true_loss.append(own.item())
        other_loss.append(cand[1:].mean().item())
        null_loss.append(loss[K].item())

    def expected_hit(k):  # P(true report within the top k) under a random tie-break
        return sum(min(max(k - lo, 0) / (ti + 1), 1.0) for lo, ti in ranks) / n

    def expected_rr(lo, ti):  # E[1/rank], rank uniform on lo+1 .. lo+ti+1
        return sum(1 / r for r in range(lo + 1, lo + ti + 2)) / (ti + 1)

    mean = lambda xs: sum(xs) / len(xs)
    return {
        "report_r1": expected_hit(1),
        "report_r5": expected_hit(5),
        "report_mrr": mean([expected_rr(lo, ti) for lo, ti in ranks]),
        "report_mean_rank": mean([1 + lo + ti / 2 for lo, ti in ranks]),
        "report_chance": 1 / K,
        "report_loss": mean(true_loss),
        "report_gap_shuffled": mean([(o - l) / l for o, l in zip(other_loss, true_loss)]),
        "report_gap_null": mean([(z - l) / l for z, l in zip(null_loss, true_loss)]),
        "report_queries": n,
    }


if __name__ == "__main__":
    import argparse
    import json

    from omegaconf import OmegaConf

    from echosyn.common import get_vae_scaler, instantiate_class_from_config
    from echosyn.common.datasets import instantiate_dataset

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the run's saved config.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint-N/denoiser_ema")
    args = ap.parse_args()
    config = OmegaConf.load(args.config)
    denoiser = instantiate_class_from_config(config.denoiser).from_pretrained(args.ckpt)
    denoiser = denoiser.to("cuda", torch.bfloat16).eval()
    dataset = instantiate_dataset(config.val_datasets, split=["VAL"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        metrics = report_alignment(denoiser, dataset, config, get_vae_scaler(config, "cuda"),
                                   **config.validation.get("alignment", {}))
    print(json.dumps(metrics, indent=1))
