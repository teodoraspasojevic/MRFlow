"""Train the pathology classifier; select the checkpoint and the thresholds on val.

    python -m evaluation.pathology.train --config evaluation/pathology/config.yaml

Writes to `output_dir`, as the modality classifier does: `config.yaml`, `splits.csv`,
`history.jsonl`, `last.pt` (resumable) and `best.pt` -- the epoch with the highest val macro
AUROC, ties to lower val BCE. `best.pt`'s meta also carries what the metrics need and must never
be re-derived elsewhere: the per-label F1-optimal thresholds on val, the train prevalence (`prior`)
and the label table's sha256. Re-running resumes from `last.pt`; `--restart` ignores it.
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from evaluation.modality import INPUT_SHAPE, file_sha256, save_checkpoint
from evaluation.modality.train import git_commit, seed_everything
from evaluation.pathology import LABELS, build_model, labels_sha256, predict_proba, read_labels
from evaluation.pathology.data import PathologyDataset
from evaluation.pathology.metrics import best_thresholds, summarize


def validate(model, dataset, device, batch_size, prior, min_positives):
    inputs = np.stack([dataset.load(i) for i in range(len(dataset))])
    probs = predict_proba(model, inputs, batch_size, device)
    thresholds = best_thresholds(dataset.labels, probs)
    return summarize(dataset.labels, probs, thresholds, prior, min_positives), thresholds


def train(config, restart=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.train
    out = config.output_dir
    os.makedirs(out, exist_ok=True)
    seed_everything(config.seed)

    labels = read_labels(config.labels_root)
    splits_path = os.path.join(config.data_root, "splits.csv")
    train_set = PathologyDataset(config.data_root, "train", labels, augment=cfg.augment)
    val_set = PathologyDataset(config.data_root, "val", labels)
    prior = train_set.labels.mean(axis=0)
    pos = train_set.labels.sum(axis=0)
    pos_weight = np.minimum((len(train_set) - pos) / np.maximum(pos, 1), cfg.pos_weight_cap)
    meta = {"config": OmegaConf.to_container(config, resolve=True), "labels": list(LABELS),
            "input_shape": list(INPUT_SHAPE), "splits_sha256": file_sha256(splits_path),
            "labels_sha256": labels_sha256(config.labels_root), "git_commit": git_commit(),
            "prior": prior.tolist(), "pos_weight": pos_weight.tolist(),
            "n_train": len(train_set), "n_val": len(val_set),
            "n_train_failed_prepare": len(train_set.rows) - len(train_set),
            "n_val_failed_prepare": len(val_set.rows) - len(val_set)}
    OmegaConf.save(config, os.path.join(out, "config.yaml"))
    shutil.copyfile(splits_path, os.path.join(out, "splits.csv"))

    model = build_model(meta["config"]["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_set) / cfg.batch_size)
    total, warmup = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
        (s + 1) / max(warmup, 1), 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))))
    weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    start, best = 0, None
    last_path = os.path.join(out, "last.pt")
    if os.path.exists(last_path) and not restart:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt["meta"]["splits_sha256"] != meta["splits_sha256"]:
            raise RuntimeError(f"{last_path} was trained on a different splits.csv; --restart")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"resumed from {last_path} at epoch {start}")
    if start == 0 and os.path.exists(os.path.join(out, "history.jsonl")):
        os.remove(os.path.join(out, "history.jsonl"))

    for epoch in range(start, cfg.epochs):
        model.train()
        torch.manual_seed(config.seed + epoch)
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            pin_memory=device == "cuda", generator=torch.Generator().manual_seed(config.seed + epoch))
        running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=cfg.bf16 and device == "cuda"):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        model.eval()
        val, thresholds = validate(model, val_set, device, cfg.batch_size, prior,
                                   config.min_positives)
        record = {"epoch": epoch, "train_loss": running / max(seen, 1),
                  "lr": scheduler.get_last_lr()[0],
                  **{f"val_{k}": val[k] for k in ("macro_auroc", "macro_auprc", "macro_f1",
                                                  "mean_bce")},
                  "val_auroc": {n: round(r["auroc"], 4) for n, r in val["per_label"].items()}}
        print(json.dumps(record))
        with open(os.path.join(out, "history.jsonl"), "a") as handle:
            handle.write(json.dumps(record) + "\n")

        score = (val["macro_auroc"], -val["mean_bce"])
        if best is None or score > (best["macro_auroc"], -best["mean_bce"]):
            best = {"epoch": epoch, "macro_auroc": val["macro_auroc"], "mean_bce": val["mean_bce"]}
            save_checkpoint(os.path.join(out, "best.pt"), model,
                            {**meta, "epoch": epoch, "val": val,
                             "thresholds": thresholds.tolist()})
            print(f"  new best at epoch {epoch}: val macro AUROC {val['macro_auroc']:.4f}")
        save_checkpoint(last_path, model, {**meta, "epoch": epoch, "val": val,
                                           "thresholds": thresholds.tolist()},
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch, best=best)
    return best


def main():
    parser = argparse.ArgumentParser(description="Train the pathology classifier.")
    parser.add_argument("--config", default="evaluation/pathology/config.yaml")
    parser.add_argument("--restart", action="store_true", help="Ignore an existing last.pt.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. train.epochs=2")
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    print(train(config, restart=args.restart))


if __name__ == "__main__":
    main()
