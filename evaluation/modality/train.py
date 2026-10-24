"""Train the modality classifier; select the checkpoint on val.

    python -m evaluation.modality.train --config evaluation/modality/config.yaml

Writes to `output_dir`: `config.yaml`, `splits.csv` (a copy of the manifest trained on),
`history.jsonl`, `last.pt` (resumable, rewritten every epoch) and `best.pt` (highest val balanced
accuracy, ties to lower val NLL). Re-running resumes from `last.pt`; `--restart` ignores it.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from evaluation.modality import (CLASSES, INPUT_SHAPE, NUM_CLASSES, build_model,
                                 file_sha256, predict_proba, save_checkpoint)
from evaluation.modality.data import ModalityDataset
from evaluation.modality.metrics import summarize


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_weights(labels, mode):
    if mode == "none":
        return torch.ones(NUM_CLASSES)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def validate(model, dataset, device, batch_size):
    inputs = np.stack([dataset.load(i) for i in range(len(dataset))])
    probs = predict_proba(model, inputs, batch_size, device)
    return summarize(dataset.labels, probs.argmax(axis=1), probs)


def is_better(metrics, best, key):
    if best is None:
        return True
    return (metrics[key], -metrics["nll_evaluated"]) > (best[key], -best["nll_evaluated"])


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def train(config, restart=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.train
    out = config.output_dir
    os.makedirs(out, exist_ok=True)
    seed_everything(config.seed)

    splits_path = os.path.join(config.data_root, "splits.csv")
    train_set = ModalityDataset(config.data_root, "train", augment=cfg.augment)
    val_set = ModalityDataset(config.data_root, "val")
    meta = {"config": OmegaConf.to_container(config, resolve=True), "classes": list(CLASSES),
            "input_shape": list(INPUT_SHAPE), "splits_sha256": file_sha256(splits_path),
            "git_commit": git_commit(),
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
    weights = class_weights(train_set.labels, cfg.class_weighting).to(device)
    meta["class_weights"] = weights.tolist()

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
        # Seeded per epoch, so a resumed run shuffles, augments and drops out as an unbroken one.
        torch.manual_seed(config.seed + epoch)
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
            num_workers=cfg.num_workers, pin_memory=device == "cuda",
            generator=torch.Generator().manual_seed(config.seed + epoch),
            persistent_workers=False)
        running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=cfg.bf16 and device == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        model.eval()
        val = validate(model, val_set, device, cfg.batch_size)
        record = {"epoch": epoch, "train_loss": running / max(seen, 1),
                  "lr": scheduler.get_last_lr()[0],
                  **{f"val_{k}": val[k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                                  "nll_evaluated", "ece_evaluated")}}
        print(json.dumps(record))
        with open(os.path.join(out, "history.jsonl"), "a") as handle:
            handle.write(json.dumps(record) + "\n")

        scalars = {k: v for k, v in val.items() if isinstance(v, (int, float))}
        if is_better(scalars, best, config.select_metric):
            best = {**scalars, "epoch": epoch}
            save_checkpoint(os.path.join(out, "best.pt"), model,
                            {**meta, "epoch": epoch, "val": val})
            print(f"  new best at epoch {epoch}: val {config.select_metric} "
                  f"{scalars[config.select_metric]:.4f}")
        save_checkpoint(last_path, model, {**meta, "epoch": epoch, "val": val},
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch, best=best)
    return best


def main():
    parser = argparse.ArgumentParser(description="Train the modality classifier.")
    parser.add_argument("--config", default="evaluation/modality/config.yaml")
    parser.add_argument("--restart", action="store_true", help="Ignore an existing last.pt.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. train.epochs=2")
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    print(train(config, restart=args.restart))


if __name__ == "__main__":
    main()
