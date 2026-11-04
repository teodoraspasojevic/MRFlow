"""Build a blank-slice-trimmed version of an existing MR-RATE latent dataset.

The source dataset is read, never written. Two stages:

  scan   Re-run the exact `preprocess_volume` (CPU only, no VAE) for every series of one source
         manifest shard and record per-slice statistics. Policy-independent, so it runs once. Each
         shard is one parquet written atomically; a finished shard is skipped on re-run.

             python lvfm/trim_mrrate.py scan --config C --split train --shard 3 --stats_dir S

  plan   Apply a trimming policy to the scan. A dry run by default: it writes the per-volume audit,
         per-bucket summaries and a threshold-sensitivity table under --report_dir, and nothing
         else. With --write it also creates the new dataset version at --out:

             python lvfm/trim_mrrate.py plan --config C --stats_dir S --report_dir R \
                 --out /path/to/mrflow_flux_trim-v1 --write

Why bounds rather than new latents: FLUX's VAE encodes every slice as an independent 2D image, so a
stored latent's slice t depends on image slice t alone, and slicing the latent is the same as
encoding the trimmed volume. The version therefore holds manifests with `trim_start`/`trim_stop`
that point at the source zips by absolute path, the two boundary tokens, an audit and a provenance
record -- a few hundred small files instead of a 6 TB copy. The dataset applies the window at load.
The cost is that the version is only valid while the source latents are unchanged, which is why the
scan records every member's CRC and `plan --write` re-checks them.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from echosyn.common.mrrate import (MANIFEST_FIELDS, list_series, preprocess_volume, read_member,
                                   sample_id, write_manifest)
from echosyn.common.trimming import (TAUS, TrimPolicy, blank_runs, blank_slices, slice_stats,
                                     trim_bounds)

TRIM_FIELDS = MANIFEST_FIELDS + ("trim_start", "trim_stop", "source_slices")
STAT_KEYS = [f"fg_{t}" for t in TAUS] + ["mean", "std", "p99", "nonfinite"]
SPLITS = ("train", "val", "test")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout
    diff = run("diff", "HEAD")
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(diff.strip()),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def source_manifests(root, split):
    return sorted(glob(os.path.join(root, "manifest", f"{split}-*.csv")))


### scan ###


_PRE = None


def _scan_one(row):
    """One series -> its stats record. Failures are recorded, never raised."""
    rec = {k: row[k] for k in ("sample_id", "split", "modality", "plane", "latent_path",
                               "embedding_path", "zip", "archive", "member", "crc", "zip_size")}
    rec["manifest_slices"] = int(row["n_slices"])
    try:
        volume, _ = preprocess_volume(read_member(row["archive"], row["member"]), row["plane"],
                                      **_PRE)
        stats = slice_stats(volume[0].numpy())
        rec["T"] = int(volume.shape[1])
        rec["status"] = "ok" if rec["T"] == rec["manifest_slices"] else "slice_count_mismatch"
        rec.update({k: stats[k].tolist() for k in STAT_KEYS})
    except Exception as e:
        rec.update(T=-1, status=f"error: {type(e).__name__}: {e}"[:300])
    return rec


def _init(pre):
    global _PRE
    _PRE = pre
    os.environ["OMP_NUM_THREADS"] = "1"


def scan(args, config):
    mri = config.mri
    manifests = source_manifests(mri.dataset_root, args.split)
    path = manifests[args.shard]
    name = os.path.basename(path)[:-4]
    out = os.path.join(args.stats_dir, f"{name}.parquet")
    manifest_sha = sha256(path)
    if os.path.exists(out) and not args.overwrite:
        meta = pq.read_schema(out).metadata or {}
        if meta.get(b"source_manifest_sha256", b"").decode() == manifest_sha:
            print(f"{out} exists for this manifest; skipping (--overwrite to redo)")
            return
        raise SystemExit(f"{out} was scanned from a different {path}; refusing to mix them")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # Where each series' raw NIfTI lives. list_series is deterministic; repeats are kept so every
    # manifest row resolves whatever max_repeats the source was built with.
    index = {sample_id(e["study_uid"], e["series_id"]): e
             for e in list_series(mri.raw_root, args.split, None)}
    crcs = {}
    for row in rows:
        if not row["zip"]:
            raise SystemExit("only zipped (--zip) source datasets are supported")
        entry = index[row["sample_id"]]
        row["archive"], row["member"] = entry["archive"], entry["member"]
        zpath = os.path.join(mri.dataset_root, row["zip"])
        if zpath not in crcs:
            crcs[zpath] = {i.filename: (i.CRC, i.file_size) for i in zipfile.ZipFile(zpath).infolist()}
        row["crc"], row["zip_size"] = crcs[zpath][row["latent_path"]]

    print(f"[scan {name}] {len(rows)} series, {args.workers} workers", flush=True)
    pre = OmegaConf.to_container(mri.preprocess, resolve=True)
    start = time.time()
    with Pool(args.workers, initializer=_init, initargs=(pre,)) as pool:
        records = pool.map(_scan_one, rows, chunksize=4)

    table = pa.Table.from_pylist(records)
    table = table.replace_schema_metadata({
        "source_manifest": path, "source_manifest_sha256": manifest_sha,
        "preprocess": json.dumps(pre), "git": json.dumps(git_state()),
        "taus": json.dumps(TAUS), "limit": str(args.limit or ""),
    })
    os.makedirs(args.stats_dir, exist_ok=True)
    tmp = f"{out}.tmp.{os.getpid()}"
    pq.write_table(table, tmp)
    os.replace(tmp, out)  # atomic: a killed task leaves no half-written shard
    status = Counter(r["status"].split(":")[0] for r in records)
    print(f"[scan {name}] wrote {out} in {time.time() - start:.0f}s: {dict(status)}")


### plan ###


def load_stats(stats_dir, root, splits):
    """Every scan shard, checked against the source manifests: one per manifest, same rows."""
    frames = []
    for split in splits:
        for path in source_manifests(root, split):
            name = os.path.basename(path)[:-4]
            shard = os.path.join(stats_dir, f"{name}.parquet")
            if not os.path.exists(shard):
                raise SystemExit(f"missing scan shard {shard}")
            meta = pq.read_schema(shard).metadata
            if meta[b"source_manifest_sha256"].decode() != sha256(path):
                raise SystemExit(f"{path} changed since it was scanned")
            if meta[b"limit"]:
                raise SystemExit(f"{shard} is a --limit scan; rescan it in full")
            df = pq.read_table(shard).to_pandas()
            df["source_manifest"] = os.path.basename(path)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df.sample_id.duplicated().any():
        raise SystemExit(f"duplicate sample_ids across scan shards: "
                         f"{df.sample_id[df.sample_id.duplicated()].head().tolist()}")
    return df


def apply_policy(df, policy):
    """Scan records -> one audit row per series."""
    rows = []
    for r in df.to_dict("records"):
        rec = {k: r[k] for k in ("sample_id", "split", "modality", "plane", "status",
                                 "source_manifest")}
        rec["source_slices"] = r["manifest_slices"]
        if r["status"] != "ok":
            rows.append(dict(rec, trim_start=None, trim_stop=None, new_slices=0, lead=None,
                             trail=None, interior=None, reason="unprocessable", included=False))
            continue
        blank = blank_slices(r, policy)
        lead, trail, interior = blank_runs(blank)
        start, stop, reason = trim_bounds(blank, policy)
        included = start is not None
        rows.append(dict(rec, trim_start=start, trim_stop=stop,
                         new_slices=(stop - start) if included else 0, lead=lead, trail=trail,
                         interior=interior, blank_frac=float(blank.mean()), reason=reason,
                         included=included))
    return pd.DataFrame(rows)


def summarize(audit, block):
    """Per split and bucket: everything the trimming decision needs, as one table."""
    audit = audit.assign(bucket=audit.modality + "__" + audit.plane)
    ok = audit[audit.status == "ok"]
    g = ok.groupby(["split", "bucket"])
    out = g.agg(volumes=("sample_id", "size"), slices=("source_slices", "sum"),
                slices_p50=("source_slices", "median"), slices_min=("source_slices", "min"),
                slices_max=("source_slices", "max"), lead_mean=("lead", "mean"),
                lead_p50=("lead", "median"), lead_p95=("lead", lambda s: s.quantile(0.95)),
                trail_mean=("trail", "mean"), trail_p50=("trail", "median"),
                trail_p95=("trail", lambda s: s.quantile(0.95)), interior_mean=("interior", "mean"),
                new_slices_p50=("new_slices", "median"))
    out["blank_slice_pct"] = 100 * g.apply(
        lambda d: (d.lead + d.trail + d.interior).sum() / d.source_slices.sum())
    out["mostly_blank_vol_pct"] = 100 * g.blank_frac.apply(lambda s: (s >= 0.9).mean())
    out["all_blank_vol_pct"] = 100 * g.reason.apply(lambda s: (s == "all_blank").mean())
    # What the training sampler sees: a blank target for `start`, a blank condition for `end`.
    out["first_block_blank_pct"] = 100 * g.lead.apply(lambda s: (s >= block).mean())
    out["last_block_blank_pct"] = 100 * g.trail.apply(lambda s: (s >= block).mean())
    out["trimmed_vol_pct"] = 100 * g.reason.apply(lambda s: s.isin(["trimmed", "min_slices"]).mean())
    out["min_slices_hit"] = g.reason.apply(lambda s: (s == "min_slices").sum())
    out["excluded"] = g.included.apply(lambda s: (~s).sum())
    out["unprocessable"] = audit.groupby(["split", "bucket"]).status.apply(
        lambda s: (s != "ok").sum()).reindex(out.index).fillna(0).astype(int)
    return out.round(2)


def sensitivity(df, base, block):
    """Headline numbers across a small grid of thresholds, so the choice can be seen to be stable."""
    rows = []
    for tau in (0.02, 0.05, 0.1):
        for min_fg in (0.005, 0.01, 0.02, 0.05):
            policy = TrimPolicy(**{**base.to_dict(), "tau": tau, "min_foreground": min_fg})
            a = apply_policy(df, policy)
            a = a[a.status == "ok"].assign(bucket=a.modality + "__" + a.plane)
            for (split, bucket), d in a.groupby(["split", "bucket"]):
                rows.append({"tau": tau, "min_foreground": min_fg, "split": split,
                             "bucket": bucket,
                             "removed_slice_pct": round(100 * (1 - d.new_slices.sum() / d.source_slices.sum()), 2),
                             "last_block_blank_pct": round(100 * (d.trail >= block).mean(), 2),
                             "first_block_blank_pct": round(100 * (d.lead >= block).mean(), 2),
                             "excluded": int((~d.included).sum())})
    return pd.DataFrame(rows)


def check_paths(out, config):
    """The version must be a new directory that does not overlap any source tree."""
    out = os.path.realpath(out)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; a dataset version is never overwritten")
    for key in ("dataset_root", "embedding_root", "raw_root"):
        src = config.mri.get(key)
        if not src:
            continue
        src = os.path.realpath(src)
        if out == src or out.startswith(src + os.sep) or src.startswith(out + os.sep):
            raise SystemExit(f"--out {out} overlaps mri.{key} {src}")


def write_version(args, config, df, audit, policy):
    root = config.mri.dataset_root
    check_paths(args.out, config)

    # Inputs unchanged since the scan: every referenced member, from each zip's central directory.
    changed = []
    for zrel, group in df.groupby("zip"):
        members = {i.filename: (i.CRC, i.file_size)
                   for i in zipfile.ZipFile(os.path.join(root, zrel)).infolist()}
        for r in group.itertuples(index=False):
            if members.get(r.latent_path) != (r.crc, r.zip_size):
                changed.append(r.sample_id)
    if changed:
        raise SystemExit(f"{len(changed)} source latents changed since the scan: {changed[:3]}")

    tmp = f"{os.path.realpath(args.out)}.tmp.{os.getpid()}"
    os.makedirs(os.path.join(tmp, "boundary"))
    for name in ("black", "white"):
        src = os.path.join(root, "boundary", f"{name}.pt")
        shutil.copy2(src, os.path.join(tmp, "boundary", f"{name}.pt"))

    merged = audit.merge(df[["sample_id", "latent_path", "embedding_path", "zip"]], on="sample_id")
    for manifest, group in merged.groupby("source_manifest"):
        rows = [{"sample_id": r.sample_id, "split": r.split, "modality": r.modality,
                 "plane": r.plane, "n_slices": int(r.new_slices), "latent_path": r.latent_path,
                 "embedding_path": r.embedding_path,
                 # absolute, so the version reads the source zips without a copy or a link
                 "zip": os.path.join(os.path.realpath(root), r.zip),
                 "trim_start": int(r.trim_start), "trim_stop": int(r.trim_stop),
                 "source_slices": int(r.source_slices)}
                for r in group.itertuples(index=False) if r.included]
        write_manifest(os.path.join(tmp, "manifest", manifest), rows, TRIM_FIELDS)
    os.makedirs(os.path.join(tmp, "audit"))
    audit.to_csv(os.path.join(tmp, "audit", "audit.csv"), index=False)

    counts = audit.groupby("split").reason.value_counts().unstack(fill_value=0)
    manifests = {os.path.basename(p): sha256(p)
                 for s in args.splits for p in source_manifests(root, s)}
    provenance = {
        "version": os.path.basename(os.path.realpath(args.out)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"dataset_root": os.path.realpath(root), "manifests_sha256": manifests,
                   "boundary_sha256": {n: sha256(os.path.join(root, "boundary", f"{n}.pt"))
                                       for n in ("black", "white")},
                   "raw_root": config.mri.raw_root,
                   "embedding_root_reused": config.mri.get("embedding_root")},
        "scan": {"stats_dir": os.path.realpath(args.stats_dir),
                 "git": json.loads(pq.read_schema(glob(os.path.join(args.stats_dir, "*.parquet"))[0]).metadata[b"git"]),
                 "preprocess": OmegaConf.to_container(config.mri.preprocess, resolve=True)},
        "plan_git": git_state(),
        "policy": policy.to_dict(),
        "counts": {s: counts.loc[s].to_dict() for s in counts.index},
        "included": audit.groupby("split").included.sum().astype(int).to_dict(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "torch": __import__("torch").__version__,
                        "nibabel": __import__("nibabel").__version__},
        "checksums": "manifests/*.csv and audit/audit.csv are listed below by sha256. Latent data "
                     "is not copied: every row's source member is pinned by the CRC32 and size "
                     "recorded in the scan shards, re-verified when this version was written.",
    }
    files = sorted(glob(os.path.join(tmp, "manifest", "*.csv"))) + [os.path.join(tmp, "audit", "audit.csv")]
    provenance["output_sha256"] = {os.path.relpath(p, tmp): sha256(p) for p in files}
    with open(os.path.join(tmp, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, default=int)
    os.rename(tmp, os.path.realpath(args.out))
    print(f"wrote dataset version {args.out}: {provenance['included']}")


def plan(args, config):
    policy = TrimPolicy(tau=args.tau, min_foreground=args.min_foreground, margin=args.margin,
                        min_slices=args.min_slices, all_blank=args.all_blank)
    df = load_stats(args.stats_dir, config.mri.dataset_root, args.splits)
    audit = apply_policy(df, policy)
    block = config.globals.target_nframes

    os.makedirs(args.report_dir, exist_ok=True)
    audit.to_csv(os.path.join(args.report_dir, "audit.csv"), index=False)
    summary = summarize(audit, block)
    summary.to_csv(os.path.join(args.report_dir, "summary.csv"))
    sens = sensitivity(df, policy, block)
    sens.to_csv(os.path.join(args.report_dir, "sensitivity.csv"), index=False)
    with open(os.path.join(args.report_dir, "policy.json"), "w") as f:
        json.dump(policy.to_dict(), f, indent=2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summary.to_string())
    print(audit.groupby("split").reason.value_counts().unstack(fill_value=0).to_string())

    if args.write:
        write_version(args, config, df, audit, policy)
    else:
        print(f"dry run: report in {args.report_dir}; nothing else written (--write to build)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--config", required=True)
    s.add_argument("--split", required=True, choices=SPLITS)
    s.add_argument("--shard", type=int, required=True, help="Index into the split's manifests.")
    s.add_argument("--stats_dir", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--limit", type=int, default=None, help="First N rows only (smoke test).")
    s.add_argument("--overwrite", action="store_true")

    q = sub.add_parser("plan")
    q.add_argument("--config", required=True)
    q.add_argument("--stats_dir", required=True)
    q.add_argument("--report_dir", required=True)
    q.add_argument("--splits", nargs="+", default=list(SPLITS))
    q.add_argument("--tau", type=float, default=0.05)
    q.add_argument("--min_foreground", type=float, default=0.01)
    q.add_argument("--margin", type=int, default=0)
    q.add_argument("--min_slices", type=int, default=None, help="Default: 2 * target_nframes.")
    q.add_argument("--all_blank", default="exclude", choices=["exclude", "keep"])
    q.add_argument("--out", default=None, help="New dataset version directory (with --write).")
    q.add_argument("--write", action="store_true", help="Build the version, not just the report.")
    return p.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.cmd == "scan":
        scan(args, config)
        return
    if args.min_slices is None:
        args.min_slices = 2 * config.globals.target_nframes
    if args.write and not args.out:
        raise SystemExit("--write needs --out")
    plan(args, config)


if __name__ == "__main__":
    main()
