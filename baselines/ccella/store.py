"""The sharded zip cache: latents per series, report embeddings per study, one manifest per shard.

Upstream writes one `_emb.nii.gz` per volume and one `text.npy` per study into a directory tree
(`diff_model_create_training_data.py:90`, `gen_json_maisi_merged.py:60`). At 575,328 train series
that is over half a million inodes against this account's ~81,000 hard limit on `/hnvme`, so the
layout -- and only the layout -- has to change.

    <cache_root>/artifacts/<split>-<shard>.latents.zip    one member per series
    <cache_root>/artifacts/<split>-<shard>.reports.zip    one member per STUDY
    <cache_root>/manifest/<split>-<shard>.csv             one row per series
    <cache_root>/cache_meta.json                          the fingerprint, written by shard 0

**Two stores, not one, because a report is one study and a volume is one series.** A FLAN-T5-XXL
`last_hidden_state` is 512 x 4096; at fp16 that is 4 MB, and MR-RATE has ~7 series per study, so
writing it per series would cost 2.3 TB instead of 330 GB. Each manifest row names the report
member it shares, and the loader reads it from the report zip.

**Writes are shard-local and atomic.** Each preprocessing task owns its two archives and nobody
else writes them, so no lock is needed; both are written to `<name>.tmp` and `os.replace`d only
after `close()`, so a killed task leaves no half-archive that a later run would read as complete.
The manifest is written last, for the same reason `mrflow_verify.py` treats a bundle with no
manifest as a failed task: the manifest is the commit record.

`ZIP_STORED`, not deflate: the members are fp16 latents and fp16 embeddings, both already dense, and
storing uncompressed keeps a member's bytes contiguous so `read` is one seek and one read. Measured
on MRFlow's own cache, a 5.4 MB member comes back in ~2 ms.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile

import numpy as np

ARTIFACT_DIR = "artifacts"
MANIFEST_DIR = "manifest"
CACHE_META_NAME = "cache_meta.json"
CACHE_FORMAT_VERSION = 1

MANIFEST_FIELDS = (
    "sample_id",       # sha1(study|series)[:16] -- stable, carries no identifier
    "split",
    "study_key",       # sha1(study)[:16] -- the report member name, and the label join key
    "series_key",      # sha1(study|series)[:16], == sample_id; named for the manifest contract
    "latent_zip",      # relative to cache_root
    "latent_member",
    "report_zip",
    "report_member",
    "modality",
    "modality_id",
    "plane",
    "spacing_mm",      # native (x, y, z) voxel size, ';'-joined -- the model's spacing input
    "grid",            # the fixed RAS grid this latent was encoded at, ';'-joined
    "labels",          # 14 chars of '0'/'1', in labels.LABELS_14 order
    "label_mask",      # 14 chars of '0'/'1'; all-zero when the study has no label row
)


def study_key(study_uid):
    """Identifier-free, stable key for one study. Same construction as `mrrate.sample_id`."""
    return hashlib.sha1(study_uid.encode()).hexdigest()[:16]


def sample_key(study_uid, series_id):
    """Identifier-free, stable key for one series."""
    return hashlib.sha1(f"{study_uid}|{series_id}".encode()).hexdigest()[:16]


class ShardStore:
    """The two archives one preprocessing task owns. Not thread-safe and not meant to be."""

    def __init__(self, cache_root, split, shard):
        self.cache_root = cache_root
        self.latent_name = os.path.join(ARTIFACT_DIR, f"{split}-{shard:04d}.latents.zip")
        self.report_name = os.path.join(ARTIFACT_DIR, f"{split}-{shard:04d}.reports.zip")
        os.makedirs(os.path.join(cache_root, ARTIFACT_DIR), exist_ok=True)
        self._latent_tmp = os.path.join(cache_root, self.latent_name + ".tmp")
        self._report_tmp = os.path.join(cache_root, self.report_name + ".tmp")
        self._latents = zipfile.ZipFile(self._latent_tmp, "w", zipfile.ZIP_STORED)
        self._reports = zipfile.ZipFile(self._report_tmp, "w", zipfile.ZIP_STORED)
        self._written_reports = set()

    def save_latent(self, key, array):
        member = f"{key}.npy"
        self._latents.writestr(member, _to_npy(array))
        return member

    def save_report(self, key, array):
        """Write a study's embedding once. Returns the member name whether or not it was new."""
        member = f"{key}.npy"
        if key not in self._written_reports:
            self._reports.writestr(member, _to_npy(array))
            self._written_reports.add(key)
        return member

    def has_report(self, key):
        return key in self._written_reports

    @property
    def n_reports(self):
        return len(self._written_reports)

    def close(self):
        """Seal both archives. Only after this does a reader see either of them."""
        self._latents.close()
        self._reports.close()
        os.replace(self._latent_tmp, os.path.join(self.cache_root, self.latent_name))
        os.replace(self._report_tmp, os.path.join(self.cache_root, self.report_name))

    def abort(self):
        """Drop both temporaries. A killed task leaves nothing a later run could mistake for done."""
        for handle, path in ((self._latents, self._latent_tmp), (self._reports, self._report_tmp)):
            try:
                handle.close()
            finally:
                if os.path.exists(path):
                    os.remove(path)


def _to_npy(array):
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


### Reading ###########################################################################################

class ZipReader:
    """Lazily-opened zip handles, one set per process.

    A `zipfile.ZipFile` holds a file object with its own offset, so sharing one across DataLoader
    workers after a fork returns interleaved garbage. Handles are therefore keyed by pid and opened
    on first use *in the worker*, which is also why `CcellaDataset` must not open anything in its
    constructor.
    """

    def __init__(self, cache_root, max_open=16):
        self.cache_root = cache_root
        self.max_open = max_open
        self._handles = {}
        self._pid = None

    def _zip(self, name):
        pid = os.getpid()
        if self._pid != pid:                      # forked: the parent's handles are not ours
            self._handles = {}
            self._pid = pid
        if name not in self._handles:
            path = os.path.join(self.cache_root, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"cache archive missing: {path}")
            if len(self._handles) >= self.max_open:
                _, old = self._handles.popitem()
                old.close()
            self._handles[name] = zipfile.ZipFile(path)
        return self._handles[name]

    def read_array(self, zip_name, member):
        """One member as a numpy array. Raises on a missing or unreadable member -- never zeros."""
        try:
            blob = self._zip(zip_name).read(member)
        except KeyError as exc:
            raise KeyError(f"{member} not in {zip_name}") from exc
        try:
            return np.load(io.BytesIO(blob), allow_pickle=False)
        except Exception as exc:
            raise IOError(f"{member} in {zip_name} is not a readable .npy: {exc}") from exc


def write_manifest(cache_root, split, shard, rows):
    path = os.path.join(cache_root, MANIFEST_DIR, f"{split}-{shard:04d}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


def read_manifest(cache_root, split):
    """Every shard manifest for one split, in filename order. Raises if the split has none."""
    from glob import glob

    paths = sorted(glob(os.path.join(cache_root, MANIFEST_DIR, f"{split}-*.csv")))
    if not paths:
        raise FileNotFoundError(f"no {split} manifest under {cache_root}/{MANIFEST_DIR}")
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


### Fingerprint #######################################################################################

def cache_meta(config, label_fingerprint, autoencoder_sha, tokenizer_settings):
    """Everything a cached array's meaning depends on. A mismatch is refused, never worked around."""
    from .labels import LABELS_14, LABEL_SOURCE_ID
    from .upstream import PINNED_COMMIT

    return {
        "format_version": CACHE_FORMAT_VERSION,
        "upstream_commit": PINNED_COMMIT,
        "volume": dict(config["volume"]),
        # `encoder_path` is where the weights happen to sit, not what they are: moving the files
        # must not invalidate a cache, so it is deliberately left out of the fingerprint.
        "text": {k: v for k, v in config["text"].items() if k != "encoder_path"},
        "tokenizer": tokenizer_settings,
        "autoencoder_sha256": autoencoder_sha,
        "latent_channels": config["model"]["latent_channels"],
        "label_source_id": LABEL_SOURCE_ID,
        "label_order": list(LABELS_14),
        "label_fingerprint": label_fingerprint,
    }


def write_cache_meta(cache_root, meta):
    os.makedirs(cache_root, exist_ok=True)
    path = os.path.join(cache_root, CACHE_META_NAME)
    with open(path, "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    return path


def check_cache_meta(cache_root, expected):
    """Raise unless the cache was written under exactly these settings.

    The whole point of the fingerprint is that a *silently* different cache is the failure mode that
    costs a training run, so every differing key is named rather than just the first.
    """
    path = os.path.join(cache_root, CACHE_META_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing: this cache was not written by prepare_data.py, or shard 0 never "
            f"finished. Refusing to train on arrays of unknown provenance.")
    with open(path) as handle:
        found = json.load(handle)
    differing = [k for k in expected if found.get(k) != expected[k]]
    if differing:
        detail = "\n".join(f"    {k}:\n      cache:  {found.get(k)!r}\n      config: {expected[k]!r}"
                           for k in differing)
        raise ValueError(f"cache at {cache_root} does not match this configuration:\n{detail}")
    return found


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
