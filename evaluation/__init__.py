"""Score MRFlow rollouts with the official VLM3D `mr-volume-generation` metrics.

See `evaluation/README.md` for the pipeline and what each metric means.

    challenge.py   vendored port of the official container -- the source of truth for what
                   "SSIM" or "FID" means here. Do not adjust it to taste.
    __init__.py    ChallengeAccumulator: the official score.py's aggregation, fed volume pairs
                   in memory instead of two directories of .nii.gz
    main.py        the CLI -- roll out every series of a split, score it, log to W&B

`METRIC_KEYS` is also the reporting order: FID average first, then PSNR/SSIM/MSE, then the
per-plane FIDs, then the platform's `dice` shim and the file counts.
"""

import numpy as np

# `_normalize01` is the metric's own normalization, borrowed by `comparison_frames` so the
# picture and the number can never disagree about what the pair looks like.
from evaluation.challenge import (ALLOWED_MODALITIES, FIDAccumulator, _normalize01,
                                  compute_basic_metrics, finalize_pooled)

METRIC_KEYS = (
    "FID_2p5D_Avg", "PSNR_mean", "SSIM_mean", "MSE_mean",
    "FID_2p5D_XY", "FID_2p5D_XZ", "FID_2p5D_YZ",
    "dice",
    "n_total_files", "n_scored_files", "n_missing_outputs",
    "n_excluded_out_of_scope_modality",
)


class ChallengeAccumulator:
    """One shard's (real, produced) pairs. `state()` is a picklable snapshot; `combine()` reduces
    one or many of them to the official metrics dict plus a per-case breakdown.

    Volumes are scored as they arrive and never retained -- only the small 512-d slice features
    accumulate -- because the official container works the same way and a full split would not fit
    in memory otherwise.
    """

    def __init__(self, device="auto"):
        self._fid = FIDAccumulator(device=device)
        self._per_case = []
        self.n_total = 0
        self.n_excluded = 0
        self.n_missing = 0

    @staticmethod
    def is_scored(modality):
        return (modality or "").lower() in ALLOWED_MODALITIES

    def add(self, case_id, bucket, modality, real, produced):
        """A successfully generated pair. Out-of-scope modalities are counted but never scored."""
        self.n_total += 1
        if not self.is_scored(modality):
            self.n_excluded += 1
            self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "excluded"})
            return
        metrics = compute_basic_metrics(real, produced)
        self._fid.add_pair(real, produced)
        self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "scored", **metrics})

    def add_missing(self, case_id, bucket, modality):
        """A case with no volume -- excluded if out of scope, else missing. A missing case is left
        out of the MSE/PSNR/SSIM means rather than given a worst-case value, which is what the
        official score.py's aggregation actually does."""
        self.n_total += 1
        if not self.is_scored(modality):
            self.n_excluded += 1
            status = "excluded"
        else:
            self.n_missing += 1
            status = "missing"
        self._per_case.append({"case_id": case_id, "bucket": bucket, "status": status})

    def state(self):
        return {"per_case": list(self._per_case), "n_total": self.n_total,
                "n_excluded": self.n_excluded, "n_missing": self.n_missing,
                "fid_raw": self._fid.raw_features()}


def combine(states):
    """The official aggregation over one or more shards' `ChallengeAccumulator.state()`."""
    per_case = [r for s in states for r in s["per_case"]]
    scored = [r for r in per_case if r["status"] == "scored"]

    def mean(key):
        return float(np.mean([r[key] for r in scored])) if scored else float("nan")

    n_total = sum(s["n_total"] for s in states)
    n_excluded = sum(s["n_excluded"] for s in states)
    metrics = {
        "MSE_mean": mean("MSE"), "PSNR_mean": mean("PSNR"), "SSIM_mean": mean("SSIM"),
        **finalize_pooled([s["fid_raw"] for s in states]),
        # The platform's primary-metric shim: a copy of SSIM_mean, not real Dice.
        "dice": mean("SSIM"),
        "n_total_files": n_total,
        "n_scored_files": n_total - n_excluded,
        "n_missing_outputs": sum(s["n_missing"] for s in states),
        "n_excluded_out_of_scope_modality": n_excluded,
    }
    return {"metrics": {k: metrics[k] for k in METRIC_KEYS}, "per_case": per_case}


def comparison_frames(real, produced):
    """One scored pair as a uint8 `(T, H, 2W, 3)` video: ground truth on the left, generation on
    the right, both exactly as `compute_basic_metrics` saw them -- `produced` resampled onto the
    ground truth's grid, both percentile-normalized. A metric says a volume is worse; this says how.

    Slices are the model's own axis order -- the acquisition plane's slice axis leads -- not a
    radiological display convention, so a coronal series plays through coronal slices.
    """
    if real.shape != produced.shape:
        from scipy.ndimage import zoom

        produced = zoom(produced, [r / p for r, p in zip(real.shape, produced.shape)], order=1)
    pair = np.concatenate([_normalize01(real), _normalize01(produced)], axis=2)
    return np.repeat((pair * 255).astype(np.uint8)[..., None], 3, axis=3)
