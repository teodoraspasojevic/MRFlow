"""Score MRFlow rollouts: the official challenge metrics, plus the paper's own.

See `evaluation/README.md` for the pipeline and what each metric means.

    challenge_metrics.py  vendored port of the official `mr-volume-generation` container -- the
                          source of truth for what "SSIM" or "FID_2p5D" means here. Do not adjust.
    paper_metrics.py      ours: slice-axis FID on Inception-v3 pool3, FVD over I3D clips, IS.
                          Both volumes reach these in the model's own 1 mm / 256^2 training grid.
    __init__.py           ChallengeAccumulator: fed volume pairs in memory rather than two
                          directories of .nii.gz, and it feeds each metric family the geometry
                          that family is defined on.
    main.py               the CLI -- roll out every series of a split, score it, log to W&B

**The two families see different arrays, deliberately.** The challenge metrics get the pair exactly
as the leaderboard would -- ground truth in its released geometry, generated volume resampled onto
it by the official `zoom` -- so `SSIM_mean` and `FID_2p5D_*` stay reproducible against the
container. The paper metrics get both volumes canonicalized to 1 mm isotropic / 256^2, which is the
distribution the model was trained on. Neither family may be given the other's arrays.
"""

import numpy as np

from evaluation.challenge_metrics import (ALLOWED_MODALITIES, FIDAccumulator, _normalize01,
                                          compute_basic_metrics, finalize_pooled)
from evaluation.paper_metrics import (CLIP_CONFIGS, STRATA, ClipFVDAccumulator,
                                      InceptionScoreAccumulator, SliceFIDAccumulator,
                                      canonicalize_generated, canonicalize_gt, fid_pooled,
                                      fvd_pooled, inception_score, slice_spacing, stratum_for)

# The two metric families, by provenance. Membership, not reporting order -- `METRIC_KEYS` below
# leads with the headline numbers and interleaves the families to do it.
PAPER_KEYS = (
    "FID", "FVD_f16", "FVD_f64", "IS_mean", "IS_std",
    "FID_thin", "FID_thick", "FVD_f16_thin", "FVD_f16_thick", "FVD_f64_thin", "FVD_f64_thick",
    "n_fid_slices", "n_fvd_f16_clips", "n_fvd_f64_clips", "n_thin_cases", "n_thick_cases",
)
CHALLENGE_KEYS = (
    "FID_2p5D_Avg", "FID_2p5D_XY", "FID_2p5D_XZ", "FID_2p5D_YZ", "FID_T",
    "PSNR_mean", "SSIM_mean", "MSE_mean", "dice",
)
COUNT_KEYS = ("n_total_files", "n_scored_files", "n_missing_outputs",
              "n_excluded_out_of_scope_modality")

# The numbers a run is read by, in the order they are read in: the paper's distribution distances,
# then the challenge's, then the per-voxel trio. Everything else -- the strata splits, the
# per-plane FIDs, the sample and file counts -- keeps its family order behind them.
HEADLINE_KEYS = ("FVD_f16", "FVD_f64", "FID", "FID_2p5D_Avg", "IS_mean",
                 "PSNR_mean", "MSE_mean", "SSIM_mean")

# Reporting order, and it drives the console dump, metrics.json and the W&B table at once.
METRIC_KEYS = HEADLINE_KEYS + tuple(
    k for k in PAPER_KEYS + CHALLENGE_KEYS + COUNT_KEYS if k not in HEADLINE_KEYS)


class ChallengeAccumulator:
    """One shard's (real, produced) pairs. `state()` is a picklable snapshot; `combine()` reduces
    one or many of them to the metrics dict plus a per-case breakdown.

    Volumes are scored as they arrive and never retained -- only features accumulate -- because a
    full split would not fit in memory otherwise. Per case that is a few thousand 512-d and 2048-d
    slice vectors and a few dozen 400-d clip vectors.
    """

    def __init__(self, device="auto"):
        self._fid_2p5d = FIDAccumulator(device=device)
        self._fid = SliceFIDAccumulator(device=device)
        self._fvd = ClipFVDAccumulator(device=device)
        self._is = InceptionScoreAccumulator(device=device)
        self._per_case = []
        self.n_total = 0
        self.n_excluded = 0
        self.n_missing = 0
        self.n_stratum = dict.fromkeys(STRATA, 0)

    @staticmethod
    def is_scored(modality):
        return (modality or "").lower() in ALLOWED_MODALITIES

    def add(self, case_id, bucket, modality, real, produced, spacing, plane):
        """A successfully generated pair. Out-of-scope modalities are counted but never scored.

        `spacing` is the ground truth's native (S, R, A) voxel size and `plane` its acquisition
        plane; together they say both how to canonicalize it and which stratum it falls in.
        """
        self.n_total += 1
        if not self.is_scored(modality):
            self.n_excluded += 1
            self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "excluded"})
            return

        # Challenge family: the released pair, untouched.
        metrics = compute_basic_metrics(real, produced)
        self._fid_2p5d.add_pair(real, produced)

        # Paper family: both volumes on the model's own training grid.
        slice_mm = slice_spacing(spacing, plane)
        stratum = stratum_for(slice_mm)
        self.n_stratum[stratum] += 1
        gt = canonicalize_gt(real, spacing, plane)
        gen = canonicalize_generated(produced)
        self._fid.add_pair(gt, gen, stratum)
        self._fvd.add_pair(gt, gen, stratum)
        self._is.add(gen)  # no-reference: the generated volume only

        self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "scored",
                               "slice_mm": float(slice_mm), "stratum": stratum,
                               "gt_slices_1mm": int(gt.shape[0]),
                               "generated_slices": int(gen.shape[0]), **metrics})

    def add_missing(self, case_id, bucket, modality):
        """A case with no volume -- excluded if out of scope, else missing. A missing case is left
        out of the means rather than given a worst-case value, which is what the official
        score.py's aggregation actually does."""
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
                "n_stratum": dict(self.n_stratum),
                "fid_2p5d_raw": self._fid_2p5d.raw_features(),
                "fid_raw": self._fid.raw_features(), "fvd_raw": self._fvd.raw_features(),
                "is_probs": self._is.probs()}


def combine(states):
    """Both metric families over one or more shards' `ChallengeAccumulator.state()`.

    Every distributional metric pools at the feature level, so a shard written under an older
    layout has no `fid_raw`/`fvd_raw` and raises here rather than combining into a partial number.
    """
    per_case = [r for s in states for r in s["per_case"]]
    scored = [r for r in per_case if r["status"] == "scored"]

    def mean(key):
        return float(np.mean([r[key] for r in scored])) if scored else float("nan")

    fid_raw = [s["fid_raw"] for s in states]
    fvd_raw = [s["fvd_raw"] for s in states]
    fid, n_fid, _ = fid_pooled(fid_raw)
    is_mean, is_std = inception_score(np.concatenate([s["is_probs"] for s in states]))

    n_total = sum(s["n_total"] for s in states)
    n_excluded = sum(s["n_excluded"] for s in states)
    metrics = {
        "FID": fid, "n_fid_slices": n_fid,
        "FID_thin": fid_pooled(fid_raw, ("thin",))[0],
        "FID_thick": fid_pooled(fid_raw, ("thick",))[0],
        "IS_mean": is_mean, "IS_std": is_std,
        **finalize_pooled([s["fid_2p5d_raw"] for s in states]),
        "MSE_mean": mean("MSE"), "PSNR_mean": mean("PSNR"), "SSIM_mean": mean("SSIM"),
        # The platform's primary-metric shim: a copy of SSIM_mean, not real Dice.
        "dice": mean("SSIM"),
        "n_total_files": n_total,
        "n_scored_files": n_total - n_excluded,
        "n_missing_outputs": sum(s["n_missing"] for s in states),
        "n_excluded_out_of_scope_modality": n_excluded,
        **{f"n_{k}_cases": sum(s["n_stratum"][k] for s in states) for k in STRATA},
    }
    # FID over the slice axis on the container's own squeezenet backbone. Not a second computation:
    # the official code slices array axis 0 under the label "YZ", so this *is* FID_2p5D_YZ, exposed
    # under the name the paper uses. Unlike the other two planes, an axis-0 slice is (H, W) and so
    # carries no slice-count aspect distortion.
    metrics["FID_T"] = metrics["FID_2p5D_YZ"]

    for name in CLIP_CONFIGS:
        value, n_clips, _ = fvd_pooled(fvd_raw, name)
        metrics[f"FVD_{name}"] = value
        metrics[f"n_fvd_{name}_clips"] = n_clips
        for stratum in STRATA:
            metrics[f"FVD_{name}_{stratum}"] = fvd_pooled(fvd_raw, name, (stratum,))[0]

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
