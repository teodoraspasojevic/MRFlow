"""Score MRFlow rollouts with two metric families that must never contaminate each other.

See `evaluation/README.md` for the pipeline and what each metric means.

    paper_metrics.py      FID, FVD and Inception Score, each computed the way its own reference
                          implementation computes it, with every deviation named in its docstring.
                          Both volumes reach these in the model's own 1 mm / 256^2 training grid.
    fid_inception.py      pytorch-fid v0.3.0, vendored verbatim. Do not edit.
    challenge_metrics.py  the VLM3D scoring container, vendored. MSE/PSNR/SSIM and the squeezenet
                          2.5D FID, on the pair exactly as the leaderboard saw it. Do not adjust.
    __init__.py           EvalAccumulator: fed volume pairs in memory rather than two directories
                          of .nii.gz, and it hands each family the geometry that family is defined
                          on.
    main.py               the CLI -- roll out every series of a split, score it, log to W&B

**The two families see different arrays, and the separation is the point.** The challenge is over,
so its numbers are no longer anyone's reference -- but they are still worth having, and computing
them costs one more pass over volumes that are already in memory. What may never happen is the
reverse of the old arrangement: the challenge container's habits (a squeezenet backbone, `stride=4`
slice sampling, a per-slice intensity window, a `zoom` onto the reference's shape) must not reach
`paper_metrics.py`, whose whole value is that FID/FVD/IS there are the field's standard protocols
unmodified. `PAPER_KEYS` and `CHALLENGE_KEYS` below record which family a key belongs to, which is
what says the arrays it was computed on.
"""

import numpy as np

from evaluation.challenge_metrics import (ALLOWED_MODALITIES, FIDAccumulator,
                                          compute_basic_metrics, finalize_pooled)
from evaluation.paper_metrics import (CLIP_CONFIGS, STRATA, ClipFVDAccumulator,
                                      InceptionScoreAccumulator, SliceFIDAccumulator,
                                      canonicalize_generated, canonicalize_gt, fid_pooled,
                                      fvd_pooled, inception_score, slice_spacing, stratum_for)

PAPER_KEYS = (
    "FVD_f16", "FVD_f64", "FID", "IS_mean", "IS_std",
    "FID_thin", "FID_thick", "FVD_f16_thin", "FVD_f16_thick", "FVD_f64_thin", "FVD_f64_thick",
    "n_fid_slices_real", "n_fid_slices_fake",
    "n_fvd_f16_clips_real", "n_fvd_f16_clips_fake",
    "n_fvd_f64_clips_real", "n_fvd_f64_clips_fake",
    "n_is_slices", "n_thin_cases", "n_thick_cases",
)
CHALLENGE_KEYS = (
    "FID_2p5D_Avg", "FID_2p5D_XY", "FID_2p5D_XZ", "FID_2p5D_YZ",
    "PSNR_mean", "SSIM_mean", "MSE_mean",
)
COUNT_KEYS = ("n_total_files", "n_scored_files", "n_missing_outputs",
              "n_excluded_out_of_scope_modality")

HEADLINE_KEYS = ("FVD_f16", "FID", "IS_mean", "FVD_f64",
                 "FID_2p5D_Avg", "PSNR_mean", "SSIM_mean", "MSE_mean")

METRIC_KEYS = HEADLINE_KEYS + tuple(
    k for k in PAPER_KEYS + CHALLENGE_KEYS + COUNT_KEYS if k not in HEADLINE_KEYS)


class EvalAccumulator:
    """
    One shard's (real, produced) pairs. `state()` is a picklable snapshot; `combine()` reduces
    one or many of them to the metrics dict plus a per-case breakdown.
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

        # Challenge family: the released pair, untouched. `compute_basic_metrics` normalizes both
        # volumes per its own rules and `zoom`s the generated one onto the reference's shape -- so
        # these two lines must be given `real` and `produced` as they arrived, never the
        # canonicalized arrays below.
        metrics = compute_basic_metrics(real, produced)
        self._fid_2p5d.add_pair(real, produced)

        # Paper family: both volumes onto the model's own training grid, and nothing from above
        # reaches them.
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
        """A case with no volume -- excluded if out of scope, else missing.

        It contributes no samples to the distribution metrics, since there is nothing to penalize
        a distance between distributions with, and it is left out of the MSE/PSNR/SSIM means
        rather than given a worst-case value -- which is what the official score.py's aggregation
        actually does. Watch `n_missing_outputs`.
        """
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
                "is_logits": self._is.logits()}


def combine(states):
    """Every metric over one or more shards' `EvalAccumulator.state()`.

    Every distributional metric pools at the feature level -- rows are concatenated and the
    statistic is computed once, which is what a single process that had seen every pair would do.
    The Inception Score shuffles its rows on a fixed seed before splitting, which makes each split
    a random sample of the whole set rather than a contiguous block of two or three shards -- it
    does not make the value order-invariant, and nothing does, so `IS_std` is still comparable only
    across runs with the same shard count. MSE/PSNR/SSIM are per-case and are averaged.

    A shard written under an older layout has no `fid_raw`/`fvd_raw` and raises here rather than
    combining into a partial number.
    """
    per_case = [r for s in states for r in s["per_case"]]
    scored = [r for r in per_case if r["status"] == "scored"]

    def mean(key):
        return float(np.mean([r[key] for r in scored])) if scored else float("nan")

    fid_raw = [s["fid_raw"] for s in states]
    fvd_raw = [s["fvd_raw"] for s in states]
    fid, n_fid_real, n_fid_fake = fid_pooled(fid_raw)
    is_logits = np.concatenate([s["is_logits"] for s in states])
    is_mean, is_std = inception_score(is_logits)

    n_total = sum(s["n_total"] for s in states)
    n_excluded = sum(s["n_excluded"] for s in states)
    metrics = {
        "FID": fid, "n_fid_slices_real": n_fid_real, "n_fid_slices_fake": n_fid_fake,
        "FID_thin": fid_pooled(fid_raw, ("thin",))[0],
        "FID_thick": fid_pooled(fid_raw, ("thick",))[0],
        "IS_mean": is_mean, "IS_std": is_std, "n_is_slices": int(is_logits.shape[0]),
        **finalize_pooled([s["fid_2p5d_raw"] for s in states]),
        "MSE_mean": mean("MSE"), "PSNR_mean": mean("PSNR"), "SSIM_mean": mean("SSIM"),
        "n_total_files": n_total,
        "n_scored_files": n_total - n_excluded,
        "n_missing_outputs": sum(s["n_missing"] for s in states),
        "n_excluded_out_of_scope_modality": n_excluded,
        **{f"n_{k}_cases": sum(s["n_stratum"][k] for s in states) for k in STRATA},
    }

    for name in CLIP_CONFIGS:
        value, n_real, n_fake = fvd_pooled(fvd_raw, name)
        metrics[f"FVD_{name}"] = value
        metrics[f"n_fvd_{name}_clips_real"] = n_real
        metrics[f"n_fvd_{name}_clips_fake"] = n_fake
        for stratum in STRATA:
            metrics[f"FVD_{name}_{stratum}"] = fvd_pooled(fvd_raw, name, (stratum,))[0]

    return {"metrics": {k: metrics[k] for k in METRIC_KEYS}, "per_case": per_case}


def comparison_frames(real, produced, spacing, plane):
    """One scored pair as a uint8 `(T, H, 2W, 3)` video: ground truth on the left, generation on
    the right, both exactly as the metrics saw them -- canonicalized to 1 mm / 256^2 and
    percentile-normalized, by the same two calls `add` makes. A metric says a volume is worse;
    this says how.

    The two sides can differ in length, since nothing forces a rollout to the reference's slice
    count, so the shorter one is padded with black rather than resampled: a generation that stopped
    early should look like it stopped early.

    Slices are the model's own axis order -- the acquisition plane's slice axis leads -- not a
    radiological display convention, so a coronal series plays through coronal slices.
    """
    gt, gen = canonicalize_gt(real, spacing, plane), canonicalize_generated(produced)
    t = max(gt.shape[0], gen.shape[0])
    sides = [np.pad(v, ((0, t - v.shape[0]), (0, 0), (0, 0))) for v in (gt, gen)]
    pair = np.concatenate(sides, axis=2)
    return np.repeat((pair * 255).astype(np.uint8)[..., None], 3, axis=3)
