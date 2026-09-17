"""Score MRFlow rollouts with metric families that must never contaminate each other.

See `evaluation/README.md` for the pipeline and what each metric means.

    paper_metrics.py      `fid_2d_inception`, FVD and Inception Score, each computed the way its
                          own reference implementation computes it, with every deviation named in
                          its docstring. Both volumes reach these in the model's own 1 mm / 256^2
                          training grid.
    medical_fid.py        `fid_3d_medicalnet` (volume-level, MedicalNet 3D ResNet-50) and
                          `fid_2p5d_radimagenet_*` (three anatomical planes, RadImageNet
                          ResNet-50) -- the CCELLA / Alignment-to-Synthesis protocol, on the same
                          canonicalized pair.
    hlip_metrics.py       `hlip_*`: report-to-volume agreement and retrieval under the released
                          MR-RATE-trained HLIP checkpoint. The only metrics that read the report.
    fid_inception.py      pytorch-fid v0.3.0, vendored verbatim. Do not edit.
    medicalnet_resnet.py  MedicalNet's 3D ResNet encoder, vendored. Do not edit.
    challenge_metrics.py  the VLM3D scoring container, vendored. MSE/PSNR/SSIM and the squeezenet
                          2.5D FID, on the pair exactly as the leaderboard saw it. Do not adjust.
    __init__.py           EvalAccumulator: fed volume pairs in memory rather than two directories
                          of .nii.gz, it hands each family the geometry that family is defined on,
                          and `results()` turns what it has seen into the metrics dict.
    main.py               the CLI -- generate and cache every series of a split, then score the
                          cached volumes and log to W&B

**The cached rollouts are the only thing this evaluation ever writes to disk. Features are never
persisted.** One `EvalAccumulator` sees the whole population in one process, so every distance is
computed once over every row: no cross-process pooling, no shard count that can move a number, and
no stale feature cache that could be mixed across a backbone change. The price is that scoring is
sequential where generation was parallel.

**The container family sees different arrays from everything else, and the separation is the
point.** The challenge is over, so its numbers are no longer anyone's reference -- but they are
still worth having, and computing them costs one more pass over volumes that are already in
memory. What may never happen is the reverse of the old arrangement: the challenge container's
habits (a squeezenet backbone, `stride=4` slice sampling, a per-slice intensity window, a `zoom`
onto the reference's shape) must not reach `paper_metrics.py` or `medical_fid.py`, whose whole
value is that the distances there are standard protocols on standard backbones. `PAPER_KEYS`,
`MEDICAL_KEYS`, `HLIP_KEYS` and `CHALLENGE_KEYS` below record which family a key belongs to, which
is what says the arrays it was computed on.

**Note the three FIDs are three different metrics, not three estimates of one.** `fid_2d_inception`
is slice-level on a natural-image backbone, `fid_2p5d_radimagenet_*` slice-level on a radiology
backbone in three anatomical planes, `fid_3d_medicalnet` volume-level on a 3D medical backbone.
Their scales are unrelated and none of them converts into another.
"""

import numpy as np

from evaluation import hlip_metrics, medical_fid
from evaluation.challenge_metrics import (ALLOWED_MODALITIES, FIDAccumulator,
                                          compute_basic_metrics, finalize_pooled)
from evaluation.hlip_metrics import HlipAccumulator
from evaluation.medical_fid import PLANES, MedicalNet3dAccumulator, RadImageNet2p5dAccumulator
from evaluation.paper_metrics import (CLIP_CONFIGS, ClipFVDAccumulator, Moments,
                                      Fid2dInceptionAccumulator, InceptionScoreAccumulator,
                                      canonicalize_generated, canonicalize_gt, fvd_pooled,
                                      inception_score, pooled, slice_spacing)

# The reporting order, and the only one: `METRIC_KEYS` drives the W&B table, the console dump and
# metrics.json alike. **Every score first, then every count.** A reader scanning the table sees
# the numbers that mean something before the bookkeeping that qualifies them.

SCORE_KEYS = (
    "FVD_f16", "FVD_f64",
    "fid_2d_inception",
    "fid_3d_medicalnet",
    "fid_2p5d_radimagenet_mean", "fid_2p5d_radimagenet_axial",
    "fid_2p5d_radimagenet_coronal", "fid_2p5d_radimagenet_sagittal",
    "FID_2p5D_Avg", "FID_2p5D_XY", "FID_2p5D_XZ", "FID_2p5D_YZ",
) + hlip_metrics.score_keys() + (
    "IS_mean", "IS_std",
    "PSNR_mean", "MSE_mean", "SSIM_mean",
)

COUNT_KEYS = (
    "n_fvd_f16_clips_real", "n_fvd_f16_clips_fake",
    "n_fvd_f64_clips_real", "n_fvd_f64_clips_fake",
    "n_fid_2d_inception_slices_real", "n_fid_2d_inception_slices_fake",
    "n_fid_3d_medicalnet_volumes_real", "n_fid_3d_medicalnet_volumes_fake",
) + tuple(f"n_fid_2p5d_radimagenet_{p}_{side}"
            for p in PLANES for side in ("real", "fake")) + hlip_metrics.count_keys() + (
    "n_is_slices",
    "n_total_files", "n_scored_files", "n_missing_outputs",
    "n_excluded_out_of_scope_modality",
)

METRIC_KEYS = SCORE_KEYS + COUNT_KEYS

# Which family a key belongs to -- and so which arrays it was computed on. The container's
# MSE/PSNR/SSIM and `FID_2p5D_*` see the released pair; everything else sees the canonicalized one.
CHALLENGE_KEYS = ("FID_2p5D_Avg", "FID_2p5D_XY", "FID_2p5D_XZ", "FID_2p5D_YZ",
                  "PSNR_mean", "MSE_mean", "SSIM_mean")
PAPER_KEYS = tuple(k for k in METRIC_KEYS
                   if k.startswith(("FVD_", "IS_", "n_fvd_", "n_is_", "fid_2d_", "n_fid_2d_")))
MEDICAL_KEYS = tuple(k for k in METRIC_KEYS if "medicalnet" in k or "radimagenet" in k)
HLIP_KEYS = hlip_metrics.metric_keys()


def signature():
    """What every cached feature in a shard was computed with.

    `combine` refuses to pool shards whose signatures disagree with each other or with the current
    code, so a changed backbone, checkpoint or preprocessing invalidates the cache loudly instead
    of being averaged into an older run's numbers. Bump nothing by hand: the constants the
    sub-signatures read are the ones that would have to change anyway.
    """
    return {"medical": medical_fid.signature(), "hlip": hlip_metrics.signature()}


class EvalAccumulator:
    """
    Every (real, produced) pair of a run, and `results()` for the metrics they add up to.

    Pairs arrive one at a time and are released as soon as `add` returns -- only feature rows stay
    in memory, and they never reach disk.

    `use_hlip=False` drops the HLIP tower, which is by far the heaviest extractor here and a
    ~1.6 GB download; its keys then read nan.
    """

    def __init__(self, device="auto", batch_size=32, hlip_batch_size=8, use_hlip=True):
        # Every extractor is built here, so `--combine` pays the model-loading cost once.
        self._fid_2p5d = FIDAccumulator(device=device)
        self._fid_2d = Fid2dInceptionAccumulator(device=device, batch_size=batch_size)
        self._fvd = ClipFVDAccumulator(device=device)
        self._is = InceptionScoreAccumulator(device=device, batch_size=batch_size)
        self._fid_3d = MedicalNet3dAccumulator(device=device)
        self._fid_rad = RadImageNet2p5dAccumulator(device=device, batch_size=batch_size)
        self._hlip = (HlipAccumulator(device=device, batch_size=hlip_batch_size)
                      if use_hlip else None)
        self._per_case = []
        self.n_total = 0
        self.n_excluded = 0
        self.n_missing = 0

    @staticmethod
    def is_scored(modality):
        return (modality or "").lower() in ALLOWED_MODALITIES

    def add(self, case_id, bucket, modality, real, produced, spacing, plane, report,
            study_uid):
        """A successfully generated pair. Out-of-scope modalities are counted but never scored.

        `spacing` is the ground truth's native (S, R, A) voxel size and `plane` its acquisition
        plane; together they say how to canonicalize it. `report` and `study_uid` are for HLIP: the
        report builds its two text variants, and the study id says what a correct retrieval is.
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

        # Everything else: both volumes onto the model's own training grid, and nothing from above
        # reaches them.
        gt = canonicalize_gt(real, spacing, plane)
        gen = canonicalize_generated(produced)
        self._fid_2d.add_pair(gt, gen)
        self._fvd.add_pair(gt, gen)
        self._is.add(gen)  # no-reference: the generated volume only
        self._fid_3d.add_pair(gt, gen, plane)
        self._fid_rad.add_pair(gt, gen, plane)
        if self._hlip is not None:
            # The generation only, against its study's own report.
            self._hlip.add(report, modality, plane, study_uid, gen)

        # `slice_mm` is the ground truth's native slice spacing -- recorded per case because it
        # says whether its 1 mm z detail was acquired or interpolated. It drives no metric.
        self._per_case.append({"case_id": case_id, "bucket": bucket, "status": "scored",
                               "slice_mm": float(slice_spacing(spacing, plane)),
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
        """A picklable snapshot of everything this shard saw: streamed moments, the rows the two
        metrics that need rows keep, and the per-case record.

        This is what an array task writes beside its volumes. It is **transient** -- `--combine`
        pools it and then deletes it, because features are an intermediate and the volumes are the
        thing worth keeping.
        """
        return {
            "per_case": list(self._per_case),
            "n_total": self.n_total, "n_excluded": self.n_excluded, "n_missing": self.n_missing,
            "signature": signature(),
            "fid_2d": _sides_state(self._fid_2d.moments()),
            "fid_3d": _sides_state(self._fid_3d.moments()),
            "fid_rad": {p: _sides_state(m) for p, m in self._fid_rad.moments().items()},
            "fvd": {name: _sides_state(m) for name, m in self._fvd.moments().items()},
            "fid_2p5d": self._fid_2p5d.raw_features(),
            "is_logits": self._is.logits(),
            "hlip": self._hlip.state() if self._hlip is not None else None,
        }

    def results(self):
        """This accumulator's own metrics, without going through a file."""
        return metrics_from(merge_states([self.state()]))


def _sides_state(sides):
    return {side: m.state() for side, m in sides.items()}


def _sides_merged(states):
    return {side: Moments.merged([st[side] for st in states]) for side in ("real", "fake")}


def merge_states(states):
    """Several shards' `state()` into one. Moments are additive, so pooling is exact -- the same
    numbers a single process over every pair would have produced, with no rows to concatenate.

    Raises when two shards disagree about which extractors produced them, rather than averaging a
    changed backbone into an older run.
    """
    if any(st["signature"] != states[0]["signature"] for st in states):
        raise ValueError("shards were computed with different feature extractors or preprocessing; "
                         "re-run the array rather than pooling them")
    hlip = [st["hlip"] for st in states if st["hlip"] is not None]
    return {
        "per_case": [r for st in states for r in st["per_case"]],
        "n_total": sum(st["n_total"] for st in states),
        "n_excluded": sum(st["n_excluded"] for st in states),
        "n_missing": sum(st["n_missing"] for st in states),
        "signature": states[0]["signature"],
        "fid_2d": _sides_merged([st["fid_2d"] for st in states]),
        "fid_3d": _sides_merged([st["fid_3d"] for st in states]),
        "fid_rad": {p: _sides_merged([st["fid_rad"][p] for st in states]) for p in PLANES},
        "fvd": {n: _sides_merged([st["fvd"][n] for st in states]) for n in CLIP_CONFIGS},
        "fid_2p5d": [st["fid_2p5d"] for st in states],
        "is_logits": np.concatenate([st["is_logits"] for st in states]),
        "hlip": hlip_metrics.merge_states(hlip) if hlip else None,
    }


def metrics_from(state):
    """Every metric from a `merge_states` result.

    The Inception Score shuffles its rows on a fixed seed before splitting, which makes its ten
    splits random samples of the whole set rather than contiguous blocks; it does not make the
    value order-invariant. MSE/PSNR/SSIM are per case and averaged. Every distance is computed
    once, from pooled moments.
    """
    scored = [r for r in state["per_case"] if r["status"] == "scored"]

    def mean(key):
        return float(np.mean([r[key] for r in scored])) if scored else float("nan")

    fid_2d, n_fid_real, n_fid_fake = pooled(state["fid_2d"])
    is_logits = state["is_logits"]
    is_mean, is_std = inception_score(is_logits)
    metrics = {
        "fid_2d_inception": fid_2d,
        "n_fid_2d_inception_slices_real": n_fid_real,
        "n_fid_2d_inception_slices_fake": n_fid_fake,
        "IS_mean": is_mean, "IS_std": is_std, "n_is_slices": int(is_logits.shape[0]),
        **finalize_pooled(state["fid_2p5d"]),
        **hlip_metrics.summarize(state["hlip"]),
        "MSE_mean": mean("MSE"), "PSNR_mean": mean("PSNR"), "SSIM_mean": mean("SSIM"),
        "n_total_files": state["n_total"],
        "n_scored_files": state["n_total"] - state["n_excluded"],
        "n_missing_outputs": state["n_missing"],
        "n_excluded_out_of_scope_modality": state["n_excluded"],
    }

    value, n_real, n_fake = pooled(state["fid_3d"])
    metrics["fid_3d_medicalnet"] = value
    metrics["n_fid_3d_medicalnet_volumes_real"] = n_real
    metrics["n_fid_3d_medicalnet_volumes_fake"] = n_fake

    # Three independent distances, then their unweighted arithmetic mean -- the planes' rows are
    # never pooled into one FID. A plane whose distance is undefined drops out of the mean rather
    # than dragging it to nan. Slice counts are per plane and per side, because every slice is used
    # and so a longer volume genuinely weighs more; read them against `n_scored_files`.
    for plane in PLANES:
        value, n_real, n_fake = pooled(state["fid_rad"][plane])
        metrics[f"fid_2p5d_radimagenet_{plane}"] = value
        metrics[f"n_fid_2p5d_radimagenet_{plane}_real"] = n_real
        metrics[f"n_fid_2p5d_radimagenet_{plane}_fake"] = n_fake
    finite = [metrics[f"fid_2p5d_radimagenet_{p}"] for p in PLANES
              if not np.isnan(metrics[f"fid_2p5d_radimagenet_{p}"])]
    metrics["fid_2p5d_radimagenet_mean"] = float(np.mean(finite)) if finite else float("nan")

    for name in CLIP_CONFIGS:
        value, n_real, n_fake = pooled(state["fvd"][name])
        metrics[f"FVD_{name}"] = value
        metrics[f"n_fvd_{name}_clips_real"] = n_real
        metrics[f"n_fvd_{name}_clips_fake"] = n_fake

    return {"metrics": {k: metrics[k] for k in METRIC_KEYS},
            "per_case": list(state["per_case"])}


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
