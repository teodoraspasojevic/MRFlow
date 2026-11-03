"""Value tests for the reference-protocol and container metric families.

Plumbing (shapes, key sets) is cheap to get right and cheap to notice when it breaks. These test
the things that fail *silently*: a metric that no longer returns zero for an identical pair, a
corruption ladder that stops ordering, a distance that depends on the order cases arrived in --
and, since these numbers are meant to be quotable against the literature, the three protocol facts
that make them comparable at all: the TF-ported Inception backbone, every slice being used, and one
clip per volume.

    pytest tests/test_evaluation_metrics.py -v

Volumes are small and volume counts low so this runs on a login node; the Inception and I3D
forwards dominate, ~10-15 min on CPU and well under a minute on a GPU node. `device="auto"` picks
whatever is there. The Inception weights (91 MB) download on first use.
"""

import numpy as np
import pytest

from evaluation import CHALLENGE_KEYS, EvalAccumulator, METRIC_KEYS, PAPER_KEYS
from evaluation.challenge_metrics import compute_basic_metrics
from evaluation.fid_inception import FID_WEIGHTS_URL, FIDInceptionA
from evaluation.paper_metrics import (CLIP_CONFIGS, ClipFVDAccumulator,
                                      Fid2dInceptionAccumulator, InceptionScoreAccumulator,
                                      _quantize, canonicalize_generated, canonicalize_gt,
                                      fvd_pooled, inception_score, normalize01, pooled,
                                      slice_spacing)

SIZE = (80, 96, 96)          # 80 slices -> one f16 clip and one f64 clip per volume
LADDER = (16, 96, 96)        # the shortest volume that still yields an f16 clip
LADDER_N = 16                # volumes per rung: one clip each, so this is the FVD sample count


def phantom(shape=SIZE, sigma=0.0, seed=0):
    """Broadband structure rather than band-limited noise: a blurred-noise phantom makes FVD
    order a noise ladder *backwards* below ~400 samples (measured 3159/2841/840 for sigma
    0.05/0.2/0.8), while a sum of random-frequency sinusoids orders monotonically.
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    vol = np.zeros(shape, np.float32)
    for f in (2, 5, 11, 23):
        p = rng.uniform(0, 2 * np.pi, 3)
        vol += np.sin(2 * np.pi * f * zz + p[0]) * np.sin(2 * np.pi * f * yy + p[1]) \
            * np.sin(2 * np.pi * f * xx + p[2]) / f
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    if sigma:
        vol = vol + rng.normal(0, sigma, shape).astype(np.float32)
    return np.clip(vol, 0, 1).astype(np.float32)


### The reference protocols ###


def test_fid_runs_on_the_tf_ported_inception_not_torchvisions():
    """The single fact that decides whether an FID is comparable to a published one. torchvision's
    ImageNet Inception gives a different number with no conversion factor, so `use_fid_inception`
    must stay on and the weights must stay `pt_inception-2015-12-05`."""
    assert "pt_inception-2015-12-05" in FID_WEIGHTS_URL
    acc = Fid2dInceptionAccumulator()
    # block 2 holds the Mixed_5* stack, whose first entry is patched for the TF pooling.
    assert isinstance(acc.model.blocks[2][0], FIDInceptionA)
    assert acc._features(phantom((4, 96, 96))).shape[1] == 2048


def test_fid_uses_every_slice():
    """No reference implementation subsamples its image set. The container's `stride=4` used to
    leak in here; one feature row per slice is what says it no longer does."""
    for n in (7, 16, 80):
        assert Fid2dInceptionAccumulator()._features(phantom((n, 96, 96))).shape[0] == n


def test_fvd_takes_exactly_one_clip_per_volume():
    """StyleGAN-V loads `num_frames` consecutive frames per video and discards videos shorter than
    that: one video is one sample and clips never overlap. Sliding windows would report a sample
    count the covariance does not actually have."""
    acc = ClipFVDAccumulator()
    acc.add_pair(phantom(SIZE), phantom(SIZE, seed=1))
    for name in CLIP_CONFIGS:
        sides = acc.moments()[name]
        assert sides["real"].n == 1 and sides["fake"].n == 1, name
        assert sides["real"].dim == 400, name


def test_a_volume_shorter_than_a_clip_contributes_nothing_to_that_fvd():
    """`discard_short_videos=True`. A rollout that stopped early must lower the generated sample
    count rather than be stretched to fit -- which is why both counts are reported."""
    acc = ClipFVDAccumulator()
    acc.add_pair(phantom((80, 96, 96)), phantom((32, 96, 96), seed=1))
    assert acc.moments()["f64"]["real"].n == 1
    assert acc.moments()["f64"]["fake"].n == 0


def test_inputs_are_quantized_to_uint8_like_the_references():
    """pytorch-fid reads PNGs and StyleGAN-V reads uint8 frames, so the feature extractors never
    see more precision than 8 bits and neither may we. Round-half-to-even would put `k/255` on the
    wrong level for some k, so the mapping is checked on every level."""
    levels = (np.arange(256, dtype=np.float32) / 255.0).reshape(1, 16, 16)
    assert np.array_equal(_quantize(levels).ravel(), np.arange(256))
    assert _quantize(levels).dtype == np.uint8
    assert _quantize(np.array([[[0.0, 1.0]]], np.float32)).tolist() == [[[0, 255]]]


### Geometry ###


@pytest.mark.parametrize("shape,spacing,plane,expect_t", [
    ((25, 448, 512), (6.00, 0.45, 0.45), "AXIAL", 150),      # 2D thick, 150 mm of extent
    ((176, 256, 272), (1.00, 1.00, 0.94), "SAGITTAL", 176),  # 3D thin, already 1 mm
    ((360, 512, 512), (0.55, 0.50, 0.50), "CORONAL", 180),   # finer than 1 mm -> decimated
    # CORONAL stacks along A, so the slice spacing is spacing[2]=0.50, NOT spacing[0].
    # 360 x 0.50 = 180 mm. Getting this wrong is the (S, R, A) trap this row exists for.
])
def test_canonicalize_gt_preserves_extent_and_fixes_inplane(shape, spacing, plane, expect_t):
    """The slice axis lands on the volume's own extent in mm, never on a fixed count -- stretching
    to a constant T would rescale anatomy and would hide a rollout of the wrong length."""
    out = canonicalize_gt(phantom(shape), spacing, plane)
    assert out.shape == (expect_t, 256, 256)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_generated_is_left_alone():
    gen = phantom((158, 256, 256))
    out = canonicalize_generated(gen)
    assert out.shape == gen.shape


def test_slice_spacing_uses_the_plane_permutation_not_axis_zero():
    """`spacing` stays (S, R, A) while the array is plane-permuted. Indexing it with an array axis
    is the standing trap; a coronal series is what catches it."""
    spacing = (6.0, 0.5, 0.7)  # S, R, A
    assert slice_spacing(spacing, "AXIAL") == 6.0     # stack axis S
    assert slice_spacing(spacing, "SAGITTAL") == 0.5  # stack axis R
    assert slice_spacing(spacing, "CORONAL") == 0.7   # stack axis A


def test_normalization_is_scale_invariant():
    """Absolute intensity must never reach a metric: the same anatomy at twice the gain is the
    same input."""
    vol = phantom((8, 96, 96))
    assert np.allclose(normalize01(vol), normalize01(vol * 2.0), atol=1e-5)


### Identical inputs must score zero ###


def test_fid_and_fvd_are_zero_for_an_identical_pair():
    """The single most useful regression: any change that breaks the feature path, the clip
    windowing or the Frechet arithmetic moves this off zero.

    Several volumes, because one clip per volume means one volume is one FVD sample and a
    covariance needs at least two.
    """
    vols = [canonicalize_generated(phantom((80, 96, 96), seed=i)) for i in range(4)]
    fid_acc, fvd_acc = Fid2dInceptionAccumulator(), ClipFVDAccumulator()
    for vol in vols:
        fid_acc.add_pair(vol, vol)
        fvd_acc.add_pair(vol, vol)

    assert abs(pooled(fid_acc.moments())[0]) < 1e-3
    for name in CLIP_CONFIGS:
        value, n_real, n_fake = fvd_pooled(fvd_acc.moments(), name)
        assert n_real == n_fake == len(vols), name
        assert abs(value) < 1e-3, (name, value)


def test_inception_score_of_a_repeated_slice_is_one():
    """One slice repeated has zero diversity, so KL(p(y|x) || p(y)) is identically 0 and IS is
    exactly exp(0). A value above 1 means the posteriors are not what they should be."""
    acc = InceptionScoreAccumulator()
    acc.add(np.repeat(phantom()[:1], 40, axis=0))
    mean, std = inception_score(acc.logits())
    assert mean == pytest.approx(1.0, abs=1e-4)
    assert std == pytest.approx(0.0, abs=1e-4)


def test_a_single_volume_cannot_have_a_covariance():
    """One volume is one clip is one sample, so the Frechet distance is undefined and must read
    nan rather than zero."""
    vol = canonicalize_generated(phantom(SIZE))
    acc = ClipFVDAccumulator()
    acc.add_pair(vol, vol)
    value, n_real, n_fake = fvd_pooled(acc.moments(), "f16")
    assert (n_real, n_fake) == (1, 1)
    assert np.isnan(value)


### Ordering ###


def test_fid_and_fvd_order_a_corruption_ladder():
    """Both must rise monotonically with noise. FVD below ~400 samples can invert on the wrong
    phantom, which is why `phantom` is broadband -- see its docstring. One clip per volume means
    the FVD sample count here is `LADDER_N`, so this is a genuinely small-sample check."""
    reals = [canonicalize_generated(phantom(LADDER, seed=i)) for i in range(LADDER_N)]
    rng = np.random.default_rng(1)
    fids, fvds = [], []
    for sigma in (0.0, 0.05, 0.20):
        fid_acc, fvd_acc = Fid2dInceptionAccumulator(), ClipFVDAccumulator()
        for r in reals:
            fake = r if sigma == 0 else np.clip(
                r + rng.normal(0, sigma, r.shape), 0, 1).astype(np.float32)
            fid_acc.add_pair(r, fake)
            fvd_acc.add_pair(r, fake)
        fids.append(pooled(fid_acc.moments())[0])
        fvds.append(fvd_pooled(fvd_acc.moments(), "f16")[0])

    assert fids == sorted(fids), fids
    assert fvds == sorted(fvds), fvds


### Order independence ###


def test_a_distance_does_not_depend_on_the_order_cases_arrived_in():
    """One accumulator sees the whole population, so nothing is pooled across processes -- but the
    rows still land in whatever order `--combine` walked the manifests. A mean and a covariance do
    not care, and float32 reordering is the only reason this is not bit-exact."""
    reals = [canonicalize_generated(phantom((80, 96, 96), seed=i)) for i in range(4)]
    rng = np.random.default_rng(2)
    fakes = [np.clip(r + rng.normal(0, 0.1, r.shape), 0, 1).astype(np.float32) for r in reals]

    def distance(order):
        acc = ClipFVDAccumulator()
        for i in order:
            acc.add_pair(reals[i], fakes[i])
        return fvd_pooled(acc.moments(), "f16")[0]

    assert distance([3, 1, 0, 2]) == pytest.approx(distance(range(4)), rel=1e-6)


def test_inception_score_shuffles_before_splitting_and_is_reproducible():
    """torch-fidelity shuffles the rows on `RandomState(2020)` before cutting the ten splits, so
    a split is a random sample of the whole set rather than a contiguous block of two or three
    shards. It does **not** make the value order-invariant -- a fixed permutation of a reordered
    set is still a different assignment -- so what is pinned here is that the shuffle happens and
    that the same input always gives the same number."""
    acc = InceptionScoreAccumulator()
    for seed in range(4):
        acc.add(phantom((20, 96, 96), seed=seed))
    logits = acc.logits()

    assert inception_score(logits) == inception_score(logits)          # reproducible
    assert inception_score(logits) != inception_score(logits, shuffle=False)  # shuffle is on
    # and the mean barely moves under a reordering, even though the std does
    reordered = np.concatenate([logits[20:], logits[:20]])
    assert inception_score(logits)[0] == pytest.approx(inception_score(reordered)[0], rel=0.05)


### Wiring ###


def test_combine_emits_every_key_and_counts_what_it_says():
    acc = EvalAccumulator(use_hlip=False)
    for i, (spacing, plane) in enumerate([((1.0, 1.0, 1.0), "AXIAL"),
                                          ((4.0, 1.0, 1.0), "AXIAL")]):
        real = phantom((20 if i else 80, 96, 96), seed=i)
        acc.add(f"{i}" * 16, "t1w__axial", "t1w", real, phantom((80, 256, 256), seed=9),
                spacing, plane, {"findings": f"case {i}", "impression": "normal"},
                f"study-{i}")
    acc.add_missing("f" * 16, "t1w__axial", "t1w")
    acc.add_missing("e" * 16, "mra__axial", "mra")

    result = acc.results()
    metrics = result["metrics"]
    assert set(metrics) == set(METRIC_KEYS)
    # one clip per scored volume, per side
    assert metrics["n_fvd_f16_clips_real"] == metrics["n_fvd_f16_clips_fake"] == 2
    # every slice of every scored volume, and the two sides differ because the lengths do
    assert metrics["n_fid_2d_inception_slices_fake"] == 2 * 80   # every slice, no subsampling
    assert metrics["n_is_slices"] == 2 * 80
    # all three planes, every slice: a (80, 256, 256) generated volume in (S, R, A) gives 80
    # axial, 256 sagittal and 256 coronal rows, twice over
    assert metrics["n_fid_2p5d_radimagenet_axial_fake"] == 2 * 80
    assert metrics["n_fid_2p5d_radimagenet_sagittal_fake"] == 2 * 256
    assert metrics["n_fid_2p5d_radimagenet_coronal_fake"] == 2 * 256
    # and the three-plane mean is exactly the arithmetic mean of the three distances
    planes = [metrics[f"fid_2p5d_radimagenet_{p}"] for p in ("axial", "coronal", "sagittal")]
    assert metrics["fid_2p5d_radimagenet_mean"] == pytest.approx(float(np.mean(planes)))
    # n_scored_files counts in-scope cases, NOT successfully scored ones
    assert metrics["n_total_files"] == 4
    assert metrics["n_scored_files"] == 3 and metrics["n_missing_outputs"] == 1
    # the ground truth's native slice spacing is still recorded per case; it drives no metric
    assert [r["slice_mm"] for r in result["per_case"] if r["status"] == "scored"] == [1.0, 4.0]
    # both families reported, and neither family's keys leaked into the other's tuple
    assert set(PAPER_KEYS) & set(CHALLENGE_KEYS) == set()
    assert all(np.isfinite(metrics[k]) for k in CHALLENGE_KEYS)


def test_the_challenge_family_scores_the_released_pair_not_the_canonicalized_one():
    """The separation the whole file is arranged around. `compute_basic_metrics` percentile-
    normalizes both volumes itself and `zoom`s the generated one onto the reference's shape, so it
    has to be handed `real` and `produced` exactly as they arrived; feeding it the 1 mm / 256^2
    arrays the paper metrics use would silently change what SSIM means."""
    real, produced = phantom((20, 96, 96)), phantom((80, 256, 256), seed=9)
    acc = EvalAccumulator(use_hlip=False)
    acc.add("a" * 16, "t1w__axial", "t1w", real, produced, (4.0, 1.0, 1.0), "AXIAL",
            {"findings": "a case", "impression": "normal"}, "study-a")

    scored = [r for r in acc.results()["per_case"] if r["status"] == "scored"][0]
    direct = compute_basic_metrics(real, produced)
    for key in ("MSE", "PSNR", "SSIM"):
        assert scored[key] == pytest.approx(direct[key], rel=1e-12), key


def test_official_basic_metrics_still_score_what_they_did():
    """Golden values for the container's own arithmetic. `compute_basic_metrics` is a port of
    `mrgen_evaluation/metrics_basic.py`; these move only if that file was edited, which it must
    not be."""
    rng = np.random.default_rng(0)
    real = rng.normal(0.5, 0.15, size=(48, 48, 48)).astype(np.float32)
    fake = (real + rng.normal(0, 0.05, size=real.shape)).astype(np.float32)
    out = compute_basic_metrics(real, fake)
    assert out["MSE"] == pytest.approx(0.0038177, abs=1e-7)
    assert out["PSNR"] == pytest.approx(24.181985, abs=1e-5)
    assert out["SSIM"] == pytest.approx(0.948508, abs=1e-6)


def test_examples_are_spread_across_buckets_by_shard():
    """Every shard iterates the same bucket order, so "the first n cases I see" filmed the
    alphabetically first bucket in every task of the array. Offsetting by the shard index must
    cover every bucket instead."""
    from evaluation.main import pick_example_buckets

    names = [f"{m}__{p}" for m in ("FLAIR", "SWI", "T1w", "T2w")
             for p in ("AXIAL", "CORONAL", "SAGITTAL")]
    series = [{"modality": n.split("__")[0], "plane": n.split("__")[1]} for n in names]

    covered = set()
    for shard in range(32):
        picked = pick_example_buckets(series, shard, 2)
        assert len(picked) == 2
        covered |= picked
    assert covered == set(names)
    assert pick_example_buckets(series, 0, 0) == set()
    assert pick_example_buckets(series[:1], 5, 2) == {names[0]}  # only what the shard will reach


def test_a_run_with_no_scored_cases_reads_nan_rather_than_zero():
    """Every key still has to be there, and a distance over nothing must not read as a perfect
    score. This is also the cheapest end-to-end check that `results()` wires every extractor."""
    acc = EvalAccumulator(use_hlip=False)
    acc.add_missing("a" * 16, "t1w__axial", "t1w")
    result = acc.results()

    assert set(result["metrics"]) == set(METRIC_KEYS)
    assert result["metrics"]["n_total_files"] == 1
    assert result["metrics"]["n_missing_outputs"] == 1
    for key in ("fid_2d_inception", "FVD_f16", "fid_3d_medicalnet", "fid_2p5d_radimagenet_mean"):
        assert np.isnan(result["metrics"][key]), key
