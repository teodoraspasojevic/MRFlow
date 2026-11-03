"""Value tests for the medical-backbone FIDs and the HLIP report-volume metrics.

Companion to `test_evaluation_metrics.py`, same spirit: pin the things that fail *silently*.

    pytest tests/test_medical_metrics.py -v

The retrieval and preprocessing tests are pure arithmetic and run in seconds. The four that build
a feature extractor download weights on first use -- 185 MB for MedicalNet, 195 MB for
RadImageNet's release zip, ~1.6 GB for HLIP -- and the HLIP one is slow enough on CPU
(a ViT over 1568 tokens) to be worth a GPU node. `device="cpu"` throughout so nothing depends on
one being there.
"""

import os

import numpy as np
import pytest
import torch

from evaluation import signature
from evaluation import hlip_metrics, medical_fid
from evaluation.medical_fid import (FEATURE_DIM, MEDICALNET_SHAPE, PLANES,
                                    MedicalNet3dAccumulator, RadImageNet2p5dAccumulator,
                                    _medicalnet_input, _plane_slices, radimagenet_intensity,
                                    to_canonical_axes)
from evaluation.paper_metrics import Moments, pooled
from evaluation.hlip_metrics import (CROP, EMBED_DIM, NUM_SLICES, PROMPT, HlipAccumulator,
                                     impression_sentences, preprocess_scan, quantize_scan,
                                     retrieval_metrics, sentence_texts)


def phantom(shape, seed=0):
    """Smooth blobs in [0, 1] -- structure rather than white noise, so a feature vector is not the
    same for every volume."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    vol = sum(np.sin(2 * np.pi * f * (zz + yy + xx) + rng.uniform(0, 6.28)) / f for f in (2, 5, 11))
    return ((vol - vol.min()) / (vol.max() - vol.min())).astype(np.float32)


### Shapes ###


def test_medicalnet_sees_a_fixed_grid_and_returns_one_vector_per_volume():
    """The extractor is fully convolutional, so a varying input size would silently change how
    much anatomy one receptive field covers. Every volume reaches it as the same cube."""
    for shape in ((40, 64, 64), (100, 128, 128)):
        assert tuple(_medicalnet_input(phantom(shape)).shape) == (1, 1) + MEDICALNET_SHAPE


def test_medicalnet_normalizes_over_the_foreground_only():
    """MedicalNet's own pipeline z-scores over the nonzero voxels, which is the regime its
    batch-norm statistics were estimated in. Background zeros must not drag the mean, so this is
    checked against the arithmetic spelled out -- on a volume already at the target shape, where
    the resample is a no-op."""
    vol = phantom(MEDICALNET_SHAPE)
    vol[:32] = 0.0
    foreground = vol[vol > 0]
    expected = (vol - foreground.mean()) / foreground.std()
    assert np.allclose(_medicalnet_input(vol).numpy()[0, 0], expected, atol=1e-4)


def test_medicalnet_emits_exactly_one_feature_vector_per_volume():
    """The whole volume is one sample -- no slice selection ever happens before the 3D input."""
    acc = MedicalNet3dAccumulator(device="cpu")
    for i in range(3):
        acc.add_pair(phantom((40, 64, 64), seed=i), phantom((32, 64, 64), seed=i + 9), "AXIAL")
    for side in ("real", "fake"):
        assert acc.moments()[side].n == 3
        assert acc.moments()[side].dim == FEATURE_DIM


### Three-plane slicing ###


def test_planes_are_anatomical_not_array_axes():
    """`canonicalize_*` leaves a volume plane-first, so array axis 0 means a different anatomical
    axis per series. Undoing `plane_order` is what makes "axial" one cut across the whole run."""
    vol = np.zeros((5, 7, 9), np.float32)            # a plane-first volume, three distinct lengths
    assert to_canonical_axes(vol, "AXIAL").shape == (5, 7, 9)     # S, R, A already
    assert to_canonical_axes(vol, "SAGITTAL").shape == (7, 5, 9)  # array was R, S, A
    assert to_canonical_axes(vol, "CORONAL").shape == (7, 9, 5)   # array was A, S, R


def test_every_slice_of_every_plane_is_used():
    """The policy: `sample_every_k = 1`, `center_slices = false`, `drop_empty = false`. A volume
    contributes exactly as many rows per plane as it has slices along that axis."""
    for shape in ((40, 64, 48), (7, 64, 64)):
        for axis, n in enumerate(shape):
            assert _plane_slices(phantom(shape), axis).shape[0] == n


def test_radimagenet_accumulates_three_independent_plane_distributions():
    """Three separate distributions and three separate distances -- the planes' rows are never
    pooled into one FID -- and each gets every slice of its own axis."""
    acc = RadImageNet2p5dAccumulator(device="cpu", batch_size=16)
    shape = (12, 20, 16)   # three distinct axis lengths, so a mixed-up plane shows immediately
    acc.add_pair(phantom(shape), phantom(shape, seed=1), "AXIAL")
    moments = acc.moments()
    assert set(moments) == set(PLANES)
    for plane, axis in (("axial", 0), ("sagittal", 1), ("coronal", 2)):
        assert moments[plane]["real"].n == shape[axis], plane
        assert moments[plane]["fake"].n == shape[axis], plane


### The official RadImageNet transform ###


def test_intensity_maps_0_to_minus_1_and_1_to_plus_1():
    """`pytorch_example.ipynb` does `(image - 127.5) * 2 / 255` on `[0, 255]`; on our `[0, 1]`
    representation that is `2x - 1`. No ImageNet mean subtraction on top of it."""
    x = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(radimagenet_intensity(x), torch.tensor([-1.0, 0.0, 1.0]))


def test_our_radimagenet_path_matches_the_official_notebook():
    """`pytorch_example.ipynb` verbatim: an 8-bit BGR image through `(img - 127.5) * 2 / 255`, a
    resize to 224, and `Backbone = nn.Sequential(*resnet50().children()[:9])` -- which is layer4
    followed by the global average pool, i.e. the 2048-d representation before the classifier.
    Our path starts from the `[0, 1]` window instead, and `2x - 1` is the same affine map."""
    import torchvision
    from evaluation.medical_fid import load_radimagenet

    sd = torch.load(medical_fid._fetch(**medical_fid.RADIMAGENET), map_location="cpu",
                    weights_only=True)
    official = torch.nn.Sequential(*list(torchvision.models.resnet50(weights=None).children())[:9])
    official.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()})
    official.eval()

    img8 = np.random.default_rng(0).integers(0, 256, (64, 48), dtype=np.uint8)
    a = torch.from_numpy((np.repeat(img8[:, :, None], 3, 2).astype(np.float32) - 127.5) * 2 / 255)
    a = torch.nn.functional.interpolate(a.permute(2, 0, 1)[None], size=(224, 224),
                                        mode="bilinear", align_corners=False)

    b = radimagenet_intensity(torch.from_numpy(img8.astype(np.float32) / 255.0)[None, None])
    b = torch.nn.functional.interpolate(b.repeat(1, 3, 1, 1)[:, [2, 1, 0]], size=(224, 224),
                                        mode="bilinear", align_corners=False)

    with torch.no_grad():
        assert torch.allclose(official(a).flatten(1),
                              load_radimagenet("cpu")(b).mean((2, 3)), atol=1e-5)


def test_chunking_gives_the_same_moments_as_one_pass():
    """Batching is a memory device and must change no number: streamed moments have to match what
    `np.mean` / `np.cov` would give over all the rows at once."""
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(97, 16)).astype(np.float32)

    one, chunked = Moments(16), Moments(16)
    one.add(rows)
    for start in range(0, len(rows), 10):
        chunked.add(rows[start:start + 10])

    for a, b in zip(one.finalize(), chunked.finalize()):
        assert np.allclose(a, b, atol=1e-9)
    mu, sigma = one.finalize()
    assert np.allclose(mu, rows.mean(axis=0), atol=1e-6)
    assert np.allclose(sigma, np.cov(rows, rowvar=False), atol=1e-6)


### Real and generated go through one code path ###


def test_medicalnet_gives_the_two_sides_identical_preprocessing():
    """Feed the same volume as both sides: every feature row must match bit for bit. Anything that
    normalizes or resizes the ground truth differently from the generation shows up here long
    before it shows up as a wrong number."""
    acc = MedicalNet3dAccumulator(device="cpu")
    for i in range(3):   # one feature per volume, so three pairs to have a covariance at all
        vol = phantom((40, 64, 64), seed=i)
        acc.add_pair(vol, vol, "SAGITTAL")
    assert abs(pooled(acc.moments())[0]) < 1e-3


def test_radimagenet_gives_the_two_sides_identical_preprocessing():
    """The same guard, and it matters more here: the two sides differ in slice count, so the slice
    selection and the resize are where an asymmetry would hide. Identical volumes must give
    identical moments, and so a zero distance."""
    acc = RadImageNet2p5dAccumulator(device="cpu", batch_size=16)
    vol = phantom((12, 20, 16))
    acc.add_pair(vol, vol, "CORONAL")
    for plane, sides in acc.moments().items():
        assert abs(pooled(sides)[0]) < 1e-3, plane


def test_an_identical_set_scores_zero_and_a_different_one_scores_finite_and_positive():
    """The single most useful regression for a Frechet distance: identical distributions are 0,
    different ones are a finite positive number rather than nan from a rank-deficient sqrtm."""
    same, different = MedicalNet3dAccumulator(device="cpu"), MedicalNet3dAccumulator(device="cpu")
    for i in range(6):
        vol = phantom((40, 64, 64), seed=i)
        same.add_pair(vol, vol, "AXIAL")
        different.add_pair(vol, phantom((40, 64, 64), seed=i + 100), "AXIAL")

    assert abs(pooled(same.moments())[0]) < 1e-3
    value, n_real, n_fake = pooled(different.moments())
    assert n_real == n_fake == 6
    assert np.isfinite(value) and value > 0


def test_a_single_volume_has_no_covariance():
    acc = MedicalNet3dAccumulator(device="cpu")
    acc.add_pair(phantom((40, 64, 64)), phantom((40, 64, 64), seed=1), "AXIAL")
    assert np.isnan(pooled(acc.moments())[0])


### HLIP: preprocessing ###


def test_hlip_resamples_any_series_to_the_checkpoints_input():
    """Whatever the rollout length, a series becomes the checkpoint's own 48 x 224 x 224 scan."""
    for t in (30, 158):
        assert tuple(preprocess_scan(phantom((t, 256, 256))).shape) == (1, NUM_SLICES, CROP, CROP)


def test_hlip_pads_to_square_before_resizing():
    """A non-square volume must not be stretched: the model card pads, then resizes."""
    assert tuple(preprocess_scan(phantom((40, 128, 256))).shape) == (1, NUM_SLICES, CROP, CROP)


def test_the_stored_uint8_scan_is_what_the_loader_reads_from_a_uint8_volume():
    """Shards keep scans as uint8 until the run is pooled. For the 256^2 volumes every metric is
    handed, that is exactly the loader applied to the volume quantized the way upstream's are."""
    volume = phantom((40, 256, 256))
    upstream = np.round(volume * 255) / 255
    stored = hlip_metrics._scan_input(quantize_scan(volume))
    assert stored.shape == (1, NUM_SLICES, CROP, CROP)
    assert torch.allclose(stored, preprocess_scan(upstream), atol=1e-6)


### HLIP: the report, sentence by sentence ###


def test_an_impression_is_split_into_bullets_then_sentences():
    """The forms MR-RATE impressions take: bullets, headers, placeholders, several sentences in one
    bullet, trailing commas."""
    impression = ("Cranial:\n"
                  "— Acute infarcts in the left parietal lobe; follow-up is recommended.\n"
                  "— A millimetric gliotic focus. Additionally, two more foci are present.\n\n"
                  "MR Venography:\n"
                  "— No thrombus was detected,\n"
                  "— [radiologist_1]")
    assert impression_sentences({"impression": impression}) == [
        "Acute infarcts in the left parietal lobe; follow-up is recommended.",
        "A millimetric gliotic focus.",
        "Additionally, two more foci are present.",
        "No thrombus was detected."]


def test_every_text_is_the_training_prompt_and_one_sentence():
    """Token 0 was trained on `"This study shows: "` + one report sentence -- nothing else."""
    texts = sentence_texts({"impression": "— Cerebral atrophy\n— Empty sella"})
    assert texts == ["This study shows: Cerebral atrophy.", "This study shows: Empty sella."]
    assert all(t.startswith(PROMPT) for t in texts)


def test_no_impression_gives_no_text_never_the_findings():
    for impression in ("", None, "— [radiologist_1]", "Cranial:"):
        report = {"findings": "Diffuse atrophy.", "impression": impression}
        assert impression_sentences(report) == [], impression


### HLIP: ranks ###


def test_ranks_on_a_known_similarity_matrix():
    """Rows are reports, columns studies, the diagonal the matched pairs."""
    s = np.array([[3.0, 1.0, 0.0],
                  [2.0, 1.0, 5.0],    # report 1's own study ranks third
                  [0.0, 0.0, 1.0]])
    got = retrieval_metrics(s, "t")
    assert got["t_cosine_mean"] == pytest.approx(5 / 3)
    assert got["t_report_to_study_r1"] == pytest.approx(2 / 3)
    assert got["t_report_to_study_mrr"] == pytest.approx((1 + 1 / 3 + 1) / 3)
    assert got["t_report_to_study_auroc"] == pytest.approx(2 / 3)
    # columns: study 1 ties with report 2's 1.0, study 2 is beaten by report 1's 5.0
    assert got["t_study_to_report_r1"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert got["t_study_to_report_auroc"] == pytest.approx((1 + 0.75 + 0.5) / 3)


def test_a_tie_is_scored_at_its_expected_value():
    """Two studies whose impressions are the same sentences cannot be told apart by any model.
    The tie must count as half a hit at R@1, not a full one -- and not zero."""
    got = retrieval_metrics(np.ones((2, 2)), "t")
    assert got["t_report_to_study_r1"] == pytest.approx(0.5)
    assert got["t_report_to_study_mrr"] == pytest.approx(0.75)
    assert got["t_report_to_study_auroc"] == pytest.approx(0.5)


def test_chance_and_perfect_alignment():
    n = 20
    constant = retrieval_metrics(np.zeros((n, n)), "c")
    assert constant["c_study_to_report_r1"] == pytest.approx(1 / n)
    assert constant["c_study_to_report_r5"] == pytest.approx(5 / n)
    assert constant["c_study_to_report_auroc"] == pytest.approx(0.5)
    perfect = retrieval_metrics(np.eye(n), "p")
    for d in hlip_metrics.DIRECTIONS:
        assert perfect[f"p_{d}_r1"] == perfect[f"p_{d}_mrr"] == perfect[f"p_{d}_auroc"] == 1.0


def test_recall_at_k_is_trivially_one_below_k_studies():
    """R@5 and R@10 say nothing below 5 and 10 studies; `n_hlip_studies` is what tells you."""
    got = retrieval_metrics(np.zeros((3, 3)), "t")
    assert got["t_report_to_study_r5"] == got["t_report_to_study_r10"] == 1.0


### HLIP: studies, against a stub model ###


class _StubTokenizer:
    """Text `"... s<k>."` -> token id k. `tokenizer.encode` is what truncation is counted on."""
    context_length = 6

    class tokenizer:
        @staticmethod
        def encode(text):
            return text.split()

    def __call__(self, texts):
        return torch.tensor([[int(t.rsplit("s", 1)[1].rstrip("."))] for t in texts])


class _StubModel:
    """Sentence `s<k>` embeds as e_k; a scan whose voxels all read k/255 embeds as e_k, and a study
    as the mean of its scans. Records the scan order of every study it is given."""

    def __init__(self):
        self.seen = []

    def encode_text(self, tokens):
        return torch.eye(EMBED_DIM)[tokens[:, 0]]

    def __call__(self, image):
        scans = image[0, :, 0]                                     # [n_scans, D, H, W]
        ids = ((scans.mean(dim=(1, 2, 3)) * 0.226 + 0.449) * 255).round().long()
        self.seen.append(ids.tolist())
        study = torch.eye(EMBED_DIM)[ids].mean(dim=0)
        return {"image_features": torch.stack([study, torch.zeros(EMBED_DIM)])[None]}


@pytest.fixture
def stub(monkeypatch):
    model = _StubModel()
    monkeypatch.setattr(hlip_metrics, "load_hlip", lambda device: (model, _StubTokenizer()))
    return model


def _scan(k):
    return np.full((20, 256, 256), k / 255, np.float32)


def _report(*ids):
    return {"impression": "\n".join(f"— s{k}." for k in ids)}


def test_a_studys_series_are_scored_together_whichever_shard_held_them(stub):
    """Study a's two series land in different shards. Pooling must reunite them into one study,
    and give exactly what one process over every series gives."""
    series = [("c1", "a", _report(1), _scan(1), _scan(1)),
              ("c2", "b", _report(2), _scan(2), _scan(2)),
              ("c3", "a", _report(1), _scan(1), _scan(1)),
              ("c4", "c", _report(3), _scan(1), _scan(3))]   # study c is generated wrong
    shards = [HlipAccumulator(), HlipAccumulator()]
    whole = HlipAccumulator()
    for i, s in enumerate(series):
        shards[i % 2].add(*s)
        whole.add(*s)
    pooled_ = hlip_metrics.summarize(hlip_metrics.merge_states([a.state() for a in shards]))
    assert pooled_ == hlip_metrics.summarize(whole.state())

    assert pooled_["n_hlip_studies"] == 3 and pooled_["n_hlip_scans"] == 4
    assert pooled_["hlip_real_report_to_study_r1"] == 1.0
    # generated c looks like a, so report a ties between a and c (half a hit), and report c scores
    # 0 against every generated study (a three-way tie, a third of a hit)
    assert pooled_["hlip_report_to_study_r1"] == pytest.approx((0.5 + 1 + 1 / 3) / 3)
    assert pooled_["hlip_cosine_mean"] == pytest.approx(2 / 3)


def test_a_report_scores_the_mean_cosine_of_its_sentences(stub):
    acc = HlipAccumulator()
    acc.add("c1", "a", _report(1, 2), _scan(1), _scan(1))     # matches one of its two sentences
    acc.add("c2", "b", _report(3), _scan(3), _scan(3))
    metrics = hlip_metrics.summarize(acc.state())
    assert metrics["hlip_real_cosine_mean"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["n_hlip_sentences"] == 3


def test_a_study_without_an_impression_is_excluded_from_both_sides(stub):
    acc = HlipAccumulator()
    acc.add("c1", "a", _report(1), _scan(1), _scan(1))
    acc.add("c2", "b", _report(2), _scan(2), _scan(2))
    acc.add("c3", "z", {"findings": "s9.", "impression": ""}, _scan(9), _scan(9))
    metrics = hlip_metrics.summarize(acc.state())
    assert metrics["n_hlip_studies"] == 2 and metrics["n_hlip_scans"] == 2
    assert metrics["n_hlip_excluded_no_impression"] == 1


def test_identical_reports_are_counted_and_tied_not_merged(stub):
    """Identity is the study id, never the text: two studies with one report are two answers, and
    the tie between them is scored rather than forgiven."""
    acc = HlipAccumulator()
    acc.add("c1", "a", _report(1), _scan(1), _scan(1))
    acc.add("c2", "b", _report(1), _scan(1), _scan(1))
    acc.add("c3", "c", _report(2), _scan(2), _scan(2))
    metrics = hlip_metrics.summarize(acc.state())
    assert metrics["n_hlip_studies"] == 3 and metrics["n_hlip_duplicate_reports"] == 2
    assert metrics["hlip_real_study_to_report_r1"] == pytest.approx((0.5 + 0.5 + 1) / 3)


def test_scans_reach_the_model_sorted_by_case_id(stub):
    acc = HlipAccumulator()
    for case, k in (("c3", 3), ("c1", 1), ("c2", 2)):
        acc.add(case, "a", _report(1), _scan(k), _scan(k))
    acc.add("c9", "b", _report(4), _scan(4), _scan(4))
    hlip_metrics.summarize(acc.state())
    assert stub.seen[0] == [1, 2, 3]


def test_sentences_past_the_context_are_counted(stub):
    acc = HlipAccumulator()
    acc.add("c1", "a", {"impression": "— one two three four five six seven s1."},
            _scan(1), _scan(1))
    acc.add("c2", "b", _report(2), _scan(2), _scan(2))
    assert hlip_metrics.summarize(acc.state())["n_hlip_truncated"] == 1


def test_hlip_reads_as_nan_when_the_tower_was_switched_off():
    metrics = hlip_metrics.summarize(None)
    assert set(metrics) == set(hlip_metrics.metric_keys())
    assert np.isnan(metrics["hlip_cosine_mean"]) and metrics["n_hlip_studies"] == 0


### HLIP: the real checkpoint ###


def test_hlip_loads_the_pinned_checkpoint_and_scores_studies_end_to_end():
    """The pinned revision builds, **every** weight comes from the checkpoint (`load_hlip` raises
    otherwise), and a two-scan study and a one-scan study come out as cosines."""
    acc = HlipAccumulator()
    acc.add("c1", "a", {"impression": "— Normal study."}, phantom((40, 64, 64), seed=1),
            phantom((40, 64, 64), seed=2))
    acc.add("c2", "a", {"impression": "— Normal study."}, phantom((40, 64, 64), seed=3),
            phantom((40, 64, 64), seed=4))
    acc.add("c3", "b", {"impression": "— Cerebral atrophy."}, phantom((40, 64, 64), seed=5),
            phantom((40, 64, 64), seed=6))
    metrics = hlip_metrics.summarize(acc.state(), device="cpu")
    assert metrics["n_hlip_studies"] == 2 and metrics["n_hlip_scans"] == 3
    for prefix in ("hlip", "hlip_real"):
        assert -1.0 <= metrics[f"{prefix}_cosine_mean"] <= 1.0, prefix


def test_a_study_embedding_does_not_depend_on_the_order_of_its_scans():
    """No per-scan position embedding (`max_num_scans=0`): a study is a set of scans."""
    model, _ = hlip_metrics.load_hlip("cpu")
    scans = {"x": quantize_scan(phantom((40, 64, 64), seed=1)),
             "y": quantize_scan(phantom((60, 64, 64), seed=2))}
    first = hlip_metrics._encode_study(model, "cpu", {"a": {"s": scans["x"]}, "b": {"s": scans["y"]}}, "s")
    second = hlip_metrics._encode_study(model, "cpu", {"a": {"s": scans["y"]}, "b": {"s": scans["x"]}}, "s")
    assert np.allclose(first, second, atol=1e-4)


### Parity with the official implementation ###


def _official_loader(study_path, num_slices):
    """The model card's `loader`, copied verbatim from the checkpoint's README so our preprocessing
    is checked against the upstream one rather than against itself."""
    import os
    from torchvision.transforms import Normalize
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    imgs = []
    for scan in [os.path.join(study_path, q) for q in sorted(os.listdir(study_path))]:
        img = torch.load(scan, weights_only=True)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        img = img.float() / 255.0
        _, h, w = img.shape
        size = max(h, w)
        pad_h, pad_w = size - h, size - w
        left, top = pad_w // 2, pad_h // 2
        img = torch.nn.functional.pad(img, (left, pad_w - left, top, pad_h - top),
                                      mode="constant", value=0)
        img = torch.nn.functional.interpolate(img[None, ...], size=(256, 256), mode="bilinear")[0]
        img = torch.nn.functional.interpolate(img[None, None, ...], size=(num_slices, 256, 256),
                                              mode="nearest-exact")[0, 0]
        img = img[:, 16:240, 16:240]
        normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(),
                               torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
        imgs.append(normalizer(img[None, ...]))
    return torch.stack(imgs, dim=0)[None, ...]


def _official_example_study():
    from huggingface_hub import snapshot_download
    repo = snapshot_download(repo_id=hlip_metrics.HLIP_REPO, revision=hlip_metrics.HLIP_REVISION)
    return os.path.join(repo, "docs", "BraTS-GLI-00459-000")


def test_our_scan_preprocessing_matches_the_official_loader_bit_for_bit():
    """On the study the checkpoint ships as its own example. Ours takes a `[0, 1]` volume where the
    loader takes `[0, 255]` slice tensors, so the division is done first and everything after it
    must agree exactly."""
    study = _official_example_study()
    official = _official_loader(study, hlip_metrics.NUM_SLICES)
    for i, name in enumerate(sorted(os.listdir(study))):
        volume = (torch.load(os.path.join(study, name), weights_only=True).float() / 255.0).numpy()
        assert torch.allclose(preprocess_scan(volume), official[0, i], atol=1e-6), name


def test_our_study_embedding_matches_the_official_build_on_the_official_example():
    """The whole four-scan example study and a sentence prompt, against the model card's own code
    path. This is what says the metric is the published model rather than something that merely
    loads its weights."""
    study = _official_example_study()
    model, tokenizer = hlip_metrics.load_hlip("cpu")
    text = "This study shows: Glioma."
    with torch.no_grad():
        official_image = model(image=_official_loader(study, NUM_SLICES))["image_features"][:, 0]
        official_text = model.encode_text(tokenizer([text]))

    scans = {}
    for name in sorted(os.listdir(study)):
        volume = (torch.load(os.path.join(study, name), weights_only=True).float() / 255.0).numpy()
        scans[name] = {"s": quantize_scan(volume)}
    ours_image = hlip_metrics._encode_study(model, "cpu", scans, "s")
    ours_text, _ = hlip_metrics._encode_reports(model, tokenizer, "cpu", [[text]])

    # the stored scan is uint8 after a 240 -> 256 resize here, so agreement is close, not exact
    official_image = torch.nn.functional.normalize(official_image, dim=-1)[0].numpy()
    official_text = torch.nn.functional.normalize(official_text, dim=-1)[0].numpy()
    assert float(ours_image @ official_image) > 0.999
    assert np.allclose(ours_text[0], official_text, atol=1e-5)


def test_one_scan_is_a_whole_study_and_a_blank_scan_would_corrupt_it():
    """The checkpoint has no scan mask, so every slot is read as a real scan: a study of one series
    needs no padding, and a blank second scan must never be added -- it moves the embedding."""
    model, _ = hlip_metrics.load_hlip("cpu")
    scan = preprocess_scan(phantom((40, 64, 64)))
    blank = torch.zeros_like(scan)

    with torch.no_grad():
        one = model(image=scan[None, None])["image_features"][:, 0, :]
        padded = model(image=torch.stack([scan, blank])[None])["image_features"][:, 0, :]

    assert one.shape == (1, EMBED_DIM)
    assert not torch.allclose(one, padded, atol=1e-3)


def test_changing_the_sentence_changes_the_text_embedding():
    model, tokenizer = hlip_metrics.load_hlip("cpu")
    with torch.no_grad():
        a = model.encode_text(tokenizer(["This study shows: No acute infarct."]))
        b = model.encode_text(tokenizer(["This study shows: Large left MCA infarct."]))
    assert not torch.allclose(a, b, atol=1e-3)


def permutation_test(matched, image, text, n_perm=1000, seed=0):
    """`(p, lo, hi)`: the empirical one-sided p-value that matched pairing beats a random one, and
    a 95% bootstrap CI on the matched-minus-shuffled gap.

    `p` is `(1 + #{shuffled >= matched}) / (1 + n_perm)` -- the add-one form, so a p of 0 is
    impossible and the smallest reportable value is bounded by `n_perm`.
    """
    rng = np.random.default_rng(seed)
    observed = matched.mean()
    null = np.array([(image * text[rng.permutation(len(matched))]).sum(1).mean()
                     for _ in range(n_perm)])
    p = (1 + (null >= observed).sum()) / (1 + n_perm)

    gaps = observed - null
    boot = np.array([rng.choice(gaps, len(gaps), replace=True).mean() for _ in range(1000)])
    return float(p), *(float(v) for v in np.percentile(boot, [2.5, 97.5]))


def test_matched_beats_shuffled_on_real_mr_rate_studies():
    """The check that says the metric measures anything at all: on real MR-RATE studies a report
    must sit closer to its own study than to somebody else's, by more than chance.

    The first 40 val studies with an impression, every scored series of each, through the same
    accumulator the evaluation uses. 1,000 fixed-seed permutations, so the claim is a p-value
    rather than one lucky draw. Skipped where MR-RATE is not mounted.
    """
    raw_root = os.environ.get("MRFLOW_RAW_ROOT", "/hnvme/workspace/y100dc19-MR-Rate-raw")
    if not os.path.isdir(raw_root):
        pytest.skip("MR-RATE archives not available here")

    from echosyn.common.mrrate import (list_series, load_native_volume, read_member, read_report)
    from evaluation.challenge_metrics import ALLOWED_MODALITIES
    from evaluation.paper_metrics import canonicalize_gt

    by_study = {}
    for e in list_series(raw_root, "val", 1, None, 42):
        if e["modality"].lower() in ALLOWED_MODALITIES:
            by_study.setdefault(e["study_uid"], []).append(e)
    acc, n = HlipAccumulator(), 0
    for uid, entries in by_study.items():
        report = read_report(entries[0]["archive"], uid)
        if not impression_sentences(report):
            continue
        for e in entries:
            real, spacing = load_native_volume(read_member(e["archive"], e["member"]), e["plane"])
            gt = canonicalize_gt(real, spacing, e["plane"])
            acc.add(e["series_id"], uid, report, gt, gt)
        n += 1
        if n == 40:
            break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = hlip_metrics.load_hlip(device)
    studies = acc.state()["studies"]
    uids = sorted(studies)
    text, _ = hlip_metrics._encode_reports(model, tokenizer, device,
                                           [studies[u]["sentences"] for u in uids])
    image = np.stack([hlip_metrics._encode_study(model, device, studies[u]["scans"], "real")
                      for u in uids])
    p, lo, hi = permutation_test((image * text).sum(1), image, text)
    print({"studies": len(uids), "p": p, "gap_ci": [round(lo, 4), round(hi, 4)]})
    assert p < 0.05, f"matched pairing is not better than chance: p={p}"
    assert lo > 0, f"95% CI on the gap includes zero: [{lo}, {hi}]"


### Provenance ###

# There is no feature cache to invalidate -- `--combine` recomputes every feature from the cached
# volumes in one process, so a changed backbone or preprocessing simply produces new numbers and
# can never be mixed into old ones. What a run still owes its reader is a record of *which*
# extractors produced it, which `main.py` writes into metrics.json as `extractors`.


def test_the_run_records_the_pinned_checkpoints_and_preprocessing():
    recorded = signature()
    assert recorded["medical"]["medicalnet_sha256"] == medical_fid.MEDICALNET["sha256"]
    assert recorded["medical"]["radimagenet_sha256"] == medical_fid.RADIMAGENET["sha256"]
    assert recorded["medical"]["slice_policy"] == "all_slices"
    assert recorded["medical"]["radimagenet_intensity"] == "2x-1_bgr_v2"  # not the old [0, 1]
    assert recorded["hlip"]["text_template"] == "This study shows: <sentence>"
    assert recorded["hlip"]["pooling"].startswith("image_features[:, 0]")
    assert "not MR-RATE" in recorded["hlip"]["trained_on"]
    assert recorded["hlip"]["revision"] == hlip_metrics.HLIP_REVISION


@pytest.mark.parametrize("module,attribute,value", [
    (medical_fid, "_RADIMAGENET_SIZE", 256),           # preprocessing changed
    (medical_fid, "MEDICALNET_SHAPE", (64, 64, 64)),   # target shape changed
    (hlip_metrics, "HLIP_REVISION", "0" * 40),         # checkpoint changed
])
def test_changing_an_extractor_changes_what_the_run_records(monkeypatch, module, attribute, value):
    """So two metrics.json files computed under different settings are told apart by their own
    contents rather than by whoever remembers which was which."""
    before = signature()
    monkeypatch.setattr(module, attribute, value)
    assert signature() != before
