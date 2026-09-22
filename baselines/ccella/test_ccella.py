"""Regression tests for the CCELLA MR-RATE adaptation.

    pytest baselines/ccella/test_ccella.py -q

They are grouped by what they defend:

* the label contract -- order, parsing, masking, and that a missing label never becomes a negative;
* the surgery -- six heads of 14, a fusion classifier of 6x14, sigmoid not softmax, and the
  modality/pathology/spacing conditioning order upstream already implements;
* the storage -- zip round-trip, study-level report dedup, worker-safe reads, fingerprint refusal;
* the driver -- checkpoint schedules, save/resume equality, and W&B disabled touching no network.

The model tests build a real 575M-parameter `CCELLAMRRate` on CPU, so they are slow (~40 s) and
share one module-scoped instance.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines.ccella import labels as L
from baselines.ccella.config import (ConfigError, load_config, validate_checkpoint_steps,
                                     upstream_namespace)
from baselines.ccella.data import CcellaDataset
from baselines.ccella.loss import MaskedMultiLabelBCE, pathology_metrics
from baselines.ccella.store import (ShardStore, ZipReader, check_cache_meta, read_manifest,
                                    sample_key, study_key, write_cache_meta, write_manifest)
from baselines.ccella.text import format_report

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE_CONFIG = os.path.join(HERE, "configs", "smoke.yaml")
RUN_CONFIG = os.path.join(HERE, "configs", "mrrate_ccella.yaml")


### The label contract ################################################################################

def test_the_fourteen_labels_are_in_the_source_files_own_column_order():
    """`LABELS_14` is asserted against the CSV header, not trusted from it."""
    config = load_config(RUN_CONFIG)
    root = config["mrrate"]["labels_root"]
    if not os.path.exists(os.path.join(root, L.LABELS_CSV_NAME)):
        pytest.skip("label artifact not present on this machine")
    with open(os.path.join(root, L.LABELS_CSV_NAME)) as handle:
        header = handle.readline().strip().split(",")
    assert tuple(header) == ("study_uid",) + L.LABELS_14
    assert L.NUM_LABELS == 14


def test_a_reordered_header_is_refused_rather_than_silently_accepted(tmp_path):
    swapped = ("study_uid", L.LABELS_14[1], L.LABELS_14[0]) + L.LABELS_14[2:]
    (tmp_path / L.LABELS_CSV_NAME).write_text(",".join(swapped) + "\n")
    with pytest.raises(L.LabelSourceError, match="not the expected 14 columns"):
        L.read_label_table(str(tmp_path))


def test_a_non_binary_cell_is_refused(tmp_path):
    header = ",".join(("study_uid",) + L.LABELS_14)
    (tmp_path / L.LABELS_CSV_NAME).write_text(header + "\nAAA," + ",".join(["2"] * 14) + "\n")
    with pytest.raises(L.LabelSourceError, match="non-binary"):
        L.read_label_table(str(tmp_path))


def test_a_study_with_no_label_row_is_masked_and_never_treated_as_negative():
    table = {"HAVE": (1,) + (0,) * 13}
    vector, mask = L.label_vector(table, "HAVE")
    assert vector == (1,) + (0,) * 13 and mask == (1,) * 14

    vector, mask = L.label_vector(table, "MISSING")
    assert mask == (0,) * 14, "an unlabelled study must be masked out entirely"
    assert vector == (0,) * 14, "the padded vector is zeros, but the mask is what the loss reads"


def test_pos_weight_is_neg_over_pos_and_is_capped():
    table = {f"S{i}": (1 if i < 10 else 0,) + (0,) * 13 for i in range(100)}
    weights, counts, total = L.pos_weight(table, list(table), cap=20.0)
    assert total == 100 and counts[0] == 10
    assert weights[0] == pytest.approx(90 / 10)          # neg / pos
    assert weights[1] == 20.0, "a label with no positives gets the cap, which is inert"

    rare = {f"S{i}": (1 if i == 0 else 0,) + (0,) * 13 for i in range(1000)}
    capped, _, _ = L.pos_weight(rare, list(rare), cap=20.0)
    assert capped[0] == 20.0, "999/1 must be capped, not 999"


def test_pos_weight_ignores_studies_with_no_label_row():
    table = {"A": (1,) + (0,) * 13, "B": (0,) * 14}
    weights, counts, total = L.pos_weight(table, ["A", "B", "NOT_IN_TABLE"], cap=20.0)
    assert total == 2, "an unlabelled study must not count as a negative in the weighting"
    assert counts[0] == 1


### The multi-label objective #########################################################################

def test_the_loss_is_unreduced_so_upstreams_masking_lines_still_work():
    loss = MaskedMultiLabelBCE(pos_weight=[1.0] * 14)
    logits = torch.zeros(3, 14, requires_grad=True)
    target = torch.zeros(3, 14)
    out = loss(logits, target)
    assert out.shape == (3, 14), "train_one_epoch masks a [B, C] tensor and reduces it itself"
    assert torch.allclose(out, torch.full((3, 14), float(np.log(2))), atol=1e-6)


def test_the_upstream_masking_arithmetic_reduces_to_a_per_sample_sum_over_labels():
    """Reproduces `train_one_epoch`'s two lines exactly, with the patched width."""
    loss = MaskedMultiLabelBCE()
    logits = torch.zeros(4, 14)
    target = torch.zeros(4, 14)
    isnull = torch.tensor([0.0, 1.0, 0.0, 1.0])

    class_loss = loss(logits, target)
    isnull_unsqueeze = isnull.unsqueeze(1).repeat(1, class_loss.shape[-1])
    reduced = torch.sum(class_loss * (1 - isnull_unsqueeze)) / torch.sum(1 - isnull)
    assert reduced == pytest.approx(14 * float(np.log(2)), rel=1e-5)


def test_pos_weight_raises_the_loss_only_on_positives():
    plain = MaskedMultiLabelBCE()
    weighted = MaskedMultiLabelBCE(pos_weight=[5.0] * 14)
    logits, positive, negative = torch.zeros(1, 14), torch.ones(1, 14), torch.zeros(1, 14)
    assert weighted(logits, positive).sum() > plain(logits, positive).sum()
    assert torch.allclose(weighted(logits, negative), plain(logits, negative))


def test_the_loss_rejects_a_width_that_is_not_fourteen():
    with pytest.raises(ValueError):
        MaskedMultiLabelBCE()(torch.zeros(2, 5), torch.zeros(2, 5))
    with pytest.raises(ValueError):
        MaskedMultiLabelBCE(pos_weight=[1.0] * 5)


def test_labels_are_independent_not_a_distribution():
    """A softmax objective would couple the 14; BCE must not."""
    loss = MaskedMultiLabelBCE()
    both = loss(torch.zeros(1, 14), torch.ones(1, 14))
    assert both.sum() == pytest.approx(14 * float(np.log(2)), rel=1e-5), (
        "all-positive is a legal target: the 14 groups are not mutually exclusive")


def test_pathology_metrics_leave_a_one_sided_label_undefined_rather_than_scoring_it():
    logits = np.random.RandomState(0).randn(50, 14)
    targets = np.zeros((50, 14))
    targets[:25, 0] = 1                     # only label 0 has both classes
    mask = np.ones((50, 14))
    out = pathology_metrics(logits, targets, mask)
    assert out["n_labels_scored"] == 1
    assert out["per_label"][L.LABELS_14[0]]["auroc"] is not None
    assert out["per_label"][L.LABELS_14[1]]["auroc"] is None
    assert out["auroc_macro"] == pytest.approx(out["per_label"][L.LABELS_14[0]]["auroc"])


def test_pathology_metrics_drop_masked_samples():
    logits = np.zeros((10, 14))
    targets = np.zeros((10, 14))
    mask = np.ones((10, 14))
    mask[5:] = 0
    assert pathology_metrics(logits, targets, mask)["n_samples_scored"] == 5


### The report string #################################################################################

REPORT = {
    "report": "full text",
    "clinical_information": "headache",
    "technique": "axial T1w FLAIR 5 mm",
    "findings": "There is mild gliosis in the left frontal lobe.",
    "impression": "Chronic small vessel disease.",
}


def test_the_report_string_is_findings_then_impression_joined_plainly():
    assert format_report(REPORT) == (
        "There is mild gliosis in the left frontal lobe.\n\nChronic small vessel disease.")


def test_an_empty_section_is_dropped_and_an_empty_report_is_the_empty_string():
    assert format_report({"findings": "A.", "impression": ""}) == "A."
    assert format_report({"findings": "", "impression": ""}) == ""


def test_no_modality_plane_or_spacing_leaks_into_the_report_string():
    text = format_report(REPORT)
    for token in ("T1w", "T2w", "FLAIR", "SWI", "MRA", "AXIAL", "SAGITTAL", "CORONAL",
                  "[MODALITY]", "[PLANE]", "[SPACING]", "mm"):
        assert token not in text, f"{token!r} must not be in the conditioning text"


def test_no_pathology_label_name_leaks_into_the_report_string():
    text = format_report(REPORT)
    for name in L.LABELS_14:
        assert name not in text
        assert name.split("_", 1)[1].replace("_", " ") not in text


### The model surgery #################################################################################

@pytest.fixture(scope="module")
def model():
    config = load_config(SMOKE_CONFIG, ["model.use_flash_attention=false"])
    from baselines.ccella.model import build_model

    return build_model(config, torch.device("cpu"))


def test_six_cascade_heads_each_output_fourteen_logits(model):
    heads = model.ella.connector.cascade_blocks
    assert len(heads) == 6, "upstream's six-block cascade must be preserved"
    assert [h.fc2.out_features for h in heads] == [14] * 6


def test_the_fusion_classifier_consumes_six_times_fourteen(model):
    assert model.classifier.in_features == 6 * 14
    assert model.classifier.out_features == 14


def test_the_adapter_still_emits_upstreams_cross_attention_tokens(model):
    assert model.ella.connector.latents.shape == (256, 768), (
        "the tokens handed to cross-attention are upstream's 256 x 768 and must not move")


def test_the_unet_projects_a_fourteen_wide_pathology_vector(model):
    assert model.unet.pirads_layer[0].in_features == 14


def test_modality_is_a_seven_way_embedding_from_mrflows_own_table(model):
    from echosyn.common.mrrate import MODALITY_TO_ID

    assert model.unet.class_embedding.num_embeddings == len(MODALITY_TO_ID) == 7
    assert MODALITY_TO_ID["CFG_NULL"] == 0, "id 0 is the guidance null and may never move"


def test_pathology_conditioning_is_sigmoid_not_softmax(model):
    """Two labels can both be near 1. Under a softmax they could not."""
    torch.manual_seed(0)
    captured = {}
    original = model.unet.pirads_layer.forward
    model.unet.pirads_layer.forward = lambda v: (captured.setdefault("v", v), original(v))[1]
    try:
        with torch.no_grad():
            model(x=torch.randn(1, 4, 16, 16, 16), timesteps=torch.tensor([10]),
                  spacing_tensor=torch.rand(1, 3) * 1e2,
                  text_encoding=torch.randn(1, 512, 4096), modality_id=torch.tensor([1]))
    finally:
        model.unet.pirads_layer.forward = original
    vector = captured["v"]
    assert vector.shape == (1, 14)
    total = float(vector.sum())
    assert total > 110.0, (
        f"a softmax would sum to the conditioning scale 100; got {total:.1f}, so the 14 are "
        f"independent sigmoids")
    assert 0.0 <= float(vector.min()) and float(vector.max()) <= 100.0


def test_modality_is_added_to_the_timestep_embedding_before_any_concatenation(model):
    """Upstream's `_get_time_and_class_embedding` adds; `_get_input_embeddings` concatenates."""
    x = torch.zeros(1, 4, 16, 16, 16)
    t = torch.tensor([7])
    with torch.no_grad():
        base = model.unet._get_time_and_class_embedding(x, t, torch.tensor([0]))
        other = model.unet._get_time_and_class_embedding(x, t, torch.tensor([3]))
        delta = other - base
        expected = (model.unet.class_embedding(torch.tensor([3]))
                    - model.unet.class_embedding(torch.tensor([0])))
    assert base.shape == (1, 256), "the timestep condition is one 256-wide vector before concat"
    assert torch.allclose(delta, expected, atol=1e-6), (
        "changing modality must shift the timestep embedding by exactly the embedding difference, "
        "i.e. it is added elementwise and not concatenated")


def test_pathology_then_spacing_is_the_concatenation_order(model):
    """Upstream's order: [timestep(+modality) | pathology | spacing]."""
    torch.manual_seed(0)
    emb = torch.randn(1, 256)
    pathology = torch.rand(1, 14) * 100
    spacing = torch.rand(1, 3) * 100
    with torch.no_grad():
        combined = model.unet._get_input_embeddings(emb, pathology, spacing)
        assert combined.shape == (1, 768)
        assert torch.allclose(combined[:, :256], emb)
        assert torch.allclose(combined[:, 256:512], model.unet.pirads_layer(pathology))
        assert torch.allclose(combined[:, 512:], model.unet.spacing_layer(spacing))


def test_a_forward_pass_returns_noise_and_raw_logits(model):
    torch.manual_seed(0)
    with torch.no_grad():
        out, logits = model(x=torch.randn(1, 4, 16, 16, 16), timesteps=torch.tensor([5]),
                            spacing_tensor=torch.rand(1, 3) * 1e2,
                            text_encoding=torch.randn(1, 512, 4096),
                            modality_id=torch.tensor([2]))
    assert out.shape == (1, 4, 16, 16, 16)
    assert logits.shape == (1, 14)
    assert logits.abs().max() > 0, "the classification head must return logits, not probabilities"


def test_the_model_refuses_to_run_without_a_modality_id(model):
    with pytest.raises(ValueError, match="modality_id is required"):
        model(x=torch.zeros(1, 4, 16, 16, 16), timesteps=torch.tensor([1]),
              spacing_tensor=torch.zeros(1, 3), text_encoding=torch.zeros(1, 512, 4096))


def test_the_diffusion_loss_reaches_the_pathology_head_and_the_modality_embedding():
    """The predicted pathology vector is not detached before it conditions the U-Net.

    Every resnet block's `conv2` and the output conv are `zero_module`-initialised upstream, so at
    step 0 the residual branches contribute nothing and *every* conditioning gradient is exactly
    zero -- including the plain timestep's. That is standard identity initialisation, not a dead
    path, so the test perturbs those convs the way the first optimizer step would before probing.
    """
    config = load_config(SMOKE_CONFIG, ["model.use_flash_attention=false"])
    from baselines.ccella.model import build_model

    torch.manual_seed(0)
    net = build_model(config, torch.device("cpu"))
    for module in net.unet.modules():
        if type(module).__name__ == "DiffusionUNetResnetBlock":
            torch.nn.init.normal_(module.conv2.conv.weight, std=0.02)
    torch.nn.init.normal_(net.unet.out[-1].conv.weight, std=0.02)

    out, _ = net(x=torch.randn(1, 4, 16, 16, 16), timesteps=torch.tensor([500]),
                 spacing_tensor=torch.rand(1, 3) * 1e2,
                 text_encoding=torch.randn(1, 512, 4096), modality_id=torch.tensor([3]))
    out.pow(2).mean().backward()

    for name, param in [("ELLA head", net.ella.connector.cascade_blocks[0].fc2.weight),
                        ("fusion classifier", net.classifier.weight),
                        ("pirads projection", net.unet.pirads_layer[0].weight),
                        ("modality embedding", net.unet.class_embedding.weight)]:
        assert param.grad is not None and param.grad.abs().sum() > 0, (
            f"the diffusion loss does not reach {name}")


### Storage ###########################################################################################

def test_zip_round_trip_and_atomic_publication(tmp_path):
    store = ShardStore(str(tmp_path), "val", 3)
    latent = np.random.RandomState(0).randn(4, 8, 8, 8).astype(np.float16)
    member = store.save_latent("abc", latent)
    assert not os.path.exists(os.path.join(tmp_path, store.latent_name)), (
        "an unsealed archive must not be visible")
    store.close()

    reader = ZipReader(str(tmp_path))
    assert np.array_equal(reader.read_array(store.latent_name, member), latent)


def test_an_aborted_shard_leaves_nothing_behind(tmp_path):
    store = ShardStore(str(tmp_path), "val", 0)
    store.save_latent("abc", np.zeros((4, 4, 4, 4), dtype=np.float16))
    store.abort()
    assert not os.path.exists(os.path.join(tmp_path, store.latent_name))
    assert not os.path.exists(os.path.join(tmp_path, store.latent_name + ".tmp"))


def test_a_report_is_written_once_per_study_however_many_series_reference_it(tmp_path):
    store = ShardStore(str(tmp_path), "train", 0)
    embedding = np.zeros((8, 16), dtype=np.float16)
    key = study_key("STUDY-A")
    members = [store.save_report(key, embedding) for _ in range(5)]
    store.save_report(study_key("STUDY-B"), embedding)
    store.close()
    assert len(set(members)) == 1
    import zipfile
    with zipfile.ZipFile(os.path.join(tmp_path, store.report_name)) as handle:
        assert len(handle.namelist()) == 2, "five series of one study must store one embedding"


def test_the_keys_carry_no_identifier_and_are_stable():
    assert study_key("ABC") == study_key("ABC")
    assert "ABC" not in study_key("ABC")
    assert sample_key("ABC", "t1w-raw-axi") != sample_key("ABC", "t1w-raw-sag")
    assert len(sample_key("ABC", "x")) == 16


def test_a_missing_member_raises_rather_than_returning_zeros(tmp_path):
    store = ShardStore(str(tmp_path), "val", 0)
    store.save_latent("present", np.zeros((4, 4, 4, 4), dtype=np.float16))
    store.close()
    reader = ZipReader(str(tmp_path))
    with pytest.raises(KeyError):
        reader.read_array(store.latent_name, "absent.npy")
    with pytest.raises(FileNotFoundError):
        reader.read_array("artifacts/nope.zip", "x.npy")


def _tiny_cache(tmp_path, n_series=6, n_studies=2, channels=4, size=4, hidden=16, length=8):
    store = ShardStore(str(tmp_path), "train", 0)
    rng = np.random.RandomState(0)
    rows = []
    for i in range(n_series):
        skey = study_key(f"STUDY-{i % n_studies}")
        store.save_report(skey, rng.randn(length, hidden).astype(np.float16))
        key = sample_key(f"STUDY-{i % n_studies}", f"series-{i}")
        member = store.save_latent(key, rng.randn(channels, size, size, size).astype(np.float16))
        labelled = i % 3 != 0
        rows.append({
            "sample_id": key, "split": "train", "study_key": skey, "series_key": key,
            "latent_zip": store.latent_name, "latent_member": member,
            "report_zip": store.report_name, "report_member": f"{skey}.npy",
            "modality": "T1w", "modality_id": 1, "plane": "AXIAL",
            "spacing_mm": "1.000000;1.000000;5.000000", "grid": f"{size};{size};{size}",
            "labels": "1" + "0" * 13, "label_mask": ("1" if labelled else "0") * 14,
        })
    store.close()
    write_manifest(str(tmp_path), "train", 0, rows)
    return rows


def test_a_training_batch_has_the_keys_dtypes_and_shapes_upstream_reads(tmp_path):
    rows = _tiny_cache(tmp_path)
    dataset = CcellaDataset(str(tmp_path), rows)
    item = dataset[0]
    assert set(item) == {"image", "text", "spacing", "pirads", "text_isnull", "modality_id"}
    assert item["image"].shape == (4, 4, 4, 4) and item["image"].dtype == torch.float32
    assert item["text"].shape == (8, 16) and item["text"].dtype == torch.float32
    assert item["pirads"].shape == (14,) and item["pirads"].dtype == torch.float32
    assert item["spacing"].dtype == torch.float32
    assert item["modality_id"].dtype == torch.int64
    assert item["text_isnull"].shape == ()


def test_spacing_keeps_upstreams_hundredfold_scale_and_labels_do_not(tmp_path):
    rows = _tiny_cache(tmp_path)
    item = CcellaDataset(str(tmp_path), rows)[0]
    assert torch.allclose(item["spacing"], torch.tensor([100.0, 100.0, 500.0]))
    assert item["pirads"].max() <= 1.0, (
        "a BCE target must lie in [0, 1]; upstream's * 1e2 on pirads cannot carry over")


def test_an_unlabelled_sample_is_flagged_for_the_classification_mask(tmp_path):
    rows = _tiny_cache(tmp_path)
    dataset = CcellaDataset(str(tmp_path), rows)
    flags = [float(dataset[i]["text_isnull"]) for i in range(len(dataset))]
    assert flags == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_the_dataset_is_usable_from_multiple_dataloader_workers(tmp_path):
    rows = _tiny_cache(tmp_path, n_series=12)
    dataset = CcellaDataset(str(tmp_path), rows)
    loader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=2, shuffle=False)
    seen = [batch["image"] for batch in loader]
    assert sum(b.shape[0] for b in seen) == 12
    single = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=0, shuffle=False)
    for a, b in zip(seen, single):
        assert torch.equal(a, b["image"]), "worker reads must match single-process reads"


def test_the_manifest_round_trips(tmp_path):
    rows = _tiny_cache(tmp_path)
    back = read_manifest(str(tmp_path), "train")
    assert len(back) == len(rows)
    assert back[0]["labels"] == rows[0]["labels"]


### Cache fingerprint #################################################################################

def test_a_cache_written_under_different_settings_is_refused(tmp_path):
    write_cache_meta(str(tmp_path), {"volume": {"grid": [64, 64, 64]}, "label_order": ["a"]})
    check_cache_meta(str(tmp_path), {"volume": {"grid": [64, 64, 64]}})
    with pytest.raises(ValueError, match="does not match this configuration"):
        check_cache_meta(str(tmp_path), {"volume": {"grid": [192, 224, 192]}})
    with pytest.raises(ValueError, match="label_order"):
        check_cache_meta(str(tmp_path), {"label_order": ["b"]})


def test_a_cache_with_no_fingerprint_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="cache_meta"):
        check_cache_meta(str(tmp_path), {"anything": 1})


### Configuration and the checkpoint schedule #########################################################

def test_the_sixty_thousand_step_schedule_is_the_specified_one():
    steps = load_config(RUN_CONFIG)["train"]["checkpoint_steps"]
    assert steps == [20000, 40000, 50000, 55000, 60000]


def test_the_hundred_and_twenty_thousand_step_schedule_validates():
    steps = [20000, 40000, 60000, 80000, 100000, 110000, 120000]
    assert validate_checkpoint_steps(steps, 120000) == steps


def test_a_schedule_that_is_not_strictly_increasing_is_refused():
    with pytest.raises(ConfigError, match="strictly increasing"):
        validate_checkpoint_steps([20000, 20000, 60000], 60000)
    with pytest.raises(ConfigError, match="strictly increasing"):
        validate_checkpoint_steps([40000, 20000, 60000], 60000)


def test_a_schedule_past_the_end_or_not_ending_on_it_is_refused():
    with pytest.raises(ConfigError, match="past max_train_steps"):
        validate_checkpoint_steps([20000, 70000], 60000)
    with pytest.raises(ConfigError, match="must include the final step"):
        validate_checkpoint_steps([20000, 40000], 60000)


def test_an_unknown_config_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  nope: 1\n")
    with pytest.raises(ConfigError, match="unknown config key 'train.nope'"):
        load_config(str(path))


def test_a_grid_the_autoencoder_and_unet_cannot_halve_is_refused(tmp_path):
    path = tmp_path / "grid.yaml"
    path.write_text("volume:\n  grid: [192, 200, 192]\n")
    with pytest.raises(ConfigError, match="divisible by 32"):
        load_config(str(path))


def test_the_text_width_must_match_the_adapters_input_dim(tmp_path):
    path = tmp_path / "text.yaml"
    path.write_text("text:\n  hidden_size: 768\n")
    with pytest.raises(ConfigError, match="input_dim"):
        load_config(str(path))


def test_the_upstream_namespace_still_feeds_upstreams_define_instance():
    config = load_config(SMOKE_CONFIG, ["model.use_flash_attention=false"])
    namespace = upstream_namespace(config)
    assert namespace.diffusion_unet_def["_target_"] == "baselines.ccella.model.CCELLAMRRate"
    assert namespace.diffusion_unet_def["num_classes"] == 14
    assert namespace.diffusion_unet_def["num_class_embeds"] == 7
    from baselines.ccella.upstream import upstream_module

    scheduler = upstream_module("scripts.utils").define_instance(namespace, "noise_scheduler")
    assert scheduler.num_train_timesteps == 1000


### The pinned upstream checkout ######################################################################

def test_the_upstream_checkout_is_the_pinned_commit_plus_the_stored_patch():
    from baselines.ccella.upstream import upstream_root, verify_upstream

    if not os.path.isdir(upstream_root()):
        pytest.skip("upstream checkout not present on this machine")
    ok, problems = verify_upstream()
    assert ok, "\n".join(problems)


def test_the_patch_touches_only_train_one_epoch():
    from baselines.ccella.upstream import PATCH_PATH, PATCHED_FILES

    patch = open(PATCH_PATH).read()
    changed = {line.split(" b/")[-1].strip() for line in patch.splitlines()
               if line.startswith("diff --git ")}
    assert changed == set(PATCHED_FILES)
    added = [l for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")]
    assert len(added) <= 25 and len(removed) <= 5, (
        f"the upstream patch has grown to +{len(added)}/-{len(removed)} lines; it is meant to "
        f"stay minimal")


### The driver ########################################################################################

def test_checkpoint_save_and_resume_restore_every_piece_of_state(tmp_path):
    """Save/resume equality on the exact objects a run carries, without a GPU or a cache."""
    from baselines.ccella.train import rng_state, set_rng_state

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10, power=2.0)
    for _ in range(3):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    payload = {
        "unet_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "global_step": 3, "epoch": 1,
        "rng": rng_state(), "label_order": list(L.LABELS_14),
    }
    torch.save(payload, tmp_path / "ckpt.pt")
    after_save = torch.randn(3)

    loaded = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    fresh = torch.nn.Linear(4, 4)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.PolynomialLR(fresh_opt, total_iters=10, power=2.0)
    fresh.load_state_dict(loaded["unet_state_dict"])
    fresh_opt.load_state_dict(loaded["optimizer"])
    fresh_sched.load_state_dict(loaded["scheduler"])
    set_rng_state(loaded["rng"])

    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)
    assert fresh_sched.get_last_lr() == scheduler.get_last_lr()
    assert loaded["global_step"] == 3
    assert loaded["label_order"] == list(L.LABELS_14)
    assert torch.equal(torch.randn(3), after_save), "RNG must resume where it left off"


def test_wandb_disabled_mode_needs_no_network(monkeypatch):
    import socket

    import wandb

    def no_network(*args, **kwargs):
        raise AssertionError("wandb touched the network in disabled mode")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    run = wandb.init(project="ccella-test", mode="disabled", name="offline-check")
    wandb.log({"x": 1.0}, step=1)
    run.finish()


def test_the_validation_subset_is_deterministic_and_rank_independent(tmp_path):
    from baselines.ccella.data import fixed_validation_subset

    rows = _tiny_cache(tmp_path, n_series=10)
    os.rename(os.path.join(tmp_path, "manifest", "train-0000.csv"),
              os.path.join(tmp_path, "manifest", "val-0000.csv"))
    config = load_config(SMOKE_CONFIG, [f"data.cache_root={tmp_path}", "validation.subset=4"])
    first = fixed_validation_subset(config, "val")
    second = fixed_validation_subset(config, "val")
    assert [r["sample_id"] for r in first] == [r["sample_id"] for r in second]
    assert len(first) == 4
    assert first == sorted(first, key=lambda r: r["sample_id"])
    _ = rows
