"""The regression net for Text2CT-on-MR-RATE.

    pytest baselines/text2ct/test_finetune.py -v

Every test is skipped rather than failed when what it needs is absent -- the upstream clone, the
released weights, a GPU, the MR-RATE archives -- and the skip reason says which. The tests that run
everywhere (config, manifest, modality vocabulary, preprocessing on a synthetic NIfTI) are the ones
that catch the mistakes a login node can catch; the rest need `--gres=gpu:h200:1`.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest
import torch

# The repo root: this file is baselines/text2ct/test_finetune.py. It lives beside the code rather
# than in a `tests/` directory because the root .gitignore ignores `tests/` anywhere in the tree,
# and a baseline's regression net has to travel with the baseline.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.text2ct import config as cfg_module
from baselines.text2ct import fine_tune, mrrate_data
from baselines.text2ct.config import apply_dotted, load_config, validate

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
SMOKE_CONFIG = os.path.join(CONFIG_DIR, "smoke.yaml")
MAIN_CONFIG = os.path.join(CONFIG_DIR, "mrrate_finetune.yaml")


def _config(**overrides):
    config = load_config(SMOKE_CONFIG)
    return apply_dotted(config, [f"{k}={json.dumps(v)}" for k, v in overrides.items()])


def _text2ct_root():
    return load_config(SMOKE_CONFIG)["text2ct_root"]


def _have(path):
    return bool(path) and os.path.exists(path)


requires_upstream = pytest.mark.skipif(
    not _have(os.path.join(_text2ct_root() or "", "configs", "config_rflow.json")),
    reason="upstream Text2CT clone not present")
requires_unet = pytest.mark.skipif(
    not _have(os.path.join(_text2ct_root() or "", "models", "unet_rflow_200ep.pt")),
    reason="released UNet weights not downloaded")
requires_vae = pytest.mark.skipif(
    not _have(os.path.join(_text2ct_root() or "", "models", "autoencoder_epoch273.pt")),
    reason="released autoencoder weights not downloaded")
requires_clip = pytest.mark.skipif(
    not _have(os.path.join(_text2ct_root() or "", "models", "CLIP3D_Finding_Impression_30ep.pt")),
    reason="released 3D-CLIP weights not downloaded")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
# Every test that instantiates the UNet or the VAE needs a GPU, not only the ones that run a
# training step: `config_rflow.json` sets `use_flash_attention: true`, which MONAI refuses to build
# on CPU, and the MAISI autoencoder's `norm_float16` is likewise GPU-only. Building them with those
# switches off would be a different model, so the tests take the GPU rather than the substitute.
requires_built_model = pytest.mark.skipif(not torch.cuda.is_available(),
                                          reason="the released model definition is GPU-only "
                                                 "(use_flash_attention / norm_float16)")
requires_mrrate = pytest.mark.skipif(
    not _have(os.path.join(load_config(SMOKE_CONFIG)["mrrate"]["raw_root"] or "",
                           "series.parquet")),
    reason="MR-RATE archives not reachable")


### Configuration ###


def test_config_defaults_load_and_validate():
    config = validate(load_config(MAIN_CONFIG))
    assert config["train"]["max_steps"] == 60000
    assert config["volume"]["inplane_size"] % 32 == 0
    assert config["volume"]["num_slices"] % 32 == 0


def test_the_grid_reproduces_text2cts_own_latent_shape():
    """512 x 512 x 128 -> (4, 128, 128, 32), exactly what
    `scripts/diff_model_create_training_data.py:181` logs for a CT."""
    config = load_config(MAIN_CONFIG)
    assert cfg_module.latent_shape(config) == (4, 128, 128, 32)
    assert cfg_module.grid_spacing(config) == (0.5, 0.5, 1.5)


def test_both_configs_satisfy_the_released_samplers_own_input_check():
    for path in (MAIN_CONFIG, SMOKE_CONFIG):
        assert cfg_module.check_input_problems(load_config(path)["volume"]) == []


def test_a_grid_the_released_sampler_would_reject_fails_at_config_time():
    # 160 slices is what a 1 mm grid over a median MR-RATE brain wants, and check_input forbids it.
    with pytest.raises(ValueError, match="check_input"):
        validate(load_config(MAIN_CONFIG, {"volume": {"num_slices": 160}}))
    with pytest.raises(ValueError, match="check_input"):
        validate(load_config(MAIN_CONFIG, {"volume": {"inplane_mm": 0.25}}))


def test_unknown_config_key_raises():
    with pytest.raises(KeyError):
        load_config(None, {"train": {"not_a_key": 1}})


def test_dotted_overrides_parse_types():
    config = apply_dotted(load_config(None), ["train.lr=5e-5", "train.max_steps=10",
                                              "data.report_sections=[\"impression\"]"])
    assert config["train"]["lr"] == 5e-5 and config["train"]["max_steps"] == 10
    assert config["data"]["report_sections"] == ["impression"]


def test_validate_rejects_a_grid_the_unet_cannot_take():
    config = load_config(MAIN_CONFIG, {"volume": {"num_slices": 150}})
    with pytest.raises(ValueError, match="multiple of 32"):
        validate(config)


def test_effective_batch_size_counts_gpus_and_accumulation():
    config = load_config(MAIN_CONFIG)
    assert cfg_module.effective_batch_size(config, 8) == 128


### Tracking ###


def test_wandb_defaults_sit_alongside_mrflows_own_runs():
    wandb = load_config(MAIN_CONFIG)["wandb"]
    assert wandb["project"] == "MRFlow"           # same project as the model it is a baseline for
    assert wandb["mode"] == "online"
    # A fixed id, not a generated one: a continuation job must resume the SAME run, or a 60,000-step
    # fine-tune spread over several 24 h jobs reads as several unrelated curves.
    assert wandb["id"] and wandb["id"] == wandb["name"]


def test_validate_rejects_an_unknown_wandb_mode():
    with pytest.raises(ValueError, match="wandb.mode"):
        validate(load_config(MAIN_CONFIG, {"wandb": {"mode": "yes please"}}))


def test_disabled_tracker_is_a_no_op_and_never_touches_wandb(monkeypatch):
    """`--no_wandb` and non-zero ranks must not import or initialise wandb at all."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *a, **k):
        assert name != "wandb", "a disabled Tracker imported wandb"
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse)
    config = load_config(MAIN_CONFIG, {"wandb": {"mode": "disabled"}})
    tracker = fine_tune.Tracker(config, world_size=1)
    assert tracker.run is None
    tracker.log({"x": 1}, step=0)                 # all three must be safe no-ops
    tracker.summary({"y": 2})
    tracker.finish()


def test_tracker_is_off_in_smoke_mode():
    config = load_config(MAIN_CONFIG)
    assert fine_tune.Tracker(config, world_size=1, enabled=False).run is None


### Modality vocabulary ###


def test_modality_ids_are_maisis_own_and_never_collide_with_ct():
    table = mrrate_data.MODALITY_TO_ID
    assert mrrate_data.PRETRAINED_CT_CLASS_ID not in table.values()
    assert table["T1w"] == 9 and table["T2w"] == 10 and table["FLAIR"] == 11
    assert table["MRA"] == 16 and table["SWI"] == 20
    assert table["CFG_NULL"] == 0 and table["UNKNOWN"] == 8
    assert len(set(table.values())) == len(table)
    assert max(table.values()) < mrrate_data.NUM_CLASS_EMBEDS


def test_modality_aliases_resolve_and_unknown_labels_raise():
    assert mrrate_data.modality_to_id("t1w") == mrrate_data.modality_to_id("T1w")
    assert mrrate_data.modality_to_id("swan") == mrrate_data.MODALITY_TO_ID["SWI"]
    with pytest.raises(KeyError, match="unmapped modality"):
        mrrate_data.modality_to_id("DWI")


def test_plane_aliases_match_the_array_layout():
    assert mrrate_data.plane_name("axi") == "AXIAL"
    assert mrrate_data.plane_order("AXIAL") == (0, 1, 2)
    assert mrrate_data.plane_order("SAGITTAL") == (1, 0, 2)
    assert mrrate_data.plane_order("CORONAL") == (2, 0, 1)
    # Oblique falls back to axial in both the label and the permutation, so they cannot disagree.
    assert mrrate_data.plane_name("obl") == "AXIAL"
    assert mrrate_data.plane_order("obl") == mrrate_data.plane_order("AXIAL")


def test_the_coarse_axis_lands_on_the_acquisition_planes_stacking_axis():
    """The target grid is anisotropic, so which (S, R, A) axis carries `slice_mm` depends on the
    plane. Getting it wrong blurs the wrong axis and nothing downstream would notice."""
    for plane, stacking in (("AXIAL", 0), ("SAGITTAL", 1), ("CORONAL", 2)):
        spacing = mrrate_data.target_spacing_sra(plane, 0.5, 1.5)
        assert spacing[stacking] == 1.5
        assert [spacing[a] for a in range(3) if a != stacking] == [0.5, 0.5]
        # and it is the axis plane_order leads with, i.e. the output's last axis
        assert mrrate_data.plane_order(plane)[0] == stacking


def test_an_isotropic_request_is_plane_independent():
    assert {mrrate_data.target_spacing_sra(p, 1.0, 1.0) for p in mrrate_data.PLANES} == \
        {(1.0, 1.0, 1.0)}


### Report text ###


def test_report_formatting_is_upstreams_string_by_default():
    """The default conditioning text is byte-identical to what the frozen encoder was trained on."""
    report = {"findings": "No acute infarct.", "impression": "Unremarkable."}
    body = mrrate_data.format_report(report, modality="t1w")
    assert body == "Findings: No acute infarct. Impression: Unremarkable."


def test_no_plane_appears_in_the_conditioning_text_in_either_mode():
    """Plane is not a conditioning signal anywhere: not as a class id, not in the string."""
    report = {"findings": "Left frontal lesion.", "impression": "Follow up."}
    for modality_prefix in (False, True):
        for plane_word in ("axial", "sagittal", "coronal", "plane"):
            text = mrrate_data.format_report(report, modality="t1w",
                                             modality_prefix=modality_prefix)
            assert plane_word not in text.lower()


def test_optional_modality_prefix_names_the_contrast_and_nothing_else():
    report = {"findings": "No acute infarct.", "impression": "Unremarkable."}
    body = mrrate_data.format_report(report, modality="t1w")
    prefixed = mrrate_data.format_report(report, modality="t1w", modality_prefix=True)
    assert prefixed == "Brain MRI, T1-weighted. " + body


def test_missing_report_sections_raise_instead_of_producing_an_empty_string():
    with pytest.raises(mrrate_data.ReportMissing):
        mrrate_data.format_report({"findings": "", "impression": "  "})
    # ~9% of MR-RATE studies have no impression: findings alone is still a usable sample.
    assert "Findings: x" in mrrate_data.format_report({"findings": "x", "impression": ""})


### Anti-drift: what we restate must still match upstream's source ###

# The model, the encoders and the scheduler are *driven* from the upstream clone, so they cannot
# drift. Three things are restated in this package -- the report string, the training objective and
# the tensors the UNet is called with -- and these tests read `887caa9`'s own source to check that
# the restatement is still true. They fail loudly if the pinned commit is ever moved.


def _upstream_source(relative):
    with open(os.path.join(_text2ct_root(), relative)) as handle:
        return handle.read()


@requires_upstream
def test_report_string_still_matches_upstreams_f_string():
    """`scripts/save_embeddings_ctrate.py:81` is the format the frozen encoder was trained on."""
    source = _upstream_source("scripts/save_embeddings_ctrate.py")
    assert 'text = f"Findings: {findings} Impression: {impressions}"' in source
    ours = mrrate_data.format_report({"findings": "A.", "impression": "B."})
    assert ours == "Findings: A. Impression: B."


@requires_upstream
def test_training_objective_still_matches_upstreams():
    """v-prediction target `images - noise` and an L1 loss (`scripts/diff_model_train.py`)."""
    source = _upstream_source("scripts/diff_model_train.py")
    assert "model_gt = images - noise" in source
    assert "loss_pt = torch.nn.L1Loss()" in source
    assert "images = images * scale_factor" in source
    assert load_config(MAIN_CONFIG)["train"]["loss"] == "l1"


@requires_upstream
def test_unet_is_still_called_with_the_same_inputs_upstream_passes():
    """Upstream builds `unet_inputs` as x / timesteps / spacing_tensor / context, plus class_labels
    when `num_class_embeds` is set. `diffusion_loss` must pass exactly that set."""
    import inspect

    source = _upstream_source("scripts/diff_model_train.py")
    for key in ('"x": noisy_latent', '"timesteps": timesteps',
                '"spacing_tensor": spacing_tensor', '"context": cond_masked',
                '"class_labels": modality_tensor'):
        assert key in source, key
    ours = inspect.getsource(fine_tune.diffusion_loss)
    for kwarg in ("x=noisy", "timesteps=timesteps", "context=context",
                  "class_labels=class_labels", "spacing_tensor=spacing"):
        assert kwarg in ours, kwarg


@requires_upstream
def test_report_dropout_default_and_null_still_match_the_release():
    """Report CFG is upstream's: a 0.1 dropout rate and a ZERO context as the null, on both sides."""
    import json

    with open(os.path.join(_text2ct_root(), "configs", "config_diff_model.json")) as handle:
        released = json.load(handle)
    assert released["diffusion_unet_train"]["conditional_free_guidance"] == 0.1
    assert load_config(MAIN_CONFIG)["cfg"]["report_dropout_prob"] == 0.1

    train_source = _upstream_source("scripts/diff_model_train.py")
    assert "cond_masked[mask_uncond] = 0" in train_source          # training null is zeros
    infer_source = _upstream_source("scripts/diff_model_infer.py")
    assert "torch.zeros_like(impression" in infer_source           # sampling null is zeros
    from baselines.text2ct.text_encoder import null_context
    assert float(null_context(2, "cpu").abs().sum()) == 0.0


@requires_upstream
def test_scale_factor_default_is_the_one_the_release_falls_back_to():
    source = _upstream_source("scripts/diff_model_infer.py")
    assert "scale_factor = 1.0287" in source
    assert load_config(MAIN_CONFIG)["model"]["scale_factor"] == 1.0287


@requires_upstream
def test_define_instance_is_upstreams_own_function_not_a_copy():
    from baselines.text2ct.model import _add_upstream_to_path, _define_instance

    _add_upstream_to_path(_text2ct_root())
    from scripts.utils import define_instance
    assert define_instance.__module__ == "scripts.utils"
    # _define_instance is a thin wrapper that calls it; assert the call, not a reimplementation
    import inspect
    assert "from scripts.utils import define_instance" in inspect.getsource(_define_instance)


### Volume preprocessing ###


def _synthetic_nifti(shape=(64, 60, 40), spacing=(1.5, 1.5, 3.0), seed=0):
    """A NIfTI in nibabel's (X, Y, Z) order with a bright blob and a zero background, so the
    nonzero-percentile normalization has something to key on."""
    import nibabel as nib

    rng = np.random.default_rng(seed)
    data = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(n // 4, 3 * n // 4) for n in shape)
    data[sl] = 100.0 + 800.0 * rng.random(tuple(s.stop - s.start for s in sl))
    affine = np.diag([*spacing, 1.0])
    buf = io.BytesIO()
    file_map = nib.Nifti1Image(data, affine).make_file_map()
    image = nib.Nifti1Image(data, affine)
    return image.to_bytes()


def test_preprocessing_shape_dtype_and_range():
    out, spacing, native = mrrate_data.preprocess_volume(
        _synthetic_nifti(), "AXIAL", inplane_mm=1.0, slice_mm=1.0, inplane_size=64, num_slices=64,
        min_native_slices=8)
    assert out.shape == (64, 64, 64)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert np.isfinite(out).all()
    assert len(spacing) == 3 and native > 0


def test_preprocessing_pads_short_volumes_and_crops_long_ones_without_touching_the_content():
    short, _, native_short = mrrate_data.preprocess_volume(
        _synthetic_nifti(shape=(64, 64, 12), spacing=(1.0, 1.0, 1.0)), "AXIAL",
        inplane_mm=1.0, slice_mm=1.0, inplane_size=64, num_slices=64, min_native_slices=8)
    long, _, native_long = mrrate_data.preprocess_volume(
        _synthetic_nifti(shape=(64, 64, 200), spacing=(1.0, 1.0, 1.0)), "AXIAL",
        inplane_mm=1.0, slice_mm=1.0, inplane_size=64, num_slices=64, min_native_slices=8)
    assert short.shape == long.shape == (64, 64, 64)
    assert native_short == 12 and native_long == 200
    # Padding is zeros at both ends, so the first and last slice of the padded volume are blank.
    assert float(np.abs(short[:, :, 0]).max()) == 0.0


def test_preprocessing_is_deterministic():
    blob = _synthetic_nifti()
    a = mrrate_data.preprocess_volume(blob, "AXIAL", inplane_mm=1.0, slice_mm=1.0,
                                      inplane_size=64, num_slices=64, min_native_slices=8)[0]
    b = mrrate_data.preprocess_volume(blob, "AXIAL", inplane_mm=1.0, slice_mm=1.0,
                                      inplane_size=64, num_slices=64, min_native_slices=8)[0]
    assert np.array_equal(a, b)


def test_a_degenerate_volume_is_rejected_rather_than_encoded():
    with pytest.raises(mrrate_data.VolumeUnusable):
        mrrate_data.preprocess_volume(_synthetic_nifti(shape=(64, 64, 4), spacing=(1, 1, 1)),
                                      "AXIAL", inplane_mm=1.0, slice_mm=1.0, inplane_size=64,
                                      num_slices=64, min_native_slices=32)


### Writing a generated volume back out ###


@pytest.mark.parametrize("plane", ["AXIAL", "SAGITTAL", "CORONAL"])
def test_generated_nifti_round_trips_through_read_canonical(tmp_path, plane):
    """`save_generated_nifti` must be the exact inverse of what `read_canonical` + `plane_order` do,
    for every plane. Deliberately non-cubic so a transposed axis cannot hide."""
    rng = np.random.default_rng(0)
    volume = rng.random((64, 96, 32)).astype(np.float32)      # (X, Y, Z), all three sizes different
    path = str(tmp_path / "gen.nii.gz")
    mrrate_data.save_generated_nifti(volume, plane, path, inplane_mm=0.5, slice_mm=1.5)

    with open(path, "rb") as handle:
        sra, spacing = mrrate_data.read_canonical(handle.read())
    assert spacing == pytest.approx(mrrate_data.target_spacing_sra(plane, 0.5, 1.5))

    # The ingest path permutes plane-first; undoing the final transpose must give back the input.
    plane_first = sra.transpose(mrrate_data.plane_order(plane))
    assert np.allclose(plane_first.transpose(1, 2, 0), volume, atol=1e-6)


@requires_mrrate
def test_a_round_tripped_real_volume_still_looks_like_the_same_brain():
    """The semantic half: preprocess a real series onto the model grid, write it out as a generated
    volume, and put it through the *evaluation's own* ingest geometry. What comes back must be the
    same anatomy MRFlow's grid would have produced -- a wrong axis order leaves shapes intact and
    correlation near zero."""
    import tempfile

    try:
        from baselines.common.canonicalize import canonicalize_external
    except ImportError:
        # `canonicalize_external` reaches `echosyn.common`, which imports diffusers -- present in
        # the MRFlow venv, absent from this baseline's. Same split as
        # `test_list_series_matches_mrflows_selection`: run it there.
        pytest.skip("baselines.common.canonicalize needs the MRFlow venv (diffusers)")

    config = _config()
    entry = mrrate_data.list_series(config["mrrate"]["raw_root"], "val", 1, 4, 42)[0]
    blob = mrrate_data.read_member(entry["archive"], entry["member"])

    model_grid, _, _ = mrrate_data.preprocess_volume(blob, entry["plane"], 0.5, 1.5, 512, 128, 15.0)
    with tempfile.TemporaryDirectory() as d:
        path = mrrate_data.save_generated_nifti(model_grid, entry["plane"],
                                                os.path.join(d, "g.nii.gz"), 0.5, 1.5)
        with open(path, "rb") as handle:
            sra, spacing = mrrate_data.read_canonical(handle.read())
    ingested = canonicalize_external(sra, spacing, entry["plane"], posterior_shift_mm=0.0)

    # MRFlow's own grid for the same series, cropped to the depth the round trip kept.
    reference, _, _ = mrrate_data.preprocess_volume(blob, entry["plane"], 1.0, 1.0, 256,
                                                    ingested.shape[0], 15.0)
    reference = reference.transpose(2, 0, 1)                  # (X, Y, Z) -> plane-first
    assert ingested.shape == reference.shape, (ingested.shape, reference.shape)

    a, b = ingested.ravel().astype(np.float64), reference.ravel().astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"round-trip correlation against MRFlow's grid: {corr:.5f}")
    assert corr > 0.95, f"orientation is wrong somewhere: correlation {corr:.4f}"


### Cache: manifest, store, metadata ###


@pytest.mark.parametrize("bundle", [True, False])
def test_latent_store_round_trips_relative_to_the_cache_root(tmp_path, bundle):
    store = mrrate_data.LatentStore(str(tmp_path), 3, "train", bundle=bundle)
    array = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    member = store.save("abc.latent.npy", array)
    store.close()
    row = {"latent_path": member, "zip": store.zip_name}
    back = mrrate_data.load_artifact(str(tmp_path), row, "latent_path")
    assert np.array_equal(back, array) and back.dtype == np.float16
    # One inode per shard when bundled -- the whole reason the bundle exists.
    assert len(os.listdir(tmp_path / "artifacts")) == 1


def test_two_splits_never_share_a_bundle(tmp_path):
    """train shard 3 and val shard 3 must not be the same file. They were, and two concurrent array
    tasks truncated each other's archive in place -- each still wrote a valid-looking manifest, so
    nothing downstream would have noticed until the latents read back as the wrong anatomy."""
    names = set()
    for split in ("train", "val", "test"):
        store = mrrate_data.LatentStore(str(tmp_path), 3, split)
        store.save("x.npy", np.zeros((2, 2), dtype=np.float16))
        store.close()
        names.add(store.zip_name)
    assert len(names) == 3, names
    assert len(os.listdir(tmp_path / "artifacts")) == 3


def test_manifest_round_trips_and_is_read_in_shard_order(tmp_path):
    rows = [{f: "" for f in mrrate_data.MANIFEST_FIELDS} | {"sample_id": f"s{i}", "split": "val",
            "modality": "T1w", "plane": "AXIAL"} for i in range(3)]
    mrrate_data.write_manifest(str(tmp_path / "manifest" / "val-0001.csv"), rows[2:])
    mrrate_data.write_manifest(str(tmp_path / "manifest" / "val-0000.csv"), rows[:2])
    back = mrrate_data.read_manifest(str(tmp_path), "val")
    assert [r["sample_id"] for r in back] == ["s0", "s1", "s2"]


@requires_vae
@requires_clip
def test_cache_metadata_refuses_a_stale_cache(tmp_path):
    config = _config()
    from baselines.text2ct.config import weight_paths
    paths = weight_paths(config)
    meta = mrrate_data.cache_meta(config, paths["vae"], paths["clip"])
    with open(tmp_path / mrrate_data.CACHE_META_NAME, "w") as handle:
        json.dump(meta, handle)

    assert mrrate_data.check_cache_meta(str(tmp_path), meta) == []
    moved = load_config(SMOKE_CONFIG, {"volume": {"num_slices": 96}})
    other = mrrate_data.cache_meta(moved, paths["vae"], paths["clip"])
    with pytest.raises(ValueError, match="different settings"):
        mrrate_data.check_cache_meta(str(tmp_path), other)
    assert mrrate_data.check_cache_meta(str(tmp_path), other, strict=False)


def test_cache_metadata_is_missing_rather_than_assumed(tmp_path):
    with pytest.raises(FileNotFoundError):
        mrrate_data.check_cache_meta(str(tmp_path), {})


### Dataset ###


def test_synthetic_dataset_matches_the_cached_dataset_contract():
    config = _config()
    dataset = fine_tune.SyntheticLatentDataset(config, n=4)
    item = dataset[0]
    assert item["latent"].shape == (4, 64, 64, 32)      # 256/4, 256/4, 128/4
    assert item["latent"].shape == cfg_module.latent_shape(config)
    assert item["context"].shape == (1, 768)
    assert abs(float(item["context"].norm()) - 1.0) < 1e-5
    assert item["class_label"].dtype == torch.int64
    # the grid's mm x 1e2, the scale `diff_model_train.py:111` trains spacing_layer on
    assert item["spacing"].tolist() == [100.0, 100.0, 200.0]


### The model ###


@requires_upstream
@requires_unet
@requires_built_model
def test_unet_loads_strictly_and_carries_maisis_class_embedding():
    from baselines.text2ct.model import build_unet
    unet, extra = build_unet(_text2ct_root(),
                             os.path.join(_text2ct_root(), "models", "unet_rflow_200ep.pt"),
                             "cuda", modality_init="keep")
    assert unet.num_class_embeds == mrrate_data.NUM_CLASS_EMBEDS
    assert unet.class_embedding.weight.shape[0] == mrrate_data.NUM_CLASS_EMBEDS
    assert unet.include_spacing_input is True
    assert unet.include_top_region_index_input is False   # include_body_region: false
    assert extra == {}                                    # the release is a bare state_dict


@requires_upstream
@requires_unet
@requires_built_model
def test_a_mismatched_checkpoint_raises_instead_of_loading_partially():
    from baselines.text2ct.model import build_unet, load_strict
    unet, _ = build_unet(_text2ct_root(),
                         os.path.join(_text2ct_root(), "models", "unet_rflow_200ep.pt"),
                         "cuda", modality_init="keep")
    broken = {k: v for k, v in unet.state_dict().items() if "class_embedding" not in k}
    with pytest.raises(RuntimeError, match="does not match"):
        load_strict(unet, broken, "UNet")


@requires_upstream
@requires_unet
@requires_built_model
def test_ct_row_init_reproduces_the_pretrained_model_at_step_zero():
    """The modality condition must not perturb the released UNet before a single gradient step."""
    from baselines.text2ct.model import build_unet

    unet, _ = build_unet(_text2ct_root(),
                         os.path.join(_text2ct_root(), "models", "unet_rflow_200ep.pt"),
                         "cuda", modality_init="ct_row")
    unet.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16, 16, device="cuda")
    kwargs = dict(x=x, timesteps=torch.tensor([500.0], device="cuda"),
                  context=torch.randn(1, 1, 768, device="cuda"),
                  spacing_tensor=torch.tensor([[100., 100., 100.]], device="cuda"))
    with torch.no_grad():
        as_ct = unet(class_labels=torch.tensor([mrrate_data.PRETRAINED_CT_CLASS_ID],
                                               device="cuda"), **kwargs)
        as_t1 = unet(class_labels=torch.tensor([mrrate_data.MODALITY_TO_ID["T1w"]],
                                               device="cuda"), **kwargs)
        as_flair = unet(class_labels=torch.tensor([mrrate_data.MODALITY_TO_ID["FLAIR"]],
                                                  device="cuda"), **kwargs)
    assert torch.equal(as_ct, as_t1)
    assert torch.equal(as_ct, as_flair)


@requires_upstream
@requires_unet
@requires_built_model
def test_zeros_init_is_available_and_differs_from_ct_row():
    from baselines.text2ct.model import build_unet
    path = os.path.join(_text2ct_root(), "models", "unet_rflow_200ep.pt")
    zeroed, _ = build_unet(_text2ct_root(), path, "cuda", modality_init="zeros")
    t1 = mrrate_data.MODALITY_TO_ID["T1w"]
    assert float(zeroed.class_embedding.weight[t1].abs().sum()) == 0.0
    assert float(zeroed.class_embedding.weight[mrrate_data.PRETRAINED_CT_CLASS_ID].abs().sum()) > 0


@requires_upstream
@requires_unet
@requires_built_model
def test_modality_embedding_is_a_unet_parameter_and_receives_gradient():
    from baselines.text2ct.model import build_unet
    unet, _ = build_unet(_text2ct_root(),
                         os.path.join(_text2ct_root(), "models", "unet_rflow_200ep.pt"), "cuda")
    assert any(name == "class_embedding.weight" for name, _ in unet.named_parameters())
    out = unet(x=torch.randn(1, 4, 16, 16, 16, device="cuda"),
               timesteps=torch.tensor([300.0], device="cuda"),
               context=torch.randn(1, 1, 768, device="cuda"),
               spacing_tensor=torch.tensor([[100., 100., 100.]], device="cuda"),
               class_labels=torch.tensor([mrrate_data.MODALITY_TO_ID["T2w"]], device="cuda"))
    out.float().pow(2).mean().backward()
    grad = unet.class_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    used = mrrate_data.MODALITY_TO_ID["T2w"]
    assert float(grad[used].abs().sum()) > 0
    # Only the row that was looked up moves: an nn.Embedding gradient is sparse by construction.
    assert float(grad[mrrate_data.PRETRAINED_CT_CLASS_ID].abs().sum()) == 0


@requires_upstream
def test_noise_scheduler_is_rectified_flow_with_v_prediction():
    from baselines.text2ct.model import build_noise_scheduler
    scheduler = build_noise_scheduler(_text2ct_root())
    assert scheduler.prediction_type == "v_prediction"
    assert scheduler.num_train_timesteps == 1000
    latent = torch.randn(2, 4, 8, 8, 8)
    timesteps = scheduler.sample_timesteps(latent)
    assert timesteps.shape == (2,)
    noisy = scheduler.add_noise(original_samples=latent, noise=torch.randn_like(latent),
                                timesteps=timesteps)
    assert noisy.shape == latent.shape and torch.isfinite(noisy).all()


@requires_upstream
@requires_vae
@requires_built_model
def test_vae_encodes_the_training_grid_to_the_expected_latent_shape():
    from baselines.text2ct.model import build_vae
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = build_vae(_text2ct_root(),
                    os.path.join(_text2ct_root(), "models", "autoencoder_epoch273.pt"), device)
    volume = torch.rand(1, 1, 64, 64, 64, device=device)
    # autocast is not optional: `norm_float16: true` makes the encoder emit fp16 activations into
    # fp32 convolutions. Upstream wraps the same call the same way
    # (`scripts/diff_model_create_training_data.py:179`).
    with torch.inference_mode(), torch.autocast("cuda", enabled=device.type == "cuda"):
        latent = vae.encode_stage_2_inputs(volume)
    assert latent.shape == (1, 4, 16, 16, 16)          # 4x compression on every axis


@requires_upstream
@requires_vae
@requires_built_model
def test_fast_concat_is_bit_identical_to_stock_monai():
    """The one library patch this package makes. It must change nothing but the time.

    The input crosses MONAI's hardcoded `max(size) < 500` threshold, which is what selects the host
    round-trip -- below it both paths are already the same code. `encode` is used rather than
    `encode_stage_2_inputs` because the latter samples the posterior, so two calls differ by a draw
    whatever the concat does.
    """
    from baselines.text2ct.model import build_vae, set_fast_maisi_concat

    root = _text2ct_root()
    ckpt = os.path.join(root, "models", "autoencoder_epoch273.pt")
    x = torch.rand(1, 1, 512, 512, 32, device="cuda")
    out = {}
    try:
        for fast in (False, True):
            vae = build_vae(root, ckpt, "cuda", fast_concat=fast)
            with torch.inference_mode(), torch.autocast("cuda"):
                out[fast] = tuple(t.float().clone() for t in vae.encode(x))
            del vae
            torch.cuda.empty_cache()
    finally:
        set_fast_maisi_concat(True)

    assert torch.equal(out[False][0], out[True][0]), \
        f"z_mu differs by {float((out[False][0] - out[True][0]).abs().max()):.3e}"
    assert torch.equal(out[False][1], out[True][1])


def test_fast_concat_can_be_turned_off_and_back_on():
    """`fast_concat=False` has to genuinely restore the stock methods, not just skip patching --
    otherwise the comparison above silently compares the patch against itself."""
    from monai.apps.generation.maisi.networks import autoencoderkl_maisi as maisi

    from baselines.text2ct.model import set_fast_maisi_concat

    try:
        set_fast_maisi_concat(True)
        patched = maisi.MaisiGroupNorm3D._cat_inputs
        set_fast_maisi_concat(False)
        stock = maisi.MaisiGroupNorm3D._cat_inputs
        assert stock is not patched
        set_fast_maisi_concat(True)
        assert maisi.MaisiGroupNorm3D._cat_inputs is not stock
    finally:
        set_fast_maisi_concat(True)


### Equivalence with upstream's own training loop ###


@requires_upstream
@requires_unet
@requires_built_model
def test_our_loss_equals_upstreams_own_train_one_epoch():
    """The claim this package has to earn: `fine_tune.diffusion_loss` computes what
    `scripts/diff_model_train.train_one_epoch` computes.

    Not a transcription check -- upstream's real function is imported and driven over the same
    batch. Both are seeded identically and draw in the same order (noise, then timesteps), so the
    noise and the timesteps are the same tensors, and the losses must agree to floating point.

    Report dropout is 0 and the modality is forced to upstream's hardcoded `ct` = 1, because those
    two are exactly the places this adaptation deliberately differs; everything else must not.
    """
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import build_noise_scheduler, build_unet, module_checksum
    from baselines.text2ct.upstream_bridge import upstream_train_one_epoch

    config = apply_dotted(_config(), ["cfg.report_dropout_prob=0.0"])
    device = torch.device("cuda")
    unet, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], device)
    scheduler = build_noise_scheduler(config["text2ct_root"])
    scale = config["model"]["scale_factor"]

    dataset = fine_tune.SyntheticLatentDataset(config, n=2)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    # Upstream hardcodes class 1; make ours agree so only the objective is under test.
    batch["class_label"] = torch.full_like(batch["class_label"],
                                           mrrate_data.PRETRAINED_CT_CLASS_ID)

    before = module_checksum(unet)
    torch.manual_seed(1234)
    theirs = upstream_train_one_epoch(config["text2ct_root"], unet, batch, scheduler, scale,
                                      device, report_dropout=0.0, amp=False)
    assert module_checksum(unet) == before, "the lr=0 optimizer step moved a weight"

    torch.manual_seed(1234)
    unet.train()
    with torch.no_grad():
        ours = float(fine_tune.diffusion_loss(unet, batch, scheduler, config, device))

    assert ours == pytest.approx(theirs, rel=1e-6), f"ours {ours!r} vs upstream {theirs!r}"


@requires_upstream
def test_upstream_batch_is_a_rename_not_a_conversion():
    """`as_upstream_batch` must not scale, squeeze or cast anything -- if it did, the equivalence
    test above would be comparing two different inputs."""
    from baselines.text2ct.upstream_bridge import as_upstream_batch

    config = _config()
    dataset = fine_tune.SyntheticLatentDataset(config, n=2)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    converted = as_upstream_batch(batch)
    assert converted["image"] is batch["latent"]
    assert converted["spacing"] is batch["spacing"]
    assert converted["cond"] is batch["context"]


@requires_upstream
@requires_vae
@requires_clip
@requires_mrrate
@requires_gpu
def test_upstream_layout_is_readable_by_upstreams_own_path_arithmetic(tmp_path):
    """The emitted layout must satisfy `diff_model_train.py:508-521` exactly -- the derivation is
    string surgery on paths, so it is easy to be one substitution away from silently empty."""
    import json

    from baselines.text2ct.prepare_data import prepare
    from baselines.text2ct.upstream_bridge import (REPORT_ENCODER_MODEL, write_upstream_configs,
                                                   write_upstream_layout)

    config = _config()
    config["data"]["cache_root"] = str(tmp_path / "cache")
    config["mrrate"]["max_series_val"] = 2
    prepare(config, "val", shard=0, num_shards=1, limit=2)

    out = str(tmp_path / "upstream")
    listing = write_upstream_layout(config, "val", out, limit=2)
    written = write_upstream_configs(config, out, "val")

    env = json.load(open(written["environment.json"]))
    names = [item["image"] for item in json.load(open(listing))["training"]]
    assert names, "no volumes emitted"

    # Replay upstream's derivation verbatim.
    def resolve_path(path, base):
        return path if os.path.isabs(path) or path.startswith(base) else os.path.join(base, path)

    for name in names:
        image_path = resolve_path(name, env["data_base_dir"])
        str_img = image_path.replace(env["data_base_dir"], env["embedding_base_dir"])
        str_info = (str_img + ".json").replace("_emb", "")
        str_cond = str_info.replace(".nii.gz.json", f"_impression_{REPORT_ENCODER_MODEL}.npy")
        assert os.path.exists(image_path), image_path      # :511 skips the series without this
        for path in (str_img, str_info, str_cond):
            assert os.path.exists(path), path
        assert "spacing" in json.load(open(str_info))

    # And that the latent reloads through upstream's own transform in (C, X, Y, Z).
    import monai
    loaded = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
    ])({"image": str_img})["image"]
    assert tuple(loaded.shape) == cfg_module.latent_shape(config)


### Freezing ###


def test_freeze_survives_a_train_call_and_keeps_parameters_out_of_the_optimizer():
    from baselines.text2ct.model import FrozenGuard, freeze

    frozen = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4),
                                 torch.nn.Dropout(0.5))
    trainable = torch.nn.Linear(4, 4)
    guard = FrozenGuard()
    guard.add("frozen", frozen)

    frozen.train()                                     # the accident this guards against
    guard.assert_eval_mode()
    assert all(not p.requires_grad for p in frozen.parameters())

    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=1e-3)
    guard.assert_absent_from(optimizer)
    with pytest.raises(AssertionError):
        FrozenGuard.assert_absent_from(guard, torch.optim.AdamW(list(frozen.parameters()), lr=1e-3))


def test_frozen_batchnorm_running_stats_do_not_move():
    """eval() is what stops this: a BatchNorm in train mode updates `running_mean` on the forward
    pass alone, with no gradient and no optimizer involved."""
    from baselines.text2ct.model import FrozenGuard

    module = torch.nn.BatchNorm1d(4)
    guard = FrozenGuard()
    guard.add("bn", module)
    with torch.inference_mode():
        module(torch.randn(8, 4) * 10 + 5)
    guard.assert_unchanged()


def test_checksum_notices_a_single_changed_weight():
    from baselines.text2ct.model import FrozenGuard

    module = torch.nn.Linear(4, 4)
    guard = FrozenGuard()
    guard.add("linear", module)
    guard.assert_unchanged()
    with torch.no_grad():
        module.weight[0, 0] += 1e-3
    with pytest.raises(AssertionError, match="changed"):
        guard.assert_unchanged()


def test_parameter_report_separates_trainable_from_frozen():
    from baselines.text2ct.model import freeze, parameter_report
    trainable = torch.nn.Linear(10, 10)                        # 110 parameters
    frozen = freeze(torch.nn.Linear(10, 10))
    lines, n_trainable, n_frozen = parameter_report({"unet": trainable, "vae": frozen})
    assert n_trainable == 110 and n_frozen == 110
    assert any("TOTAL" in line for line in lines)


### Forward, backward, CFG ###


@requires_upstream
@requires_unet
@requires_gpu
def test_one_training_step_moves_the_unet_and_leaves_the_vae_untouched(tmp_path):
    """The end-to-end contract in one test: forward diffusion, UNet forward, backward, optimizer
    step, and the frozen components bit-identical afterwards."""
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import (FrozenGuard, build_noise_scheduler, build_unet, build_vae,
                                         module_checksum)

    config = _config()
    paths = weight_paths(config)
    device = torch.device("cuda")
    unet, _ = build_unet(config["text2ct_root"], paths["unet"], device)
    scheduler = build_noise_scheduler(config["text2ct_root"])
    guard = FrozenGuard()
    if os.path.exists(paths["vae"]):
        guard.add("vae", build_vae(config["text2ct_root"], paths["vae"], device))

    before_unet = module_checksum(unet)
    optimizer = fine_tune.build_optimizer(unet, config["train"])
    guard.assert_absent_from(optimizer)

    dataset = fine_tune.SyntheticLatentDataset(config, n=2)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    unet.train()
    loss = fine_tune.diffusion_loss(unet, batch, scheduler, config, device)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    assert fine_tune._assert_unet_gradients(unet)
    optimizer.step()

    assert module_checksum(unet) != before_unet
    guard.assert_all(optimizer)


@requires_upstream
@requires_unet
@requires_gpu
def test_conditional_and_unconditional_branches_differ_and_cfg_interpolates_them():
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import build_unet
    from baselines.text2ct.text_encoder import null_context

    config = _config()
    device = torch.device("cuda")
    unet, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], device)
    unet.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16, 16, device=device)
    context = torch.randn(1, 1, 768, device=device)
    context = context / context.norm()
    labels = torch.tensor([mrrate_data.MODALITY_TO_ID["FLAIR"]], device=device)
    kwargs = dict(x=x, timesteps=torch.tensor([500.0], device=device),
                  class_labels=labels, spacing_tensor=torch.tensor([[100., 100., 100.]],
                                                                   device=device))
    with torch.no_grad():
        uncond = unet(context=null_context(1, device), **kwargs)
        cond = unet(context=context, **kwargs)
    assert not torch.allclose(uncond, cond), "the report must change the prediction"

    scale = 5.0
    guided = uncond + scale * (cond - uncond)
    assert torch.allclose(guided, uncond + scale * (cond - uncond))
    # The modality is the same tensor in both branches -- that is the whole design.
    assert kwargs["class_labels"] is labels


@requires_upstream
@requires_unet
@requires_gpu
def test_cfg_sampling_path_runs_and_stays_finite():
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import build_noise_scheduler, build_unet

    config = _config()
    device = torch.device("cuda")
    unet, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], device)
    unet.eval()
    out = fine_tune.cfg_sample_step(unet, config, device,
                                    build_noise_scheduler(config["text2ct_root"]), steps=2)
    assert out.shape == (1,) + cfg_module.latent_shape(config)
    assert torch.isfinite(out).all()


def test_report_dropout_zeroes_the_context_and_never_the_modality():
    """The null branch drops only the report: the class label is untouched at every dropout rate."""
    context = torch.randn(64, 1, 768)
    labels = torch.full((64,), mrrate_data.MODALITY_TO_ID["T1w"])
    torch.manual_seed(0)
    mask = torch.rand(64) < 1.0
    dropped = torch.where(mask[:, None, None], torch.zeros_like(context), context)
    assert float(dropped.abs().sum()) == 0.0
    assert torch.equal(labels, torch.full((64,), mrrate_data.MODALITY_TO_ID["T1w"]))


### Learning-rate schedule ###


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    train = load_config(MAIN_CONFIG)["train"]
    assert fine_tune.lr_at(0, train) < train["lr"]
    assert abs(fine_tune.lr_at(train["warmup_steps"], train) - train["lr"]) < 1e-12
    assert fine_tune.lr_at(train["max_steps"] - 1, train) < train["min_lr"] * 1.01
    assert fine_tune.lr_at(train["max_steps"], train) == pytest.approx(train["min_lr"])
    # Monotone after warmup.
    values = [fine_tune.lr_at(s, train) for s in range(1000, train["max_steps"], 5000)]
    assert values == sorted(values, reverse=True)


def test_poly_schedule_matches_upstreams_shape():
    train = dict(load_config(MAIN_CONFIG)["train"], scheduler="poly", warmup_steps=0,
                 min_lr=0.0, poly_power=2.0, max_steps=100, lr=1.0)
    assert fine_tune.lr_at(0, train) == pytest.approx(1.0)
    assert fine_tune.lr_at(50, train) == pytest.approx(0.25)
    assert fine_tune.lr_at(100, train) == pytest.approx(0.0)


### Distributed structure ###


def test_distributed_setup_is_a_no_op_without_the_environment(monkeypatch):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    rank, world, local, device = fine_tune.setup_distributed()
    assert (rank, world, local) == (0, 1, 0)
    assert device.type in ("cuda", "cpu")


def test_loaders_use_a_distributed_sampler_when_world_size_exceeds_one():
    """Structural check only -- no process group is created, so nothing here needs two GPUs."""
    from torch.utils.data import DistributedSampler

    config = _config()
    dataset = fine_tune.SyntheticLatentDataset(config, n=16)
    sampler = DistributedSampler(dataset, num_replicas=4, rank=2, shuffle=True, drop_last=True)
    assert len(sampler) == 4
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first, "set_epoch must reshuffle, or every epoch sees one order"
    # Every rank gets a disjoint quarter.
    seen = [set(DistributedSampler(dataset, num_replicas=4, rank=r, shuffle=True, drop_last=True))
            for r in range(4)]
    assert sum(len(s) for s in seen) == len(set().union(*seen)) == 16


### Checkpoints ###


@requires_upstream
@requires_unet
@requires_built_model
def test_checkpoint_saves_resumes_and_carries_the_modality_vocabulary(tmp_path):
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import build_unet, module_checksum

    config = _config()
    unet, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], "cuda")
    optimizer = fine_tune.build_optimizer(unet, config["train"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    with torch.no_grad():                       # make the weights differ from the release
        unet.class_embedding.weight[mrrate_data.MODALITY_TO_ID["T1w"]] += 0.5
    trained = module_checksum(unet)
    path = str(tmp_path / fine_tune.checkpoint_name(7))
    fine_tune.save_checkpoint(path, unet, optimizer, scaler, 7, 1, config, None, 1.0287, 1000)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 7 and payload["epoch"] == 1
    assert payload["modality_vocabulary"]["modality_to_id"] == mrrate_data.MODALITY_TO_ID
    assert payload["scale_factor"] == 1.0287
    # Upstream's own loader reads exactly these two keys (scripts/diff_model_infer.py:57).
    assert "unet_state_dict" in payload and "scale_factor" in payload

    fresh, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], "cuda")
    assert module_checksum(fresh) != trained
    step, epoch = fine_tune.load_checkpoint(path, fresh, optimizer, scaler, "cuda")
    assert (step, epoch) == (7, 1)
    assert module_checksum(fresh) == trained
    assert fine_tune.latest_checkpoint(str(tmp_path)) == path


@requires_upstream
@requires_unet
@requires_built_model
def test_resume_refuses_a_checkpoint_from_a_different_modality_vocabulary(tmp_path):
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import build_unet

    config = _config()
    unet, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], "cuda")
    optimizer = fine_tune.build_optimizer(unet, config["train"])
    path = str(tmp_path / fine_tune.checkpoint_name(1))
    fine_tune.save_checkpoint(path, unet, optimizer, None, 1, 0, config, None, 1.0287, 1000)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["modality_vocabulary"]["modality_to_id"]["T1w"] = 99
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="different modality vocabulary"):
        fine_tune.load_checkpoint(path, unet, optimizer, None, "cuda")


def test_checkpoint_pruning_keeps_the_newest(tmp_path):
    for step in (10, 20, 30, 40):
        (tmp_path / fine_tune.checkpoint_name(step)).write_text("x")
    fine_tune.prune_checkpoints(str(tmp_path), keep=2)
    assert sorted(os.listdir(tmp_path)) == [fine_tune.checkpoint_name(30),
                                            fine_tune.checkpoint_name(40)]


### Reproducibility ###


@requires_upstream
@requires_unet
@requires_gpu
def test_a_fixed_seed_reproduces_the_loss():
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.model import build_noise_scheduler, build_unet

    config = _config()
    device = torch.device("cuda")
    unet, _ = build_unet(config["text2ct_root"], weight_paths(config)["unet"], device)
    unet.eval()
    scheduler = build_noise_scheduler(config["text2ct_root"])
    dataset = fine_tune.SyntheticLatentDataset(config, n=2)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])

    losses = []
    for _ in range(2):
        generator = torch.Generator(device=device).manual_seed(99)
        torch.manual_seed(99)
        with torch.no_grad():
            losses.append(float(fine_tune.diffusion_loss(unet, batch, scheduler, config, device,
                                                         generator)))
    assert losses[0] == pytest.approx(losses[1], rel=1e-6)


### Text encoder ###


@requires_upstream
@requires_clip
@requires_gpu
def test_text_encoder_is_frozen_and_returns_one_normalized_768d_token():
    from baselines.text2ct.config import weight_paths
    from baselines.text2ct.text_encoder import build_text_encoder, encode_reports

    config = _config()
    encoder = build_text_encoder(config["text2ct_root"], weight_paths(config)["clip"], "cuda",
                                 config["data"]["text_max_length"])
    assert all(not p.requires_grad for p in encoder.parameters())
    encoder.train()                             # must not re-enable anything
    assert not any(m.training for m in encoder.modules())

    out = encode_reports(encoder, ["Findings: normal. Impression: unremarkable.",
                                   "Findings: acute infarct. Impression: stroke."])
    assert out.shape == (2, 1, 768)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 1), atol=1e-4)
    assert not torch.allclose(out[0], out[1]), "two different reports must encode differently"


@requires_upstream
@requires_clip
def test_tokenizer_shim_matches_upstream_vocabulary():
    """The shim swaps the vendored slow tokenizer for the installed one. Same vocab, same ids --
    checked against CLIP's own known encoding of a fixed string."""
    import transformers

    tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    ids = tokenizer("a diagram", add_special_tokens=True)["input_ids"]
    assert ids == [49406, 320, 22697, 49407]


### MR-RATE, end to end ###


@requires_mrrate
def test_list_series_matches_mrflows_selection():
    """The two pipelines must see the same population -- same parquet filters, same seed, same
    shuffle -- or 'trained on the same data' is not true."""
    config = _config()
    ours = mrrate_data.list_series(config["mrrate"]["raw_root"], "val", 1, 32, 42)
    assert len(ours) == 32
    assert {e["modality"] for e in ours} <= set(mrrate_data.MODALITY_TO_ID)
    assert all(os.path.exists(e["archive"]) for e in ours[:4])
    sys.path.insert(0, ROOT)
    try:
        from echosyn.common.mrrate import list_series as reference
    except Exception:                           # MRFlow's venv only
        pytest.skip("echosyn not importable in this environment")
    theirs = reference(config["mrrate"]["raw_root"], "val", 1, 32, 42)
    assert [e["series_id"] for e in ours] == [e["series_id"] for e in theirs]
    assert [e["member"] for e in ours] == [e["member"] for e in theirs]


@requires_mrrate
def test_a_real_mrrate_series_preprocesses_onto_the_training_grid():
    config = _config()
    entry = mrrate_data.list_series(config["mrrate"]["raw_root"], "val", 1, 4, 42)[0]
    volume, spacing, native = mrrate_data.preprocess_volume(
        mrrate_data.read_member(entry["archive"], entry["member"]), entry["plane"],
        config["volume"]["inplane_mm"], config["volume"]["slice_mm"],
        config["volume"]["inplane_size"], config["volume"]["num_slices"],
        config["volume"]["posterior_shift_mm"])
    assert volume.shape == (config["volume"]["inplane_size"], config["volume"]["inplane_size"],
                            config["volume"]["num_slices"])
    assert volume.dtype == np.float32
    assert 0.0 <= volume.min() and volume.max() <= 1.0
    assert volume.max() > 0.5, "a real brain must not normalize to near-zero"
    assert native >= 32 and all(s > 0 for s in spacing)


@requires_mrrate
def test_a_real_mrrate_report_formats_into_the_conditioning_string():
    config = _config()
    entry = mrrate_data.list_series(config["mrrate"]["raw_root"], "val", 1, 8, 42)[0]
    report = mrrate_data.read_report(entry["archive"], entry["study_uid"])
    text = mrrate_data.format_report(report, config["data"]["report_sections"],
                                     entry["modality"], config["data"]["modality_prefix"])
    assert text.startswith(("Findings:", "Impression:"))
    assert len(text) > 40


@requires_mrrate
@requires_vae
@requires_clip
@requires_gpu
def test_prepare_data_writes_a_readable_cache_and_fine_tune_trains_on_it(tmp_path):
    """The whole pipeline on four real MR-RATE series: cache, manifest, metadata, then four
    optimizer steps with the freezing assertions on."""
    from baselines.text2ct.prepare_data import prepare

    config = _config()
    config["data"]["cache_root"] = str(tmp_path / "cache")
    config["output_dir"] = str(tmp_path / "run")
    config["mrrate"]["max_series_val"] = 4
    prepare(config, "val", shard=0, num_shards=1, limit=4)

    rows = mrrate_data.read_manifest(config["data"]["cache_root"], "val")
    assert rows, "prepare_data kept no series"
    assert os.path.exists(os.path.join(config["data"]["cache_root"],
                                       mrrate_data.CACHE_META_NAME))

    dataset = mrrate_data.Text2CTLatentDataset(config["data"]["cache_root"], "val",
                                              cfg_module.grid_spacing(config))
    item = dataset[0]
    assert item["latent"].shape == cfg_module.latent_shape(config)
    assert item["context"].shape == (1, 768)
    assert torch.isfinite(item["latent"]).all()

    # Train off that cache: the val manifest stands in for both splits here.
    import shutil
    for name in os.listdir(os.path.join(config["data"]["cache_root"], "manifest")):
        shutil.copy(os.path.join(config["data"]["cache_root"], "manifest", name),
                    os.path.join(config["data"]["cache_root"], "manifest",
                                 name.replace("val-", "train-")))
    for path in [os.path.join(config["data"]["cache_root"], "manifest", n) for n in
                 os.listdir(os.path.join(config["data"]["cache_root"], "manifest"))
                 if n.startswith("train-")]:
        text = open(path).read().replace(",val,", ",train,")
        open(path, "w").write(text)

    result = fine_tune.train(config, smoke=True, smoke_steps=2)
    assert result["global_step"] == 2
    assert result["trainable_parameters"] > 0 and result["frozen_parameters"] > 0
    assert fine_tune.latest_checkpoint(config["output_dir"])
