"""Modality/plane class conditioning, null report, condition states and hierarchical CFG.

Run from the repo root with the root on PYTHONPATH:

    PYTHONPATH=. pytest tests/test_conditioning.py -v

The forward-pass tests need a GPU: STDiT's cross-attention goes through xformers'
memory_efficient_attention, which has no CPU kernel. They skip without one.
"""

import csv

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from echosyn.common import sample_condition_states, stamp_label_mapping, check_label_mapping
from echosyn.common.datasets import MRRateLatentBlockDataset
from echosyn.common.models import DiffuserSTDiT
from echosyn.common.mrrate import (CFG_NULL_MODALITY_ID, MODALITY_TO_ID, NUM_MODALITY_CLASSES,
                                   NUM_PLANE_CLASSES, PLANE_TO_ID, label_counts, modality_to_id,
                                   plane_to_id)

gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="xformers has no CPU kernel")

BLOCK, LATENT_C, RES = 4, 4, 8
SERIES = [("T1w", "AXIAL"), ("T2w", "SAGITTAL"), ("FLAIR", "CORONAL"), ("SWI", "AXIAL")]


@pytest.fixture
def dataset_root(tmp_path):
    """A tiny loose-file dataset: 4 series, one per (modality, plane) pair above."""
    root = tmp_path / "data"
    for name in ("latents", "embeddings", "boundary", "manifest"):
        (root / name).mkdir(parents=True)
    for name in ("black", "white"):
        torch.save(torch.zeros(2 * LATENT_C, RES, RES), root / "boundary" / f"{name}.pt")

    rows = []
    for i, (modality, plane) in enumerate(SERIES):
        sid = f"sample{i}"
        torch.save(torch.randn(2 * LATENT_C, 2 * BLOCK, RES, RES).half(),
                   root / "latents" / f"{sid}.pt")
        torch.save(torch.randn(1, 768), root / "embeddings" / f"{sid}.pt")
        rows.append({"sample_id": sid, "split": "train", "modality": modality, "plane": plane,
                     "n_slices": 2 * BLOCK, "latent_path": f"latents/{sid}.pt",
                     "embedding_path": f"embeddings/{sid}.pt", "zip": ""})

    with open(root / "manifest" / "train-0000.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(root)


def make_model(**kwargs):
    args = dict(input_size=(BLOCK, RES, RES), in_channels=2 * LATENT_C, out_channels=LATENT_C,
                patch_size=(2, 2, 2), hidden_size=64, depth=2, num_heads=4, caption_channels=768,
                model_max_length=1, num_modality_classes=NUM_MODALITY_CLASSES,
                num_plane_classes=NUM_PLANE_CLASSES)
    args.update(kwargs)
    return DiffuserSTDiT(**args)


def make_inputs(B, device, modality=1, plane=0):
    return dict(
        x=torch.randn(B, LATENT_C, BLOCK, RES, RES, device=device),
        timestep=torch.full((B,), 0.5, device=device),
        encoder_hidden_states=torch.randn(B, 1, 768, device=device),
        cond_image=torch.randn(B, LATENT_C, BLOCK, RES, RES, device=device),
        modality_id=torch.full((B,), modality, dtype=torch.long, device=device),
        plane_id=torch.full((B,), plane, dtype=torch.long, device=device),
    )


def wake(model, classes=True):
    """Stand in for a trained model.

    A freshly built STDiT is deliberately inert: `initialize_weights` zeroes `final_layer.linear`
    and every `cross_attn.proj`, and `initialize_temporal` zeroes `attn_temp.proj`. So an untouched
    model outputs exactly 0 whatever it is conditioned on, and a conditioning test against it would
    pass without testing anything. Fill those in, plus the class embeddings unless the test is
    specifically about their zero init.
    """
    with torch.no_grad():
        model.model.final_layer.linear.weight.normal_(0, 0.05)
        for block in model.model.blocks:
            block.cross_attn.proj.weight.normal_(0, 0.05)
            block.attn_temp.proj.weight.normal_(0, 0.05)
        model.model.y_embedder.y_embedding.normal_(0, 0.5)
        if classes:
            model.model.modality_embedder.weight.normal_(0, 0.5)
            model.model.plane_embedder.weight.normal_(0, 0.5)
    return model


### 1-3. Dataset output and collation ###


def test_dataset_returns_class_labels(dataset_root):
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK)
    sample = ds[0]
    assert "modality_id" in sample and "plane_id" in sample


def test_labels_are_scalar_longs(dataset_root):
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK)
    for i in range(len(ds)):
        for key in ("modality_id", "plane_id"):
            label = ds[i][key]
            assert label.dtype == torch.long, key
            assert label.shape == (), f"{key} must be a scalar, got {tuple(label.shape)}"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_default_collation_gives_batched_longs(dataset_root, num_workers):
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK)
    loader = DataLoader(ds, batch_size=4, num_workers=num_workers)
    batch = next(iter(loader))
    for key in ("modality_id", "plane_id"):
        assert batch[key].shape == (4,), f"{key}: {tuple(batch[key].shape)}"
        assert batch[key].dtype == torch.long


### 4-6. Mapping stability and manifest counts ###


def test_mapping_is_stable_and_null_is_reserved():
    # Literal ids, so these values are the contract every checkpoint is trained against.
    assert MODALITY_TO_ID == {"CFG_NULL": 0, "T1w": 1, "T2w": 2, "FLAIR": 3, "SWI": 4,
                             "MRA": 5, "UNKNOWN": 6}
    assert PLANE_TO_ID == {"AXIAL": 0, "SAGITTAL": 1, "CORONAL": 2}
    # UNKNOWN is a real dataset category and must stay distinct from the guidance null.
    assert MODALITY_TO_ID["UNKNOWN"] != CFG_NULL_MODALITY_ID
    assert CFG_NULL_MODALITY_ID == 0


def test_mapping_is_identical_across_splits_and_workers(dataset_root):
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK)
    single = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    forked = next(iter(DataLoader(ds, batch_size=4, num_workers=2)))
    assert torch.equal(single["modality_id"], forked["modality_id"])
    assert torch.equal(single["plane_id"], forked["plane_id"])
    assert single["modality_id"].tolist() == [modality_to_id(m) for m, _ in SERIES]


def test_aliases_resolve_and_unknown_labels_fail():
    assert modality_to_id("t1w") == modality_to_id("T1w") == 1
    assert modality_to_id("swan") == modality_to_id("SWI")
    # `obl` is in the challenge's vocabulary but not MR-RATE's; it falls back to axial, which is
    # what plane_order does with it, so the label and the array layout agree.
    assert plane_to_id("obl") == plane_to_id("AXIAL")
    with pytest.raises(KeyError, match="unmapped modality"):
        modality_to_id("T1rho")
    with pytest.raises(KeyError, match="unmapped plane"):
        plane_to_id("diagonal")


def test_manifest_counts(dataset_root):
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK)
    modality, plane, pair = label_counts(ds.rows)
    assert modality == {"T1w": 1, "T2w": 1, "FLAIR": 1, "SWI": 1}
    assert plane == {"AXIAL": 2, "SAGITTAL": 1, "CORONAL": 1}
    assert pair[("SWI", "AXIAL")] == 1 and len(pair) == 4


### 7-8. Embeddings ###


def test_embedding_shapes_and_zero_init():
    model = make_model()
    assert model.model.modality_embedder.weight.shape == (NUM_MODALITY_CLASSES, 64)
    assert model.model.plane_embedder.weight.shape == (NUM_PLANE_CLASSES, 64)
    # Zero-init: a CTFlow checkpoint predicts exactly as it did before these existed.
    assert (model.model.modality_embedder.weight == 0).all()
    assert (model.model.plane_embedder.weight == 0).all()


@gpu_only
def test_zero_init_makes_class_conditioning_a_no_op():
    """The checkpoint-compatibility guarantee: with the embeddings at their zero init, the class
    labels cannot change the prediction, so a CTFlow checkpoint predicts exactly what it used to."""
    model = wake(make_model(), classes=False).cuda().eval()
    inputs = make_inputs(2, "cuda")
    with torch.no_grad():
        baseline = model(**inputs).sample
        assert baseline.abs().sum() > 0, "wake() failed: the model is still inert"
        for modality, plane in ((0, 0), (4, 2), (6, 1)):
            other = {**inputs,
                     "modality_id": torch.full((2,), modality, dtype=torch.long, device="cuda"),
                     "plane_id": torch.full((2,), plane, dtype=torch.long, device="cuda")}
            assert torch.equal(model(**other).sample, baseline), \
                f"class ids changed the output at zero init ({modality}, {plane})"


@gpu_only
def test_gradients_reach_the_selected_rows():
    model = wake(make_model()).cuda()
    inputs = make_inputs(2, "cuda", modality=MODALITY_TO_ID["FLAIR"], plane=PLANE_TO_ID["CORONAL"])
    model(**inputs).sample.square().mean().backward()

    for embedder, used in ((model.model.modality_embedder, MODALITY_TO_ID["FLAIR"]),
                           (model.model.plane_embedder, PLANE_TO_ID["CORONAL"])):
        grad = embedder.weight.grad
        assert grad[used].abs().sum() > 0, "the selected row got no gradient"
        others = [i for i in range(grad.shape[0]) if i != used]
        assert grad[others].abs().sum() == 0, "an unselected row got a gradient"


### 9-11. Condition states ###


def test_condition_states_produce_the_right_ids_and_masks():
    config = OmegaConf.create({"condition_states": {"full": 0.8, "modality_only": 0.1,
                                                    "semantic_null": 0.1}})
    real = torch.full((20000,), MODALITY_TO_ID["T2w"], dtype=torch.long)
    modality_id, report_drop = sample_condition_states(config, real)

    # Exactly three states occur, and each is internally consistent.
    is_null = modality_id == CFG_NULL_MODALITY_ID
    full = ~report_drop & ~is_null
    modality_only = report_drop & ~is_null
    semantic_null = report_drop & is_null
    assert (full | modality_only | semantic_null).all(), "a sample is in no valid state"
    # A nulled modality always implies a dropped report; the reverse is not required.
    assert not (is_null & ~report_drop).any()

    for mask, target in ((full, 0.80), (modality_only, 0.10), (semantic_null, 0.10)):
        rate = mask.float().mean().item()
        assert abs(rate - target) < 0.015, f"{target} state realized at {rate}"


def test_condition_state_probabilities_must_sum_to_one():
    config = OmegaConf.create({"condition_states": {"full": 0.8, "modality_only": 0.1,
                                                    "semantic_null": 0.2}})
    with pytest.raises(AssertionError, match="must sum to 1"):
        sample_condition_states(config, torch.ones(4, dtype=torch.long))


def test_plane_is_never_dropped(dataset_root):
    """The state sampler only ever touches modality: plane is geometry, not semantics."""
    config = OmegaConf.create({"condition_states": {"full": 0.0, "modality_only": 0.0,
                                                    "semantic_null": 1.0}})
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    before = batch["plane_id"].clone()
    modality_id, _ = sample_condition_states(config, batch["modality_id"])
    assert (modality_id == CFG_NULL_MODALITY_ID).all()          # every modality nulled
    assert torch.equal(batch["plane_id"], before)               # plane untouched
    # And there is no null plane class to drop to in the first place.
    assert "CFG_NULL" not in PLANE_TO_ID


### 10. Forced report dropout ###


@gpu_only
def test_forced_report_drop_selects_the_learned_null_on_the_right_device():
    model = wake(make_model()).cuda().eval()
    embedder = model.model.y_embedder
    # A Parameter, not a buffer: buffers are skipped by the optimizer and by EMA saves.
    assert isinstance(embedder.y_embedding, torch.nn.Parameter)
    assert "model.y_embedder.y_embedding" in dict(model.named_parameters())

    with torch.no_grad():
        embedder.y_embedding.copy_(torch.full_like(embedder.y_embedding, 3.0))

    inputs = make_inputs(4, "cuda")
    kept = model(**inputs, force_drop_ids=torch.zeros(4, dtype=torch.bool, device="cuda")).sample
    dropped = model(**inputs, force_drop_ids=torch.ones(4, dtype=torch.bool, device="cuda")).sample
    assert not torch.allclose(kept, dropped), "force_drop_ids had no effect"

    # Per-sample selection: samples 0 and 2 dropped, 1 and 3 kept.
    mask = torch.tensor([True, False, True, False], device="cuda")
    mixed = model(**inputs, force_drop_ids=mask).sample
    assert torch.allclose(mixed[0], dropped[0], atol=1e-5)
    assert torch.allclose(mixed[1], kept[1], atol=1e-5)
    assert torch.allclose(mixed[2], dropped[2], atol=1e-5)
    assert torch.allclose(mixed[3], kept[3], atol=1e-5)


@gpu_only
def test_null_report_is_not_the_none_path():
    """`encoder_hidden_states=None` skips cross-attention entirely -- a different, untrained
    function. The learned null must not be confused with it."""
    model = wake(make_model()).cuda().eval()
    inputs = make_inputs(2, "cuda")
    dropped = model(**inputs, force_drop_ids=torch.ones(2, dtype=torch.bool, device="cuda")).sample
    none_path = model(**{**inputs, "encoder_hidden_states": None}).sample
    assert not torch.allclose(dropped, none_path)


### 12-14. Hierarchical CFG ###


def guided(model, inputs, s_mod, s_rep, batched):
    """The three-branch guided velocity, computed either batched or sequentially."""
    B = inputs["x"].shape[0]
    device = inputs["x"].device
    null = torch.full((B,), CFG_NULL_MODALITY_ID, dtype=torch.long, device=device)
    branches = [(null, True), (inputs["modality_id"], True), (inputs["modality_id"], False)]

    if batched:
        K = 3
        drop = torch.tensor([True, True, False], device=device).repeat_interleave(B)
        out = model(
            x=inputs["x"].repeat(K, 1, 1, 1, 1), timestep=inputs["timestep"].repeat(K),
            encoder_hidden_states=inputs["encoder_hidden_states"].repeat(K, 1, 1),
            cond_image=inputs["cond_image"].repeat(K, 1, 1, 1, 1),
            modality_id=torch.cat([m for m, _ in branches]),
            plane_id=inputs["plane_id"].repeat(K), force_drop_ids=drop,
        ).sample
        v_base, v_mod, v_full = out.chunk(K)
    else:
        preds = [model(**{**inputs, "modality_id": m},
                       force_drop_ids=torch.full((B,), d, dtype=torch.bool, device=device)).sample
                 for m, d in branches]
        v_base, v_mod, v_full = preds

    return v_base + s_mod * (v_mod - v_base) + s_rep * (v_full - v_mod), v_full


@gpu_only
def test_batched_cfg_matches_sequential_calls():
    torch.manual_seed(0)
    model = wake(make_model()).cuda().eval()
    inputs = make_inputs(2, "cuda")
    with torch.no_grad():
        batched, _ = guided(model, inputs, 2.5, 1.5, batched=True)
        sequential, _ = guided(model, inputs, 2.5, 1.5, batched=False)
    assert batched.abs().sum() > 0, "wake() failed: the model is still inert"
    assert torch.allclose(batched, sequential, atol=1e-4), \
        f"max diff {(batched - sequential).abs().max().item()}"


@gpu_only
def test_cfg_scales_of_one_recover_the_full_prediction():
    torch.manual_seed(0)
    model = wake(make_model()).cuda().eval()
    inputs = make_inputs(2, "cuda")
    with torch.no_grad():
        v_guided, v_full = guided(model, inputs, 1.0, 1.0, batched=True)
    assert v_full.abs().sum() > 0, "wake() failed: the model is still inert"
    assert torch.allclose(v_guided, v_full, atol=1e-5), \
        f"max diff {(v_guided - v_full).abs().max().item()}"

    # And the branches must genuinely differ, or the identity above is vacuous.
    with torch.no_grad():
        other, _ = guided(model, inputs, 3.0, 2.0, batched=True)
    assert not torch.allclose(other, v_full, atol=1e-3), "guidance had no effect at all"


@gpu_only
def test_modality_term_cannot_see_the_report():
    """`v_modality - v_base` is the modality guidance term, and the report is dropped in *both*
    branches -- so no report text can reach it, including the `[MODALITY]/[PLANE]` prefix the
    embeddings carry. Proven by swapping the report for a completely different one: with
    force_drop_ids set, the caption is replaced by the learned null and the input is ignored."""
    model = wake(make_model()).cuda().eval()
    inputs = make_inputs(2, "cuda")
    other = {**inputs, "encoder_hidden_states": torch.randn(2, 1, 768, device="cuda")}
    dropped = torch.ones(2, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        for name, modality in (("base", CFG_NULL_MODALITY_ID), ("modality", MODALITY_TO_ID["T1w"])):
            ids = torch.full((2,), modality, dtype=torch.long, device="cuda")
            a = model(**{**inputs, "modality_id": ids}, force_drop_ids=dropped).sample
            b = model(**{**other, "modality_id": ids}, force_drop_ids=dropped).sample
            assert torch.equal(a, b), f"the {name} branch still depends on the report text"
        # And the report does matter when it is not dropped, or the check above is vacuous.
        kept = torch.zeros(2, dtype=torch.bool, device="cuda")
        assert not torch.allclose(model(**inputs, force_drop_ids=kept).sample,
                                  model(**other, force_drop_ids=kept).sample)


@gpu_only
def test_cond_image_and_plane_are_identical_across_branches():
    """The three branches may differ only in modality and report -- everything else must be the
    same tensor repeated, or the guidance differences measure the wrong thing."""
    model = wake(make_model()).cuda().eval()
    seen = {}

    original = model.model.forward

    def spy(x, timestep, y=None, mask=None, modality_id=None, plane_id=None, force_drop_ids=None):
        seen["x"], seen["plane_id"], seen["timestep"] = x, plane_id, timestep
        return original(x, timestep, y, mask, modality_id, plane_id, force_drop_ids)

    model.model.forward = spy
    inputs = make_inputs(2, "cuda", plane=PLANE_TO_ID["SAGITTAL"])
    with torch.no_grad():
        guided(model, inputs, 3.0, 2.0, batched=True)

    # x here is cat([latent, cond_image]) -- one tensor covers both.
    for key in ("x", "plane_id", "timestep"):
        chunks = seen[key].chunk(3)
        for chunk in chunks[1:]:
            assert torch.equal(chunks[0], chunk), f"{key} differs across CFG branches"


### 15. No spacing conditioning ###


def test_no_spacing_input_is_required():
    """Every preprocessed voxel is 1 mm, so spacing is constant and carries no information."""
    import inspect
    for fn in (DiffuserSTDiT.forward, make_model().model.forward):
        assert not any("spacing" in p for p in inspect.signature(fn).parameters)
    assert not any("spacing" in n for n, _ in make_model().named_parameters())


### 16. Checkpoint compatibility ###


def test_init_from_permits_only_the_new_class_keys(tmp_path):
    from safetensors.torch import save_file
    from echosyn.common import load_init_weights

    class Log:
        def info(self, *a):
            pass

        def warning(self, *a):
            pass

    old = {k: v for k, v in make_model().state_dict().items()
           if "modality_embedder" not in k and "plane_embedder" not in k}
    path = tmp_path / "old.safetensors"
    save_file(old, str(path))

    # A CTFlow-shaped checkpoint: the two class tensors are missing, everything else matches.
    model = make_model()
    load_init_weights(OmegaConf.create({"init_from": str(path)}), model, Log())

    # An unrelated missing key must fail instead of training a partly random model.
    broken = dict(old)
    broken.pop("model.final_layer.linear.weight")
    broken_path = tmp_path / "broken.safetensors"
    save_file(broken, str(broken_path))
    with pytest.raises(RuntimeError, match="does not match the config"):
        load_init_weights(OmegaConf.create({"init_from": str(broken_path)}), make_model(), Log())

    # So must an unexpected key.
    extra = dict(old)
    extra["model.not_a_real_tensor"] = torch.zeros(3)
    extra_path = tmp_path / "extra.safetensors"
    save_file(extra, str(extra_path))
    with pytest.raises(RuntimeError, match="does not match the config"):
        load_init_weights(OmegaConf.create({"init_from": str(extra_path)}), make_model(), Log())


def test_label_mapping_round_trips_through_the_config():
    config = OmegaConf.create({"denoiser": {"args": {
        "num_modality_classes": NUM_MODALITY_CLASSES, "num_plane_classes": NUM_PLANE_CLASSES}}})
    stamp_label_mapping(config)
    assert config.labels.cfg_null_modality_id == CFG_NULL_MODALITY_ID
    assert config.labels.num_modality_classes == NUM_MODALITY_CLASSES
    check_label_mapping(config)

    # A moved id must be caught, not silently used to condition on the wrong sequence.
    config.labels.modality_to_id["T1w"] = 99
    with pytest.raises(RuntimeError, match="label mapping changed"):
        check_label_mapping(config)


def test_only_deterministic_buffers_remain():
    """Conditioning state must never live in a buffer: EMAModel.save_pretrained rebuilds the model
    from its config and copies only parameters, so buffers are re-randomized at every save."""
    model = make_model()
    assert {n for n, _ in model.named_buffers()} == {"model.pos_embed", "model.pos_embed_temporal"}


### 17. Terminal marker supervision ###


def test_terminal_transitions_are_sampled(dataset_root):
    """The stop token is learned, not a length head: the dataset must show the model both the
    black -> first-block seed and the last-block -> white terminal transition."""
    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK,
                                  p_start=0.0, p_end=1.0)
    sample = ds[0]
    assert torch.equal(sample["video"], ds.white.float()), "no anatomy -> white transition"
    assert sample["video"].shape[1] == BLOCK, "terminal block must be a full block"

    ds = MRRateLatentBlockDataset(root=dataset_root, split="train", block_size=BLOCK,
                                  p_start=1.0, p_end=0.0)
    assert torch.equal(ds[0]["image"], ds.black.float()), "no black -> first-block seed"
