"""NeuroBERT token conditioning: encoder, report format, artifacts, dataset, denoiser, one step.

    PYTHONPATH=. HF_HUB_OFFLINE=1 pytest tests/test_neurobert_conditioning.py -v

Loads the pinned davvwood/NeuroBERT from the HF cache. The denoiser tests need a GPU (STDiT's
cross-attention is xformers, no CPU kernel) and skip without one.
"""

import csv
import types

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from auto_regressive_generate import LatentAutoregressiveGenerator
from echosyn.common import load_init_weights
from echosyn.common.datasets import MRRateLatentBlockDataset
from echosyn.common.models import DiffuserSTDiT
from echosyn.common.mrrate import (EMBEDDING_MANIFEST_FIELDS, NUM_MODALITY_CLASSES,
                                   NUM_PLANE_CLASSES, encode_neurobert_null,
                                   encode_neurobert_tokens, format_neurobert, load_neurobert,
                                   neurobert_tag, text_condition)

gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="xformers has no CPU kernel")

L, D = 512, 768
BLOCK, LATENT_C, RES = 4, 4, 8

REPORT = {
    "findings": "There is restricted diffusion in the LEFT paramedian pons, 12 mm, in keeping "
                "with an acute infarct.\nNo haemorrhage.\t The ventricles are of normal size.",
    "impression": "— Acute left pontine infarct.\n— No hydrocephalus.",
}
NO_IMPRESSION = {"findings": "Normal intracranial appearances.", "impression": ""}
EMPTY = {"findings": "", "impression": None}
LONG = {"findings": " ".join(["Multiple chronic ischaemic gliotic foci are seen in the bilateral "
                              "periventricular and subcortical white matter."] * 60),
        "impression": "Small vessel disease."}


@pytest.fixture(scope="module")
def neurobert():
    return load_neurobert("cpu")


def encode(neurobert, report):
    return encode_neurobert_tokens(*neurobert, report, L)["neurobert"]


### 1. Checkpoint and tokenizer ###


def test_checkpoint_and_tokenizer_load_frozen(neurobert):
    model, tokenizer = neurobert
    assert type(model).__name__ == "RobertaModel" and model.pooler is None
    cfg = model.config
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) == (768, 6, 10000)
    assert cfg.max_position_embeddings - cfg.pad_token_id - 1 == L
    assert (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) == (0, 1, 2)
    assert tokenizer.padding_side == tokenizer.truncation_side == "right"
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_vocabulary_is_lowercase(neurobert):
    """Why the format lowercases: a capital letter is a lone byte token."""
    tokenizer = neurobert[1]
    assert len(tokenizer.tokenize(" left")) == 1
    assert len(tokenizer.tokenize(" Left")) > 2


### 2. Report format ###


def test_format_keeps_the_report_and_its_sections():
    text = format_neurobert(REPORT)
    assert text == ("findings: there is restricted diffusion in the left paramedian pons, 12 mm, in "
                    "keeping with an acute infarct. no haemorrhage. the ventricles are of normal "
                    "size. impression: — acute left pontine infarct. — no hydrocephalus.")
    assert text == text.lower() and "\n" not in text and "\t" not in text


def test_missing_sections_are_left_out():
    assert format_neurobert(NO_IMPRESSION) == "findings: normal intracranial appearances."
    assert format_neurobert({"impression": "Normal."}) == "impression: normal."
    assert format_neurobert(EMPTY) == ""


### 3. Artifacts: shapes, pooling, mask, padding, truncation ###


def test_artifact_shapes_and_pooling(neurobert):
    art = encode(neurobert, REPORT)
    n = art["n_tokens"]
    assert art["tokens"].shape == (n, D) and art["tokens"].dtype == torch.float16
    assert art["pooled"].shape == (D,) and art["pooled"].dtype == torch.float32
    assert torch.allclose(art["pooled"].norm(), torch.tensor(1.0))
    # The pooled vector is SentenceTransformer's: the mean over every token, <s> and </s> included.
    assert F.cosine_similarity(art["tokens"].float().mean(0), art["pooled"], dim=0) > 0.9999
    assert art["tag"] == neurobert_tag(L)


def test_text_condition_masks_exactly_the_report(neurobert):
    art = encode(neurobert, REPORT)
    n = art["n_tokens"]
    tokens, mask, pooled = text_condition({"neurobert": art}, ["neurobert"], L)
    assert tokens.shape == (L, D) and mask.shape == (L,) and mask.dtype == torch.bool
    assert mask[:n].all() and not mask[n:].any()
    assert torch.equal(tokens[:n], art["tokens"].float()) and not tokens[n:].any()
    assert torch.equal(pooled, art["pooled"])


def test_encoder_is_padding_invariant(neurobert):
    """A padded batch gives the single-sequence states on the valid tokens, and its masked mean."""
    model, tokenizer = neurobert
    texts = [format_neurobert(REPORT), format_neurobert(NO_IMPRESSION)]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    for i, report in enumerate((REPORT, NO_IMPRESSION)):
        art = encode(neurobert, report)
        n = int(batch["attention_mask"][i].sum())
        assert n == art["n_tokens"]
        assert torch.allclose(hidden[i, :n].half().float(), art["tokens"].float(), atol=2e-3)
        mean = hidden[i, :n].mean(0)
        assert torch.allclose(mean / mean.norm(), art["pooled"], atol=1e-5)


def test_truncation_keeps_the_end_token_and_records_the_length(neurobert):
    tokenizer = neurobert[1]
    art = encode(neurobert, LONG)
    assert art["n_tokens"] > L and art["tokens"].shape == (L, D)
    ids = tokenizer(format_neurobert(LONG), truncation=True, max_length=L)["input_ids"]
    assert len(ids) == L and ids[0] == tokenizer.bos_token_id and ids[-1] == tokenizer.eos_token_id
    text_condition({"neurobert": art}, ["neurobert"], L)  # exactly full, still fits


def test_caches_cannot_be_mixed(neurobert):
    art = {"neurobert": encode(neurobert, REPORT)}
    with pytest.raises(ValueError, match="text artifact"):
        text_condition(art, ["neurobert"], 256)  # encoded for a different context
    hlip = {"findings": {"tokens": torch.zeros(3, D), "pooled": torch.zeros(D)}}
    with pytest.raises(KeyError):
        text_condition(hlip, ["neurobert"], L)


def test_empty_report_is_the_two_special_tokens(neurobert):
    art = encode(neurobert, EMPTY)
    assert art["n_tokens"] == 2 and art["tokens"].shape == (2, D)
    assert torch.isfinite(art["tokens"]).all() and torch.isfinite(art["pooled"]).all()


### 4. Determinism and precision ###


def test_eval_mode_is_deterministic(neurobert):
    a, b = encode(neurobert, REPORT), encode(neurobert, REPORT)
    assert torch.equal(a["tokens"], b["tokens"]) and torch.equal(a["pooled"], b["pooled"])


def test_fp16_storage_is_finite_and_in_range(neurobert):
    model, tokenizer = neurobert
    for report in (REPORT, LONG, EMPTY):
        ids = tokenizer(format_neurobert(report), truncation=True, max_length=L,
                        return_tensors="pt")
        with torch.no_grad():
            hidden = model(**ids).last_hidden_state
        assert hidden.abs().max() < 1e3
        assert torch.isfinite(encode(neurobert, report)["tokens"]).all()


def test_null_text_is_the_padded_empty_string(neurobert):
    null = encode_neurobert_null(*neurobert, L)
    assert null.shape == (L, D) and torch.isfinite(null).all()
    assert torch.allclose(null[:2].half().float(), encode(neurobert, EMPTY)["tokens"].float(),
                          atol=2e-3)


### 5. Dataset ###


@pytest.fixture
def roots(tmp_path, neurobert):
    """Two series with latents, and their two studies' NeuroBERT artifacts in their own tree."""
    root, emb = tmp_path / "data", tmp_path / "emb"
    for d in (root / "latents", root / "manifest", root / "boundary", emb / "embeddings",
              emb / "manifest"):
        d.mkdir(parents=True)
    for name in ("black", "white"):
        torch.save(torch.zeros(2 * LATENT_C, RES, RES), root / "boundary" / f"{name}.pt")
    rows, emb_rows = [], []
    for i, report in enumerate((REPORT, NO_IMPRESSION)):
        sid = f"sample{i}"
        torch.save(torch.randn(2 * LATENT_C, 2 * BLOCK, RES, RES).half(),
                   root / "latents" / f"{sid}.pt")
        torch.save(encode_neurobert_tokens(*neurobert, report, L), emb / "embeddings" / f"s{i}.pt")
        rows.append({"sample_id": sid, "split": "train", "modality": "T1w", "plane": "AXIAL",
                     "n_slices": 2 * BLOCK, "latent_path": f"latents/{sid}.pt",
                     "embedding_path": "", "zip": ""})
        emb_rows.append({"sample_id": sid, "split": "train",
                         "embedding_path": f"embeddings/s{i}.pt", "zip": ""})
    for path, fields, data in ((root / "manifest" / "train-0000.csv", list(rows[0]), rows),
                               (emb / "manifest" / "train-0000.csv", EMBEDDING_MANIFEST_FIELDS,
                                emb_rows)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    return str(root), str(emb)


def make_dataset(roots, **kwargs):
    return MRRateLatentBlockDataset(roots[0], "train", block_size=BLOCK, p_start=0, p_end=0,
                                    embedding_root=roots[1], text_sections=["neurobert"],
                                    max_text_tokens=L, **kwargs)


def test_dataset_batches_tokens_mask_and_pooled(roots, neurobert):
    batch = next(iter(DataLoader(make_dataset(roots), batch_size=2, shuffle=False)))
    assert batch["embedding"].shape == (2, L, D)
    assert batch["embedding_mask"].shape == (2, L) and batch["embedding_mask"].dtype == torch.bool
    assert batch["pooled"].shape == (2, D)
    lengths = [encode(neurobert, r)["n_tokens"] for r in (REPORT, NO_IMPRESSION)]
    assert batch["embedding_mask"].sum(1).tolist() == lengths


def test_global_condition_none_leaves_pooled_out(roots):
    assert "pooled" not in make_dataset(roots, global_condition="none")[0]


### 6. Denoiser ###


def make_model(**kwargs):
    args = dict(input_size=(BLOCK, RES, RES), in_channels=2 * LATENT_C, out_channels=LATENT_C,
                patch_size=(2, 2, 2), hidden_size=64, depth=2, num_heads=4, caption_channels=D,
                model_max_length=L, pooled_channels=D, qk_norm=True, y_norm=True,
                y_norm_scale_factor=0.01, num_modality_classes=NUM_MODALITY_CLASSES,
                num_plane_classes=NUM_PLANE_CLASSES)
    args.update(kwargs)
    return DiffuserSTDiT(**args)


def wake(model):
    """A fresh STDiT outputs exactly 0 (final layer, cross-attention and temporal projections are
    zero-init); fill them in so conditioning tests test something."""
    with torch.no_grad():
        model.model.final_layer.linear.weight.normal_(0, 0.05)
        for block in model.model.blocks:
            block.cross_attn.proj.weight.normal_(0, 0.05)
            block.attn_temp.proj.weight.normal_(0, 0.05)
        model.model.pooled_embedder.weight.normal_(0, 0.05)
        model.model.caption_norm.weight.fill_(1.0)
    return model


def text_inputs(neurobert, reports, device):
    conds = [text_condition(encode_neurobert_tokens(*neurobert, r, L), ["neurobert"], L)
             for r in reports]
    tokens, mask, pooled = (torch.stack(x).to(device) for x in zip(*conds))
    return {"encoder_hidden_states": tokens, "mask": mask, "pooled": pooled}


def inputs(B, device):
    return dict(x=torch.randn(B, LATENT_C, BLOCK, RES, RES, device=device),
                timestep=torch.full((B,), 0.5, device=device),
                cond_image=torch.randn(B, LATENT_C, BLOCK, RES, RES, device=device),
                modality_id=torch.ones(B, dtype=torch.long, device=device),
                plane_id=torch.zeros(B, dtype=torch.long, device=device))


@gpu_only
def test_denoiser_forward_bf16_is_finite(neurobert):
    model = wake(make_model()).cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs(2, "cuda"), **text_inputs(neurobert, [REPORT, LONG], "cuda")).sample
    assert out.shape == (2, LATENT_C, BLOCK, RES, RES) and torch.isfinite(out).all()


@gpu_only
def test_padded_rows_are_invisible_to_cross_attention(neurobert):
    model = wake(make_model()).cuda().eval()
    x, text = inputs(1, "cuda"), text_inputs(neurobert, [REPORT], "cuda")
    garbage = dict(text, encoder_hidden_states=torch.where(
        text["mask"][..., None], text["encoder_hidden_states"], 100.0))
    with torch.no_grad():
        assert torch.equal(model(**x, **text).sample, model(**x, **garbage).sample)


@gpu_only
def test_report_changes_the_prediction(neurobert):
    model = wake(make_model()).cuda().eval()
    x = inputs(1, "cuda")
    with torch.no_grad():
        a = model(**x, **text_inputs(neurobert, [REPORT], "cuda")).sample
        b = model(**x, **text_inputs(neurobert, [NO_IMPRESSION], "cuda")).sample
    assert (a - b).abs().max() > 1e-4


@gpu_only
def test_pooled_must_match_pooled_channels(neurobert):
    x, text = inputs(1, "cuda"), text_inputs(neurobert, [REPORT], "cuda")
    with torch.no_grad(), pytest.raises(AssertionError, match="pooled"):
        make_model().cuda().eval()(**x, **{k: v for k, v in text.items() if k != "pooled"})
    with torch.no_grad(), pytest.raises(AssertionError, match="pooled"):
        make_model(pooled_channels=0).cuda().eval()(**x, **text)
    with torch.no_grad():  # none: no pooled, no pooled_embedder
        make_model(pooled_channels=0).cuda().eval()(
            **x, **{k: v for k, v in text.items() if k != "pooled"})


@gpu_only
def test_null_report_drops_tokens_and_pooled_together(neurobert):
    model = wake(make_model()).cuda().eval()
    model.model.y_embedder.y_embedding.data.copy_(encode_neurobert_null(*neurobert, L))
    x, drop = inputs(1, "cuda"), torch.ones(1, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        a = model(**x, **text_inputs(neurobert, [REPORT], "cuda"), force_drop_ids=drop).sample
        b = model(**x, **text_inputs(neurobert, [LONG], "cuda"), force_drop_ids=drop).sample
        c = model(**x, **text_inputs(neurobert, [REPORT], "cuda")).sample
    assert torch.equal(a, b) and not torch.equal(a, c)


@gpu_only
def test_hierarchical_cfg_with_token_conditioning(neurobert):
    """The generator's three-branch velocity, at scales 1, equals the plain conditional call."""
    model = wake(make_model()).cuda().eval()
    x = inputs(1, "cuda")
    text = text_inputs(neurobert, [REPORT], "cuda")
    gen = types.SimpleNamespace(denoiser=model, modality_cfg_scale=1.0, report_cfg_scale=1.0)
    args = (x["timestep"], x["x"], text, x["cond_image"], x["modality_id"], x["plane_id"])
    with torch.no_grad():
        single = LatentAutoregressiveGenerator.velocity(gen, *args)
        gen.modality_cfg_scale = gen.report_cfg_scale = 1.0 + 1e-9  # forces the batched path
        batched = LatentAutoregressiveGenerator.velocity(gen, *args)
    assert torch.allclose(single, batched, atol=1e-4)


@gpu_only
def test_one_training_step(neurobert):
    """train.py's step: frozen encoder untouched and gradient-free, text adapters and trunk move."""
    encoder = neurobert[0]
    before = {n: p.clone() for n, p in encoder.named_parameters()}
    # A fresh STDiT is inert (final_layer and every cross_attn.proj zero-init), so on its first
    # step nothing upstream of the output gets a gradient. Wake only those two, not the text
    # adapters under test, so the zero-init pooled_embedder still has to leave zero below.
    model = make_model().cuda().train()
    with torch.no_grad():
        model.model.final_layer.linear.weight.normal_(0, 0.05)
        for block in model.model.blocks:
            block.cross_attn.proj.weight.normal_(0, 0.05)
    model.model.y_embedder.y_embedding.data.copy_(encode_neurobert_null(*neurobert, L))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = {n: p.detach().clone() for n, p in model.named_parameters()}

    text = text_inputs(neurobert, [REPORT, NO_IMPRESSION], "cuda")
    assert not any(v.requires_grad for v in text.values())
    x = inputs(2, "cuda")
    t = torch.rand(2, device="cuda").view(-1, 1, 1, 1, 1)
    z_0, z_1 = torch.randn_like(x["x"]), torch.randn_like(x["x"])
    x["x"], x["timestep"] = (1 - t) * z_0 + t * z_1, t.flatten()
    drop = torch.tensor([False, True], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.mse_loss(model(**x, **text, force_drop_ids=drop).sample, z_1 - z_0)
    loss.backward()
    assert torch.isfinite(loss)
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in grads.values())
    for name in ("model.y_embedder.y_proj.fc1.weight", "model.pooled_embedder.weight",
                 "model.pooled_embedder.bias", "model.caption_norm.weight",
                 "model.y_embedder.y_embedding", "model.blocks.0.cross_attn.kv_linear.weight"):
        assert grads[name].abs().sum() > 0, name
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.equal(p, before[n]) for n, p in encoder.named_parameters())
    assert model.model.pooled_embedder.weight.abs().sum() > 0  # left its zero init
    moved = [n for n, p in model.named_parameters() if not torch.equal(p, start[n])]
    assert "model.y_embedder.y_proj.fc1.weight" in moved and "model.blocks.0.mlp.fc1.weight" in moved


### 7. Warm start from an HLIP-tokens checkpoint ###


def test_init_from_hlip_tokens_keeps_the_trunk_and_reinits_the_text_adapters(tmp_path):
    hlip = make_model(model_max_length=256)
    with torch.no_grad():
        for p in hlip.parameters():
            p.normal_()
    save_file({k: v.contiguous() for k, v in hlip.state_dict().items()},
              str(tmp_path / "diffusion_pytorch_model.safetensors"))
    logger = types.SimpleNamespace(info=lambda *a: None)
    skip = ["model.y_embedder.", "model.caption_norm.", "model.pooled_embedder."]

    with pytest.raises(RuntimeError):  # the null report's shape differs: 256 vs 512 tokens
        load_init_weights(OmegaConf.create({"init_from": str(tmp_path)}), make_model(), logger)

    model = make_model()
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    load_init_weights(OmegaConf.create({"init_from": str(tmp_path), "init_from_skip": skip}),
                      model, logger)
    hlip_state = hlip.state_dict()
    for k, v in model.state_dict().items():
        if k.startswith(tuple(skip)):
            assert torch.equal(v, fresh[k]), k
        elif k in hlip_state:
            assert torch.equal(v, hlip_state[k]), k
    assert not model.model.pooled_embedder.weight.any()  # zero init: a no-op at step 0
