"""Report encoding with NeuroBERT, and the conversion of an encoded report into model inputs."""

import hashlib
import os

import torch


# NeuroBERT (Wood et al., Radiology: AI 2025): tokenizer and config from the release at a pinned revision.
NEUROBERT_REPO = "davvwood/NeuroBERT"
NEUROBERT_REVISION = "69c974a5a846953179f2737f5ef8d402f4a9e0d5"
NEUROBERT_FORMAT = "neurobert-v4"  # stored in every artifact; bump when format_neurobert changes
# Weights fine-tuned on MR-RATE train reports, pinned by sha256; MRFLOW_NEUROBERT_WEIGHTS sets their path.
NEUROBERT_WEIGHTS = os.environ.get(
    "MRFLOW_NEUROBERT_WEIGHTS",
    "models/neurobert-mlm-rsm/model.safetensors")
NEUROBERT_WEIGHTS_SHA256 = "00dfdb0ce31264dae54ba989ef2d0a578448f852983123442b52eda6dbbffa33"


def neurobert_tag(max_length):
    """What a cached NeuroBERT artifact was encoded with; `text_condition` checks it."""
    return (f"{NEUROBERT_REPO}@{NEUROBERT_REVISION[:8]}+mlm-rsm-{NEUROBERT_WEIGHTS_SHA256[:8]}"
            f"|max{max_length}|mean|{NEUROBERT_FORMAT}")


def load_neurobert(device):
    """Frozen fine-tuned NeuroBERT -> (model, tokenizer).

    Config and tokenizer come from the release snapshot at the pinned revision; the weights are
    the sha256-pinned fine-tuned file (`NEUROBERT_WEIGHTS`), loaded strictly into
    `RobertaModel(config)` with no pooler (the release's pooler was never trained either).
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, RobertaConfig, RobertaModel

    path = snapshot_download(NEUROBERT_REPO, revision=NEUROBERT_REVISION,
                             allow_patterns=["config.json", "vocab.json", "merges.txt",
                                             "tokenizer.json", "tokenizer_config.json",
                                             "special_tokens_map.json"])
    with open(NEUROBERT_WEIGHTS, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != NEUROBERT_WEIGHTS_SHA256:
            raise RuntimeError(f"{NEUROBERT_WEIGHTS} does not match the pinned sha256")
    state = load_file(NEUROBERT_WEIGHTS)
    state = {k: v for k, v in state.items() if k != "embeddings.position_ids"}
    model = RobertaModel(RobertaConfig.from_pretrained(path), add_pooling_layer=False)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, AutoTokenizer.from_pretrained(path)


def format_neurobert(report):
    """MR-RATE report -> NeuroBERT input `findings: ... impression: ...`.

    Lowercased (the vocabulary is lowercase) with whitespace collapsed; empty sections are left out.
    """
    parts = [f"{s}: {' '.join(report[s].split())}"
             for s in ("findings", "impression") if (report.get(s) or "").strip()]
    return " ".join(parts).lower()


def encode_neurobert_tokens(model, tokenizer, report, max_length=512):
    """Report -> `{"neurobert": {"tokens": [n, 768] fp16, "pooled": [768], "n_tokens", "tag"}}`.

    `tokens` is the last hidden state over every position, `<s>` and `</s>` included, cut at
    `max_length` (right truncation keeps `</s>`). `pooled` is the SentenceTransformer embedding --
    the mean of those same states, the representation NeuroBERT's sentence objective trained --
    L2-normalized, since that objective is a cosine. `n_tokens` is the untruncated length.
    """
    text = format_neurobert(report)
    ids = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids.to(model.device)).last_hidden_state[0].float()
    pooled = hidden.mean(0)
    return {"neurobert": {
        "tokens": hidden.half().cpu(),
        "pooled": (pooled / pooled.norm()).cpu(),
        "n_tokens": len(tokenizer(text, verbose=False)["input_ids"]),
        "tag": neurobert_tag(max_length),
    }}


def encode_neurobert_null(model, tokenizer, max_length=512):
    """[max_length, 768]: NeuroBERT over the empty string, padded to the context, every position
    kept -- Sana's null caption, so the denoiser's learned null report starts on the same scale as
    real report tokens."""
    ids = tokenizer("", padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**ids.to(model.device)).last_hidden_state
    return hidden[0].float().cpu()


def text_condition(artifact, sections, max_tokens):
    """A token-sequence artifact -> (tokens [max_tokens, D], mask [max_tokens] bool,
    pooled [len(sections) * D]): the sections' sequences concatenated and zero-padded, and their
    pooled vectors concatenated. NeuroBERT artifacts carry a tag that must match `max_tokens`."""
    for s in sections:
        if "tag" in artifact[s] and artifact[s]["tag"] != neurobert_tag(max_tokens):
            raise ValueError(f"text artifact {artifact[s]['tag']!r} != {neurobert_tag(max_tokens)!r}")
    tokens = torch.cat([artifact[s]["tokens"] for s in sections]).float()
    if len(tokens) > max_tokens:
        raise ValueError(f"{len(tokens)} text tokens > model_max_length {max_tokens}")
    mask = torch.zeros(max_tokens, dtype=torch.bool)
    mask[:len(tokens)] = True
    padded = torch.zeros(max_tokens, tokens.shape[1])
    padded[:len(tokens)] = tokens
    pooled = torch.cat([artifact[s]["pooled"] for s in sections]).float()
    return padded, mask, pooled
