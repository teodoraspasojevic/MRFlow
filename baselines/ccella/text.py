"""CCELLA's report representation, with MR-RATE's sections mapped into it.

**Traced from upstream and preserved exactly** (`scripts/data_processing/gen_json_maisi_merged.py`):

    encoder            google/flan-t5-xxl, `T5EncoderModel.from_pretrained`, frozen, eval, no grad
    tokenizer          `T5Tokenizer.from_pretrained(..., truncation_side="left")`
    call               `encode_plus(text, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=512)` -- see the note in
                       `encode_report` on transformers v5 removing `encode_plus`
    representation     `model(input_ids, attention_mask).last_hidden_state.squeeze(0)` -> [512, 4096]
    empty report       encoded as the empty string, and the sample's `text_isnull` is set

The encoder is never trained here and is not even resident during training: like upstream, the
embedding is computed once offline and cached. Only the dtype differs -- stored fp16 rather than
fp32, halving 330 GB to 165 GB, and cast back to float on load. FLAN-T5's hidden states sit well
inside fp16 range; the cast is the same one MRFlow already makes for its own cached latents.

**What goes into the string, and what deliberately does not.**

MR-RATE stores a report as five sections (`report`, `clinical_information`, `technique`,
`findings`, `impression`). Upstream feeds "the plaintext radiology report for that exam" -- one
string, no markers. The smallest faithful mapping is therefore a plain join of the two sections
that describe the study, with no bracketed headings of the kind MRFlow adds:

    "<findings>\\n\\n<impression>"

Empty sections are dropped rather than emitted as a blank line, and a report with neither is the
empty string, which is upstream's own null-report case.

**Modality, plane, spacing and the 14 labels are NOT in the string.**

  * Modality is structured metadata and reaches the U-Net as `class_labels`, upstream's own class
    embedding added to the timestep. Putting it in the report and predicting it back would make the
    auxiliary head trivial.
  * Spacing already reaches the U-Net as `spacing_tensor`, upstream's own input. Restating it in
    text would be a second, uncontrolled copy of the same condition.
  * Plane is not represented at all -- see `volume.py`: CCELLA's preprocessing reorients every
    volume to RAS, so the acquisition plane survives only as through-plane blur. Adding it to the
    text would be a new conditioning channel upstream does not have.
  * The labels are the prediction target. They are derived *from* the findings upstream of this
    repository, which is exactly CCELLA's own situation with PI-RADS, but no label name and no
    label value is ever inserted into the string.

`test_ccella.py` asserts each of those four absences on real section text.
"""

from __future__ import annotations

import numpy as np
import torch

# Upstream's, verbatim.
ENCODER = "google/flan-t5-xxl"
MAX_LENGTH = 512
TRUNCATION_SIDE = "left"
PADDING = "max_length"

SECTIONS = ("findings", "impression")
SECTION_SEPARATOR = "\n\n"


def format_report(report, sections=SECTIONS):
    """MR-RATE's report dict -> the one plaintext string upstream's encoder expects.

    Returns `""` when every requested section is empty, which the caller turns into `text_isnull`.
    """
    parts = [(report.get(name) or "").strip() for name in sections]
    return SECTION_SEPARATOR.join(part for part in parts if part)


def tokenizer_settings(encoder=ENCODER, max_length=MAX_LENGTH):
    """The part of the fingerprint that describes how text became tokens."""
    return {
        "encoder": encoder,
        "max_length": max_length,
        "truncation_side": TRUNCATION_SIDE,
        "padding": PADDING,
        "sections": list(SECTIONS),
        "separator": SECTION_SEPARATOR,
    }


def build_text_encoder(device, encoder=ENCODER, dtype=torch.float32):
    """The frozen FLAN-T5 encoder and its tokenizer, loaded the way upstream loads them."""
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(encoder, truncation_side=TRUNCATION_SIDE)
    model = T5EncoderModel.from_pretrained(encoder, dtype=dtype).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


@torch.no_grad()
def encode_report(tokenizer, model, text, max_length=MAX_LENGTH):
    """One report string -> `[max_length, hidden]` fp16, upstream's `last_hidden_state`."""
    # Upstream calls `tokenizer.encode_plus(...)`, which transformers v5 removed. Calling the
    # tokenizer directly is what `encode_plus` always delegated to for a single string, and every
    # keyword below is upstream's, unchanged.
    batch = tokenizer(
        text, return_tensors="pt", padding=PADDING, truncation=True, max_length=max_length)
    device = next(model.parameters()).device
    hidden = model(batch["input_ids"].to(device),
                   attention_mask=batch["attention_mask"].to(device)).last_hidden_state
    return hidden.squeeze(0).to(torch.float16).cpu().numpy()


def truncation_fraction(tokenizer, texts, max_length=MAX_LENGTH):
    """How many of `texts` the encoder would cut, for the preprocessing log. Diagnostic only."""
    over = sum(1 for t in texts if len(tokenizer(t)["input_ids"]) > max_length)
    return over / max(1, len(texts))


def empty_embedding(hidden_size, max_length=MAX_LENGTH):
    """The shape an embedding must have, for tests and for a smoke cache with no encoder."""
    return np.zeros((max_length, hidden_size), dtype=np.float16)
