"""Does the generated volume match the text it was asked for? HLIP says.

Every other metric here compares distributions, or compares a rollout with one particular patient's
scan. These ask the question the task actually poses. They are the only metrics in the project that
read the conditioning at all.

**Three text variants, scored side by side and never mixed.** They differ in what the text is and
in what makes a retrieval hit correct:

    hlip_condition_*    the exact string the generator was conditioned on: acquisition markers
                        (`[MODALITY] .. [PLANE] ..`) followed by the bracketed `[FINDINGS]` and
                        `[IMPRESSION]` sections. **This is not MR-RATE's `report` column** -- it is
                        those two sections plus markers, assembled by `encode_conditioning`. Its
                        identity is the `conditioning_uid`: one acquisition condition of one study.
    hlip_findings_*     `"This study looks like: " + findings`, HLIP's own MR-RATE template.
                        Identity is `study_uid`.
    hlip_impression_*   `"This study shows: " + impression`, HLIP's own MR-RATE template.
                        Identity is `study_uid`.

Each emits `_volume_cosine_mean` / `_std`, `_to_volume_r{1,5,10}` and the same within-stratum, plus
its own counts. **There is no fallback between findings and impression**: a case missing a section
is excluded from that variant alone and counted in `n_hlip_<variant>_excluded_no_text`.

HLIP was trained on MR-RATE with `--text-process-cfg "sentence and findings"`, i.e. a single
impression sentence against the joined findings -- so `findings` and `impression` are close to the
text distribution its tower saw, while `condition` deliberately is not. Compare a model against
another model on the same variant; do not compare variants with each other.

Model: HLIP (github.com/zch0414/hlip, TMLR 2026), the released brain-MRI checkpoint
`zch0414/clip-vit_large-scan_study-dualdinotxt1568` pinned at `HLIP_REVISION`. The build follows
that repo's model card exactly -- its vendored `hlip` package registers the visual encoder with
timm, its `model_configs/*.json` is the architecture, its tokenizer files are the tokenizer -- and
the checkpoint is loaded over a randomly initialised skeleton (`pretrained_image=False,
pretrained_text=False`, so no ImageNet or BiomedBERT weights are fetched and then overwritten).
`load_hlip` **raises unless every parameter came from the checkpoint**.

Image preprocessing is the official checkpoint loader's, not a reimplementation: pad to square,
bilinear resize to 256^2, then **full-depth deterministic resampling of the whole slice axis to
`NUM_SLICES` = 48 with `nearest-exact`** -- every slice of the volume informs that resample; none
is dropped, cropped away or sampled out. 48 is an architectural constant, not a policy choice: the
checkpoint's `img_size=(48, 224, 224)` at patch `(6, 16, 16)` is what makes its 1568 visual tokens.
Then a centre crop to 224^2 and the *scalar* means of the ImageNet channel statistics, which is how
HLIP normalises a one-channel scan.

Two things these metrics are not:

  independent of the generator, if the generator ever conditions on HLIP. It does not -- MRFlow's
  text encoder is CXR-BERT -- and `warn_if_not_independent` checks the config and says so loudly if
  that changes. A model conditioned on the same encoder it is scored by is scoring itself.
  the conditioning embedding. The generator's 768-d CXR-BERT vector never reaches this file; every
  string here is re-encoded by HLIP's own tokenizer and text tower.

**Retrieval identity is an id, never the text.** Two different studies whose reports happen to read
alike are two different answers, and merging them on string equality would silently forgive a wrong
retrieval. Queries are distinct ids; candidates are every generated volume in the run; a candidate
is a positive when its id matches the query's, so the diagonal counts and an id covering several
volumes has several positives. `_within_stratum` restricts candidates to the query's own **full
acquisition stratum -- modality *and* plane** -- so the acquisition markers in `condition` cannot
solve retrieval by naming the contrast and the orientation.

When a run holds fewer than K candidates `R@K` is computed at `min(K, n_candidates)` and is
uninformative; read `n_hlip_<variant>_candidates` first.

Tokenizer truncation is reported per variant. Measured on 300 val series, **42.7% of `condition`
strings run past HLIP's 256-token context** (median 238 tokens, p90 403, max 655) and lose their
tail; `findings` and `impression` are shorter. `n_hlip_<variant>_token_collisions` additionally
counts ids whose *truncated* token sequence is shared with a different id -- retrieval cannot tell
those apart, so a nonzero count caps the achievable R@K.
"""

import hashlib
import importlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from echosyn.common.mrrate import acquisition_prefix, format_report

HLIP_REPO = "zch0414/clip-vit_large-scan_study-dualdinotxt1568"
HLIP_REVISION = "183bae10b472004007251daa91b3382a07bcae6e"
HLIP_MODEL = "ablate_seqposemb_clip_vit_large_multiscan_h2_dualdinotxt1568"

EMBED_DIM = 768
NUM_SLICES = 48   # the checkpoint's img_size is (48, 224, 224) at patch (6, 16, 16) -> 1568 tokens
RESIZE = 256
CROP = 224
RECALL_K = (1, 5, 10)

# The three text variants: how the string is built, and what makes a hit correct. `condition` is
# identified by the acquisition condition it was generated from; the two HLIP-native templates are
# identified by the study, because one study's report is one report however many series it has.
VARIANTS = {"condition": "cond_uid", "findings": "study_uid", "impression": "study_uid"}

# HLIP's own MR-RATE wording. Upstream is inconsistent about the space after the colon
# (`get_sentence` has one, `get_findings` does not); both are spaced here.
TEMPLATES = {"findings": "This study looks like: ", "impression": "This study shows: "}


def score_keys():
    """The scores, most important first: the paired cosine and the retrieval each variant is read
    by, then the harder within-stratum retrieval, then the spread and the positives per query.
    Grouped by metric across the three variants, so the same number is read side by side."""
    k = [f"hlip_{v}_volume_cosine_mean" for v in VARIANTS]
    k += [f"hlip_{v}_to_volume_r{n}" for v in VARIANTS for n in RECALL_K]
    k += [f"hlip_{v}_to_volume_r{n}_within_stratum" for v in VARIANTS for n in RECALL_K]
    k += [f"hlip_{v}_volume_cosine_std" for v in VARIANTS]
    k += [f"hlip_{v}_positives_per_query_mean" for v in VARIANTS]
    return tuple(k)


def count_keys():
    """The sample counts, which all live at the end of the table."""
    return tuple(f"n_hlip_{v}_{c}" for v in VARIANTS
                 for c in ("queries", "candidates", "pairs", "excluded_no_text", "truncated",
                           "token_collisions"))


def metric_keys():
    return score_keys() + count_keys()


def conditioning_uid(study_uid, modality, plane):
    """A stable id for one acquisition condition of one study.

    `run_shard` writes this into the manifest so the id a score is computed against is the one
    generation recorded, not something re-derived later from fields that might have moved.
    """
    return hashlib.sha1(f"{study_uid}|{modality}|{plane}".encode()).hexdigest()[:16]


def variant_text(variant, report, modality, plane):
    """The string for one variant, or None when the report has no text for it -- never a fallback
    to the other section."""
    if variant == "condition":
        body = format_report(report)
        return f"{acquisition_prefix(modality, plane)}\n{body}" if body.strip() else None
    section = (report.get(variant) or "").strip()
    return TEMPLATES[variant] + section if section else None

# HLIP normalises a single-channel scan with the scalar mean of the ImageNet per-channel constants,
# which is what `timm.data.constants` gives it in the model card's loader.
_MEAN, _STD = 0.449, 0.226


def load_hlip(device):
    """`(model, tokenizer)`, built and loaded the way the checkpoint's model card builds them."""
    from huggingface_hub import snapshot_download
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import _MODEL_CONFIGS
    import safetensors.torch as st

    repo = snapshot_download(repo_id=HLIP_REPO, revision=HLIP_REVISION)
    if repo not in sys.path:
        sys.path.append(repo)
        importlib.invalidate_caches()
    importlib.import_module("hlip.visual_encoder")  # registers the visual encoder with timm

    with open(f"{repo}/hlip/model_configs/{HLIP_MODEL}.json") as handle:
        config = json.load(handle)
    config["text_cfg"]["hf_tokenizer_name"] = HLIP_REPO  # the tokenizer is vendored in the repo
    _MODEL_CONFIGS[HLIP_MODEL] = config

    # Nothing pretrained is fetched for either tower: every weight below comes from the checkpoint,
    # and the load is checked rather than trusted.
    model, _, _ = create_model_and_transforms(HLIP_MODEL, device=device, output_dict=True,
                                              pretrained_image=False, pretrained_text=False)
    missing, unexpected = model.load_state_dict(st.load_file(f"{repo}/model.safetensors"),
                                                strict=False)
    if missing or unexpected:
        raise RuntimeError(f"HLIP checkpoint {HLIP_REVISION} does not fit the model: "
                           f"{len(missing)} missing, {len(unexpected)} unexpected "
                           f"(first: {(missing + unexpected)[:3]})")
    # open_clip logs "Model initialized randomly" while building the skeleton above; this line is
    # the one that says the checkpoint then replaced every one of those weights.
    print(f"[hlip] loaded {HLIP_REPO}@{HLIP_REVISION[:8]}, every parameter from the checkpoint")
    return model.eval(), get_tokenizer(HLIP_MODEL)


def warn_if_not_independent(text_checkpoint):
    """HLIP scoring a model that was conditioned on HLIP is a model grading its own homework."""
    if "hlip" in str(text_checkpoint).lower() or str(text_checkpoint) == HLIP_REPO:
        print("!" * 78)
        print(f"!! WARNING: the generator's text encoder is {text_checkpoint}, which is the HLIP")
        print("!! checkpoint these metrics are computed with. The hlip_* numbers are NOT an")
        print("!! independent evaluation -- the model is being scored by its own conditioning.")
        print("!" * 78)


def preprocess_scan(volume):
    """A canonicalized `(T, H, W)` volume in `[0, 1]` -> HLIP's `(1, 48, 224, 224)` scan tensor.

    The model card's `loader`, step for step, with the one difference that our volume arrives as
    floats in `[0, 1]` rather than as `[0, 255]` slice tensors -- the division by 255 that opens
    that function is therefore already done.
    """
    img = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
    h, w = img.shape[1:]
    size = max(h, w)
    pad_w, pad_h = size - w, size - h
    img = F.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    img = F.interpolate(img[None], size=(RESIZE, RESIZE), mode="bilinear")[0]
    img = F.interpolate(img[None, None], size=(NUM_SLICES, RESIZE, RESIZE),
                        mode="nearest-exact")[0, 0]
    offset = (RESIZE - CROP) // 2
    img = img[:, offset:offset + CROP, offset:offset + CROP]
    return ((img - _MEAN) / _STD)[None]


def conditioning_text(report, modality, plane):
    """**The exact string the generator was conditioned on** -- not a report field chosen to suit
    the metric.

    `encode_conditioning` builds `acquisition_prefix` + the bracketed `[FINDINGS]`/`[IMPRESSION]`
    sections and encodes that one string; this calls the very same two builders, so the two cannot
    drift. Returns `None` when the report has neither section, and that case is excluded with a
    count rather than scored against an empty string.

    It is therefore *not* either of HLIP's own MR-RATE templates. HLIP was trained on MR-RATE with
    `--text-process-cfg "sentence and findings"`, i.e. `f'This study shows: {impression}'` against
    `'This study looks like:' + findings`. Ours carries both sections at once behind bracket
    markers and an acquisition prefix. **No HLIP template is prepended**, because prepending one
    would score a string the model was never asked to generate from. The consequence to state when
    quoting these numbers: the text distribution is not the one HLIP's text tower was trained on,
    so the absolute cosine is not comparable with HLIP's own MR-RATE figures -- only across our own
    runs, which all use this same string.
    """
    body = format_report(report)
    if not body.strip():
        return None
    return f"{acquisition_prefix(modality, plane)}\n{body}"


class HlipAccumulator:
    """One volume embedding per case, and up to three text embeddings beside it.

    **One generated series is a study holding exactly one scan**: `[B, 1, 1, 48, 224, 224]`. The
    released checkpoint is built with `max_num_scans=0`, so it carries no per-scan position
    embedding and no attention or padding mask anywhere; `num_scans` is read off the input's shape
    and every slot present is a real scan (the official `StudyDataset` runs at `batch_size=1` for
    exactly this reason). There is nothing to mask, and a blank scan must never be fabricated to
    pad: `_scan2study` averages prefix tokens over the scan axis, so an empty slot would drag the
    study embedding. Real and generated volumes take this identical one-scan path.

    The image is encoded once and shared by all three variants -- a ViT-L over 1568 tokens is the
    expensive half; three text encodes are nearly free. Cases are buffered to `batch_size` and the
    buffer holds preprocessed scans (9.6 MB each), never volumes.
    """

    def __init__(self, device="auto", batch_size=8):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model, self.tokenizer = load_hlip(device)
        self._pending = []
        self._image, self._meta = [], []
        # per variant: the row of `image` each text belongs to, its embedding, and its token ids
        self._text = {v: {"row": [], "vec": [], "tokens": []} for v in VARIANTS}
        self._excluded = dict.fromkeys(VARIANTS, 0)
        self._truncated = dict.fromkeys(VARIANTS, 0)

    def add(self, report, modality, plane, study_uid, cond_uid, fake):
        """One generated volume, the ids that say what a correct retrieval is, and whichever of the
        three texts its report supports."""
        texts = {v: variant_text(v, report, modality, plane) for v in VARIANTS}
        for v, text in texts.items():
            if text is None:
                self._excluded[v] += 1
            elif len(self.tokenizer.tokenizer.encode(text)) > self.tokenizer.context_length:
                self._truncated[v] += 1   # past HLIP's context: the tail never reaches the tower
        self._meta.append({"study_uid": str(study_uid), "cond_uid": str(cond_uid),
                           "stratum": f"{modality}__{plane}"})
        self._pending.append((preprocess_scan(fake), texts))
        if len(self._pending) >= self.batch_size:
            self._flush()

    @torch.no_grad()
    def _flush(self):
        if not self._pending:
            return
        row0 = sum(a.shape[0] for a in self._image)
        scans = torch.stack([scan for scan, _ in self._pending]).unsqueeze(1).to(self.device)
        # [B, n_scans=1, 1, D, H, W] -- one scan per study, never padded.
        self._image.append(_unit(self.model(image=scans)["image_features"][:, 0, :])
                           .float().cpu().numpy())

        for v in VARIANTS:
            rows = [row0 + i for i, (_, t) in enumerate(self._pending) if t[v] is not None]
            texts = [t[v] for _, t in self._pending if t[v] is not None]
            if not texts:
                continue
            tokens = self.tokenizer(texts)
            self._text[v]["row"] += rows
            self._text[v]["tokens"] += [tuple(row.tolist()) for row in tokens]
            self._text[v]["vec"].append(
                _unit(self.model.encode_text(tokens.to(self.device))).float().cpu().numpy())
        self._pending = []

    def state(self):
        self._flush()
        empty = np.zeros((0, EMBED_DIM), np.float32)
        return {
            "image": np.concatenate(self._image) if self._image else empty,
            "meta": list(self._meta),
            "text": {v: {"row": t["row"], "tokens": t["tokens"],
                         "vec": np.concatenate(t["vec"]) if t["vec"] else empty}
                     for v, t in self._text.items()},
            "excluded": dict(self._excluded), "truncated": dict(self._truncated),
        }


def _unit(x):
    """L2-normalize, so every similarity below is a cosine."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def merge_states(states):
    """Several shards' `state()` into one. Embeddings concatenate and the row indices shift by the
    number of images already merged, so retrieval still runs over the whole population at once --
    the candidate pool is every generated volume in the run, never one shard's."""
    merged = {"image": np.concatenate([st["image"] for st in states]),
              "meta": [m for st in states for m in st["meta"]],
              "text": {v: {"row": [], "vec": [], "tokens": []} for v in VARIANTS},
              "excluded": {v: sum(st["excluded"][v] for st in states) for v in VARIANTS},
              "truncated": {v: sum(st["truncated"][v] for st in states) for v in VARIANTS}}
    offset = 0
    for st in states:
        for v in VARIANTS:
            merged["text"][v]["row"] += [r + offset for r in st["text"][v]["row"]]
            merged["text"][v]["tokens"] += st["text"][v]["tokens"]
            merged["text"][v]["vec"].append(st["text"][v]["vec"])
        offset += st["image"].shape[0]
    empty = np.zeros((0, EMBED_DIM), np.float32)
    for v in VARIANTS:
        vecs = [a for a in merged["text"][v]["vec"] if a.shape[0]]
        merged["text"][v]["vec"] = np.concatenate(vecs) if vecs else empty
    return merged


def summarize(state):
    """Every HLIP number, or nan throughout if the tower was switched off (`state` is None).

    Each variant is scored independently: its own queries, its own exclusions, its own truncation.
    Candidates are always **all** generated volumes in the run -- a volume whose report lacked an
    impression is still a candidate that an impression query could wrongly retrieve.
    """
    if state is None:
        return {k: (0 if k.startswith("n_") else float("nan")) for k in metric_keys()}

    image, meta = state["image"], state["meta"]
    metrics = {}
    for variant, id_field in VARIANTS.items():
        metrics.update(_variant_metrics(variant, state["text"][variant], image, meta, id_field,
                                        state["excluded"][variant],
                                        state["truncated"][variant]))
    return metrics


def _variant_metrics(variant, text, image, meta, id_field, n_excluded, n_truncated):
    prefix, n_cand = f"hlip_{variant}", len(meta)
    out = {f"{prefix}_volume_cosine_mean": float("nan"),
           f"{prefix}_volume_cosine_std": float("nan"),
           f"{prefix}_positives_per_query_mean": float("nan"),
           f"n_{prefix}_queries": 0, f"n_{prefix}_candidates": n_cand,
           f"n_{prefix}_pairs": len(text["row"]),
           f"n_{prefix}_excluded_no_text": n_excluded,
           f"n_{prefix}_truncated": n_truncated, f"n_{prefix}_token_collisions": 0}
    out.update({f"{prefix}_to_volume_r{k}{suffix}": float("nan")
                for k in RECALL_K for suffix in ("", "_within_stratum")})
    if not text["row"]:
        return out

    rows = np.array(text["row"])
    ident = np.array([m[id_field] for m in meta])
    stratum = np.array([m["stratum"] for m in meta])

    paired = (text["vec"] * image[rows]).sum(axis=1)
    out[f"{prefix}_volume_cosine_mean"] = float(np.mean(paired))
    out[f"{prefix}_volume_cosine_std"] = float(np.std(paired))
    out[f"n_{prefix}_token_collisions"] = _n_collisions(text["tokens"], ident[rows])

    # One query per distinct id, its first text standing for it.
    first = {}
    for i, key in enumerate(ident[rows]):
        first.setdefault(key, i)
    q = list(first.values())
    similarity = text["vec"][q] @ image.T
    positive = ident[None, :] == ident[rows][q][:, None]
    out[f"n_{prefix}_queries"] = len(q)
    out[f"{prefix}_positives_per_query_mean"] = float(positive.sum(axis=1).mean())
    out.update({f"{prefix}_to_volume_r{k}": v for k, v in
                recall_at_k(similarity, positive).items()})

    # The full acquisition stratum -- modality *and* plane -- so the markers cannot answer it.
    same = stratum[None, :] == stratum[rows][q][:, None]
    out.update({f"{prefix}_to_volume_r{k}_within_stratum": v for k, v in
                recall_at_k(np.where(same, similarity, -np.inf), positive & same).items()})
    return out


def _n_collisions(tokens, ident):
    """Ids whose *truncated* token sequence is shared with a different id. Retrieval cannot tell
    those apart, so a nonzero count is a ceiling on R@K rather than a model failing."""
    by_tokens = {}
    for row, key in zip(tokens, ident):
        by_tokens.setdefault(row, set()).add(key)
    return int(sum(len(keys) for keys in by_tokens.values() if len(keys) > 1))


def recall_at_k(similarity, positive, ks=RECALL_K):
    """`{K: R@K}`: the fraction of query rows whose top-K columns hold at least one positive.

    `positive` is a boolean matrix rather than an assumed diagonal, because one conditioning string
    can have more than one correct volume -- and the diagonal itself is a positive, never excluded.
    K is capped at the number of candidates, so a run with 4 volumes reports `R@5 = R@10 = 1.0`:
    **R@5 and R@10 say nothing below 5 and 10 candidates**, and `n_hlip_candidates` is what tells
    you. A production pool should be hundreds; the smoke run checks plumbing only.
    """
    order = np.argsort(-similarity, axis=1)
    hits = np.take_along_axis(positive, order, axis=1)
    n = similarity.shape[1]
    return {k: float(hits[:, :min(k, n)].any(axis=1).mean()) for k in ks}


def signature():
    return {"repo": HLIP_REPO, "revision": HLIP_REVISION, "model": HLIP_MODEL,
            "num_slices": NUM_SLICES, "crop": CROP, "pooling": "image_features[:, 0]",
            "scans_per_study": 1, "padding": "none (checkpoint has no scan mask)",
            # What the text tower is given, recorded because it is the one choice a reader of these
            # numbers most needs to know: the generator's own conditioning string, no HLIP template.
            "text_source": "generator_conditioning: acquisition_prefix + [FINDINGS]/[IMPRESSION]",
            "text_template": None}
