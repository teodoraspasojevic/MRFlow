"""Does the generated volume match the text it was asked for? HLIP says.

Every other metric here compares distributions, or compares a rollout with one particular patient's
scan. These ask the question the task actually poses. They are the only metrics in the project that
read the conditioning at all.

**Two text variants, scored side by side and never mixed.** Both are HLIP's own MR-RATE wording,
because the checkpoint below was trained on MR-RATE with exactly these templates, and both are
identified by `study_uid` -- one study's report is one report however many series it has:

    hlip_findings_*     `"This study looks like: " + findings`
    hlip_impression_*   `"This study shows: " + impression`

**Read `impression` as the primary number.** Measured on 57 real MR-RATE val series, a matched
impression clears the shuffled mean by 4.1 sigma where findings manages 1.4. Upstream's own
zero-shot prompts are all `"This study shows: ..."` ones and its findings head is trained through
the `logit_bias` stand-in `train.py` marks FIXME, so the impression head is the better-exercised of
the two. `findings` is kept because it is a second, independent reading of the same volume.

Each emits `_volume_cosine_mean` / `_std`, `_to_volume_r{1,5,10}`, the same within-stratum, and its
own counts. **There is no fallback between findings and impression**: a case missing a section is
excluded from that variant alone and counted in `n_hlip_<variant>_excluded_no_text`.

The generator's own conditioning string used to be scored here as a third variant. It is gone: it
is deliberately outside the text distribution HLIP's tower was trained on (bracketed section
markers, both sections concatenated, acquisition markers up front), and 42.7% of those strings ran
past the 256-token context, so the number was a truncation artifact as much as a model score.
Compare a model against another model on the same variant; do not compare variants with each other.

Model: HLIP (github.com/zch0414/hlip, TMLR 2026), the **MR-RATE-trained** release
`zch0414/clip-vit_base-scan_study-dualdinotxt1568` pinned at `HLIP_REVISION` -- upstream trains
this one on the MR-RATE training split with `--text-process-cfg "sentence and findings"`, which is
where `TEMPLATES` comes from. The larger `clip-vit_large-scan_study-*` release is a stronger brain
MRI tower but was trained on institutional data and only *evaluated* on MR-RATE; in-domain is worth
more here than raw strength. The build follows the model card exactly -- its vendored `hlip`
package registers the visual encoder with timm, its `model_configs/*.json` is the architecture, its
tokenizer files are the tokenizer -- and the checkpoint is loaded over a randomly initialised
skeleton (`pretrained_image=False, pretrained_text=False`, so no ImageNet or BiomedBERT weights are
fetched and then overwritten). `load_hlip` **raises unless every parameter came from the
checkpoint**.

**`dualdinotxt` means two image embeddings, and each variant must be scored against its own.** The
visual head emits `num_prefix_tokens = 2` (the trunk's `cls_token` and `reg_token`), and upstream
trains them against different text with two separate contrastive losses -- `image_features[:, 0]`
against the impression sentence, `image_features[:, 1]` against the findings
(`src/hlip_train/train.py`, `image_features_sentence` / `image_features_report`). `VARIANTS` holds
that mapping. Crossing them compares a text with a projection that was never trained on it, which
is silent: the cosines stay in range and merely mean less. Upstream's zero-shot scripts index
`[:, 0]` unconditionally because every prompt they ship is a `"This study shows: ..."` one.

Image preprocessing is the official checkpoint loader's, not a reimplementation: pad to square,
bilinear resize to 256^2, then **full-depth deterministic resampling of the whole slice axis to
`NUM_SLICES` = 48 with `nearest-exact`** -- every slice of the volume informs that resample; none
is dropped, cropped away or sampled out. 48 is an architectural constant, not a policy choice: the
checkpoint's `img_size=(48, 224, 224)` at patch `(6, 16, 16)` is what makes its 1568 visual tokens.
Then a centre crop to 224^2 and the *scalar* means of the ImageNet channel statistics, which is how
HLIP normalises a one-channel scan. What it does **not** pin is the intensity mapping: upstream's
scans arrive as uint8 tensors built by data-processing code the repo has since removed, so our
0.5/99.5 percentile window over nonzero voxels is not provably the same transform. It is the same
transform on every model scored here, so comparisons hold; absolute cosines are not comparable to
upstream's published numbers.

Two things these metrics are not:

  independent of the generator, if the generator ever conditions on HLIP. It does not -- MRFlow's
  text encoder is CXR-BERT -- and `warn_if_not_independent` checks the config and says so loudly if
  that changes. A model conditioned on the same encoder it is scored by is scoring itself.
  the conditioning embedding. The generator's 768-d CXR-BERT vector never reaches this file; every
  string here is re-encoded by HLIP's own tokenizer and text tower.

**Retrieval identity is an id, never the text.** Two different studies whose reports happen to read
alike are two different answers, and merging them on string equality would silently forgive a wrong
retrieval. Queries are distinct `study_uid`s; candidates are every generated volume in the run; a
candidate is a positive when its study matches the query's, so the diagonal counts and a study with
several series has several positives. `_within_stratum` restricts candidates to the query's own
**full acquisition stratum -- modality *and* plane**.

When a run holds fewer than K candidates `R@K` is computed at `min(K, n_candidates)` and is
uninformative; read `n_hlip_<variant>_candidates` first.

Tokenizer truncation is counted per variant against HLIP's 256-token context; both templates here
are short enough that it is rare. `n_hlip_<variant>_token_collisions` additionally counts ids whose
*truncated* token sequence is shared with a different id -- retrieval cannot tell those apart, so a
nonzero count caps the achievable R@K.
"""

import importlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

HLIP_REPO = "zch0414/clip-vit_base-scan_study-dualdinotxt1568"
HLIP_REVISION = "6e4b8ab1a1330c59f64a72773c454e513591cf89"
HLIP_MODEL = "ablate_seqposemb_clip_vit_base_multiscan_h2_dualdinotxt1568"

EMBED_DIM = 768
NUM_IMAGE_TOKENS = 2   # cls + reg, trained against the impression and the findings respectively
NUM_SLICES = 48   # the checkpoint's img_size is (48, 224, 224) at patch (6, 16, 16) -> 1568 tokens
RESIZE = 256
CROP = 224
RECALL_K = (1, 5, 10)

# The two text variants, and **which image prefix token each is scored against** -- see the module
# docstring. Both are identified by `study_uid`.
VARIANTS = {"findings": 1, "impression": 0}

# HLIP's own MR-RATE wording. Upstream is inconsistent about the space after the colon
# (`get_sentence` has one, `get_findings` and `get_impressions` do not); both are spaced here.
# `impression` takes the whole section, which is upstream's `get_impressions` rather than the single
# sentence its `sentence and findings` config samples at training time.
TEMPLATES = {"findings": "This study looks like: ", "impression": "This study shows: "}


def score_keys():
    """The scores, most important first: the paired cosine and the retrieval each variant is read
    by, then the harder within-stratum retrieval, then the spread and the positives per query.
    Grouped by metric across the variants, so the same number is read side by side."""
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


def variant_text(variant, report):
    """The string for one variant, or None when the report has no text for it -- never a fallback
    to the other section."""
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


class HlipAccumulator:
    """Both image embeddings per case, and up to two text embeddings beside them.

    **One generated series is a study holding exactly one scan**: `[B, 1, 1, 48, 224, 224]`. The
    released checkpoint is built with `max_num_scans=0`, so it carries no per-scan position
    embedding and no attention or padding mask anywhere; `num_scans` is read off the input's shape
    and every slot present is a real scan (the official `StudyDataset` runs at `batch_size=1` for
    exactly this reason). There is nothing to mask, and a blank scan must never be fabricated to
    pad: `_scan2study` averages prefix tokens over the scan axis, so an empty slot would drag the
    study embedding. Real and generated volumes take this identical one-scan path.

    The image is encoded once and both of its prefix tokens kept -- a ViT over 1568 tokens is the
    expensive half; two text encodes are nearly free. Cases are buffered to `batch_size` and the
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

    def add(self, report, modality, plane, study_uid, fake):
        """One generated volume, the study it belongs to, and whichever of the two texts its report
        supports."""
        texts = {v: variant_text(v, report) for v in VARIANTS}
        for v, text in texts.items():
            if text is None:
                self._excluded[v] += 1
            elif len(self.tokenizer.tokenizer.encode(text)) > self.tokenizer.context_length:
                self._truncated[v] += 1   # past HLIP's context: the tail never reaches the tower
        self._meta.append({"study_uid": str(study_uid), "stratum": f"{modality}__{plane}"})
        self._pending.append((preprocess_scan(fake), texts))
        if len(self._pending) >= self.batch_size:
            self._flush()

    @torch.no_grad()
    def _flush(self):
        if not self._pending:
            return
        row0 = sum(a.shape[0] for a in self._image)
        scans = torch.stack([scan for scan, _ in self._pending]).unsqueeze(1).to(self.device)
        # [B, n_scans=1, 1, D, H, W] -- one scan per study, never padded. Out comes
        # [B, 2, EMBED_DIM]: both prefix tokens, each scored by the variant trained against it.
        self._image.append(_unit(self.model(image=scans)["image_features"]).float().cpu().numpy())

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
            "image": (np.concatenate(self._image) if self._image
                      else np.zeros((0, NUM_IMAGE_TOKENS, EMBED_DIM), np.float32)),
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

    Each variant is scored independently, against its own image token: its own queries, its own
    exclusions, its own truncation. Candidates are always **all** generated volumes in the run -- a
    volume whose report lacked an impression is still a candidate an impression query could wrongly
    retrieve.
    """
    if state is None:
        return {k: (0 if k.startswith("n_") else float("nan")) for k in metric_keys()}

    metrics = {}
    for variant, token in VARIANTS.items():
        metrics.update(_variant_metrics(variant, state["text"][variant],
                                        state["image"][:, token, :], state["meta"],
                                        state["excluded"][variant], state["truncated"][variant]))
    return metrics


def _variant_metrics(variant, text, image, meta, n_excluded, n_truncated):
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
    ident = np.array([m["study_uid"] for m in meta])
    stratum = np.array([m["stratum"] for m in meta])

    paired = (text["vec"] * image[rows]).sum(axis=1)
    out[f"{prefix}_volume_cosine_mean"] = float(np.mean(paired))
    out[f"{prefix}_volume_cosine_std"] = float(np.std(paired))
    out[f"n_{prefix}_token_collisions"] = _n_collisions(text["tokens"], ident[rows])

    # One query per distinct study, its first text standing for it.
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

    # The full acquisition stratum -- modality *and* plane.
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

    `positive` is a boolean matrix rather than an assumed diagonal, because one study can have more
    than one correct volume -- and the diagonal itself is a positive, never excluded. K is capped at
    the number of candidates, so a run with 4 volumes reports `R@5 = R@10 = 1.0`: **R@5 and R@10 say
    nothing below 5 and 10 candidates**, and `n_hlip_<variant>_candidates` is what tells you. A
    production pool should be hundreds; the smoke run checks plumbing only.
    """
    order = np.argsort(-similarity, axis=1)
    hits = np.take_along_axis(positive, order, axis=1)
    n = similarity.shape[1]
    return {k: float(hits[:, :min(k, n)].any(axis=1).mean()) for k in ks}


def signature():
    return {"repo": HLIP_REPO, "revision": HLIP_REVISION, "model": HLIP_MODEL,
            "trained_on": "MR-RATE train split, --text-process-cfg 'sentence and findings'",
            "num_slices": NUM_SLICES, "crop": CROP,
            # Which image prefix token each variant is scored against: the two are trained by
            # separate losses, so this is part of what the number means.
            "pooling": {v: f"image_features[:, {t}]" for v, t in VARIANTS.items()},
            "scans_per_study": 1, "padding": "none (checkpoint has no scan mask)",
            "text_source": "HLIP MR-RATE templates over the study's own report sections",
            "text_template": dict(TEMPLATES)}
