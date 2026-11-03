"""Does the generated study match the report it was generated from? HLIP says.

Every other metric here compares distributions, or compares a rollout with one particular patient's
scan. These ask the question the task actually poses. They are the only metrics in the project that
read the conditioning at all.

**One study, one report, one comparison.** An MR-RATE report describes a whole study, and HLIP
embeds a whole study. So every scored series of one `study_uid` is given to the model together, as
one multi-scan study `[1, n_scans, 1, 48, 224, 224]`, and compared with that study's report. The
same is done with the real series of the same cases, which gives the ceiling: `hlip_*` scores the
generated studies, `hlip_real_*` the real ones, over the identical studies and scans. A series that
was not generated (missing, collapsed, excluded by a trimmed version) is left out of both sides of
its study, so the two always hold the same scans.

**The report is read the way this checkpoint was trained to read it: one sentence at a time.**
Image token 0 was trained against `"This study shows: "` + one sentence of the report.
`impression_sentences` splits the impression into its bullets, and the bullets into sentences, and
each becomes one `"This study shows: <sentence>"` text. A report's score against a study is the mean
cosine of its sentences with that study. Findings are not scored: no released checkpoint was
trained on a findings template.

Each side emits, with `n` studies and one correct answer per row (the diagonal):

    hlip_cosine_mean / _std            the matched report-study score, over studies
    hlip_report_to_study_r{1,5,10}     a report's own study among the top K of n studies
    hlip_report_to_study_mrr / _auroc  mean reciprocal rank; fraction of wrong studies below it
    hlip_study_to_report_*             the same, ranking reports for each study

Chance is `K / n` for R@K and 0.5 for AUROC. `study_to_report` is the cleaner of the two directions:
it compares one study with every report, so the number of scans a study happens to have cannot
favour it. Ties -- two studies whose impressions split into the same sentences, which normal studies
often do -- are scored at their expected value under a random tie-break, so neither side of a tie is
favoured. `n_hlip_duplicate_reports` counts those studies.

Model: HLIP, Hierarchical attention for Language-Image Pre-training (github.com/zch0414/hlip,
TMLR 2026), checkpoint `zch0414/clip-vit_base-scan_study-dualdinotxt1568` pinned at
`HLIP_REVISION`. **It is not trained on MR-RATE.** It is one of the four February 2026 releases
built on the `hlip-ablation` branch, which the ablation blog (zch0414.github.io/hlip-ablation)
says were trained on University of Michigan data (BrainMRI220K + HeadCT240K):
- the image side saw whole multi-scan studies;
- the text side saw LLM-summarized reports (Prima's pipeline), behind `"This study shows: "`.

Upstream's MR-RATE-trained models have no released weights, so no released HLIP checkpoint has
seen MR-RATE. Because the training data is Michigan's, there is no MR-RATE train/test leakage. The
larger `clip-vit_large-scan_study-*` release has the same provenance. The choice between it and
this one was made on the mistaken belief that this one was in-domain, and has not been re-made on
evidence. The build follows the model card exactly -- its vendored `hlip` package registers the
visual encoder with timm, its `model_configs/*.json` is the architecture, its tokenizer files are
the tokenizer -- and the checkpoint is loaded over a randomly initialised skeleton
(`pretrained_image=False, pretrained_text=False`, so no ImageNet or BiomedBERT weights are fetched
and then overwritten). `load_hlip` **raises unless every parameter came from the checkpoint**.

**`dualdinotxt` means two image embeddings.** The visual head emits the trunk's `cls_token` and
`reg_token`, trained under two separate contrastive losses (`hlip-ablation` branch,
`src/hlip_train/data.py` `get_sentence` / `get_report`, `train.py` `image_features_sentence` /
`image_features_report`):
- `image_features[:, 0]` against one randomly chosen report sentence -- the one scored here;
- `image_features[:, 1]` against the whole report, sentences permuted.

The study is encoded hierarchically: layers 0, 1, 3, 4, 6, 7, 9 and 10 attend within each scan,
layers 2, 5, 8 and 11 across the whole study. The checkpoint has `max_num_scans=0`: no per-scan
position embedding and no mask, so the scans are an unordered set and every slot must be a real
scan. Scans are still sorted by case id, so the input is reproducible.

Image preprocessing is the official checkpoint loader's, not a reimplementation: pad to square,
bilinear resize to 256^2, then `nearest-exact` along the slice axis to `NUM_SLICES` = 48. That
**keeps 48 slices and drops the rest** -- 110 of a 158-slice volume never reach the model. 48 is an
architectural constant, not a policy choice: the checkpoint's `img_size=(48, 224, 224)` at patch
`(6, 16, 16)` is what makes its 1568 visual tokens per scan. Then a centre crop to 224^2 and the
*scalar* means of the ImageNet channel statistics, which is how HLIP normalises a one-channel scan.
What it does **not** pin is the intensity mapping: upstream's scans arrive as uint8 tensors built by
data-processing code the repo has since removed. The paper states a [0.5, 99.5] percentile clip for
brain MRI but not over which voxels, so our 0.5/99.5 window over all voxels
(`paper_metrics.normalize01`) is not provably the same transform. It is the same transform on every
model scored here, so comparisons hold; absolute cosines are not comparable to upstream's published
numbers.

**The model runs once, over the pooled run.** A study's series land in different array tasks, and
study attention needs all of them in one forward pass. So a shard keeps each scan as the uint8
`48 x 224 x 224` the model would read (2.4 MB a side), and `summarize` encodes every study after the
shards are pooled. uint8 is what upstream's own scans are; for the 256^2 volumes every metric here
is handed, the in-plane resize is the identity, so the stored scan is exactly what the loader would
have computed from a uint8 volume.

Two things these metrics are not:

  independent of the generator, if the generator ever conditions on HLIP. It does not -- MRFlow's
  text encoder is CXR-BERT -- and `warn_if_not_independent` checks the config and says so loudly if
  that changes. A model conditioned on the same encoder it is scored by is scoring itself.
  the conditioning embedding. The generator's 768-d CXR-BERT vector never reaches this file; every
  string here is re-encoded by HLIP's own tokenizer and text tower.

**Retrieval identity is the study id, never the text.** Two studies whose reports read alike stay
two answers; the tie between them is scored, not merged away.
"""

import functools
import importlib
import json
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HLIP_REPO = "zch0414/clip-vit_base-scan_study-dualdinotxt1568"
HLIP_REVISION = "6e4b8ab1a1330c59f64a72773c454e513591cf89"
HLIP_MODEL = "ablate_seqposemb_clip_vit_base_multiscan_h2_dualdinotxt1568"

EMBED_DIM = 768
TOKEN = 0         # cls: the head trained against one report sentence
NUM_SLICES = 48   # the checkpoint's img_size is (48, 224, 224) at patch (6, 16, 16) -> 1568 tokens
RESIZE = 256
CROP = 224
RECALL_K = (1, 5, 10)
PROMPT = "This study shows: "   # the ablation recipe's prefix for every training text
TEXT_BATCH = 64

SIDES = {"fake": "hlip", "real": "hlip_real"}     # which volumes, and the key prefix they report under
DIRECTIONS = ("report_to_study", "study_to_report")


def score_keys():
    """The scores, most important first, each read generated-then-real side by side."""
    names = ["cosine_mean"]
    names += [f"{d}_r{k}" for d in DIRECTIONS for k in RECALL_K]
    names += [f"{d}_{m}" for d in DIRECTIONS for m in ("auroc", "mrr")]
    names += ["cosine_std"]
    return tuple(f"{p}_{n}" for n in names for p in SIDES.values())


def count_keys():
    """The sample counts, which all live at the end of the table. Both sides share them."""
    return tuple(f"n_hlip_{c}" for c in ("studies", "scans", "sentences", "duplicate_reports",
                                         "excluded_no_impression", "truncated"))


def metric_keys():
    return score_keys() + count_keys()


# MR-RATE impressions are bulleted ("— finding"), sometimes under a header ("Cranial:", "MR
# Venography:") and with anonymisation placeholders ("[radiologist_1]").
_BULLET = re.compile(r"^\s*(?:[—–•*-]+|\d+[.)]\s)\s*")
_HEADER = re.compile(r"^[^.!?]{0,40}:$")
_PLACEHOLDER = re.compile(r"\[[A-Za-z][A-Za-z _]*_\d+\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def impression_sentences(report):
    """The impression as a list of sentences: one per bullet, split further where a bullet holds
    several. Headers and placeholders are dropped, trailing commas become full stops, and a report
    with no impression gives `[]` -- never a fallback to the findings."""
    sentences = []
    for line in (report.get("impression") or "").splitlines():
        line = _PLACEHOLDER.sub("", _BULLET.sub("", line)).strip()
        if not line or _HEADER.match(line):
            continue
        for s in _SENTENCE_END.split(line):
            s = s.strip().rstrip(",;:").strip()
            if re.search(r"[A-Za-z0-9]", s):
                sentences.append(s if s[-1] in ".!?" else s + ".")
    return sentences


def sentence_texts(report):
    return [PROMPT + s for s in impression_sentences(report)]

# HLIP normalises a single-channel scan with the scalar mean of the ImageNet per-channel constants,
# which is what `timm.data.constants` gives it in the model card's loader.
_MEAN, _STD = 0.449, 0.226


@functools.lru_cache(maxsize=None)
def load_hlip(device):
    """`(model, tokenizer)`, built and loaded the way the checkpoint's model card builds them.
    Cached, so scoring and the tests build it once per device."""
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
    # the tokenizer is vendored in the snapshot; its path pins it to the revision, not to `main`
    config["text_cfg"]["hf_tokenizer_name"] = repo
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


def resample_scan(volume):
    """A `(T, H, W)` volume in `[0, 1]` -> the `(48, 224, 224)` scan HLIP reads, still in `[0, 1]`.

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
    return img[:, offset:offset + CROP, offset:offset + CROP]


def preprocess_scan(volume):
    """A canonicalized `(T, H, W)` volume in `[0, 1]` -> HLIP's `(1, 48, 224, 224)` scan tensor."""
    return ((resample_scan(volume) - _MEAN) / _STD)[None]


def quantize_scan(volume):
    """`resample_scan` as uint8 -- what a shard keeps until the run is pooled."""
    return (resample_scan(volume).clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def _scan_input(scan_u8):
    return ((torch.from_numpy(scan_u8).float() / 255 - _MEAN) / _STD)[None]


class HlipAccumulator:
    """Every scored series, grouped by study: its report's sentences and both sides' scans.

    Nothing is encoded here -- a shard rarely holds every series of a study, and HLIP's study
    attention needs them all at once. `summarize` runs the model over the pooled studies.
    """

    def __init__(self):
        self._studies = {}

    def add(self, case_id, study_uid, report, fake, real):
        """One series: its generated and its real volume, both canonicalized, and its study's
        report (the same for every series of the study)."""
        study = self._studies.setdefault(str(study_uid), {"sentences": sentence_texts(report),
                                                          "scans": {}})
        study["scans"][str(case_id)] = {"fake": quantize_scan(fake), "real": quantize_scan(real)}

    def state(self):
        return {"studies": self._studies}


def _unit(x):
    """L2-normalize, so every similarity below is a cosine."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def merge_states(states):
    """Several shards' `state()` into one: a study's series reunite, whichever shard scored them."""
    studies = {}
    for st in states:
        for uid, study in st["studies"].items():
            merged = studies.setdefault(uid, {"sentences": study["sentences"], "scans": {}})
            merged["scans"].update(study["scans"])
    return {"studies": studies}


def summarize(state, device="auto"):
    """Every HLIP number, or nan throughout if the tower was switched off (`state` is None).

    Loads the model and encodes every study and sentence of the pooled run. A study whose
    impression gives no sentence is excluded from both sides and counted.
    """
    if state is None:
        return {k: (0 if k.startswith("n_") else float("nan")) for k in metric_keys()}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    studies = state["studies"]
    uids = sorted(uid for uid, s in studies.items() if s["sentences"])
    out = {"n_hlip_studies": len(uids),
           "n_hlip_scans": sum(len(studies[u]["scans"]) for u in uids),
           "n_hlip_sentences": sum(len(studies[u]["sentences"]) for u in uids),
           "n_hlip_excluded_no_impression": len(studies) - len(uids),
           "n_hlip_truncated": 0, "n_hlip_duplicate_reports": 0}
    if not uids:
        return {**{k: float("nan") for k in score_keys()}, **out}

    model, tokenizer = load_hlip(device)
    reports, out["n_hlip_truncated"] = _encode_reports(model, tokenizer, device,
                                                       [studies[u]["sentences"] for u in uids])
    keys = [tuple(sorted(studies[u]["sentences"])) for u in uids]
    counts = {k: keys.count(k) for k in set(keys)}
    out["n_hlip_duplicate_reports"] = sum(1 for k in keys if counts[k] > 1)

    for side, prefix in SIDES.items():
        image = np.stack([_encode_study(model, device, studies[u]["scans"], side) for u in uids])
        out.update(retrieval_metrics(reports @ image.T, prefix))
    return out


@torch.no_grad()
def _encode_reports(model, tokenizer, device, reports):
    """`(vectors [n, D], n_truncated)`: each report as the mean of its unit sentence embeddings, so
    a report-study similarity is the mean cosine over the report's sentences. Each distinct
    sentence is encoded once and summed in sorted order, so two reports made of the same
    sentences get bit-identical vectors and tie exactly."""
    unique = sorted({s for r in reports for s in r})
    vectors = []
    for i in range(0, len(unique), TEXT_BATCH):
        tokens = tokenizer(unique[i:i + TEXT_BATCH]).to(device)
        vectors.append(_unit(model.encode_text(tokens)).float().cpu().numpy())
    vectors = np.concatenate(vectors)
    index = {s: i for i, s in enumerate(unique)}
    limit = tokenizer.context_length
    truncated = sum(len(tokenizer.tokenizer.encode(s)) > limit for r in reports for s in r)
    mean = np.stack([vectors[sorted(index[s] for s in r)].mean(axis=0) for r in reports])
    return mean, int(truncated)


@torch.no_grad()
def _encode_study(model, device, scans, side):
    """One study's scans, sorted by case id, through the model at once -> its unit token-0 vector."""
    image = torch.stack([_scan_input(scans[c][side]) for c in sorted(scans)])  # [n_scans, 1, D, H, W]
    features = model(image=image[None].to(device))["image_features"]          # [1, 2, EMBED_DIM]
    return _unit(features[0, TOKEN]).float().cpu().numpy()


def retrieval_metrics(similarity, prefix):
    """Cosine, R@K, AUROC and MRR from a square `[report, study]` similarity matrix whose diagonal
    holds the matched pairs, in both directions.

    Each row's correct answer has `greater` candidates scoring above it and `equal` tied with it.
    Under a random tie-break its rank is uniform on `greater + 1 ... greater + equal + 1`, and every
    number below is that rank's expected value -- so a tie is neither a hit nor a miss. AUROC is the
    fraction of wrong candidates scored below the right one, ties counted half.
    """
    n = similarity.shape[0]
    matched = np.diag(similarity)
    out = {f"{prefix}_cosine_mean": float(matched.mean()),
           f"{prefix}_cosine_std": float(matched.std())}
    for direction, s in zip(DIRECTIONS, (similarity, similarity.T)):
        greater = (s > matched[:, None]).sum(axis=1)
        equal = (s == matched[:, None]).sum(axis=1) - 1
        for k in RECALL_K:
            out[f"{prefix}_{direction}_r{k}"] = float(
                np.mean(np.clip(k - greater, 0, equal + 1) / (equal + 1)))
        out[f"{prefix}_{direction}_mrr"] = float(np.mean(
            [np.mean(1.0 / np.arange(g + 1, g + e + 2)) for g, e in zip(greater, equal)]))
        out[f"{prefix}_{direction}_auroc"] = (
            float(np.mean((n - 1 - greater - equal / 2) / (n - 1))) if n > 1 else float("nan"))
    return out


def signature():
    return {"repo": HLIP_REPO, "revision": HLIP_REVISION, "model": HLIP_MODEL,
            "trained_on": "Michigan BrainMRI220K + HeadCT240K (hlip-ablation), not MR-RATE",
            "num_slices": NUM_SLICES, "crop": CROP, "scan_storage": "uint8",
            "pooling": f"image_features[:, {TOKEN}] (sentence head)",
            "scans_per_study": "every scored series of the study, sorted by case id",
            "text_source": "impression sentences, one text each",
            "text_template": PROMPT + "<sentence>",
            "report_score": "mean cosine over the report's sentences"}
