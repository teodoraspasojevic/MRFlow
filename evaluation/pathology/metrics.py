"""Multi-label metrics that keep failures in the denominator, and study-clustered bootstraps.

`probs` rows of a case with no prediction (missing or unreadable volume) are NaN. Two readings:

    strict      (no suffix, primary) every expected case; a failed row is replaced by `prior`, the
                train-split prevalence of each label -- what a model that never saw an image would
                output. It therefore ranks at chance and is thresholded like any other case.
    evaluated   (`_evaluated` suffix) only the cases that produced a prediction.

Macro means run over the **scored** labels only: those with at least `min_positives` positives
and as many negatives in the population. The others are still reported per label, flagged
`scored: false` -- an AUROC from two positives is not a number worth averaging.

Labels belong to studies and a study contributes several cases, so the bootstrap resamples
**studies** (`groups`), not cases.
"""

import numpy as np

from evaluation.modality.metrics import auroc
from evaluation.pathology import LABELS


def average_precision(scores, positive):
    """Area under the precision-recall curve as a step sum (sklearn's `average_precision_score`,
    ties included). NaN without a positive."""
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], positive[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # end of each block of tied scores
    tp = np.cumsum(t)[last]
    precision = tp / (last + 1)
    recall_step = np.diff(np.r_[0, tp]) / n_pos
    return float(np.sum(precision * recall_step))


def best_thresholds(y, probs):
    """Per label, the probability threshold that maximizes F1 -- chosen on val, applied elsewhere."""
    out = []
    for j in range(y.shape[1]):
        p, t = probs[:, j], y[:, j].astype(bool)
        best, best_f1 = 0.5, -1.0
        for thr in np.unique(p):
            pred = p >= thr
            tp = np.sum(pred & t)
            f1 = 2 * tp / (pred.sum() + t.sum()) if pred.sum() + t.sum() else 0.0
            if f1 > best_f1:
                best, best_f1 = float(thr), f1
        out.append(best)
    return np.array(out)


def _calibration(p, t, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(t[bins == b].mean() - p[bins == b].mean()) * np.mean(bins == b)
                     for b in range(n_bins) if np.any(bins == b)))


def _per_label(y, probs, thresholds):
    rows = {}
    for j, name in enumerate(LABELS):
        p, t = probs[:, j], y[:, j].astype(bool)
        pred = p >= thresholds[j]
        tp, fp = int(np.sum(pred & t)), int(np.sum(pred & ~t))
        fn, tn = int(np.sum(~pred & t)), int(np.sum(~pred & ~t))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else float("nan")
        rows[name] = {
            "n_pos": int(t.sum()), "n_neg": int((~t).sum()), "prevalence": float(t.mean()),
            "auroc": auroc(p, t), "auprc": average_precision(p, t),
            "threshold": float(thresholds[j]), "precision": precision, "recall": recall,
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "brier": float(np.mean((p - t) ** 2)), "ece": _calibration(p, t),
        }
    return rows


def _macro(per_label, scored, key):
    values = [per_label[n][key] for n in scored if not np.isnan(per_label[n][key])]
    return float(np.mean(values)) if values else float("nan")


def scored_labels(y, min_positives):
    return [n for j, n in enumerate(LABELS)
            if y[:, j].sum() >= min_positives and (1 - y[:, j]).sum() >= min_positives]


def summarize(y, probs, thresholds, prior, min_positives=10, scored=None):
    """Every metric for one set of cases. `scored` fixes the macro label set (default: from `y`)."""
    y = np.asarray(y, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    ok = ~np.isnan(probs).any(axis=1)
    strict = np.where(ok[:, None], probs, np.asarray(prior)[None])
    scored = scored if scored is not None else scored_labels(y, min_positives)

    per_label = _per_label(y, strict, thresholds)
    per_label_ev = _per_label(y[ok], probs[ok], thresholds) if ok.any() else None
    for name in LABELS:
        per_label[name]["scored"] = name in scored
    hard = strict >= np.asarray(thresholds)[None]
    eps = 1e-7
    bce = -(y * np.log(np.clip(strict, eps, 1)) + (1 - y) * np.log(np.clip(1 - strict, eps, 1)))
    out = {"n": len(y), "n_evaluated": int(ok.sum()), "n_failed": int((~ok).sum()),
           "scored_labels": list(scored),
           "exact_match": float(np.mean((hard == y.astype(bool)).all(axis=1))) if len(y) else float("nan"),
           "hamming_accuracy": float(np.mean(hard == y.astype(bool))) if len(y) else float("nan"),
           "mean_bce": float(bce.mean()) if len(y) else float("nan"),
           "per_label": per_label, "per_label_evaluated": per_label_ev}
    for key in ("auroc", "auprc", "f1", "brier", "ece"):
        out[f"macro_{key}"] = _macro(per_label, scored, key)
        out[f"macro_{key}_evaluated"] = (_macro(per_label_ev, scored, key) if per_label_ev
                                         else float("nan"))
    return out


BOOTSTRAP_KEYS = ("macro_auroc", "macro_auprc", "macro_f1")


def _headline(y, probs, thresholds, prior, scored):
    ok = ~np.isnan(probs).any(axis=1)
    strict = np.where(ok[:, None], probs, np.asarray(prior)[None])
    idx = [LABELS.index(n) for n in scored]
    values = {"macro_auroc": [], "macro_auprc": [], "macro_f1": []}
    for j in idx:
        p, t = strict[:, j], y[:, j].astype(bool)
        pred = p >= thresholds[j]
        tp, fp, fn = np.sum(pred & t), np.sum(pred & ~t), np.sum(~pred & t)
        values["macro_auroc"].append(auroc(p, t))
        values["macro_auprc"].append(average_precision(p, t))
        values["macro_f1"].append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
            for k, v in values.items()}


def _cluster_draws(groups, n_boot, seed):
    """Case indices for `n_boot` resamples of whole groups, with replacement."""
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([members[g] for g in rng.integers(0, len(unique), len(unique))])


def _percentiles(draws, level):
    alpha = (1 - level) / 2
    return {k: [float(np.nanquantile(v, alpha)), float(np.nanquantile(v, 1 - alpha))]
            for k, v in draws.items()}


def bootstrap_ci(y, probs, groups, thresholds, prior, scored, n_boot=2000, seed=0, level=0.95):
    """Percentile CIs for `BOOTSTRAP_KEYS`, resampling studies. The macro label set stays fixed."""
    y, probs = np.asarray(y, dtype=np.float64), np.asarray(probs, dtype=np.float64)
    draws = {k: [] for k in BOOTSTRAP_KEYS}
    for idx in _cluster_draws(groups, n_boot, seed):
        s = _headline(y[idx], probs[idx], thresholds, prior, scored)
        for k in BOOTSTRAP_KEYS:
            draws[k].append(s[k])
    return _percentiles(draws, level)


def paired_bootstrap(y, probs_a, probs_b, groups, thresholds, prior, scored, n_boot=2000, seed=0,
                     level=0.95):
    """CIs for `metric(a) - metric(b)`, the same resampled studies for both sides."""
    y = np.asarray(y, dtype=np.float64)
    probs_a, probs_b = np.asarray(probs_a, np.float64), np.asarray(probs_b, np.float64)
    draws = {k: [] for k in BOOTSTRAP_KEYS}
    for idx in _cluster_draws(groups, n_boot, seed):
        a = _headline(y[idx], probs_a[idx], thresholds, prior, scored)
        b = _headline(y[idx], probs_b[idx], thresholds, prior, scored)
        for k in BOOTSTRAP_KEYS:
            draws[k].append(a[k] - b[k])
    return _percentiles(draws, level)


def markdown(summary, ci=None, title=""):
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    lines = [f"### {title}", "",
             f"{summary['n']} cases, {summary['n_evaluated']} evaluated, {summary['n_failed']} "
             f"failed (failed cases score the train prior in the strict metrics). Macro means over "
             f"{len(summary['scored_labels'])} scored labels.", "",
             "| metric | strict | 95% CI (study bootstrap) | evaluated only |", "|---|---|---|---|"]
    for key in ("auroc", "auprc", "f1", "brier", "ece"):
        k = f"macro_{key}"
        interval = f"[{ci[k][0]:.4f}, {ci[k][1]:.4f}]" if ci and k in ci else ""
        lines.append(f"| {k} | {f(summary[k])} | {interval} | {f(summary[k + '_evaluated'])} |")
    lines += [f"| exact_match | {f(summary['exact_match'])} | | |",
              f"| hamming_accuracy | {f(summary['hamming_accuracy'])} | | |",
              f"| mean_bce | {f(summary['mean_bce'])} | | |", "",
              "| label | scored | pos | prevalence | AUROC | AUPRC | thr | precision | recall | "
              "specificity | F1 | Brier |", "|---" * 12 + "|"]
    for name, r in summary["per_label"].items():
        lines.append(f"| {name} | {'yes' if r['scored'] else 'no'} | {r['n_pos']} | "
                     f"{r['prevalence']:.3f} | {f(r['auroc'])} | {f(r['auprc'])} | "
                     f"{r['threshold']:.3f} | {f(r['precision'])} | {f(r['recall'])} | "
                     f"{f(r['specificity'])} | {f(r['f1'])} | {f(r['brier'])} |")
    return "\n".join(lines) + "\n"
