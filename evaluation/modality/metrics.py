"""Classification metrics that account for failures instead of dropping them.

A case whose volume is missing or could not be classified has `y_pred = -1` and a NaN probability
row. Two readings are reported, and every key says which it is:

    strict      (no suffix) the denominator is every expected case; a failure counts as wrong,
                gets probability 0 for its expected class, and sits in the confusion matrix's
                extra `failed` column. This is the primary number.
    evaluated   (`_evaluated` suffix, and every probability-based metric) only the cases that
                produced a prediction.

The confusion matrix is `[true, predicted]`: row = expected class, column = predicted class.
"""

import numpy as np
from scipy.stats import rankdata

from evaluation.modality import CLASSES, NUM_CLASSES

FAILED = -1


def _per_class(y_true, y_pred):
    """Per-class precision/recall/F1 and support, and their macro means over the classes present.

    Precision of a class nothing was predicted as is 0 (sklearn's `zero_division=0`); a class with
    no support is left out of the macro means rather than counted as 0 or 1.
    """
    rows, present = {}, []
    for c, name in enumerate(CLASSES):
        support = int(np.sum(y_true == c))
        predicted = int(np.sum(y_pred == c))
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else (0.0 if support else float("nan"))
        f1 = (2 * precision * recall / (precision + recall) if support and precision + recall > 0
              else (0.0 if support else float("nan")))
        rows[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            present.append(name)
    macro = {k: float(np.mean([rows[n][k] for n in present])) if present else float("nan")
             for k in ("precision", "recall", "f1")}
    return rows, macro


def auroc(scores, positive):
    """One-vs-rest AUROC by the Mann-Whitney rank statistic; ties count half. NaN without both
    classes."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(probs, y_true, n_bins=15):
    """Top-1 ECE over equal-width confidence bins, weighted by bin size."""
    if len(probs) == 0:
        return float("nan")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(correct[bins == b].mean() - confidence[bins == b].mean()) * np.mean(bins == b)
                     for b in range(n_bins) if np.any(bins == b)))


def summarize(y_true, y_pred, probs):
    """Every metric for one set of cases. `probs` columns are in `CLASSES` order."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    probs = np.asarray(probs, dtype=np.float64).reshape(len(y_true), NUM_CLASSES)
    ok = y_pred != FAILED
    n = len(y_true)

    per_class, macro = _per_class(y_true, y_pred)
    per_class_ev, macro_ev = _per_class(y_true[ok], y_pred[ok])
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p if p != FAILED else NUM_CLASSES] += 1

    p_ok, t_ok = probs[ok], y_true[ok]
    p_expected = p_ok[np.arange(len(t_ok)), t_ok] if len(t_ok) else np.zeros(0)
    aurocs = {name: auroc(p_ok[:, c], t_ok == c) for c, name in enumerate(CLASSES)}
    defined = [v for v in aurocs.values() if not np.isnan(v)]

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    return {
        "n": n, "n_evaluated": int(ok.sum()), "n_failed": int(n - ok.sum()),
        "accuracy": mean(y_pred == y_true),
        "balanced_accuracy": macro["recall"],
        "macro_precision": macro["precision"], "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "mean_p_expected": float(p_expected.sum() / n) if n else float("nan"),
        "accuracy_evaluated": mean(y_pred[ok] == y_true[ok]),
        "balanced_accuracy_evaluated": macro_ev["recall"],
        "macro_f1_evaluated": macro_ev["f1"],
        "mean_p_expected_evaluated": mean(p_expected),
        "nll_evaluated": mean(-np.log(np.clip(p_expected, 1e-12, 1.0))),
        "ece_evaluated": expected_calibration_error(p_ok, t_ok),
        "auroc_evaluated": aurocs,
        "macro_auroc_evaluated": float(np.mean(defined)) if defined else float("nan"),
        "per_class": per_class,
        "per_class_evaluated": per_class_ev,
        "confusion_matrix": {"rows_true": list(CLASSES),
                             "cols_predicted": list(CLASSES) + ["failed"],
                             "counts": confusion.tolist()},
    }


BOOTSTRAP_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "mean_p_expected")


def _headline(y_true, y_pred, probs):
    """The strict `BOOTSTRAP_KEYS`, and nothing else -- `summarize` is too slow to resample."""
    _, macro = _per_class(y_true, y_pred)
    ok = y_pred != FAILED
    p_expected = probs[ok, y_true[ok]].sum() / len(y_true)
    return {"accuracy": float(np.mean(y_pred == y_true)), "balanced_accuracy": macro["recall"],
            "macro_f1": macro["f1"], "mean_p_expected": float(p_expected)}


def bootstrap_ci(y_true, y_pred, probs, n_boot=2000, seed=0, level=0.95):
    """Percentile CIs over case resamples for `BOOTSTRAP_KEYS`. Fixed seed, so reproducible."""
    y_true, y_pred, probs = np.asarray(y_true), np.asarray(y_pred), np.asarray(probs)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in BOOTSTRAP_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        s = _headline(y_true[idx], y_pred[idx], probs[idx])
        for k in BOOTSTRAP_KEYS:
            draws[k].append(s[k])
    return _percentiles(draws, level)


def _percentiles(draws, level):
    alpha = (1 - level) / 2
    return {k: [float(np.nanquantile(v, alpha)), float(np.nanquantile(v, 1 - alpha))]
            for k, v in draws.items()}


def paired_bootstrap(y_true, pred_a, probs_a, pred_b, probs_b, n_boot=2000, seed=0, level=0.95):
    """CIs for `metric(a) - metric(b)`, resampling the same case indices for both sides -- the
    right test when a and b are the same cases (a ground truth and the volume generated for it)."""
    y_true = np.asarray(y_true)
    pred_a, probs_a = np.asarray(pred_a), np.asarray(probs_a)
    pred_b, probs_b = np.asarray(pred_b), np.asarray(probs_b)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in BOOTSTRAP_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        a = _headline(y_true[idx], pred_a[idx], probs_a[idx])
        b = _headline(y_true[idx], pred_b[idx], probs_b[idx])
        for k in BOOTSTRAP_KEYS:
            draws[k].append(a[k] - b[k])
    return _percentiles(draws, level)


### Per-case records and reports ###


def case_record(expected, probs=None, error="", status="failed"):
    """One case's prediction fields. `probs=None` is a failure -- `status` "failed" (a volume that
    could not be read or classified) or "missing" (no volume at all) -- and carries `error`."""
    record = {"expected": expected, "predicted": "", "confidence": float("nan"),
              "correct": False, "status": status if probs is None else "ok", "error": error,
              **{f"p_{name}": float("nan") for name in CLASSES}}
    if probs is not None:
        k = int(np.argmax(probs))
        record.update(predicted=CLASSES[k], confidence=float(probs[k]),
                      correct=CLASSES[k] == expected,
                      **{f"p_{name}": float(p) for name, p in zip(CLASSES, probs)})
    return record


def arrays(records):
    """Case records -> (y_true, y_pred, probs) for `summarize`."""
    y_true = np.array([CLASSES.index(r["expected"]) for r in records], dtype=int)
    y_pred = np.array([CLASSES.index(r["predicted"]) if r["status"] == "ok" else FAILED
                       for r in records], dtype=int)
    probs = np.array([[r[f"p_{name}"] for name in CLASSES] for r in records],
                     dtype=np.float64).reshape(len(records), NUM_CLASSES)
    return y_true, y_pred, probs


def markdown(summary, ci=None, title=""):
    """A human-readable version of one `summarize` result."""
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    lines = [f"### {title}", "",
             f"{summary['n']} cases, {summary['n_evaluated']} evaluated, "
             f"{summary['n_failed']} failed (failures count as wrong in the strict metrics).", "",
             "| metric | strict | 95% CI | evaluated only |", "|---|---|---|---|"]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mean_p_expected"):
        interval = f"[{ci[key][0]:.4f}, {ci[key][1]:.4f}]" if ci else ""
        lines.append(f"| {key} | {f(summary[key])} | {interval} | "
                     f"{f(summary[key + '_evaluated'])} |")
    lines += [f"| macro_precision | {f(summary['macro_precision'])} | | |",
              f"| NLL | | | {f(summary['nll_evaluated'])} |",
              f"| ECE (15 bins) | | | {f(summary['ece_evaluated'])} |",
              f"| macro AUROC (OvR) | | | {f(summary['macro_auroc_evaluated'])} |", "",
              "| class | precision | recall | F1 | support | AUROC |", "|---|---|---|---|---|---|"]
    for name in CLASSES:
        row = summary["per_class"][name]
        lines.append(f"| {name} | {f(row['precision'])} | {f(row['recall'])} | {f(row['f1'])} | "
                     f"{row['support']} | {f(summary['auroc_evaluated'][name])} |")
    cm = summary["confusion_matrix"]
    lines += ["", "Confusion matrix (rows = expected, columns = predicted):", "",
              "| | " + " | ".join(cm["cols_predicted"]) + " |",
              "|---" * (len(cm["cols_predicted"]) + 1) + "|"]
    lines += [f"| **{name}** | " + " | ".join(str(v) for v in row) + " |"
              for name, row in zip(cm["rows_true"], cm["counts"])]
    return "\n".join(lines) + "\n"
