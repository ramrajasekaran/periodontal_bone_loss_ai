"""
metrics.py  —  Pure NumPy/PyTorch metric functions.
Replaces sklearn.metrics entirely so that scipy (blocked by WDAC) is never imported.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_np(y_true, y_pred, num_classes=3):
    """Return (num_classes, num_classes) confusion matrix as numpy array."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Per-class and macro metrics
# ---------------------------------------------------------------------------

def per_class_prf(y_true, y_pred, num_classes=3):
    """
    Compute per-class precision, recall, F1 from a confusion matrix.

    Returns
    -------
    precision : ndarray, shape (num_classes,)
    recall    : ndarray, shape (num_classes,)
    f1        : ndarray, shape (num_classes,)
    support   : ndarray, shape (num_classes,)  — number of true samples per class
    """
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    precision = np.zeros(num_classes, dtype=float)
    recall    = np.zeros(num_classes, dtype=float)
    f1        = np.zeros(num_classes, dtype=float)
    support   = cm.sum(axis=1)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[i] + recall[i]
        f1[i]        = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0

    return precision, recall, f1, support


def macro_prf(y_true, y_pred, num_classes=3):
    """Macro-averaged precision, recall, F1 (unweighted mean over classes)."""
    precision, recall, f1, _ = per_class_prf(y_true, y_pred, num_classes)
    return precision.mean(), recall.mean(), f1.mean()


def accuracy_np(y_true, y_pred):
    """Simple accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true == y_pred))


def macro_f1_np(y_true, y_pred, num_classes=3):
    """Convenience wrapper — returns only macro F1."""
    _, _, f1 = macro_prf(y_true, y_pred, num_classes)
    return float(f1)


# ---------------------------------------------------------------------------
# One-vs-Rest ROC-AUC  (no scipy)
# ---------------------------------------------------------------------------

def _auc_binary(y_binary, y_score):
    """Area under the ROC curve via trapezoidal rule (no scipy)."""
    y_binary = np.asarray(y_binary, dtype=float)
    y_score  = np.asarray(y_score,  dtype=float)

    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_binary[desc_idx]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    tpr = np.concatenate([[0], tp / n_pos])
    fpr = np.concatenate([[0], fp / n_neg])
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trapz_fn(tpr, fpr)) if trapz_fn else float('nan')


def roc_auc_ovr(y_true, y_prob, num_classes=3):
    """
    Macro One-vs-Rest AUC without scipy.

    Parameters
    ----------
    y_true  : array-like, shape (n,)          — integer class labels
    y_prob  : array-like, shape (n, num_classes) — softmax probabilities
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    aucs = []
    for c in range(num_classes):
        binary = (y_true == c).astype(float)
        auc = _auc_binary(binary, y_prob[:, c])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float('nan')


# ---------------------------------------------------------------------------
# Classification report (text)
# ---------------------------------------------------------------------------

def classification_report_str(y_true, y_pred, class_names, num_classes=3):
    """
    Human-readable classification report identical in format to sklearn's.
    """
    precision, recall, f1, support = per_class_prf(y_true, y_pred, num_classes)
    acc = accuracy_np(y_true, y_pred)

    # Header
    lines = []
    header = f"{'':>20s}  {'precision':>9s}  {'recall':>9s}  {'f1-score':>9s}  {'support':>9s}"
    lines.append(header)
    lines.append('')

    for i, name in enumerate(class_names):
        lines.append(
            f"{name:>20s}  {precision[i]:9.4f}  {recall[i]:9.4f}  {f1[i]:9.4f}  {support[i]:9d}"
        )

    lines.append('')
    macro_p, macro_r, macro_f = precision.mean(), recall.mean(), f1.mean()
    total = int(support.sum())
    lines.append(f"{'macro avg':>20s}  {macro_p:9.4f}  {macro_r:9.4f}  {macro_f:9.4f}  {total:9d}")

    weighted_p = float(np.average(precision, weights=support))
    weighted_r = float(np.average(recall,    weights=support))
    weighted_f = float(np.average(f1,        weights=support))
    lines.append(f"{'weighted avg':>20s}  {weighted_p:9.4f}  {weighted_r:9.4f}  {weighted_f:9.4f}  {total:9d}")
    lines.append('')
    lines.append(f"  Accuracy : {acc:.4f}  ({int(np.sum(np.asarray(y_true)==np.asarray(y_pred)))}/{len(y_true)})")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Confidence / calibration helpers
# ---------------------------------------------------------------------------

def prediction_entropy(probs):
    """Shannon entropy of probability vector (higher = less certain)."""
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def is_low_confidence(probs, threshold=0.60):
    """True when maximum softmax probability is below threshold."""
    return float(np.max(probs)) < threshold
