from typing import Sequence

import numpy as np


def gou_gtxr(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """
    Python translation of elitism::mtp(..., method="gtxr").

    Returns a 0/1 rejection vector in the original p-value order.
    """
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be a 1D sequence.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("pvals must lie in [0, 1].")

    n = p.size
    if n == 0:
        return np.array([], dtype=int)

    order = np.argsort(p, kind="quicksort")
    p_sorted = p[order]
    rej_idx = 0

    # Direct translation of elitism:::mtp.gtxr0b.
    for i in range(1, n + 1):
        idx = n - i
        if p_sorted[idx] <= ((i + 1) / (2 * i)) * alpha:
            # BinarySearch(...) in the package returns the count of sorted
            # p-values <= alpha / i among the first n-i+1 entries.
            rej_idx = int(np.searchsorted(p_sorted[: idx + 1], alpha / i, side="right"))
            break

    rejected_sorted = np.zeros(n, dtype=int)
    if rej_idx > 0:
        rejected_sorted[:rej_idx] = 1

    rejected = np.zeros(n, dtype=int)
    rejected[order] = rejected_sorted
    return rejected


def gou_gtxr_batch(pvals_matrix: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Batch version for shape (n_samples, K).
    """
    p = np.asarray(pvals_matrix, dtype=float)
    if p.ndim == 1:
        return gou_gtxr(p, alpha=alpha)[None, :]
    if p.ndim != 2:
        raise ValueError("pvals_matrix must be a 1D or 2D array.")

    out = np.zeros_like(p, dtype=int)
    for i in range(p.shape[0]):
        out[i] = gou_gtxr(p[i], alpha=alpha)
    return out
