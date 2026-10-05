"""Multiple-testing corrections."""
from __future__ import annotations

import numpy as np


def holm(p_values, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni: controls family-wise error rate. Returns a reject mask."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    reject = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            reject[i] = True
        else:
            break
    return reject


def benjamini_hochberg(p_values, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: controls false discovery rate. Returns a reject mask."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, bool)
    if passed.any():
        k = np.max(np.flatnonzero(passed))
        reject[order[: k + 1]] = True
    return reject
