"""CUPED: variance reduction with a pre-experiment covariate."""
from __future__ import annotations

import numpy as np


def cuped_adjust(y_c, x_c, y_t, x_t) -> tuple[np.ndarray, np.ndarray, float]:
    """Return CUPED-adjusted metrics for both groups and theta.

    y: metric during the experiment, x: same metric for the same user before it.
    theta is estimated on the pooled data, so the adjustment is identical for both
    groups and the treatment effect estimate stays unbiased.
    Variance shrinks by a factor of (1 - corr(x, y)^2).
    """
    y = np.concatenate([y_c, y_t]).astype(float)
    x = np.concatenate([x_c, x_t]).astype(float)
    theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    mx = x.mean()
    adj_c = np.asarray(y_c, float) - theta * (np.asarray(x_c, float) - mx)
    adj_t = np.asarray(y_t, float) - theta * (np.asarray(x_t, float) - mx)
    return adj_c, adj_t, float(theta)
