"""Hypothesis tests for the most common product metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    control: float
    treatment: float
    diff: float              # treatment - control (absolute)
    rel_diff: float          # diff / control
    ci_low: float
    ci_high: float
    p_value: float
    method: str

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def __str__(self) -> str:
        return (f"{self.metric}: {self.control:.4g} -> {self.treatment:.4g} "
                f"({self.rel_diff:+.2%}), diff CI [{self.ci_low:.4g}, {self.ci_high:.4g}], "
                f"p={self.p_value:.4f} [{self.method}]")


def _q(alpha: float) -> float:
    return stats.norm.ppf(1 - alpha / 2)


def ztest_proportions(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05,
                      metric: str = "conversion") -> TestResult:
    """Two-proportion z-test: pooled SE for p-value, unpooled SE for the CI."""
    p_c, p_t = x_c / n_c, x_t / n_t
    p = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p * (1 - p) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    d = p_t - p_c
    return TestResult(metric, p_c, p_t, d, d / p_c, d - _q(alpha) * se, d + _q(alpha) * se,
                      float(2 * stats.norm.sf(abs(z))), "z-test")


def welch_ttest(c: np.ndarray, t: np.ndarray, alpha: float = 0.05, metric: str = "mean") -> TestResult:
    """Welch t-test for means (unequal variances). Fine for heavy tails at large n (CLT)."""
    c, t = np.asarray(c, float), np.asarray(t, float)
    m_c, m_t = c.mean(), t.mean()
    v_c, v_t = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = np.sqrt(v_c + v_t)
    df = (v_c + v_t) ** 2 / (v_c ** 2 / (len(c) - 1) + v_t ** 2 / (len(t) - 1))
    tstat = (m_t - m_c) / se
    q = stats.t.ppf(1 - alpha / 2, df)
    d = m_t - m_c
    return TestResult(metric, m_c, m_t, d, d / m_c, d - q * se, d + q * se,
                      float(2 * stats.t.sf(abs(tstat), df)), "Welch t-test")


def bootstrap_diff(c: np.ndarray, t: np.ndarray, stat=np.mean, n_boot: int = 5000,
                   alpha: float = 0.05, seed: int = 0, metric: str = "stat") -> TestResult:
    """Percentile bootstrap for any statistic (median, quantiles, trimmed mean...).
    p-value: share of bootstrap differences on the other side of zero, two-sided."""
    rng = np.random.default_rng(seed)
    c, t = np.asarray(c, float), np.asarray(t, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = stat(rng.choice(t, len(t))) - stat(rng.choice(c, len(c)))
    s_c, s_t = stat(c), stat(t)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return TestResult(metric, s_c, s_t, s_t - s_c, (s_t - s_c) / s_c, lo, hi, min(p, 1.0), "bootstrap")


def _ratio_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Delta-method variance of sum(num)/sum(den) with user-level randomization."""
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cov = np.cov(num, den, ddof=1)[0, 1]
    r = mx / my
    var = (vx - 2 * r * cov + r ** 2 * vy) / (n * my ** 2)
    return r, var


def delta_method_ratio(num_c, den_c, num_t, den_t, alpha: float = 0.05,
                       metric: str = "ratio") -> TestResult:
    """Ratio metrics (CTR = clicks/views, AOV = revenue/orders) when the unit of
    randomization is the user, not the view. A naive per-view test underestimates
    variance because views of one user are correlated."""
    r_c, v_c = _ratio_var(np.asarray(num_c, float), np.asarray(den_c, float))
    r_t, v_t = _ratio_var(np.asarray(num_t, float), np.asarray(den_t, float))
    d, se = r_t - r_c, np.sqrt(v_c + v_t)
    return TestResult(metric, r_c, r_t, d, d / r_c, d - _q(alpha) * se, d + _q(alpha) * se,
                      float(2 * stats.norm.sf(abs(d / se))), "delta method")
