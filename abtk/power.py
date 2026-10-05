"""Sample size, MDE and duration planning."""
from __future__ import annotations

import math

from scipy import stats


def _z(alpha: float, power: float, two_sided: bool = True) -> tuple[float, float]:
    za = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    return za, stats.norm.ppf(power)


def sample_size_proportions(p_base: float, mde: float, relative: bool = True,
                            alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> int:
    """Users needed in the CONTROL group to detect `mde` on a conversion rate.

    mde: relative (0.05 = +5%) or absolute (0.01 = +1 pp) lift.
    ratio: n_treatment / n_control.
    Uses the unpooled normal approximation.
    """
    p2 = p_base * (1 + mde) if relative else p_base + mde
    if not 0 < p2 < 1:
        raise ValueError("treatment rate must be in (0, 1)")
    za, zb = _z(alpha, power)
    var = p_base * (1 - p_base) + p2 * (1 - p2) / ratio
    return math.ceil((za + zb) ** 2 * var / (p2 - p_base) ** 2)


def sample_size_means(std: float, mde_abs: float, alpha: float = 0.05,
                      power: float = 0.8, ratio: float = 1.0) -> int:
    """Users per control group to detect an absolute difference in means."""
    za, zb = _z(alpha, power)
    return math.ceil((za + zb) ** 2 * std ** 2 * (1 + 1 / ratio) / mde_abs ** 2)


def mde_proportions(p_base: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Approximate absolute MDE for a given sample size (equal groups)."""
    za, zb = _z(alpha, power)
    return (za + zb) * math.sqrt(2 * p_base * (1 - p_base) / n_per_group)


def test_duration_days(n_per_group: int, daily_users: int, n_groups: int = 2,
                       traffic_share: float = 1.0, min_days: int = 7) -> int:
    """Days needed; never less than one full week to cover weekday seasonality."""
    days = math.ceil(n_per_group * n_groups / (daily_users * traffic_share))
    return max(days, min_days)
