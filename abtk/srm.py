"""Sample Ratio Mismatch check."""
from __future__ import annotations

from typing import Sequence

from scipy import stats


def srm_check(observed: Sequence[int], expected_shares: Sequence[float] | None = None,
              threshold: float = 0.001) -> dict:
    """Chi-square goodness-of-fit test on group sizes.

    A strict threshold (p < 0.001) is standard: SRM means the split is broken
    (bucketing bug, bot filtering, redirects), and the test result can't be trusted.
    """
    total = sum(observed)
    k = len(observed)
    shares = expected_shares or [1 / k] * k
    expected = [total * s for s in shares]
    chi2, p = stats.chisquare(observed, expected)
    return {"chi2": float(chi2), "p_value": float(p), "srm": bool(p < threshold),
            "observed_shares": [o / total for o in observed]}
