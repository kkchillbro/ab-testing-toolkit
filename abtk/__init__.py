"""abtk: a small, dependency-light A/B testing toolkit (NumPy + SciPy)."""
from .power import sample_size_proportions, sample_size_means, mde_proportions, test_duration_days
from .srm import srm_check
from .tests import ztest_proportions, welch_ttest, bootstrap_diff, delta_method_ratio, TestResult
from .cuped import cuped_adjust
from .multiple import holm, benjamini_hochberg

__all__ = [
    "sample_size_proportions", "sample_size_means", "mde_proportions", "test_duration_days",
    "srm_check", "ztest_proportions", "welch_ttest", "bootstrap_diff", "delta_method_ratio",
    "TestResult", "cuped_adjust", "holm", "benjamini_hochberg",
]
