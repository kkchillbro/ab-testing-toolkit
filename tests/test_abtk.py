import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abtk  # noqa: E402


class TestPower(unittest.TestCase):
    def test_sample_size_proportions_formula(self):
        # p=10% -> 11%: (1.95996+0.84162)^2 * (0.09+0.0979) / 0.01^2
        n = abtk.sample_size_proportions(0.10, 0.10)
        self.assertAlmostEqual(n, 14749, delta=3)

    def test_mde_roundtrip(self):
        n = abtk.sample_size_proportions(0.2, 0.02, relative=False)
        mde = abtk.mde_proportions(0.2, n)
        self.assertAlmostEqual(mde, 0.02, delta=0.001)

    def test_duration_at_least_a_week(self):
        self.assertEqual(abtk.test_duration_days(100, 10_000), 7)


class TestTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_welch_matches_scipy(self):
        c, t = self.rng.exponential(10, 3000), self.rng.exponential(10.5, 2500)
        r = abtk.welch_ttest(c, t)
        ref = stats.ttest_ind(t, c, equal_var=False)
        self.assertAlmostEqual(r.p_value, ref.pvalue, places=10)

    def test_ztest_detects_effect(self):
        r = abtk.ztest_proportions(1000, 10000, 1150, 10000)
        self.assertTrue(r.significant)
        self.assertAlmostEqual(r.diff, 0.015)

    def test_delta_method_close_to_bootstrap(self):
        views = self.rng.poisson(5, 4000) + 1
        clicks = self.rng.binomial(views, 0.1)
        _, var = abtk.tests._ratio_var(clicks, views)
        boots = []
        for _ in range(1500):
            i = self.rng.integers(0, len(views), len(views))
            boots.append(clicks[i].sum() / views[i].sum())
        self.assertAlmostEqual(np.sqrt(var), np.std(boots), delta=0.1 * np.std(boots))

    def test_srm(self):
        self.assertFalse(abtk.srm_check([50_100, 49_900])["srm"])
        self.assertTrue(abtk.srm_check([51_000, 49_000])["srm"])


class TestCuped(unittest.TestCase):
    def test_unbiased_and_lower_variance(self):
        rng = np.random.default_rng(2)
        x = rng.gamma(2, 50, 20000)
        y = 0.8 * x + rng.normal(0, 30, 20000)
        y_c, y_t, x_c, x_t = y[:10000], y[10000:] + 5, x[:10000], x[10000:]
        a_c, a_t, theta = abtk.cuped_adjust(y_c, x_c, y_t, x_t)
        self.assertAlmostEqual(a_t.mean() - a_c.mean(), 5, delta=1.0)
        self.assertLess(a_c.var(), 0.3 * y_c.var())
        self.assertAlmostEqual(theta, 0.8, delta=0.05)


class TestMultiple(unittest.TestCase):
    def test_holm_and_bh(self):
        p = [0.01, 0.04, 0.03, 0.005]
        np.testing.assert_array_equal(abtk.holm(p), [True, False, False, True])
        np.testing.assert_array_equal(abtk.benjamini_hochberg(p), [True, True, True, True])


if __name__ == "__main__":
    unittest.main()
