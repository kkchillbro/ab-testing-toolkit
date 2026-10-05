"""
End-to-end experiment readout: one-page checkout vs current 3-step checkout.

Fictional e-commerce store, synthetic data with a known true effect.
Steps: design (sample size) -> SRM check -> primary metric -> secondary metrics
(CUPED, delta method) -> guardrail -> multiple-testing correction -> decision.

Run:  python examples/checkout_experiment.py   -> reports/checkout_experiment.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abtk  # noqa: E402

TRUE = {"conv_lift_rel": 0.08, "aov_lift_rel": -0.02}


def simulate(n_per_group: int = 32_000, seed: int = 11) -> dict:
    """Each user has a latent purchase propensity and basket size. The same traits
    drive the 6-month pre-period, so pre-period revenue is a good CUPED covariate."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in ["control", "treatment"]:
        n = n_per_group + int(rng.integers(-150, 150))           # realistic imbalance
        propensity = rng.beta(1.2, 21, n)                         # mean ~5.4% per opportunity
        basket = rng.lognormal(3.6, 0.6, n)                       # user's typical order value
        pre_orders = rng.binomial(26, propensity)                  # 6 months of history
        pre_rev = pre_orders * basket * rng.lognormal(0, 0.2, n)

        lift = 1 + TRUE["conv_lift_rel"] if g == "treatment" else 1.0
        orders = rng.binomial(2, np.clip(propensity * lift, 0, 1))
        aov_mult = 1 + TRUE["aov_lift_rel"] if g == "treatment" else 1.0
        revenue = orders * basket * aov_mult * rng.lognormal(0, 0.2, n)
        refunds = rng.binomial(orders, 0.04)
        out[g] = {"n": n, "pre_rev": pre_rev, "orders": orders, "revenue": revenue,
                  "converted": (orders > 0).astype(int), "refunds": refunds}
    return out


def main() -> None:
    lines = ["# Checkout experiment readout (auto-generated)\n",
             "> Fictional store, synthetic data. True effect planted in the generator: "
             f"conversion {TRUE['conv_lift_rel']:+.0%}, AOV {TRUE['aov_lift_rel']:+.0%}.\n"]

    # ---------------------------------------------------------------- design
    base_conv, mde_rel = 0.10, 0.08
    n_needed = abtk.sample_size_proportions(base_conv, mde_rel, alpha=0.05, power=0.8)
    days = abtk.test_duration_days(n_needed, daily_users=4_600)
    lines += ["## 1. Design (before launch)\n",
              f"- Primary metric: purchase conversion, baseline ≈ {base_conv:.1%}",
              f"- MDE: {mde_rel:+.0%} relative → **{n_needed:,} users per group**, "
              f"≈ {days} days at 4,600 eligible users/day (rounded up to full weeks: 14 days)",
              "- Secondary: revenue per user (with CUPED), AOV (ratio metric, delta method)",
              "- Guardrail: refund rate per order. Decision rule fixed in advance: ship if the primary is "
              "significant and the guardrail does not get significantly worse.\n"]

    d = simulate()
    c, t = d["control"], d["treatment"]

    # ---------------------------------------------------------------- SRM
    srm = abtk.srm_check([c["n"], t["n"]])
    lines += ["## 2. Sanity check: sample ratio mismatch\n",
              f"Control {c['n']:,} / treatment {t['n']:,}, χ² p = {srm['p_value']:.3f} → "
              + ("**SRM detected, stop and debug the split**" if srm["srm"] else "no SRM, split is healthy.") + "\n"]

    # ---------------------------------------------------------------- metrics
    conv = abtk.ztest_proportions(c["converted"].sum(), c["n"], t["converted"].sum(), t["n"],
                                  metric="conversion")
    rpu_raw = abtk.welch_ttest(c["revenue"], t["revenue"], metric="revenue/user (raw)")
    a_c, a_t, theta = abtk.cuped_adjust(c["revenue"], c["pre_rev"], t["revenue"], t["pre_rev"])
    rpu_cuped = abtk.welch_ttest(a_c, a_t, metric="revenue/user (CUPED)")
    var_red = 1 - np.var(np.r_[a_c, a_t]) / np.var(np.r_[c["revenue"], t["revenue"]])
    aov = abtk.delta_method_ratio(c["revenue"], c["orders"], t["revenue"], t["orders"], metric="AOV")
    refund = abtk.delta_method_ratio(c["refunds"], c["orders"], t["refunds"], t["orders"],
                                     metric="refund rate")

    def fmt_p(p):
        return "<0.0001" if p < 1e-4 else f"{p:.4f}"

    def row(r):
        return (f"| {r.metric} | {r.control:.4g} | {r.treatment:.4g} | {r.rel_diff:+.1%} | "
                f"[{r.ci_low:+.4g}, {r.ci_high:+.4g}] | {fmt_p(r.p_value)} | {r.method} |")

    lines += ["## 3. Results\n",
              "| metric | control | treatment | lift | 95% CI of diff | p-value | method |",
              "|---|---|---|---|---|---|---|",
              row(conv), row(rpu_raw), row(rpu_cuped), row(aov), row(refund), "",
              f"CUPED (θ = {theta:.2f}) removed only **{var_red:.0%}** of revenue variance: ~89% of users "
              "have zero revenue in a 2-week window, so the pre-period explains little. On dense metrics "
              "(sessions, minutes) with ρ ≈ 0.7 the saving is ~50% (see `reports/simulations.md`). "
              "Lesson: check ρ on historical data before promising a shorter test.\n"]

    # ---------------------------------------------------------------- multiple testing
    secondary = [rpu_cuped.p_value, aov.p_value]
    holm = abtk.holm(secondary)
    lines += ["## 4. Multiple testing\n",
              "Primary metric is tested at α = 5%. The two secondary metrics are corrected with Holm: "
              f"revenue/user (CUPED) → {'significant' if holm[0] else 'not significant'}, "
              f"AOV → {'significant' if holm[1] else 'not significant'}.\n"]

    # ---------------------------------------------------------------- decision
    ship = conv.significant and conv.diff > 0 and not (refund.significant and refund.diff > 0)
    aov_note = (f"AOV {aov.rel_diff:+.1%} is significant: more but smaller baskets, check the revenue trade-off."
                if aov.significant else
                f"AOV {aov.rel_diff:+.1%} is not significant, so the revenue gain comes from conversion.")
    refund_note = (f"Refund rate is not significantly worse (p = {refund.p_value:.2f}), but its CI reaches "
                   f"{refund.ci_high*100:+.1f} pp, so it stays on the post-launch dashboard."
                   if not refund.significant else
                   f"Refund rate changed significantly ({refund.rel_diff:+.1%}).")
    lines += ["## 5. Decision\n",
              f"**{'SHIP' if ship else 'DO NOT SHIP'}.** Conversion {conv.rel_diff:+.1%} "
              f"(CI {conv.ci_low*100:+.2f}…{conv.ci_high*100:+.2f} pp). {aov_note} {refund_note} "
              "Watch for a novelty effect: compare the lift for new vs returning users over the next 4 weeks.\n"]

    report = "\n".join(lines)
    (ROOT / "reports" / "checkout_experiment.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
