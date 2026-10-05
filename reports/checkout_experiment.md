# Checkout experiment readout (auto-generated)

> Fictional store, synthetic data. True effect planted in the generator: conversion +8%, AOV -2%.

## 1. Design (before launch)

- Primary metric: purchase conversion, baseline ≈ 10.0%
- MDE: +8% relative → **22,853 users per group**, ≈ 10 days at 4,600 eligible users/day (rounded up to full weeks: 14 days)
- Secondary: revenue per user (with CUPED), AOV (ratio metric, delta method)
- Guardrail: refund rate per order. Decision rule fixed in advance: ship if the primary is significant and the guardrail does not get significantly worse.

## 2. Sanity check: sample ratio mismatch

Control 31,890 / treatment 31,888, χ² p = 0.994 → no SRM, split is healthy.

## 3. Results

| metric | control | treatment | lift | 95% CI of diff | p-value | method |
|---|---|---|---|---|---|---|
| conversion | 0.1046 | 0.1145 | +9.4% | [+0.005038, +0.01473] | <0.0001 | z-test |
| revenue/user (raw) | 4.825 | 5.288 | +9.6% | [+0.1807, +0.7449] | 0.0013 | Welch t-test |
| revenue/user (CUPED) | 4.828 | 5.285 | +9.5% | [+0.1829, +0.731] | 0.0011 | Welch t-test |
| AOV | 43.82 | 43.59 | -0.5% | [-1.66, +1.197] | 0.7505 | delta method |
| refund rate | 0.0356 | 0.04317 | +21.3% | [-0.001273, +0.01642] | 0.0934 | delta method |

CUPED (θ = 0.04) removed only **6%** of revenue variance: ~89% of users have zero revenue in a 2-week window, so the pre-period explains little. On dense metrics (sessions, minutes) with ρ ≈ 0.7 the saving is ~50% (see `reports/simulations.md`). Lesson: check ρ on historical data before promising a shorter test.

## 4. Multiple testing

Primary metric is tested at α = 5%. The two secondary metrics are corrected with Holm: revenue/user (CUPED) → significant, AOV → not significant.

## 5. Decision

**SHIP.** Conversion +9.4% (CI +0.50…+1.47 pp). AOV -0.5% is not significant, so the revenue gain comes from conversion. Refund rate is not significantly worse (p = 0.09), but its CI reaches +1.6 pp, so it stays on the post-launch dashboard. Watch for a novelty effect: compare the lift for new vs returning users over the next 4 weeks.
