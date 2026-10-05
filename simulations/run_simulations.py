"""
Three Monte-Carlo simulations that show common A/B testing mistakes.

1. Peeking: checking an A/A test every day and stopping at the first p < 0.05.
2. Wrong unit of analysis: testing CTR per view when users were randomized.
3. CUPED: how much variance (and sample size) a pre-period covariate saves.

Output: reports/figures/sim_*.png and reports/simulations.md
Run:    python simulations/run_simulations.py   (~1 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import abtk  # noqa: E402

FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
BLUE, RED, GREEN = "#2563eb", "#dc2626", "#16a34a"


def sim_peeking(n_sims: int = 3000, days: int = 21, users_per_day: int = 500,
                p: float = 0.10, seed: int = 0) -> dict:
    """A/A tests (no real effect). Cumulative z-test every day."""
    rng = np.random.default_rng(seed)
    conv_c = rng.binomial(users_per_day, p, (n_sims, days)).cumsum(1)
    conv_t = rng.binomial(users_per_day, p, (n_sims, days)).cumsum(1)
    n = users_per_day * np.arange(1, days + 1)
    pool = (conv_c + conv_t) / (2 * n)
    z = (conv_t - conv_c) / n / np.sqrt(pool * (1 - pool) * 2 / n)
    sig = np.abs(z) > stats.norm.ppf(0.975)
    fpr_by_day = sig.cumsum(1).astype(bool).mean(0)      # stopped at first significant look
    return {"fpr_fixed": sig[:, -1].mean(), "fpr_peeking": fpr_by_day[-1], "curve": fpr_by_day}


def sim_unit_of_analysis(n_sims: int = 2000, users: int = 2000, seed: int = 1) -> dict:
    """A/A tests on CTR. Users differ in their own click propensity, so views of one
    user are correlated. Naive per-view z-test vs delta method."""
    rng = np.random.default_rng(seed)
    naive_sig = delta_sig = 0
    for _ in range(n_sims):
        res = []
        for _g in range(2):
            views = rng.geometric(0.15, users)                   # heavy users exist
            ctr_user = rng.beta(2, 18, users)                    # user-level heterogeneity
            clicks = rng.binomial(views, ctr_user)
            res.append((clicks, views))
        (cc, vc), (ct, vt) = res
        naive = abtk.ztest_proportions(cc.sum(), vc.sum(), ct.sum(), vt.sum())
        delta = abtk.delta_method_ratio(cc, vc, ct, vt)
        naive_sig += naive.p_value < 0.05
        delta_sig += delta.p_value < 0.05
    return {"fpr_naive": naive_sig / n_sims, "fpr_delta": delta_sig / n_sims}


def sim_cuped(n: int = 20_000, seed: int = 2) -> dict:
    """Revenue per user: pre-period vs experiment period at different correlations."""
    rng = np.random.default_rng(seed)
    rows = []
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
        x = rng.normal(0, 1, n)
        y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
        y_c, y_t, x_c, x_t = y[: n // 2], y[n // 2:], x[: n // 2], x[n // 2:]
        a_c, a_t, _ = abtk.cuped_adjust(y_c, x_c, y_t, x_t)
        ratio = np.var(np.r_[a_c, a_t]) / np.var(y)
        rows.append((rho, ratio))
    return {"rows": rows}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pk = sim_peeking()
    ua = sim_unit_of_analysis()
    cu = sim_cuped()

    # --- figure 1: peeking
    fig, ax = plt.subplots(figsize=(7, 3.4))
    days = np.arange(1, len(pk["curve"]) + 1)
    ax.plot(days, pk["curve"] * 100, color=RED, lw=2, marker=".", label="peek daily, stop at first p<0.05")
    ax.axhline(5, color=BLUE, ls="--", label="nominal α = 5%")
    ax.set_xlabel("days of experiment")
    ax.set_ylabel("false positive rate, %")
    ax.set_title(f"A/A tests: daily peeking inflates false positives to {pk['fpr_peeking']:.0%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "sim_peeking.png")
    plt.close(fig)

    # --- figure 2: unit of analysis
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.barh(["delta method\n(user-level)", "naive z-test\n(per view)"],
            [ua["fpr_delta"] * 100, ua["fpr_naive"] * 100], color=[GREEN, RED])
    ax.axvline(5, color=BLUE, ls="--")
    for i, v in enumerate([ua["fpr_delta"], ua["fpr_naive"]]):
        ax.text(v * 100 + 0.3, i, f"{v:.1%}", va="center")
    ax.set_xlabel("false positive rate in A/A tests, %")
    ax.set_title("CTR with user randomization: per-view test is overconfident")
    fig.tight_layout()
    fig.savefig(FIG / "sim_unit_of_analysis.png")
    plt.close(fig)

    # --- figure 3: CUPED
    rho, ratio = zip(*cu["rows"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(rho, np.array(ratio) * 100, marker="o", color=GREEN, lw=2, label="simulated")
    grid = np.linspace(0, 0.9, 50)
    ax.plot(grid, (1 - grid ** 2) * 100, color="#6b7280", ls="--", label="theory: 1 − ρ²")
    ax.set_xlabel("correlation between pre-period and experiment metric (ρ)")
    ax.set_ylabel("variance / sample size left, %")
    ax.set_title("CUPED: with ρ = 0.7 you need ~half the users")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "sim_cuped.png")
    plt.close(fig)

    md = f"""# Simulation results (auto-generated)

| simulation | setup | result |
|---|---|---|
| Peeking | 3,000 A/A tests, 21 daily looks, 500 users/group/day, p=10% | FPR fixed horizon **{pk['fpr_fixed']:.1%}**, with daily peeking **{pk['fpr_peeking']:.1%}** |
| Unit of analysis | 2,000 A/A tests on CTR, 2,000 users/group, heterogeneous users | FPR naive per-view test **{ua['fpr_naive']:.1%}**, delta method **{ua['fpr_delta']:.1%}** |
| CUPED | variance left after adjustment | """ + ", ".join(f"ρ={r:.1f}: {v:.0%}" for r, v in cu["rows"]) + " |\n"
    (ROOT / "reports" / "simulations.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
