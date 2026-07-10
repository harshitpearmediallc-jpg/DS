"""The money chart: misclassification rate AND dollar cost vs conversions-at-decision.

Expectation from the premise: error is high at 5-10 conversions and decays toward
~0 by 50-100. The chart makes the "don't trust a call below N conversions"
argument visible in one picture.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def money_chart(curve, trust_n, typical_point, out_path: str | Path,
                tolerable_rate: float | None = None) -> Path:
    """curve: dataframe with n_conversions, misclassification_rate,
    total_cost_lower_bound. Saves a two-axis figure to out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = curve["n_conversions"].to_numpy()
    err = curve["misclassification_rate"].to_numpy() * 100.0
    dollars = curve["total_cost_lower_bound"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color1 = "#c0392b"
    ax1.plot(x, err, "-o", color=color1, linewidth=2, label="Misclassification rate")
    ax1.set_xlabel("Conversions accumulated at decision time")
    ax1.set_ylabel("Naive-rule misclassification rate (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=0)

    if tolerable_rate is not None:
        ax1.axhline(tolerable_rate * 100, color=color1, linestyle=":", alpha=0.6,
                    label=f"Tolerable error ({tolerable_rate*100:.0f}%)")

    ax2 = ax1.twinx()
    color2 = "#2c3e50"
    ax2.bar(x, dollars, width=2.2, alpha=0.25, color=color2,
            label="Cost of acting (lower bound, $)")
    ax2.set_ylabel("Cost of acting on the call — lower bound ($)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(bottom=0)

    if trust_n is not None:
        ax1.axvline(trust_n, color="#27ae60", linestyle="--", linewidth=2,
                    label=f"Trust threshold N*={trust_n}")
    ax1.axvline(typical_point, color="#e67e22", linestyle="--", linewidth=1.5,
                label=f"Typical decision point ({typical_point})")

    # merge legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    ax1.set_title("Naive scale/kill rule: error and $ cost vs. sample size at decision")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
