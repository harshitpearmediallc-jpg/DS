"""Dollar cost of the naive rule's two error types, by decision point.

false_kill  — killed a true winner. Cost = the net profit the campaign actually
              earned AFTER the decision instant (observable, because these
              campaigns survived). This is money the buyer would have forgone.

false_scale — scaled a true loser. Cost = the net loss incurred AFTER the
              decision instant. CONSERVATIVE: we model reversion only — we do
              NOT apply a scaling multiplier in v1, so this is the loss the
              campaign racks up at its *existing* spend rate, not an amplified one.

Every false-kill figure is a LOWER BOUND: we only observe campaigns that were
kept alive. Winners that buyers actually killed early are invisible, so the true
cost of the naive kill rule is strictly larger than what we can measure.
"""

from __future__ import annotations

import pandas as pd


def cost_by_point(sim: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dollar cost of each error type per decision point.

    `sim` must carry `post_profit` (net profit after the decision instant) from
    the reconstructed state.
    """
    rows = []
    for n, grp in sim.groupby("n_conversions"):
        fk = grp[grp["error_type"] == "false_kill"]
        fs = grp[grp["error_type"] == "false_scale"]
        # false-kill cost = forgone profit = post_profit of killed winners (>=0)
        fk_cost = fk["post_profit"].clip(lower=0).sum()
        # false-scale cost = loss after = -post_profit of scaled losers (>=0)
        fs_cost = (-fs["post_profit"]).clip(lower=0).sum()
        rows.append({
            "n_conversions": int(n),
            "false_kill_cost_lower_bound": float(fk_cost),
            "false_scale_cost": float(fs_cost),
            "total_cost_lower_bound": float(fk_cost + fs_cost),
            "n_false_kill": len(fk),
            "n_false_scale": len(fs),
        })
    return pd.DataFrame(rows).sort_values("n_conversions").reset_index(drop=True)


def per_campaign_costs(sim: pd.DataFrame, n_conversions: int) -> pd.DataFrame:
    """Per-campaign cost at a single decision point, sorted by magnitude.

    Used to surface the biggest individual money leaks for anecdotes.
    """
    grp = sim[sim["n_conversions"] == n_conversions].copy()
    grp["dollar_cost"] = 0.0
    fk = grp["error_type"] == "false_kill"
    fs = grp["error_type"] == "false_scale"
    grp.loc[fk, "dollar_cost"] = grp.loc[fk, "post_profit"].clip(lower=0)
    grp.loc[fs, "dollar_cost"] = (-grp.loc[fs, "post_profit"]).clip(lower=0)
    return grp.sort_values("dollar_cost", ascending=False)
