"""Extract named anecdote cases for the Phase 1 writeup.

The most persuasive artefact for a skeptical media buyer is a concrete story:
"Campaign X looked like -40% at 8 conversions; the naive rule kills it; it went
on to earn +$Z over the next 2,000 conversions." We pull the biggest such cases
straight from the reconstructed states + costs.
"""

from __future__ import annotations

import pandas as pd

from ..datasource import EventCols
from ..backtest import cost


def build_anecdotes(sim: pd.DataFrame, ground_truth: pd.DataFrame,
                    campaigns: pd.DataFrame, decision_point: int,
                    top_n: int = 5) -> list[dict]:
    """Return up to `top_n` highest-dollar misclassification stories at a point."""
    ranked = cost.per_campaign_costs(sim, decision_point)
    ranked = ranked[ranked["dollar_cost"] > 0]
    if ranked.empty:
        return []

    name_map = campaigns.set_index(EventCols.ID)["name"].to_dict() if "name" in campaigns else {}
    gt_map = ground_truth.set_index(EventCols.ID)

    stories = []
    for _, r in ranked.head(top_n).iterrows():
        cid = r["campaign_id"]
        gt = gt_map.loc[cid] if cid in gt_map.index else None
        story = {
            "campaign_id": cid,
            "name": name_map.get(cid, cid),
            "decision_point": decision_point,
            "error_type": r["error_type"],
            "observed_roi_at_decision": round(float(r["observed_roi"]), 3),
            "call": r["call"],
            "true_verdict": r["verdict"],
            "final_roi": round(float(gt["roi"]), 3) if gt is not None else None,
            "total_conversions": int(gt["total_conversions"]) if gt is not None else None,
            "dollar_cost": round(float(r["dollar_cost"]), 2),
        }
        story["headline"] = _format_story(story)
        stories.append(story)
    return stories


def _format_story(s: dict) -> str:
    pct = lambda x: f"{x*100:+.0f}%"
    if s["error_type"] == "false_kill":
        return (f"{s['name']}: {pct(s['observed_roi_at_decision'])} at "
                f"{s['decision_point']} conv would have been KILLED, but ran to "
                f"{s['total_conversions']:,} conv at {pct(s['final_roi'])} ROI "
                f"= +${s['dollar_cost']:,.0f} of forgone profit.")
    return (f"{s['name']}: {pct(s['observed_roi_at_decision'])} at "
            f"{s['decision_point']} conv looked like a winner and would have been "
            f"SCALED, but was truly a loser ({pct(s['final_roi'])} final) "
            f"= ${s['dollar_cost']:,.0f} lost after that point.")
