"""Run Phase 1 against the REAL tracker data prepared into data/real/.

Assumes prepare_real_data.py has written data/real/{events,campaigns}.parquet in
canonical form. Points a FileDataSource at it with an identity column_map and
runs the same backtest + reporting used on synthetic data.
"""

from __future__ import annotations

import pandas as pd

from src.config import load_config, resolve_path, Cfg
from src.datasource import make_datasource, EventCols
from src.backtest.engine import run_phase1, export_thresholds
from src.reporting.charts import money_chart
from src.reporting.anecdotes import build_anecdotes

CANON_EVENTS = ["campaign_id", "timestamp", "postback_time", "spend", "clicks",
                "conversions", "revenue", "sub_id", "creative", "lander", "offer",
                "audience", "region", "city"]
CANON_CAMPS = ["campaign_id", "name", "offer", "start_time", "end_time", "status"]


def _real_cfg() -> Cfg:
    cfg = load_config()
    d = cfg.to_dict()
    d["data_source"]["kind"] = "file"
    d["data_source"]["file"]["path"] = "data/real"
    d["data_source"]["file"]["format"] = "parquet"
    # identity map: prepared parquet already uses canonical names
    d["data_source"]["column_map"]["events"] = {c: c for c in CANON_EVENTS}
    d["data_source"]["column_map"]["campaigns"] = {c: c for c in CANON_CAMPS}
    # real campaigns are larger; extend the range so the 10% crossing is visible
    d["backtest"]["decision_conversion_points"] = [5, 10, 20, 30, 50, 75, 100, 125, 150]
    # point outputs at real_* so we don't clobber the synthetic run
    d["io"]["money_chart_out"] = "outputs/real_money_chart.png"
    d["io"]["thresholds_out"] = "outputs/real_phase1_thresholds.yaml"
    return Cfg(d)


def main():
    cfg = _real_cfg()
    ds = make_datasource(cfg)
    result = run_phase1(ds, cfg)
    curve = result.confusion.merge(result.costs, on="n_conversions").sort_values("n_conversions")

    pd.set_option("display.width", 160)
    print("\n============ PHASE 1 ON REAL TRACKER DATA (spend_revenue) ============\n")
    for k, v in result.meta.items():
        print(f"  {k}: {v}")
    print("\nError & $ cost by conversions-at-decision:")
    show = curve[["n_conversions", "n_campaigns_scored", "misclassification_rate",
                  "false_kill", "false_scale", "false_kill_cost_lower_bound",
                  "false_scale_cost", "total_cost_lower_bound"]].copy()
    show["misclassification_rate"] = (show["misclassification_rate"] * 100).round(1)
    for c in ["false_kill_cost_lower_bound", "false_scale_cost", "total_cost_lower_bound"]:
        show[c] = show[c].round(0)
    print(show.to_string(index=False))

    print(f"\nTrust threshold N* (error <= {cfg.backtest.tolerable_error_rate*100:.0f}%): "
          f"{result.trust_threshold_n} conversions")

    h = result.headline
    if h:
        print("\n---- HEADLINE (priced at the real decision point) ----")
        print(f"  At {h['typical_decision_conversions']} conversions, naive rule misclassified "
              f"{h['misclassification_rate']*100:.1f}% of {h['n_campaigns_scored']} campaigns,")
        print(f"  costing >= ${h['total_cost_lower_bound']:,.0f} "
              f"(>= ${h['false_kill_cost_lower_bound']:,.0f} forgone from {h['n_false_kill']} killed winners "
              f"+ ${h['false_scale_cost']:,.0f} on {h['n_false_scale']} scaled losers). LOWER BOUND.")

    ls = result.lag_summary
    print("\n---- LAG vs GENUINE RECOVERY ----")
    if ls["n_campaigns"] and ls["lag_share"] is not None and ls["lag_share"] > 0.01:
        print(f"  {ls['n_campaigns']} early-loss winners; lag {ls['lag_share']*100:.0f}% / "
              f"genuine {ls['genuine_share']*100:.0f}%.")
    else:
        print("  Not separable from this source: revenue is click-attributed upstream "
              "(no postback timestamps), so observed ROI == lag-aligned ROI.")

    campaigns = ds.load_campaigns()
    anecdotes = build_anecdotes(result.sim, result.ground_truth, campaigns,
                                decision_point=result.typical_point, top_n=5)
    print("\n---- NAMED ANECDOTES ----")
    for a in anecdotes:
        print(f"  - {a['headline']}")

    chart = resolve_path(cfg, cfg.io.money_chart_out)
    money_chart(curve, result.trust_threshold_n, result.typical_point, chart,
                tolerable_rate=cfg.backtest.tolerable_error_rate)
    export_thresholds(result, cfg)
    _write_real_summary(cfg, result, curve, anecdotes)
    print(f"\nWrote {chart}")
    print(f"Wrote {resolve_path(cfg, 'outputs/real_phase1_summary.md')}")


def _write_real_summary(cfg, result, curve, anecdotes):
    h, m = result.headline, result.meta
    L = []
    L.append("# Phase 1 on REAL tracker data (`spend_revenue`)\n")
    L.append("_Source: RedTrack-style hourly rollup, May 4 – Jul 10 2026. Revenue is "
             "click-attributed upstream (no postback timestamps), so ROI is not lag-biased "
             "in this source. All false-kill figures are **lower bounds** — only campaigns "
             "kept alive to a mature verdict are observable._\n")
    L.append("## Cohort\n")
    L.append(f"- Mature campaigns scored (≥{cfg.ground_truth.maturity_min_conversions} conv): "
             f"**{m['n_with_verdict']:,}** ({m['n_winners']:,} winners, {m['n_losers']:,} losers; "
             f"{m['n_borderline_excluded']:,} borderline excluded).")
    L.append(f"- Winners outnumber losers ~{m['n_winners']/max(m['n_losers'],1):.1f}:1 — the fingerprint "
             f"of survivorship (losers were killed; winners kept), which is exactly why the false-kill "
             f"cost is only a lower bound.\n")
    L.append("## Headline\n")
    if h:
        L.append(f"At the **{h['typical_decision_conversions']}-conversion** point where buyers act, the "
                 f"naive `if profit: scale else: kill` rule **misclassified {h['misclassification_rate']*100:.1f}%** "
                 f"of {h['n_campaigns_scored']:,} campaigns — costing **≥ ${h['total_cost_lower_bound']:,.0f}** over "
                 f"this 2-month window (≥ ${h['false_kill_cost_lower_bound']:,.0f} of forgone profit from "
                 f"{h['n_false_kill']} killed winners + ${h['false_scale_cost']:,.0f} lost on {h['n_false_scale']} "
                 f"scaled losers).\n")
    tn = result.trust_threshold_n
    L.append(f"**Trust threshold N\\*** (error ≤ {cfg.backtest.tolerable_error_rate*100:.0f}%): "
             f"**{tn} conversions**. Caveat: by ~125 conv ~20% of campaigns have dropped out of the scored "
             f"set (they never reached that many), so the tail is optimistically biased — treat ~100–125 as "
             f"the practical 'don't trust a call below this' line.\n")
    L.append("## Error & $ cost by sample size\n")
    L.append("| conv @ decision | campaigns scored | misclass % | false kills | false scales | $ cost (lower bound) |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in curve.iterrows():
        L.append(f"| {int(r['n_conversions'])} | {int(r['n_campaigns_scored']):,} | "
                 f"{r['misclassification_rate']*100:.1f}% | {int(r['false_kill'])} | {int(r['false_scale'])} "
                 f"| ${r['total_cost_lower_bound']:,.0f} |")
    L.append("\n![real money chart](real_money_chart.png)\n")
    L.append("## Named cases (real campaigns)\n")
    for a in anecdotes:
        L.append(f"- {a['headline']}")
    L.append("\n## Data caveats to confirm\n")
    L.append("- A minority of winners show extreme ROIs (e.g. +200x). The dollar costs are computed from "
             "*actual observed post-decision profit* (bounded), not these ratios, so headline $ is robust — but "
             "the extremes suggest some campaigns may have under-tracked spend worth spot-checking.")
    L.append("- `revenue` is assumed to be realised payout (matches the generated `profit = revenue - spend` "
             "column). If any of it is projected EPC, verdicts for those campaigns would shift.")
    L.append("- No postback stream here, so the lag-vs-genuine-recovery guard is inactive. If a raw RedTrack "
             "conversions export exists, we can add it.")
    resolve_path(cfg, "outputs/real_phase1_summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
