"""Phase 1 backtest run — end to end.

Usage:
    python -m notebooks.run_phase1            # uses config.yaml, synthetic data
    python -m notebooks.run_phase1 --regen    # regenerate synthetic data first

Produces:
    outputs/money_chart.png
    outputs/phase1_thresholds.yaml
    outputs/phase1_summary.md
and prints the headline number + curve to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.datasource import make_datasource
from src.synthetic import write_dataset, GenConfig
from src.backtest.engine import run_phase1, export_thresholds
from src.reporting.charts import money_chart
from src.reporting.anecdotes import build_anecdotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate synthetic data")
    args = ap.parse_args()

    cfg = load_config()
    if args.regen:
        write_dataset(cfg, GenConfig(seed=cfg.random_seed))
        print("regenerated synthetic dataset")

    ds = make_datasource(cfg)
    result = run_phase1(ds, cfg)

    curve = result.confusion.merge(result.costs, on="n_conversions").sort_values("n_conversions")

    pd.set_option("display.width", 140)
    print("\n================ PHASE 1: BACKTEST OF THE NAIVE RULE ================\n")
    print("Cohort:")
    for k, v in result.meta.items():
        print(f"  {k}: {v}")

    print("\nError & $ cost by conversions-at-decision:")
    show = curve[["n_conversions", "n_campaigns_scored", "misclassification_rate",
                  "false_kill", "false_scale",
                  "false_kill_cost_lower_bound", "false_scale_cost",
                  "total_cost_lower_bound"]].copy()
    show["misclassification_rate"] = (show["misclassification_rate"] * 100).round(1)
    for c in ["false_kill_cost_lower_bound", "false_scale_cost", "total_cost_lower_bound"]:
        show[c] = show[c].round(0)
    print(show.to_string(index=False))

    print(f"\nTrust threshold N* (error <= {result.meta['tolerable_error_rate']*100:.0f}%): "
          f"{result.trust_threshold_n} conversions")

    print("\n---- HEADLINE (priced at the real decision point) ----")
    h = result.headline
    if h:
        print(f"  At {h['typical_decision_conversions']} conversions (where buyers actually act), "
              f"the naive rule misclassified {h['misclassification_rate']*100:.1f}% of "
              f"{h['n_campaigns_scored']} campaigns with ground truth,")
        print(f"  costing >= ${h['total_cost_lower_bound']:,.0f} "
              f"(>= ${h['false_kill_cost_lower_bound']:,.0f} forgone from {h['n_false_kill']} killed winners "
              f"+ ${h['false_scale_cost']:,.0f} lost on {h['n_false_scale']} scaled losers).")
        print("  NOTE: false-kill cost is a LOWER BOUND — killed winners are unobserved (survivorship).")

    print("\n---- LAG vs GENUINE RECOVERY (early-loss winners at typical point) ----")
    ls = result.lag_summary
    if ls["n_campaigns"]:
        print(f"  {ls['n_campaigns']} winners looked like a loss early. Of the ROI recovered by maturity, "
              f"{ls['lag_share']*100:.0f}% is just delayed postbacks (lag) and "
              f"{ls['genuine_share']*100:.0f}% is genuine improvement.")
    else:
        print("  (no early-loss winners at the typical point)")

    print(f"\n---- SURVIVOR COHORT (early loss, kept running, true winners): "
          f"{len(result.survivor_cohort)} campaigns ----")
    if not result.survivor_cohort.empty:
        sc = result.survivor_cohort.head(5)[["campaign_id", "early_observed_roi", "roi", "total_conversions"]]
        print(sc.to_string(index=False))

    # anecdotes
    campaigns = ds.load_campaigns()
    anecdotes = build_anecdotes(result.sim, result.ground_truth, campaigns,
                                decision_point=result.typical_point, top_n=5)
    print("\n---- NAMED ANECDOTES ----")
    for a in anecdotes:
        print(f"  - {a['headline']}")

    # artefacts
    chart_path = resolve_path(cfg, cfg.io.money_chart_out)
    money_chart(curve, result.trust_threshold_n, result.typical_point, chart_path,
                tolerable_rate=cfg.backtest.tolerable_error_rate)
    thr_path = export_thresholds(result, cfg)

    _write_summary_md(cfg, result, curve, anecdotes)
    print(f"\nWrote: {chart_path}")
    print(f"Wrote: {thr_path}")
    print(f"Wrote: {resolve_path(cfg, 'outputs/phase1_summary.md')}")


def _write_summary_md(cfg, result, curve, anecdotes):
    h = result.headline
    ls = result.lag_summary
    lines = []
    lines.append("# Phase 1 — Backtest of the naive scale/kill rule\n")
    lines.append("_All false-kill figures are **lower bounds**: we only observe campaigns that "
                 "survived to a verdict. Winners that buyers actually killed early are invisible._\n")
    lines.append("## Headline\n")
    if h:
        lines.append(
            f"Among **{h['n_campaigns_scored']} campaigns with mature ground truth**, at the "
            f"**{h['typical_decision_conversions']}-conversion** point where buyers actually pull the "
            f"trigger, the naive `if profit: scale else: kill` rule **misclassified "
            f"{h['misclassification_rate']*100:.1f}%** of them — costing **>= ${h['total_cost_lower_bound']:,.0f}** "
            f"(>= ${h['false_kill_cost_lower_bound']:,.0f} of forgone profit from "
            f"{h['n_false_kill']} killed winners, plus ${h['false_scale_cost']:,.0f} lost on "
            f"{h['n_false_scale']} scaled losers).\n")
    lines.append(f"**Trust threshold N\\*** (error drops to <= {cfg.backtest.tolerable_error_rate*100:.0f}%): "
                 f"**{result.trust_threshold_n} conversions**. Below this, a scale/kill call is not "
                 f"trustworthy — Phase 2 defaults to HOLD.\n")
    lines.append("## Error & cost by sample size\n")
    lines.append("| conv @ decision | campaigns | misclass % | false kills | false scales | $ cost (lower bound) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in curve.iterrows():
        lines.append(f"| {int(r['n_conversions'])} | {int(r['n_campaigns_scored'])} | "
                     f"{r['misclassification_rate']*100:.1f}% | {int(r['false_kill'])} | "
                     f"{int(r['false_scale'])} | ${r['total_cost_lower_bound']:,.0f} |")
    lines.append("\n![money chart](money_chart.png)\n")
    lines.append("## Lag vs. genuine recovery\n")
    if ls["n_campaigns"]:
        lines.append(f"Of {ls['n_campaigns']} winners that looked like an early loss, "
                     f"**{ls['lag_share']*100:.0f}%** of the recovery is delayed postbacks (lag) and "
                     f"**{ls['genuine_share']*100:.0f}%** is genuine improvement. Waiting for the lag to "
                     f"resolve — not the naive rule — explains most 'recoveries'.\n")
    lines.append("## Named cases\n")
    for a in anecdotes:
        lines.append(f"- {a['headline']}")
    lines.append("")
    out = resolve_path(cfg, "outputs/phase1_summary.md")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
