# Results summary

Findings from running the tool against a real affiliate tracker (RedTrack-style
`spend_revenue` rollup, ~2 months, 2,664 mature campaigns). Aggregate figures
only — no campaign-level data is committed to this repo (all `outputs/` and
`data/` are git-ignored). Reproduce with the notebooks below.

## Phase 1 — the naive rule is expensive

The naive `if profit: scale else: kill` rule was backtested against mature
ground-truth verdicts (campaigns with ≥100 conversions), reconstructing each
campaign's state at 5/10/20/.../150 cumulative conversions.

- Misclassification decays with sample size: **~39% at 5 conversions → ~10% at
  125 → ~6% by 300**.
- **Trust threshold ≈ 125 conversions** (where error drops to ≤10%). Practical
  read: don't trust a scale/kill call below ~100–125 conversions.
- At the **10-conversion point** where buyers actually act, the rule
  misclassified **~35%** of campaigns, costing **≥ ~$1.4M** over the window — a
  **lower bound**, because we only observe campaigns that survived to a verdict
  (survivorship censoring; winners killed early are invisible).
- Winners outnumber losers ~4:1 in the mature set — the fingerprint of that
  survivorship.

Two correctness guards are implemented, not hand-waved: survivorship (every
false-kill dollar is labelled a lower bound; a cohort finder surfaces
early-loss-but-kept winners) and conversion/postback lag (two-clock
reconstruction; the guard is correctly inactive on this source because RedTrack
click-attributes revenue upstream).

## Data quality — investigated, conclusion robust

- **Revenue is realised payout** (never present without a conversion; continuous
  payout-per-conversion), so verdicts are sound.
- **Extreme ROIs trace to ~107 under-tracked-spend campaigns** (zero spend, or
  sub-cent CPC with >300% ROI). Removing all of them moves the headline only
  **−4.7%** (≥ ~$1.38M). A reusable guard flags them every run.
- **No per-conversion/postback data exists** anywhere in the source, confirming
  revenue is click-attributed upstream.

## Phase 2 — SCALE / HOLD / KILL confidence layer

Intercepts the decision moment and returns a calibrated call with a checkable
reason:

1. **Sample-size gate first** — HOLD below the trust threshold, stating how many
   more conversions are needed.
2. **Regime check** — bootstrap recent-vs-prior + CUSUM + per-dimension breakdown
   to separate a genuine adverse *change* from noise (a consistent loser is a
   plain KILL, not a "regime change").
3. **Significance** — bootstrap 95% ROI interval vs break-even → SCALE / KILL /
   HOLD.

Confidence is an **isotonic-calibrated** P(profitable) fit on Phase-1 ground
truth; held-out calibration error **ECE 0.095 → 0.065**. Every call is logged
(append-only JSONL) with empty outcome slots for a future trajectory predictor.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.synthetic                 # synthetic data with known ground truth
python -m notebooks.run_phase1          # Phase 1 on synthetic
pytest -q                               # 31 tests

# against real data:
#   1) export the tracker table (see export_tracker.py / SCHEMA_ASSUMPTIONS.md)
#   2) python prepare_real_data.py path/to/export.csv
#   3) python -m notebooks.run_phase1_real
#   4) python -m notebooks.run_phase2_real
```
