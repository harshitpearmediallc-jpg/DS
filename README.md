# Scale / Kill Confidence Tool

Tells a media buyer whether a campaign's scale/kill signal is **trustworthy yet,
or just noise** — and recommends **SCALE / HOLD / KILL** with an honest,
explainable reason and a confidence level. The decision unit is the **whole
campaign**.

The naive rule buyers use today — *if in profit, scale; if in loss, kill* — acts
on tiny samples where day-to-day swings are mostly random. It scales lucky flukes
and kills real winners. This tool intercepts that exact moment.

Built in two phases. **Phase 1 (this delivery) validates the premise and
produces the trust threshold Phase 2 will use. Phase 2 is not built yet — it is
gated on your review of Phase 1.**

## What Phase 1 found (on synthetic data with known ground truth)

Run: `python -m notebooks.run_phase1 --regen`

- The naive rule's error **decays with sample size**: ~45% wrong at 5
  conversions → ~6% by 100. See `outputs/money_chart.png`.
- **Trust threshold N\*** (error ≤ 10%): **75 conversions** on this dataset —
  "don't trust a scale/kill call below ~75 conversions."
- At the **10-conversion point** where buyers actually act, the rule misclassified
  **~42%** of campaigns, costing **≥ ~$1.0M** (a lower bound — see below).
- **~70%** of apparent early-loss "recoveries" are just **delayed postbacks
  (lag)**, not genuine turnarounds — a trap the tool must not reward.

These numbers come from a **synthetic** dataset built to resemble the problem;
they demonstrate the machinery is correct. The real headline comes from running
Phase 1 against your tracker once the schema is mapped.

## Two correctness guards (implemented, not hand-waved)

1. **Survivorship / censoring.** We only observe campaigns that *survived* to a
   verdict. Winners buyers actually killed early are invisible, so every
   false-kill dollar figure is labelled a **lower bound** everywhere. A dedicated
   `survivor_cohort` finder surfaces campaigns that had a bad early stretch but
   were kept running and won anyway — direct proof that early-loss ≠ doomed.
2. **Conversion / postback lag.** Postbacks arrive after the spend that generated
   them, so early ROI is structurally biased low. Reconstruction uses two clocks
   (spend-time vs postback-time) and the tool separately reports how much
   early-loss reversal is **lag vs genuine recovery**.

## Layout

```
config.yaml            all tunables (no magic numbers in code)
SCHEMA_ASSUMPTIONS.md  what we assume about the tracker + open questions for you
src/
  config.py            YAML loader
  synthetic.py         synthetic data generator with KNOWN ground truth
  datasource/          DataSource interface + file (parquet/csv) + SQL (read-only)
  backtest/            Phase 1: reconstruct, cohorts, simulate, cost, lag, engine
  reporting/           money chart + named anecdotes
  logging_/            recommendation+outcome log schema (anticipates Phase 2)
tests/                 statistical core validated against synthetic ground truth
notebooks/run_phase1.py  end-to-end Phase 1 run
outputs/               money_chart.png, phase1_thresholds.yaml, phase1_summary.md
```

## Setup

```bash
pip install -r requirements.txt
python -m src.synthetic          # write synthetic dataset to data/synthetic/
python -m notebooks.run_phase1   # run the backtest, emit outputs/
pytest -q                        # validate the statistical core
```

## Data source

Configured in `config.yaml → data_source`. `kind: file` reads parquet/csv
(offline + synthetic). `kind: sql` reads a Postgres/MySQL-style tracker replica
via SQLAlchemy, **read-only**, with table/column names supplied entirely through
`data_source.column_map` — no tracker column name is hardcoded. See
`SCHEMA_ASSUMPTIONS.md` for the open questions we need answered to finalise the
SQL mapping.

## Phase 2 — SCALE / HOLD / KILL confidence layer

Run: `python -m notebooks.run_phase2_real`

Intercepts the moment a buyer would apply the naive rule and returns SCALE / HOLD
/ KILL with a calibrated confidence and a checkable reason. Logic: (1) sample-size
gate — HOLD below the Phase-1 trust threshold (125 conv on the real data), stating
how many more conversions are needed; (2) regime check — bootstrap recent-vs-prior
+ CUSUM + per-dimension breakdown detect a genuine adverse *change* vs noise; (3)
significance — bootstrap 95% ROI interval vs break-even → SCALE / KILL / HOLD.

Confidence is an isotonic-calibrated P(profitable) fit on Phase-1 ground truth
(held-out ECE 0.095 → 0.065). Every call is appended to
`recommendation_log.jsonl` with empty outcome slots for a future trajectory model.
Code in `src/confidence/` (`significance`, `regime`, `calibration`, `recommend`);
see `outputs/phase2_summary.md` and `outputs/real_calibration.png`.

## Status

Phase 1 and Phase 2 complete and tested (31 tests). Validated on real tracker
data; all data caveats investigated. Next candidates: per-`n` calibration, a
production `recommend()` entry point against the live/exported data, and (future,
out of scope) the trajectory predictor the logging schema anticipates.
