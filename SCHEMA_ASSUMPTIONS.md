# SCHEMA_ASSUMPTIONS.md

This tool reads from an affiliate tracker. Until we have the **real** column
names (see "Open questions for the buyer/tracker owner" at the bottom), it is
built against the *assumed* schema documented here plus synthetic data.

**Nothing in the code hardcodes tracker column names.** Every real-world name is
wired in exactly once, in `config.yaml -> data_source.column_map`. To map the
real schema, edit that block only. This file explains what each canonical field
*means* so the mapping can be done correctly.

The tool is **read-only**. It never writes to the tracker or its database.

---

## Canonical entities

We assume two logical tables. In a real tracker these may be views, joins, or a
single denormalised stats table — the `DataSource` layer only needs to return
two dataframes with the canonical columns below.

### 1. `campaigns` — one row per campaign (the decision unit)

| canonical field | type | meaning | required |
|---|---|---|---|
| `campaign_id` | str/int | stable unique id; the join key and the unit of every recommendation | yes |
| `name` | str | human label for anecdotes/reports | no |
| `offer` | str | primary offer for the campaign | no |
| `start_time` | timestamp | when the campaign began serving | no |
| `end_time` | timestamp | when it stopped, or null if live | no |
| `status` | str | tracker status (active/paused/archived…) — used only for context | no |

The **decision unit is the whole campaign.** Sub-ids, creatives, landers,
offers, audiences, regions and cities exist only as *dimensions* on the event
rows and are used as features / corroborating evidence, never as their own
decision unit.

### 2. `events` — time series of spend & performance per campaign

The grain is assumed to be **one row per (campaign, timestamp, dimension-combo)**
— e.g. hourly or daily stats rows, optionally split by sub_id/creative/etc. The
tool aggregates as needed; it does not assume a specific grain, only that rows
carry a timestamp and additive metrics.

| canonical field | type | meaning | required |
|---|---|---|---|
| `campaign_id` | str/int | FK to `campaigns` | yes |
| `timestamp` | timestamp | **time the spend/click occurred** (see "Lag", below) | yes |
| `postback_time` | timestamp | time the conversion/revenue actually landed | no* |
| `spend` | float | media cost in this row (dollars) | yes |
| `clicks` | int | clicks in this row | no |
| `conversions` | int | conversions attributed in this row | yes |
| `revenue` | float | payout/revenue in this row (dollars), from ACTUAL payouts | yes |
| `sub_id` | str | traffic sub-id dimension | no |
| `creative` | str | creative/ad dimension | no |
| `lander` | str | landing page dimension | no |
| `offer` | str | offer dimension (may differ from campaign.offer) | no |
| `audience` | str | audience/segment dimension | no |
| `region` | str | region/state dimension | no |
| `city` | str | city dimension | no |

`*` `postback_time` is **not required but strongly desired** — it is what lets us
separate "delayed conversions finally landing" from "a losing campaign genuinely
recovering." Without it we fall back to `lag.assumed_lag_hours` as a
sensitivity check only. See below.

---

## Key assumptions the statistics depend on

1. **Additivity.** `spend`, `clicks`, `conversions`, `revenue` sum correctly
   across rows for a campaign. Profit for any slice = `sum(revenue) - sum(spend)`.
   ROI = `profit / spend`. If the tracker stores rates or ratios instead of
   additive counts, they must be converted to additive counts in the mapping.

2. **Revenue is realised payout, not assumed value.** Ground-truth verdicts use
   `revenue - spend` from actual payouts. If `revenue` is a *projected* value
   (e.g. assumed EPC × clicks), verdicts will be biased and this must be flagged.

3. **`conversions` is the sample-size clock.** All decision points are indexed by
   *cumulative conversions*, not calendar time or spend. The backtest rewinds
   each campaign to when it had accumulated 5/10/20/30/50 conversions. This
   requires conversions to be countable and monotonically accumulating.

4. **Lag / censoring (the big one).** A conversion is caused by spend at
   `timestamp` but is *recorded* at `postback_time >= timestamp`. Early
   cumulative ROI is therefore structurally biased **low**: the spend is already
   counted but its revenue hasn't posted yet. Much apparent "loss → profit
   recovery" is just this lag resolving, not a real turnaround. The tool:
   - aligns revenue back to `timestamp` within `lag.attribution_window_hours`
     when `postback_time` exists, and
   - separately reports how much early-loss reversal is explained by lag vs.
     genuine recovery.
   If `postback_time` is missing, we use `lag.assumed_lag_hours` for a
   sensitivity-only decomposition.

5. **Survivorship / censoring.** We only observe campaigns that *survived* long
   enough to earn a mature verdict. Campaigns the buyers actually killed early —
   including real winners killed by bad luck — are **invisible**. Therefore every
   false-kill cost we compute is a **lower bound** and is always labelled as
   such. A dedicated cohort finder surfaces campaigns that had a bad early
   stretch but were kept running anyway (direct proof that early-loss ≠ doomed).

6. **No PII.** The tool does not read, store, or export personal data. Dimension
   columns are treated as opaque categorical labels.

---

## What "mature" means (ground truth gate)

A campaign contributes to ground-truth accuracy claims only if it is mature:
`total_conversions >= ground_truth.maturity_min_conversions`
(default 100), optionally combined with a spend gate. The verdict is computed on
the mature portion of life. Borderline campaigns (mature ROI within
`ground_truth.borderline_roi_band` of break-even) are excluded from headline
accuracy on the first pass because they are genuinely too close to call.

---

## Open questions for the buyer / tracker owner (blockers for the SQL DataSource)

Answering these lets us finalise `column_map` and the SQL implementation. Until
then we run on synthetic data.

1. **Exact table/view + column names** for the two entities above. Especially:
   the spend-time column vs. the postback/conversion-time column — are they two
   columns on one row, or two separate event streams?
2. **Grain of the stats table** — hourly? daily? per sub_id? Is there already a
   campaign-level rollup, or do we aggregate from sub-id rows?
3. **Is `revenue` realised payout or projected?** If projected, is realised
   payout available anywhere?
4. **Typical postback lag** for your offers (drives `attribution_window_hours`).
5. **The real conversion count at which buyers pull the trigger** (drives
   `typical_decision_conversions` — the headline is priced here).
6. **A read-only DB credential / replica** we can point the SQL DataSource at.
