"""Synthetic data generator with KNOWN ground truth.

This is the harness that lets us trust the statistics before touching real data.
It builds campaigns whose *true* economics we control, then injects the two
phenomena that make the naive rule dangerous:

  1. Small-sample payout noise. Per-conversion payouts are heavy-tailed
     (lognormal), so with only 5-10 conversions the observed ROI swings far from
     the truth. This is what turns luck into a "scale" or "kill" signal.

  2. Conversion / postback lag. Each conversion's revenue posts at
     spend_time + lag (lag ~ Exponential). Spend accrues in real time while
     revenue arrives late, so EARLY cumulative ROI is structurally biased LOW.
     Much apparent "loss -> recovery" is just this lag resolving.

Two special archetypes give Phase 1 its proof cases:
  * "unlucky_winner" — truly profitable, but its first K conversions are
     engineered to look terrible. The naive rule would KILL it early. These are
     also the survivorship cohort (kept running despite a bad early stretch).
  * "lucky_loser" — truly unprofitable, but its first few conversions look
     great. The naive rule would SCALE it early.

Ground-truth parameters are written to a SEPARATE file (synthetic_truth.parquet)
that the backtest never reads. The backtest must recover the verdict from the
observable data alone; tests compare its output against this hidden truth.

Emission model (additive, so sum(revenue)-sum(spend) is exact):
  * spend rows  : one per (hour, sub_id). carry spend + clicks, conversions=0.
  * conversion rows: one per conversion. timestamp = the spend/click time that
    generated it; postback_time = when it actually posted; revenue = its payout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg, resolve_path

# --- fixed reference epoch (Date.now is unavailable / non-reproducible) -------
EPOCH = pd.Timestamp("2025-01-01 00:00:00")

# dimension vocabularies (opaque categorical labels — no meaning, no PII)
SUB_IDS = [f"sub_{i:02d}" for i in range(8)]
CREATIVES = [f"cr_{i:02d}" for i in range(5)]
LANDERS = [f"lp_{i:02d}" for i in range(3)]
AUDIENCES = ["aud_a", "aud_b", "aud_c"]
REGIONS = ["US-CA", "US-TX", "US-NY", "US-FL"]
CITIES = ["LA", "SF", "DAL", "NYC", "MIA"]
OFFERS = ["offer_alpha", "offer_beta", "offer_gamma"]


@dataclass
class Archetype:
    name: str
    true_roi_lo: float
    true_roi_hi: float
    target_conversions_lo: int
    target_conversions_hi: int
    # early-stretch injection: multiply payouts of the first `early_k`
    # conversions by `early_payout_mult`. <1 makes a good campaign look bad;
    # >1 makes a bad campaign look good.
    early_k: int = 0
    early_payout_mult: float = 1.0
    weight: float = 1.0


ARCHETYPES = [
    Archetype("clear_winner", 0.30, 1.20, 300, 1500, weight=0.34),
    Archetype("clear_loser", -0.55, -0.15, 150, 800, weight=0.34),
    Archetype("marginal", -0.05, 0.05, 200, 1000, weight=0.08),
    # unlucky winner: true winner, first ~10 conversions crushed to ~20% payout
    Archetype("unlucky_winner", 0.30, 0.80, 400, 1500,
              early_k=10, early_payout_mult=0.18, weight=0.12),
    # lucky loser: true loser, first ~6 conversions inflated ~4x
    Archetype("lucky_loser", -0.50, -0.18, 150, 800,
              early_k=6, early_payout_mult=4.0, weight=0.12),
]


@dataclass
class GenConfig:
    """Knobs for the generator. Sensible defaults; overridable in tests.

    Calibration note (why these numbers): the decision-point sample sizes are
    5-50 conversions. For the naive rule's error to be driven by SMALL-SAMPLE
    NOISE (which decays with N) rather than swamped by a constant lag bias, the
    conversion pipeline backlog (generation_rate x mean_lag) must be modest
    relative to N. generation_rate ~= clicks_per_hour * cvr. With ~15-60
    clicks/hr and cvr ~0.02-0.05 that is ~0.3-3 conv/hr; with mean lag 1-24h the
    backlog is a few conversions — a real but non-dominant early-loss bias.
    Campaigns then accumulate hundreds of conversions over days/weeks, which is
    how real affiliate campaigns actually run.
    """
    n_campaigns: int = 250
    payout_mean: float = 30.0          # $ per conversion (typical affiliate payout)
    payout_sigma: float = 0.9          # lognormal shape -> heavy tail -> early noise
    cvr_lo: float = 0.02               # click->conversion rate range
    cvr_hi: float = 0.05
    clicks_per_hour_lo: int = 15       # low traffic -> conversions accrue over days
    clicks_per_hour_hi: int = 60
    lag_mean_hours_lo: float = 1.0     # per-campaign mean postback lag range
    lag_mean_hours_hi: float = 24.0    # modest: backlog stays small vs decision N
    n_subids_per_campaign: int = 4
    seed: int = 42


# ---------------------------------------------------------------------------

def _pick_archetypes(rng: np.random.Generator, n: int) -> list[Archetype]:
    weights = np.array([a.weight for a in ARCHETYPES], dtype=float)
    weights /= weights.sum()
    idx = rng.choice(len(ARCHETYPES), size=n, p=weights)
    return [ARCHETYPES[i] for i in idx]


def _campaign_params(rng: np.random.Generator, arch: Archetype, gc: GenConfig) -> dict:
    true_roi = rng.uniform(arch.true_roi_lo, arch.true_roi_hi)
    cvr = rng.uniform(gc.cvr_lo, gc.cvr_hi)
    payout_mean = gc.payout_mean * rng.uniform(0.7, 1.4)
    # revenue per click = cvr * payout_mean ; set cpc so realised ROI == true_roi
    rev_per_click = cvr * payout_mean
    cpc = rev_per_click / (1.0 + true_roi)
    target_conv = int(rng.integers(arch.target_conversions_lo,
                                    arch.target_conversions_hi + 1))
    clicks_per_hour = int(rng.integers(gc.clicks_per_hour_lo, gc.clicks_per_hour_hi + 1))
    # hours needed to reach target conversions at this cvr/traffic
    conv_per_hour = max(clicks_per_hour * cvr, 0.5)
    hours = int(np.ceil(target_conv / conv_per_hour))
    lag_mean = rng.uniform(gc.lag_mean_hours_lo, gc.lag_mean_hours_hi)
    # each campaign uses a subset of sub_ids
    subids = list(rng.choice(SUB_IDS, size=min(gc.n_subids_per_campaign, len(SUB_IDS)),
                             replace=False))
    return dict(
        true_roi=true_roi, cvr=cvr, payout_mean=payout_mean, cpc=cpc,
        target_conv=target_conv, clicks_per_hour=clicks_per_hour, hours=hours,
        lag_mean=lag_mean, subids=subids, offer=rng.choice(OFFERS),
    )


def _generate_one(rng: np.random.Generator, cid: str, arch: Archetype,
                  p: dict, gc: GenConfig):
    """Return (campaign_row, event_rows_df, truth_row) for one campaign."""
    hours = p["hours"]
    cpc = p["cpc"]
    cvr = p["cvr"]
    subids = p["subids"]

    # --- hourly spend rows (one row per hour, sampled sub_id) ----------------
    # One row per active hour keeps row counts bounded; the hour's clicks are
    # attributed to a sampled sub_id so spend still distributes across dimensions.
    spend_records = []
    clicks_h = rng.poisson(p["clicks_per_hour"], size=hours).astype(int)
    hour_subids = rng.choice(subids, size=hours)
    for h in range(hours):
        c = int(clicks_h[h])
        if c == 0:
            continue
        ts = EPOCH + pd.Timedelta(hours=h)
        spend = c * cpc * rng.uniform(0.95, 1.05)  # small billing noise
        spend_records.append((cid, ts, pd.NaT, spend, c, 0, 0.0,
                              hour_subids[h], rng.choice(CREATIVES), rng.choice(LANDERS),
                              p["offer"], rng.choice(AUDIENCES),
                              rng.choice(REGIONS), rng.choice(CITIES)))

    # --- conversions: draw per hour, then assign payout + lag ----------------
    conv_h = rng.binomial(clicks_h, cvr)  # conversions generated each hour
    total_conv = int(conv_h.sum())
    conv_records = []
    if total_conv > 0:
        origin_hours = np.repeat(np.arange(hours), conv_h)
        # heavy-tailed payouts centred so E[payout] ~ payout_mean
        mu = np.log(p["payout_mean"]) - 0.5 * gc.payout_sigma ** 2
        payouts = rng.lognormal(mean=mu, sigma=gc.payout_sigma, size=total_conv)
        # per-conversion lag (hours), exponential with campaign mean
        lags = rng.exponential(scale=p["lag_mean"], size=total_conv)

        # inject the early stretch on the first `early_k` conversions by origin time
        order = np.argsort(origin_hours, kind="stable")
        if arch.early_k > 0:
            first = order[: arch.early_k]
            payouts[first] *= arch.early_payout_mult
            if arch.early_payout_mult < 1.0:
                # unlucky winners also tend to have their early conversions lag more
                lags[first] *= 1.8

        for j in range(total_conv):
            oh = int(origin_hours[j])
            ts = EPOCH + pd.Timedelta(hours=oh)
            pb = ts + pd.Timedelta(hours=float(lags[j]))
            conv_records.append((cid, ts, pb, 0.0, 0, 1, float(payouts[j]),
                                 rng.choice(subids), rng.choice(CREATIVES),
                                 rng.choice(LANDERS), p["offer"],
                                 rng.choice(AUDIENCES), rng.choice(REGIONS),
                                 rng.choice(CITIES)))

    cols = ["campaign_id", "timestamp", "postback_time", "spend", "clicks",
            "conversions", "revenue", "sub_id", "creative", "lander", "offer",
            "audience", "region", "city"]
    events = pd.DataFrame(spend_records + conv_records, columns=cols)

    total_spend = events["spend"].sum()
    total_rev = events["revenue"].sum()
    realised_roi = (total_rev - total_spend) / total_spend if total_spend > 0 else 0.0

    start = EPOCH
    end = EPOCH + pd.Timedelta(hours=hours)
    campaign_row = (cid, f"{arch.name}_{cid}", p["offer"], start, end, "archived")

    truth_row = (cid, arch.name, p["true_roi"], "winner" if p["true_roi"] > 0 else "loser",
                 total_conv, total_spend, total_rev, realised_roi,
                 arch.early_k, arch.early_payout_mult, p["lag_mean"])
    return campaign_row, events, truth_row


def generate(gc: GenConfig | None = None):
    """Generate the full synthetic dataset in memory.

    Returns (campaigns_df, events_df, truth_df).
    """
    gc = gc or GenConfig()
    rng = np.random.default_rng(gc.seed)
    archs = _pick_archetypes(rng, gc.n_campaigns)

    campaign_rows, event_frames, truth_rows = [], [], []
    for i, arch in enumerate(archs):
        cid = f"camp_{i:04d}"
        p = _campaign_params(rng, arch, gc)
        crow, ev, trow = _generate_one(rng, cid, arch, p, gc)
        campaign_rows.append(crow)
        event_frames.append(ev)
        truth_rows.append(trow)

    campaigns = pd.DataFrame(
        campaign_rows,
        columns=["campaign_id", "name", "offer", "start_time", "end_time", "status"],
    )
    events = pd.concat(event_frames, ignore_index=True)
    truth = pd.DataFrame(
        truth_rows,
        columns=["campaign_id", "archetype", "true_roi", "true_label",
                 "total_conversions", "total_spend", "total_revenue",
                 "realised_roi", "early_k", "early_payout_mult", "lag_mean_hours"],
    )
    return campaigns, events, truth


def write_dataset(cfg: Cfg, gc: GenConfig | None = None) -> Path:
    """Generate and write campaigns/events/truth to data_source.file.path."""
    gc = gc or GenConfig(seed=cfg.random_seed)
    campaigns, events, truth = generate(gc)
    out_dir = resolve_path(cfg, cfg.data_source.file.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg.data_source.file.format
    if fmt == "parquet":
        campaigns.to_parquet(out_dir / "campaigns.parquet", index=False)
        events.to_parquet(out_dir / "events.parquet", index=False)
        truth.to_parquet(out_dir / "synthetic_truth.parquet", index=False)
    else:
        campaigns.to_csv(out_dir / "campaigns.csv", index=False)
        events.to_csv(out_dir / "events.csv", index=False)
        truth.to_csv(out_dir / "synthetic_truth.csv", index=False)
    return out_dir


if __name__ == "__main__":
    from .config import load_config
    _cfg = load_config()
    _dir = write_dataset(_cfg)
    print(f"wrote synthetic dataset to {_dir}")
