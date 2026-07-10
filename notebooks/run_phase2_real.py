"""Phase 2 on REAL data: fit + validate calibration, then show example calls.

  1. Build a calibration dataset from mature campaigns with a verdict: at several
     decision points, compute the raw significance confidence and label by the
     eventual maturity verdict.
  2. Fit isotonic calibration on a TRAIN split; report ECE on a held-out TEST
     split (honest, not resubstitution). Save the calibrator + reliability plot.
  3. Show example SCALE / HOLD / KILL recommendations on real campaigns with reasons.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import resolve_path
from src.datasource import make_datasource, EventCols
from src.backtest import cohorts
from src.confidence import (
    recommend, features, bootstrap_roi, fit_calibrator, reliability, Calibrator,
    load_trust_threshold, to_log_record, SCALE, HOLD, KILL,
)
from src.logging_.schema import RecommendationLog
from notebooks.run_phase1_real import _real_cfg

CALIB_POINTS = [75, 100, 125, 150]     # span confidences; label by maturity verdict
BOOT = 400


def _raw_conf(events_c, n, rng):
    st = features.campaign_state(events_c, n=n, rng=rng)
    if st is None:
        return None
    sig = bootstrap_roi(st.payouts, st.spend, st.n_conversions,
                        iterations=BOOT, rng=rng)
    return sig.p_profitable


def build_calibration_set(events, gt, rng):
    verdict = gt[gt["verdict"].notna()][[EventCols.ID, "verdict"]]
    vmap = dict(zip(verdict[EventCols.ID], verdict["verdict"]))
    rows = []
    for cid, grp in events.groupby(EventCols.ID, sort=False):
        if cid not in vmap:
            continue
        y = 1 if vmap[cid] == "winner" else 0
        for n in CALIB_POINTS:
            rc = _raw_conf(grp, n, rng)
            if rc is not None:
                rows.append((cid, n, rc, y))
    return pd.DataFrame(rows, columns=["campaign_id", "n", "raw_conf", "outcome"])


def main():
    cfg = _real_cfg()
    ds = make_datasource(cfg)
    events = ds.load_events()
    gt = cohorts.assign_ground_truth(events, cfg)
    trust_n = load_trust_threshold(cfg)
    rng = np.random.default_rng(cfg.random_seed)

    print(f"Trust threshold (from Phase 1): {trust_n} conversions")
    print("Building calibration set (raw significance confidence vs maturity verdict)...")
    cal_df = build_calibration_set(events, gt, rng)
    print(f"  {len(cal_df):,} (campaign, decision-point) instances")

    # honest train/test split by CAMPAIGN (no leakage across the same campaign)
    ids = np.asarray(cal_df["campaign_id"].astype(str).unique(), dtype=object)
    rng.shuffle(ids)
    cut = int(0.5 * len(ids))
    train_ids, test_ids = set(ids[:cut]), set(ids[cut:])
    tr = cal_df[cal_df["campaign_id"].isin(train_ids)]
    te = cal_df[cal_df["campaign_id"].isin(test_ids)]

    calibrator = fit_calibrator(tr["raw_conf"].to_numpy(), tr["outcome"].to_numpy())

    raw_test = te["raw_conf"].to_numpy()
    cal_test = np.array([calibrator.predict(r) for r in raw_test])
    rel_raw, ece_raw = reliability(raw_test, te["outcome"].to_numpy())
    rel_cal, ece_cal = reliability(cal_test, te["outcome"].to_numpy())

    print(f"\nCalibration (held-out test):")
    print(f"  ECE raw significance   : {ece_raw:.3f}")
    print(f"  ECE after isotonic map : {ece_cal:.3f}")

    # save calibrator
    cal_path = resolve_path(cfg, "outputs/real_calibrator.json")
    cal_path.write_text(json.dumps(calibrator.to_dict(), indent=2))

    _plot_reliability(rel_raw, rel_cal, ece_raw, ece_cal,
                      resolve_path(cfg, "outputs/real_calibration.png"))

    # ---- example recommendations on real campaigns --------------------------
    print("\n==== EXAMPLE RECOMMENDATIONS (decide-now, full history) ====")
    buckets = {SCALE: [], HOLD: [], KILL: []}
    verdict = gt.set_index(EventCols.ID)["verdict"].to_dict()
    sample_ids = np.asarray(events[EventCols.ID].astype(str).unique(), dtype=object)
    rng.shuffle(sample_ids)
    for cid in sample_ids:
        if all(len(v) >= 2 for v in buckets.values()):
            break
        grp = events[events[EventCols.ID] == cid]
        r = recommend(grp, cfg, trust_n=trust_n, n_cap=None,
                      calibrator=calibrator, rng=rng)
        if len(buckets[r.recommendation]) < 2:
            buckets[r.recommendation].append((r, verdict.get(cid)))
    # also force one below-threshold HOLD by capping a big campaign early
    big = events.groupby(EventCols.ID).conversions.sum().idxmax()
    r_hold = recommend(events[events[EventCols.ID] == big], cfg, trust_n=trust_n,
                       n_cap=max(5, trust_n // 4), calibrator=calibrator, rng=rng)
    buckets[HOLD].insert(0, (r_hold, verdict.get(big)))

    # log every recommendation (append-only) — the future trajectory predictor's
    # training data accumulates here; outcome is backfilled later.
    log = RecommendationLog(resolve_path(cfg, cfg.io.recommendation_log))
    decided_at = pd.Timestamp.utcnow().isoformat()

    for rec_type in (SCALE, HOLD, KILL):
        print(f"\n-- {rec_type} --")
        for r, v in buckets[rec_type][:3]:
            conf = f"{r.confidence:.0%}" if r.confidence is not None else "n/a"
            print(f"  [{r.campaign_id[:14]}… true={v}] conf={conf}: {r.reason}")
            log.append(to_log_record(r, decided_at))

    print(f"\nLogged {len(log.read_all())} recommendations to {resolve_path(cfg, cfg.io.recommendation_log)}")
    print(f"Wrote {cal_path}")
    print(f"Wrote {resolve_path(cfg, 'outputs/real_calibration.png')}")


def _plot_reliability(rel_raw, rel_cal, ece_raw, ece_cal, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    for rel, color, lab, ece in [(rel_raw, "#c0392b", "raw significance", ece_raw),
                                 (rel_cal, "#27ae60", "isotonic-calibrated", ece_cal)]:
        if rel:
            x = [r["mean_conf"] for r in rel]
            y = [r["empirical_rate"] for r in rel]
            ax.plot(x, y, "-o", color=color, label=f"{lab} (ECE={ece:.3f})")
    ax.set_xlabel("Stated confidence  P(profitable)")
    ax.set_ylabel("Empirical fraction that matured profitable")
    ax.set_title("Phase 2 calibration — held-out test")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
