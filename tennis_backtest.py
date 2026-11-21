"""Leakage-free out-of-sample Tennis Oracle 2.0 backtest.

Train: earliest pre-2024 matches; calibrate: later disjoint pre-2024 matches;
test: 2024 onward, NEVER used in model fitting or probability calibration.
No fabricated flat bookmaker odds, profit or ROI.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from tennis_dataset import download_dataset, enrich_dataframe
from tennis_temporal import build_temporal_frame, chronological_masks, temporal_model_sets
from tennis_models import TARGET_COLS


TEST_START = os.environ.get("ORACLE_BACKTEST_START", "2024-01-01")
OUTFILE = os.environ.get("ORACLE_BACKTEST_JSON", "backtest_results.json")


def market_y_mask(pack, market):
    raw = pack["raw"]
    if market == "winner":
        return pack["y_winner"], np.ones(len(raw), dtype=bool)
    column = TARGET_COLS[market]
    if column not in raw:
        return np.zeros(len(raw), dtype=int), np.zeros(len(raw), dtype=bool)
    series = pd.to_numeric(raw[column], errors="coerce")
    mask = series.notna().to_numpy().copy()
    mask &= raw["best_of"].eq(3).to_numpy()
    if market == "aces_over":
        mask &= raw["w_ace"].notna().to_numpy() & raw["l_ace"].notna().to_numpy()
    return series.fillna(-1).to_numpy().astype(int), mask


def run_backtest():
    df = download_dataset(include_challenger=True, include_futures=False, strict=True)
    if len(df) < 1000:
        raise RuntimeError("Missing dataset: cannot claim valid backtest")
    pack = build_temporal_frame(enrich_dataframe(df))
    train, calibrate, test, calibration_start = chronological_masks(pack, TEST_START)
    if int(test.sum()) < 150:
        raise RuntimeError("Not enough untouched later matches for evaluation")
    trees = int(os.environ.get("ORACLE_BACKTEST_TREES", "120"))
    models, calib_metrics = temporal_model_sets(pack, train, calibrate, n_estimators=trees)
    if "winner" not in models:
        raise RuntimeError("Cannot validate without a winner model")

    report = {
        "version": "chronological-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_last_date": str(pack["dates"][train].max().date()),
        "calibration_start_date": calibration_start,
        "calibration_last_date": str(pack["dates"][calibrate].max().date()),
        "test_start": TEST_START,
        "test_last_date": str(pack["dates"][test].max().date()),
        "train_count": int(train.sum()),
        "calibration_count": int(calibrate.sum()),
        "test_count": int(test.sum()),
        "results": {},
        "calibration_diagnostics": calib_metrics,
        "roi": None,
        "roi_note": "No historical timestamped bookmaker odds: ROI intentionally not computed.",
        "source_note": "Archival noncommercial dataset, coverage may be stale.",
    }
    for market, model in models.items():
        y, valid = market_y_mask(pack, market)
        where = test & valid
        if not where.any():
            continue
        X = pack["X_winner"] if market == "winner" else pack["X_base"]
        p = model.predict_proba(X.loc[where])[:, 1]
        truth = y[where]
        past = y[(train | calibrate) & valid]
        base_rate = float(past.mean()) if len(past) else 0.5
        base_prob = np.full_like(p, base_rate, dtype=float)
        res = {
            "tested": int(where.sum()),
            "positive_rate": round(float(truth.mean()), 4),
            "accuracy": round(float(np.mean((p >= .5) == truth)), 4),
            "brier": round(float(brier_score_loss(truth, p)), 5),
            "brier_baseline_pretest_prevalence": round(float(brier_score_loss(truth, base_prob)), 5),
            "log_loss": round(float(log_loss(truth, p, labels=[0, 1])), 5),
            "log_loss_baseline_pretest_prevalence": round(float(log_loss(truth, base_prob, labels=[0, 1])), 5),
            "mean_model_probability": round(float(p.mean()), 4),
        }
        if market == "winner":
            # Lower rank is better; use the ranking visible BEFORE each match.
            rank = X.loc[where, "rank_diff"].to_numpy(dtype=float)
            res["ranking_baseline_accuracy"] = round(float(np.mean((rank <= 0) == truth)), 4)
        # Stratified reporting, to spot drift hidden by one aggregate score.
        period_years = pack["dates"].loc[where].dt.year.to_numpy()
        tours = pack["raw"].loc[where, "tour"].astype(str).to_numpy()
        def _grouped(values):
            groups = {}
            for key in sorted(set(values)):
                subset = values == key
                if subset.sum() < 100:
                    continue
                yy, pp = truth[subset], p[subset]
                groups[str(key)] = {
                    "tested": int(subset.sum()),
                    "accuracy": round(float(np.mean((pp >= .5) == yy)), 4),
                    "brier": round(float(brier_score_loss(yy, pp)), 5),
                    "positive_rate": round(float(yy.mean()), 4),
                }
            return groups
        res["by_year"] = _grouped(period_years)
        res["by_tour"] = _grouped(tours)
        report["results"][market] = res

    with open(OUTFILE, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_backtest()
