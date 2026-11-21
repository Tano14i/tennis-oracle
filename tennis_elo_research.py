"""Strict pre-match Elo research against held-out June-November 2026 results.

No production deploy, no odds as features, no ROI claims. The historical
holdout has ALREADY been inspected during development. Treat this as a
POST-HOC diagnostic, not a fresh out-of-sample validation of the project.

An Elo state on day D only sees matches BEFORE day D. Sigmoid adjustment
is fit on a PRE-test window using logistic calibration, never on test.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression

from tennis_elo import pre_match_elo, elo_configuration
from tennis_hf_audit import NAMES, download as download_hf
from tennis_hf_crosswalk import run as build_verified_crosswalk, parse_csv
from tennis_hf_research_backtest import prepare_rows, evaluate_probabilities

CALIBRATION_FROM = "2026-02-23"
HOLDOUT_FROM = "2026-06-02"


def _logit(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(q / (1.0 - q)).reshape(-1, 1)


def benchmark(frame, closing_probabilities):
    prematch = pre_match_elo(frame)
    dates = prematch["date"]
    calibration = ((dates >= CALIBRATION_FROM) & (dates < HOLDOUT_FROM)).to_numpy()
    holdout = (dates >= HOLDOUT_FROM).to_numpy()
    first = (dates < CALIBRATION_FROM).to_numpy()

    y = prematch["y_p1_wins"].to_numpy(dtype=int)
    raw = prematch["elo_p1_probability"].to_numpy(dtype=float)
    if first.sum() < 1000 or calibration.sum() < 200 or holdout.sum() < 500:
        raise ValueError("Insufficient strict date split for Elo research")
    if np.unique(y[calibration]).size < 2:
        raise ValueError("Calibration date window does not contain both classes")

    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    calibrator.fit(_logit(raw[calibration]), y[calibration])
    calibrated = calibrator.predict_proba(_logit(raw))[:, 1]

    odds_inds, consensus = [], []
    held_indices = np.flatnonzero(holdout)
    for local_i, frame_i in enumerate(held_indices):
        id_ = prematch.iloc[frame_i]["source_id"]
        reported = closing_probabilities.get(id_)
        if reported is None or not np.isfinite(reported) or not .001 < reported < .999:
            continue
        # Closing probability describes source winner. Elo p describes lower
        # numeric player ID. Orientation conversion uses true historical
        # winner ONLY to select the correct target in this EVALUATION.
        oriented = reported if y[frame_i] == 1 else 1.0 - reported
        odds_inds.append(frame_i)
        consensus.append(oriented)
    odds_inds = np.asarray(odds_inds, dtype=int)
    consensus = np.asarray(consensus, dtype=float)

    by_tour = {}
    by_surface = {}
    for col, destination in (("tour", by_tour), ("surface", by_surface)):
        for value, subset in prematch.loc[holdout].groupby(col):
            idx = subset.index.to_numpy()
            destination[str(value)] = {
                "matches": len(idx),
                "elo_calibrated": evaluate_probabilities(y[idx], calibrated[idx]),
                "elo_uncalibrated": evaluate_probabilities(y[idx], raw[idx]),
            }
    matches_seen = np.minimum(
        prematch.loc[holdout, "p1_global_matches_before"].to_numpy(dtype=int),
        prematch.loc[holdout, "p2_global_matches_before"].to_numpy(dtype=int),
    )
    return {
        "report_type": "POST_HOC_DIAGNOSTIC_ONLY",
        "deployment_allowed": False,
        "same_2026_holdout_previously_inspected": True,
        "fresh_out_of_sample_claim": False,
        "research_calibration_only_pretest": True,
        "no_prices_as_features": True,
        "elo_configuration": elo_configuration(),
        "first_date_count": int(first.sum()),
        "calibration_date_count": int(calibration.sum()),
        "holdout_match_count": int(holdout.sum()),
        "test_from": HOLDOUT_FROM,
        "test_last_date": str(prematch.loc[holdout, "date"].max().date()),
        "calibrator_coefficient": float(calibrator.coef_[0][0]),
        "calibrator_intercept": float(calibrator.intercept_[0]),
        "holdout_elo_uncalibrated": evaluate_probabilities(y[holdout], raw[holdout]),
        "holdout_elo_calibrated": evaluate_probabilities(y[holdout], calibrated[holdout]),
        "holdout_elo_vs_closing_common_subset": {
            "n": len(odds_inds),
            "elo_uncalibrated": evaluate_probabilities(y[odds_inds], raw[odds_inds]) if len(odds_inds) else None,
            "elo_calibrated": evaluate_probabilities(y[odds_inds], calibrated[odds_inds]) if len(odds_inds) else None,
            "market_closing": evaluate_probabilities(y[odds_inds], consensus) if len(odds_inds) else None,
        },
        "elo_cold_start_both_players_fewer_than_5_prior_matches":
            int((matches_seen < 5).sum()),
        "by_tour": by_tour,
        "by_surface": by_surface,
        "no_roi_or_edge_claim": True,
        "data_license_not_cleared_for_production": True,
        "notes": [
            "All Elo features use strictly earlier calendar dates, never same-day scores.",
            "The model uses neither historical odds nor current market prices as predictive features.",
            "The reported closing consensus is retrospective and cannot be wagered at prediction time.",
            "The 2026 test period was previously inspected; do not tune Elo hyperparameters on its outcomes.",
        ],
    }


def run():
    crosswalk = build_verified_crosswalk()
    with open("hf_verified_crosswalk_research.json", encoding="utf-8") as handle:
        mapping = json.load(handle)["mapping"]
    source = {"ATP": [], "WTA": []}
    for filename in NAMES:
        content, digest = download_hf(filename)
        if digest != crosswalk["source_sha256_by_filename"][filename]:
            raise ValueError("Crosswalk snapshot differs from current data: refused")
        tour = "ATP" if filename.startswith("atp_") else "WTA"
        source[tour].extend(parse_csv(content))
    data, closing, counts = prepare_rows(source, mapping, datetime.now(timezone.utc))
    report = benchmark(data, closing)
    report["verified_mappings"] = len(mapping)
    report["input_cohort"] = counts
    report["sources_sha256"] = crosswalk["source_sha256_by_filename"]
    filename = os.environ.get("ORACLE_ELO_REPORT", "hf_elo_posthoc_diagnostic.json")
    with open(filename, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
