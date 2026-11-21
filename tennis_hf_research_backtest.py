"""Reproducible 2026 tennis winner holdout vs closing market probabilities.

RESEARCH-ONLY and no model output. Inputs are the fixed HF CSV source after
integrity checks and historical crosswalk, with live/legacy deployment DISABLED.

Training uses January 2025 through early 2026; sigmoid calibration on a
later, disjoint pre-June-2026 interval; final untouched test from 2026-06-02.
Only players with independent score/date historical identity evidence enter.

Closing odds are OUT-OF-MODEL comparison only (no ROI or CLV claim).
No odds, betting markets, match outcomes or forward-looking statistics are
ever passed as features. Selection bias is expected from matched-player cohort.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss

from tennis_hf_audit import NAMES, download as download_hf, _date as parse_start
from tennis_hf_crosswalk import run as build_verified, parse_csv, score_key
from tennis_temporal import build_temporal_frame, chronological_masks
from tennis_rank_history import carry_forward_prior_ranks
from tennis_models import FEATURE_COLS_WINNER

TEST_FROM = "2026-06-02"
NUMERIC = (
    "winner_rank", "loser_rank", "winner_age", "loser_age",
    "winner_ht", "loser_ht", "winner_rank_points", "loser_rank_points",
    "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
    "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon",
    "w_2ndWon", "l_2ndWon"
)
# Deliberately excludes all closing, opening, current or future bookmaker
# columns from the frame used to construct model features.
INFER_FIELDS = NUMERIC + (
    "source_id", "tourney_date", "winner_id", "loser_id", "surface",
    "tourney_level", "round", "best_of", "score", "tour",
)
FORBIDDEN = ("close", "odds", "prob", "bookmaker", "open_")


def prepare_rows(source_by_tour, mapping, now):
    """Map verified source IDs; return model-only rows and separate odds labels."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    rows = []
    closings = {}
    counts = {"all": 0, "both_verified": 0, "eligible": 0,
              "excluded_unmapped": 0, "excluded_invalid": 0}
    for tour in ("ATP", "WTA"):
        for entry in source_by_tour.get(tour, []):
            counts["all"] += 1
            wkey = tour + ":" + str(entry.get("winner_id") or "").strip()
            lkey = tour + ":" + str(entry.get("loser_id") or "").strip()
            if wkey not in mapping or lkey not in mapping:
                counts["excluded_unmapped"] += 1
                continue
            counts["both_verified"] += 1
            start = parse_start(entry.get("start_time_utc"))
            score = score_key(entry.get("score"))
            try:
                bo = int(entry.get("best_of") or 0)
            except (ValueError, TypeError):
                bo = 0
            surface = entry.get("surface")
            if (not start or start > now or not score or surface not in
                    ("Hard", "Clay", "Grass") or bo not in (3, 5)):
                counts["excluded_invalid"] += 1
                continue
            wid = int(mapping[wkey]["legacy_id"])
            lid = int(mapping[lkey]["legacy_id"])
            if wid == lid:
                counts["excluded_invalid"] += 1
                continue
            source_id = tour + ":" + str(entry.get("match_id") or "")
            if source_id in closings:
                raise ValueError("Duplicated source ID during research preparation")
            row = {
                "source_id": source_id,
                "tourney_date": start.date().isoformat(),
                "winner_id": wid, "loser_id": lid,
                "tour": tour, "score": str(entry["score"]),
                "surface": surface, "best_of": bo,
                "tourney_level": entry.get("tourney_level") or "A",
                "round": entry.get("round") or "R32",
            }
            for key in NUMERIC:
                row[key] = pd.to_numeric(entry.get(key), errors="coerce")
            row = {key: row[key] for key in INFER_FIELDS}
            rows.append(row)
            closing = pd.to_numeric(entry.get("winner_close_prob"), errors="coerce")
            closings[source_id] = float(closing) if pd.notna(closing) else None
            counts["eligible"] += 1
    frame = pd.DataFrame(rows, columns=INFER_FIELDS)
    if any(part in column.lower() for column in frame.columns for part in FORBIDDEN):
        raise AssertionError("Forbidden bookmaker column entered ML features")
    if frame.empty:
        raise ValueError("No research cohort available")
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"])
    frame, rank_diagnostics = carry_forward_prior_ranks(frame, max_age_days=70)
    counts["causal_rank_carryforward"] = rank_diagnostics
    return frame, closings, counts


def evaluate_probabilities(y, probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1-1e-6)
    y = np.asarray(y, dtype=int)
    return {
        "tested": int(len(y)),
        "accuracy": round(float(np.mean((p >= .5) == y)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5),
    }


def fit_and_evaluate(frame, closings, n_estimators=250):
    if len(frame) < 1000:
        raise ValueError("Too few historical matches for meaningful research")
    pack = build_temporal_frame(frame)
    train, calibration, test, calibration_start = chronological_masks(pack, TEST_FROM)
    if train.sum() < 1000 or calibration.sum() < 200 or test.sum() < 500:
        raise ValueError("Missing independent training/calibration/holdout samples")
    y = pack["y_winner"]
    X = pack["X_winner"]
    if len(np.unique(y[train])) != 2 or len(np.unique(y[calibration])) != 2:
        raise ValueError("Chronological training/calibration lacks both classes")
    forest = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=8, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    forest.fit(X.loc[train], y[train])
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(forest),
                                        method="sigmoid")
    calibrated.fit(X.loc[calibration], y[calibration])

    prob = calibrated.predict_proba(X.loc[test])[:, 1]
    truth = y[test]
    rank = X.loc[test, "rank_diff"].to_numpy(dtype=float)
    rank_accuracy = float(np.mean((rank <= 0) == truth))
    raw_test = pack["raw"].loc[test]
    w_rank = pd.to_numeric(raw_test["winner_rank"], errors="coerce")
    l_rank = pd.to_numeric(raw_test["loser_rank"], errors="coerce")
    rank_known = (
        w_rank.notna().to_numpy() & l_rank.notna().to_numpy() &
        (w_rank.to_numpy() > 0) & (l_rank.to_numpy() > 0)
    )
    rank_known_n = int(rank_known.sum())
    rank_accuracy_known = (
        float(np.mean((rank[rank_known] <= 0) == truth[rank_known]))
        if rank_known_n else None
    )
    test_ids = raw_test["source_id"].tolist()

    joint_index = []
    closing_prob = []
    for idx, source_id in enumerate(test_ids):
        observed_winner_prob = closings.get(source_id)
        if (observed_winner_prob is not None and
                np.isfinite(observed_winner_prob) and
                .001 < observed_winner_prob < .999):
            # First player is the lower numeric legacy ID.
            p1_probability = (observed_winner_prob if truth[idx] == 1
                              else 1-observed_winner_prob)
            joint_index.append(idx)
            closing_prob.append(p1_probability)

    result = {
        "model_type": "RandomForest winner, time-split sigmoid calibration",
        "post_hoc_exploration": True,
        "holdout_reused_not_independent_after_feature_diagnosis": True,
        "cohort": "Historically verified source IDs only; selection-biased",
        "test_start": TEST_FROM,
        "train_n": int(train.sum()),
        "calibration_n": int(calibration.sum()),
        "test_n": int(test.sum()),
        "training_last_day": str(pack["dates"][train].max().date()),
        "calibration_first_day": calibration_start,
        "calibration_last_day": str(pack["dates"][calibration].max().date()),
        "test_last_day": str(pack["dates"][test].max().date()),
        "winner_model_full_holdout": evaluate_probabilities(truth, prob),
        "rank_only_accuracy_full_holdout": round(rank_accuracy, 5),
        "rank_baseline_diagnostics": {
            "both_ranks_known": rank_known_n,
            "rank_coverage": round(rank_known_n / int(test.sum()), 5),
            "rank_only_accuracy_both_known":
                round(rank_accuracy_known, 5) if rank_accuracy_known is not None else None,
            "unknown_rank_count": int(test.sum()) - rank_known_n,
            "note": "Ranks carried forward only from real observations on earlier dates, max 70 days; rest default to 300.",
        },
        "bookmaker_closing_common_subset": {
            "subset_n": len(joint_index),
            "model": evaluate_probabilities(truth[joint_index], prob[joint_index])
              if joint_index else None,
            "closing_consensus_probability": evaluate_probabilities(
                truth[joint_index], closing_prob) if joint_index else None,
        },
        "no_roi_claim": True,
        "models_published": False,
        "licensed_for_deployment": False,
        "training_feature_names": list(FEATURE_COLS_WINNER),
        "notes": [
            "Reported closing probabilities are retrospective; they were excluded from model training.",
            "Closing prices are not an executable pre-match backtest.",
            "A bookmaker benchmark should not be interpreted as achievable ROI.",
            "No real-money wager is made or recommended by this research run.",
        ],
    }
    return result


def run():
    crosswalk_report = build_verified()
    with open("hf_verified_crosswalk_research.json", encoding="utf-8") as handle:
        mapping = json.load(handle)["mapping"]
    inputs = {"ATP": [], "WTA": []}
    import hashlib
    for filename in NAMES:
        raw, digest = download_hf(filename)
        if digest != crosswalk_report["source_sha256_by_filename"][filename]:
            raise ValueError("Provider snapshot changed after crosswalk; stop.")
        tour = "ATP" if filename.startswith("atp_") else "WTA"
        inputs[tour].extend(parse_csv(raw))
    frame, closing, counts = prepare_rows(inputs, mapping, datetime.now(timezone.utc))
    metrics = fit_and_evaluate(frame, closing)
    metrics.update({
        "research_only": True, "dataset_integrity_passed": True,
        "verified_id_mapping_count": len(mapping), "source_cohort_counts": counts,
        "input_sha256": crosswalk_report["source_sha256_by_filename"],
        "license_review_needed": "Source upstream Flashscore/TennisExplorer rights unverified",
    })
    filename = os.environ.get("ORACLE_RESEARCH_OUTPUT", "hf_winner_research_holdout.json")
    with open(filename, "w", encoding="utf-8") as out:
        json.dump(metrics, out, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    # There is intentionally no model serialization, merge, API or deploy.
    return metrics

if __name__ == "__main__":
    run()
