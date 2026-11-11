"""Point-in-time retraining for Tennis Oracle 2.0.

Earlier dates fit the Random Forest. Later, disjoint dates calibrate it.
Calibration diagnostics are not independent out-of-sample performance.
Use tennis_backtest.py for untouched test evaluations.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd

from tennis_dataset import download_dataset, enrich_dataframe, build_name_lookup
from tennis_models import MODELS_DIR
from tennis_temporal import build_temporal_frame, chronological_masks, temporal_model_sets


def run_training():
    print("TENNIS ORACLE 2.0 — CHRONOLOGICAL TRAINING")
    df = download_dataset(include_challenger=True, include_futures=False, strict=True)
    if len(df) < 1000:
        raise RuntimeError("Insufficient historical data: training interrupted.")
    # A monthly "success" must not publish models trained on stale archives.
    newest = pd.Timestamp(df["tourney_date"].max()).date()
    age = (datetime.now(timezone.utc).date() - newest).days
    if age < 0 or (age > 21 and os.environ.get("ORACLE_ALLOW_STALE_RESEARCH") != "1"):
        raise RuntimeError(
            f"Latest research data {newest} is {age} days old: refusing live model update"
        )
    df = enrich_dataframe(df)
    df = df[df["n_sets"] > 0].copy()
    pack = build_temporal_frame(df)
    print("Valid rows:", len(pack["raw"]), "IDs with collision:", len(pack["colliding_ids"]))

    train, calibration, _, cal_start = chronological_masks(pack, "2100-01-01")
    models, calibration_diagnostics = temporal_model_sets(pack, train, calibration)
    if "winner" not in models or "games_over" not in models:
        raise RuntimeError("Missing trained winner or games model; refusing partial publication")

    os.makedirs(MODELS_DIR, exist_ok=True)
    for market, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"model_{market}.pkl"))

    # Ambiguous cross-tour numeric IDs are excluded, not silently mixed.
    raw_names = build_name_lookup(pack["raw"])
    filtered_names = {name: pid for name, pid in raw_names.items()
                      if pid in pack["player_stats"]}
    joblib.dump(pack["player_stats"], os.path.join(MODELS_DIR, "player_stats.pkl"))
    joblib.dump(pack["surface_stats"], os.path.join(MODELS_DIR, "surface_stats.pkl"))
    joblib.dump(filtered_names, os.path.join(MODELS_DIR, "name_lookup.pkl"))
    historical_cols = [
        "winner_id", "loser_id", "tourney_date", "surface", "tourney_level",
        "best_of", "round", "score", "n_sets", "total_games", "tiebreaks",
        "winner_rank", "loser_rank"
    ]
    joblib.dump(pack["raw"][historical_cols].copy(),
                os.path.join(MODELS_DIR, "df_history.pkl"))

    newest = pd.Timestamp(pack["dates"].max()).date().isoformat()
    metadata = {
        "feature_engine": "temporal-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_latest_date": newest,
        "train_matches": int(train.sum()),
        "calibration_matches": int(calibration.sum()),
        "calibration_starts": cal_start,
        "trained_markets": sorted(models),
        "calibration_only": calibration_diagnostics,
        "independent_backtest_completed": False,
        "data_license": "CC BY-NC-SA 4.0, research/non-commercial only",
        "source_note": "Aneeshers archival mirror; not a live data feed",
    }
    # Manifest is written LAST; failed earlier steps cannot claim success.
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    with open(manifest_path + ".tmp", "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, ensure_ascii=False)
    os.replace(manifest_path + ".tmp", manifest_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("Training complete. Calibration diagnostics are NOT held-out test results.")
    return metadata


if __name__ == "__main__":
    run_training()
