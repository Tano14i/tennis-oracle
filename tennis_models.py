"""
tennis_models.py
Addestramento e predizione modelli ML per Tennis Oracle.

Mercati affidabili (lift positivo):
  - winner:     mostra prob ma NO BET se < 65%
  - both_set:   +0.105 lift - affidabile
  - games_over: +0.127 lift - affidabile
  - aces_over:  +0.127 lift - affidabile

Mercati deboli (nascosti dal report):
  - sets_over:  -0.062 lift - sotto baseline
  - tiebreak:   -0.050 lift - sotto baseline
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
LEVEL_MAP = {"G": 4, "M": 3, "A": 2, "P": 1, "PM": 1, "D": 0, "F": 2, "I": 1, "O": 1, "W": 2}
ROUND_MAP = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3}

# Mercati con lift positivo — mostrati nel report
RELIABLE_MARKETS = ["winner", "both_set", "games_over", "aces_over"]

# Mercati deboli — addestrati ma nascosti nel report
ALL_MARKETS = ["winner", "both_set", "games_over", "aces_over", "sets_over", "tiebreak"]

TARGET_COLS = {
    "winner":     "target_winner",
    "sets_over":  "target_sets_over",
    "both_set":   "target_both_set",
    "games_over": "target_games_over_225",
    "aces_over":  "target_aces_over_105",
    "tiebreak":   "target_tiebreak",
}

FEATURE_COLS_BASE = [
    "rank_diff", "rank_ratio", "pts_diff", "age_diff", "ht_diff",
    "surface_enc", "level_enc", "round_enc", "best_of",
    "p1_aces_avg", "p2_aces_avg", "aces_sum",
    "p1_df_avg", "p2_df_avg",
    "p1_1st_in", "p2_1st_in",
    "p1_1st_won", "p2_1st_won",
    "p1_2nd_won", "p2_2nd_won",
    "p1_win_rate", "p2_win_rate",
    "p1_wr_surface", "p2_wr_surface",
    "h2h_p1_win_rate", "h2h_total",
    "p1_recent_wr", "p2_recent_wr",
    "p1_streak", "p2_streak",
]

# Feature extra per winner
FEATURE_COLS_WINNER = FEATURE_COLS_BASE + ["form_diff", "surf_wr_diff", "recent_form_diff", "streak_diff"]


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not np.isnan(v) and not np.isinf(v) else default
    except Exception:
        return default


def build_row(row, player_stats, surface_stats, flip=False, h2h_lookup=None, row_idx=None):
    """Costruisce una riga di feature. Se flip=True, scambia p1 e p2."""
    if flip:
        p1_id = row.get("loser_id")
        p2_id = row.get("winner_id")
        rank1 = _safe_float(row.get("loser_rank"), 300)
        rank2 = _safe_float(row.get("winner_rank"), 300)
        pts1  = _safe_float(row.get("loser_rank_points"), 0)
        pts2  = _safe_float(row.get("winner_rank_points"), 0)
        age1  = _safe_float(row.get("loser_age"), 25)
        age2  = _safe_float(row.get("winner_age"), 25)
        ht1   = _safe_float(row.get("loser_ht"), 185)
        ht2   = _safe_float(row.get("winner_ht"), 185)
    else:
        p1_id = row.get("winner_id")
        p2_id = row.get("loser_id")
        rank1 = _safe_float(row.get("winner_rank"), 300)
        rank2 = _safe_float(row.get("loser_rank"), 300)
        pts1  = _safe_float(row.get("winner_rank_points"), 0)
        pts2  = _safe_float(row.get("loser_rank_points"), 0)
        age1  = _safe_float(row.get("winner_age"), 25)
        age2  = _safe_float(row.get("loser_age"), 25)
        ht1   = _safe_float(row.get("winner_ht"), 185)
        ht2   = _safe_float(row.get("loser_ht"), 185)

    surface = str(row.get("surface", "Hard"))
    level   = str(row.get("tourney_level", "A"))
    rnd     = str(row.get("round", "R32"))

    p1_stats = player_stats.get(int(p1_id) if p1_id and not pd.isna(p1_id) else -1, {})
    p2_stats = player_stats.get(int(p2_id) if p2_id and not pd.isna(p2_id) else -1, {})
    p1_surf  = surface_stats.get(int(p1_id) if p1_id and not pd.isna(p1_id) else -1, {})
    p2_surf  = surface_stats.get(int(p2_id) if p2_id and not pd.isna(p2_id) else -1, {})
    surf_key = f"wr_{surface.lower()}"

    p1_aces    = p1_stats.get("p_aces_avg", 5.0)
    p2_aces    = p2_stats.get("p_aces_avg", 5.0)
    p1_wr      = p1_stats.get("p_win_rate", 0.5)
    p2_wr      = p2_stats.get("p_win_rate", 0.5)
    p1_swr     = p1_surf.get(surf_key, 0.5)
    p2_swr     = p2_surf.get(surf_key, 0.5)
    p1_rwr     = p1_stats.get("p_recent_wr", p1_wr)
    p2_rwr     = p2_stats.get("p_recent_wr", p2_wr)
    p1_streak  = p1_stats.get("p_streak", 0)
    p2_streak  = p2_stats.get("p_streak", 0)

    h2h_entry = h2h_lookup[row_idx] if (h2h_lookup and row_idx is not None and row_idx in h2h_lookup) else {}

    return {
        "rank_diff":     rank1 - rank2,
        "rank_ratio":    rank1 / max(rank2, 1),
        "pts_diff":      pts1 - pts2,
        "age_diff":      age1 - age2,
        "ht_diff":       ht1 - ht2,
        "surface_enc":   SURFACE_MAP.get(surface, 0),
        "level_enc":     LEVEL_MAP.get(level, 1),
        "round_enc":     ROUND_MAP.get(rnd, 3),
        "best_of":       int(row.get("best_of", 3) or 3),
        "p1_aces_avg":   p1_aces,
        "p2_aces_avg":   p2_aces,
        "aces_sum":      p1_aces + p2_aces,
        "p1_df_avg":     p1_stats.get("p_df_avg", 3.0),
        "p2_df_avg":     p2_stats.get("p_df_avg", 3.0),
        "p1_1st_in":     p1_stats.get("p_1st_in", 0.60),
        "p2_1st_in":     p2_stats.get("p_1st_in", 0.60),
        "p1_1st_won":    p1_stats.get("p_1st_won", 0.70),
        "p2_1st_won":    p2_stats.get("p_1st_won", 0.70),
        "p1_2nd_won":    p1_stats.get("p_2nd_won", 0.50),
        "p2_2nd_won":    p2_stats.get("p_2nd_won", 0.50),
        "p1_win_rate":   p1_wr,
        "p2_win_rate":   p2_wr,
        "p1_wr_surface": p1_swr,
        "p2_wr_surface": p2_swr,
        "h2h_p1_win_rate": h2h_entry.get("h2h_p1_win_rate", 0.5),
        "h2h_total":       h2h_entry.get("h2h_total", 0),
        "p1_recent_wr":  p1_rwr,
        "p2_recent_wr":  p2_rwr,
        "p1_streak":     p1_streak,
        "p2_streak":     p2_streak,
        "form_diff":       p1_wr  - p2_wr,
        "surf_wr_diff":    p1_swr - p2_swr,
        "recent_form_diff": p1_rwr - p2_rwr,
        "streak_diff":     p1_streak - p2_streak,
    }


def build_features(df, player_stats, surface_stats, h2h_lookup=None):
    """Feature matrix per training (senza winner flip — usa build_winner_features per quello)."""
    rows = [build_row(row, player_stats, surface_stats, flip=False,
                      h2h_lookup=h2h_lookup, row_idx=idx)
            for idx, row in df.iterrows()]
    return pd.DataFrame(rows, columns=FEATURE_COLS_BASE)


def build_winner_features(df, player_stats, surface_stats, h2h_lookup=None):
    """
    Feature matrix bilanciata per winner:
    Per ogni partita crea DUE righe (flip=False -> target=1, flip=True -> target=0).
    """
    rows = []
    targets = []
    for idx, row in df.iterrows():
        rows.append(build_row(row, player_stats, surface_stats, flip=False,
                               h2h_lookup=h2h_lookup, row_idx=idx))
        targets.append(1)
        # Flipped: h2h_p1_win_rate viene invertito
        r_flip = build_row(row, player_stats, surface_stats, flip=True,
                           h2h_lookup=h2h_lookup, row_idx=idx)
        r_flip["h2h_p1_win_rate"] = 1.0 - r_flip["h2h_p1_win_rate"]
        rows.append(r_flip)
        targets.append(0)
    X = pd.DataFrame(rows, columns=FEATURE_COLS_WINNER)
    return X, np.array(targets)


def train_all_models(df, player_stats, surface_stats):
    from tennis_dataset import build_h2h_lookup
    results = {}

    print("  Precalcolo H2H lookup per training (no data leakage)...")
    h2h_lookup = build_h2h_lookup(df)

    # === Winner (dataset bilanciato) ===
    print("  Training winner (dataset bilanciato)...")
    X_w, y_w = build_winner_features(df, player_stats, surface_stats, h2h_lookup=h2h_lookup)
    X_w = X_w.fillna(0)
    rf_w = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model_w = CalibratedClassifierCV(rf_w, method="sigmoid", cv=5)
    model_w.fit(X_w, y_w)
    cv_w = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10,
                               class_weight="balanced", random_state=42, n_jobs=-1),
        X_w, y_w, cv=5, scoring="accuracy"
    )
    joblib.dump(model_w, os.path.join(MODELS_DIR, "model_winner.pkl"))
    results["winner"] = {
        "accuracy": round(cv_w.mean(), 3),
        "samples": len(X_w),
        "positive_rate": 0.5,
    }
    print(f"    Accuracy CV: {cv_w.mean():.3f} | Baseline: 0.643 (fav vince)")

    # === Altri mercati ===
    print("  Costruzione feature matrix altri mercati...")
    X_base = build_features(df, player_stats, surface_stats, h2h_lookup=h2h_lookup).fillna(0)

    for market in ["sets_over", "both_set", "games_over", "aces_over", "tiebreak"]:
        target_col = TARGET_COLS[market]
        if target_col not in df.columns:
            print(f"  {market}: colonna target non trovata, skip")
            continue

        y = df[target_col].values
        mask = ~np.isnan(y.astype(float))
        X_clean = X_base[mask]
        y_clean = y[mask]

        if len(X_clean) < 100 or len(set(y_clean)) < 2:
            print(f"  {market}: dati insufficienti, skip")
            continue

        print(f"  Training {market} ({len(X_clean)} campioni)...")
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
        model.fit(X_clean, y_clean)
        cv = cross_val_score(
            RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10,
                                   class_weight="balanced", random_state=42, n_jobs=-1),
            X_clean, y_clean, cv=5, scoring="accuracy"
        )
        joblib.dump(model, os.path.join(MODELS_DIR, f"model_{market}.pkl"))

        positive_rate = float(y_clean.mean())
        results[market] = {
            "accuracy": round(cv.mean(), 3),
            "samples": len(X_clean),
            "positive_rate": round(positive_rate, 3),
        }
        print(f"    Accuracy CV: {cv.mean():.3f} | Baseline: {max(positive_rate, 1-positive_rate):.3f}")

    return results


def predict_match(p1_id, p2_id, surface, level, round_str, best_of,
                  player_stats, surface_stats, df_history):
    from tennis_dataset import get_h2h, get_h2h_on_surface

    h2h = get_h2h(df_history, p1_id, p2_id)
    h2h_surf = get_h2h_on_surface(df_history, p1_id, p2_id, surface)

    # Costruisce riga feature come se p1 fosse il winner (flip=False)
    fake_row = {
        "winner_id": p1_id, "loser_id": p2_id,
        "winner_rank": player_stats.get(p1_id, {}).get("p_rank", 300),
        "loser_rank":  player_stats.get(p2_id, {}).get("p_rank", 300),
        "winner_rank_points": player_stats.get(p1_id, {}).get("p_pts", 0),
        "loser_rank_points":  player_stats.get(p2_id, {}).get("p_pts", 0),
        "winner_age": player_stats.get(p1_id, {}).get("p_age", 25),
        "loser_age":  player_stats.get(p2_id, {}).get("p_age", 25),
        "winner_ht":  player_stats.get(p1_id, {}).get("p_ht", 185),
        "loser_ht":   player_stats.get(p2_id, {}).get("p_ht", 185),
        "surface": surface, "tourney_level": level,
        "round": round_str, "best_of": best_of,
    }

    # A prediction time non c'è lookup: passiamo h2h direttamente via fake_row
    fake_row["_h2h_p1_win_rate"] = h2h.get("h2h_p1_win_rate", 0.5)
    fake_row["_h2h_total"]       = h2h.get("h2h_total", 0)

    feat_base = build_row(fake_row, player_stats, surface_stats, flip=False)
    # Sovrascrivi con i valori H2H reali (build_row non ha lookup a prediction time)
    feat_base["h2h_p1_win_rate"] = fake_row["_h2h_p1_win_rate"]
    feat_base["h2h_total"]       = fake_row["_h2h_total"]

    X_base = pd.DataFrame([feat_base], columns=FEATURE_COLS_BASE).fillna(0)
    X_winner = pd.DataFrame([feat_base], columns=FEATURE_COLS_WINNER).fillna(0)

    predictions = {}
    for market in ALL_MARKETS:
        model_path = os.path.join(MODELS_DIR, f"model_{market}.pkl")
        if not os.path.exists(model_path):
            predictions[market] = {"prob": 0.5, "confidence": "bassa", "reliable": False}
            continue
        try:
            model = joblib.load(model_path)
            X = X_winner if market == "winner" else X_base
            prob = float(model.predict_proba(X)[0][1])

            # Winner: affidabile solo se > 65% (chiara superiorita)
            if market == "winner":
                reliable = prob >= 0.65 or prob <= 0.35
                confidence = "alta" if abs(prob - 0.5) >= 0.15 else "media" if abs(prob - 0.5) >= 0.08 else "bassa"
            else:
                reliable = market in RELIABLE_MARKETS
                confidence = "alta" if prob >= 0.70 else "media" if prob >= 0.58 else "bassa"

            predictions[market] = {
                "prob": round(prob, 3),
                "prob_pct": round(prob * 100, 1),
                "confidence": confidence,
                "reliable": reliable,
            }
        except Exception as e:
            predictions[market] = {"prob": 0.5, "confidence": "bassa", "reliable": False}

    return predictions, h2h, h2h_surf
