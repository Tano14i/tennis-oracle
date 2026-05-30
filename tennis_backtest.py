"""
tennis_backtest.py
Backtesting del modello Tennis Oracle su dati storici.

Uso:
    python tennis_backtest.py

Output:
    - Accuracy per mercato
    - Calibration (Brier score)
    - ROI simulato contro quota flat 1.90 (margine bookmaker tipico)
    - Log delle pick ad alta confidenza con esito reale
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tennis_dataset import download_dataset, enrich_dataframe, build_player_stats_lookup, build_surface_stats
from tennis_models import MODELS_DIR, FEATURE_COLS_BASE, FEATURE_COLS_WINNER, build_row, build_h2h_lookup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKTEST_FROM = "2024-01-01"   # testa su questo periodo
TRAIN_UNTIL   = "2023-12-31"   # modelli trainati su questo
FLAT_ODDS     = 1.90            # quota simulata per ROI
CONF_THRESHOLD = 0.62           # soglia per contare come "high confidence pick"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest():
    print("=" * 60)
    print("TENNIS ORACLE — BACKTESTING")
    print(f"Periodo test: {BACKTEST_FROM} → oggi")
    print(f"Quota simulata: {FLAT_ODDS} | Soglia confidenza: {CONF_THRESHOLD}")
    print("=" * 60)

    # Carica dataset completo
    print("\nCaricamento dataset...")
    df_raw = download_dataset(include_challenger=True)
    if df_raw.empty:
        print("Errore: dataset vuoto")
        return

    df = enrich_dataframe(df_raw)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    # Split train/test
    df_train = df[df["tourney_date"] <= TRAIN_UNTIL].copy()
    df_test  = df[df["tourney_date"] >= BACKTEST_FROM].copy()

    print(f"Train: {len(df_train):,} match | Test: {len(df_test):,} match")

    if df_test.empty:
        print("Nessun dato nel periodo di test.")
        return

    # Costruisci stats su dati di train only
    print("Costruzione statistiche su dati di training...")
    player_stats  = build_player_stats_lookup(df_train)
    surface_stats = build_surface_stats(df_train)
    h2h_lookup    = build_h2h_lookup(df_train)  # H2H su train only

    # Carica modelli
    print("Caricamento modelli...")
    models = {}
    for market in ["winner", "both_set", "games_over", "aces_over", "sets_over", "tiebreak"]:
        path = os.path.join(MODELS_DIR, f"model_{market}.pkl")
        if os.path.exists(path):
            models[market] = joblib.load(path)
            print(f"  OK: {market}")
        else:
            print(f"  MANCANTE: {market}")

    if not models:
        print("Nessun modello trovato. Esegui prima il training.")
        return

    # Costruisci feature per il test set
    print(f"\nCalcolo predizioni su {len(df_test):,} match di test...")
    from tennis_models import build_row, FEATURE_COLS_WINNER, FEATURE_COLS_BASE

    TARGET_MAP = {
        "winner":     "target_winner",
        "both_set":   "target_both_set",
        "games_over": "target_games_over_225",
        "aces_over":  "target_aces_over_105",
        "sets_over":  "target_sets_over",
        "tiebreak":   "target_tiebreak",
    }

    results = {m: {"y_true": [], "y_prob": [], "y_pred": []} for m in models}
    high_conf_picks = []

    for idx, row in df_test.iterrows():
        feat = build_row(row, player_stats, surface_stats, flip=False,
                         h2h_lookup=h2h_lookup, row_idx=idx)

        for market, model in models.items():
            target_col = TARGET_MAP[market]
            if target_col not in df_test.columns:
                continue

            y_true = int(row[target_col]) if not pd.isna(row.get(target_col)) else None
            if y_true is None:
                continue

            feat_cols = FEATURE_COLS_WINNER if market == "winner" else FEATURE_COLS_BASE
            X = pd.DataFrame([feat])[feat_cols].fillna(0)

            try:
                prob = float(model.predict_proba(X)[0][1])
                pred = 1 if prob >= 0.5 else 0
            except Exception:
                continue

            results[market]["y_true"].append(y_true)
            results[market]["y_prob"].append(prob)
            results[market]["y_pred"].append(pred)

            # Track high confidence picks
            if market in ["both_set", "games_over", "aces_over"] and prob >= CONF_THRESHOLD:
                high_conf_picks.append({
                    "date":    str(row.get("tourney_date", ""))[:10],
                    "market":  market,
                    "p1":      row.get("winner_name", "?"),
                    "p2":      row.get("loser_name", "?"),
                    "prob":    prob,
                    "outcome": y_true,
                    "correct": int(pred == y_true),
                })

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RISULTATI BACKTESTING")
    print("=" * 60)

    total_profit = 0.0
    total_bets   = 0

    for market, data in results.items():
        if not data["y_true"]:
            continue

        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        y_pred = np.array(data["y_pred"])

        n          = len(y_true)
        accuracy   = np.mean(y_pred == y_true)
        # Brier score (0=perfetto, 0.25=random)
        brier      = np.mean((y_prob - y_true) ** 2)
        base_rate  = y_true.mean()
        brier_base = np.mean((base_rate - y_true) ** 2)
        brier_lift = brier_base - brier  # positivo = meglio del baseline

        # ROI simulato: scommetti 1 unità ogni volta che prob >= 0.55
        bet_mask   = y_prob >= CONF_THRESHOLD
        n_bets     = bet_mask.sum()
        if n_bets > 0:
            wins    = (y_true[bet_mask] == 1).sum()
            profit  = wins * FLAT_ODDS - n_bets
            roi_pct = profit / n_bets * 100
            total_profit += profit
            total_bets   += n_bets
        else:
            roi_pct = 0
            profit  = 0

        marker = "OK" if brier_lift > 0 else "!!"
        print(f"\n[{marker}] {market.upper()}")
        print(f"  Match: {n:,} | Accuracy: {accuracy:.1%} | Base rate: {base_rate:.1%}")
        print(f"  Brier: {brier:.4f} (lift vs baseline: {brier_lift:+.4f})")
        print(f"  Picks conf>{CONF_THRESHOLD}: {n_bets} | Vincenti: {wins if n_bets>0 else 0}")
        print(f"  ROI simulato @ {FLAT_ODDS}: {roi_pct:+.1f}% ({profit:+.2f} unità su {n_bets} bet)")

    print(f"\n{'='*60}")
    print(f"TOTALE: {total_bets} bet | Profitto: {total_profit:+.2f} unità")
    if total_bets > 0:
        print(f"ROI globale: {total_profit/total_bets*100:+.1f}%")

    # Salva log pick ad alta confidenza
    if high_conf_picks:
        df_picks = pd.DataFrame(high_conf_picks)
        picks_path = os.path.join(BASE_DIR, "backtest_picks.csv")
        df_picks.to_csv(picks_path, index=False)
        win_rate = df_picks["correct"].mean()
        print(f"\nPick salvate: {picks_path}")
        print(f"Win rate pick alta confidenza: {win_rate:.1%} su {len(df_picks)} pick")

    print("=" * 60)
    return results


if __name__ == "__main__":
    run_backtest()
