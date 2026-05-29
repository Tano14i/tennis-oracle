"""
tennis_trainer.py
Training autonomo dei modelli ML per Tennis Oracle.

    python tennis_trainer.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tennis_dataset import (
    download_dataset, enrich_dataframe,
    build_player_stats_lookup, build_surface_stats, build_name_lookup,
)
from tennis_models import train_all_models, MODELS_DIR, RELIABLE_MARKETS


def run_training():
    print("=" * 50)
    print("TENNIS ORACLE — TRAINING")
    print("=" * 50)

    print("\n[1/4] Download dataset Sackmann (ATP + WTA)...")
    t0 = time.time()
    df = download_dataset()
    print(f"      Partite scaricate: {len(df)} ({time.time()-t0:.1f}s)")

    if len(df) < 1000:
        print("ERRORE: dataset insufficiente.")
        return

    print("\n[2/4] Parsing score e calcolo target...")
    df = enrich_dataframe(df)
    df = df[df["n_sets"] > 0].copy()
    print(f"      Partite valide: {len(df)}")

    bo3 = df[df["best_of"] == 3]
    print(f"\n      Baseline mercati affidabili (BO3):")
    print(f"        Both win set:    {bo3['target_both_set'].mean()*100:.1f}%")
    print(f"        Games over 22.5: {bo3['target_games_over_225'].mean()*100:.1f}%")
    print(f"        Aces over 10.5:  {df['target_aces_over_105'].mean()*100:.1f}%")

    print("\n[3/4] Calcolo statistiche rolling giocatori...")
    t0 = time.time()
    player_stats = build_player_stats_lookup(df)
    surface_stats = build_surface_stats(df)
    name_lookup = build_name_lookup(df)
    print(f"      Giocatori nel lookup: {len(player_stats)} ({time.time()-t0:.1f}s)")

    print("\n[4/4] Training modelli...")
    results = train_all_models(df, player_stats, surface_stats)

    # Salva lookup su disco
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(player_stats,  os.path.join(MODELS_DIR, "player_stats.pkl"))
    joblib.dump(surface_stats, os.path.join(MODELS_DIR, "surface_stats.pkl"))
    joblib.dump(name_lookup,   os.path.join(MODELS_DIR, "name_lookup.pkl"))
    joblib.dump(df[["winner_id", "loser_id", "tourney_date", "surface",
                    "tourney_level", "best_of", "round", "score",
                    "n_sets", "total_games", "tiebreaks",
                    "winner_rank", "loser_rank"]].copy(),
                os.path.join(MODELS_DIR, "df_history.pkl"))
    print("\n      Tutti i file salvati in models/")

    print("\n" + "=" * 50)
    print("RISULTATI TRAINING")
    print("=" * 50)
    for market, res in results.items():
        acc = res["accuracy"]
        pos = res["positive_rate"]
        baseline = 0.643 if market == "winner" else max(pos, 1 - pos)
        lift = acc - baseline
        status = "OK" if market in RELIABLE_MARKETS and lift > 0 else "!!"
        label = "(affidabile)" if market in RELIABLE_MARKETS else "(nascosto nel report)"
        print(f"  {status} {market:<15} Acc: {acc:.3f} | Baseline: {baseline:.3f} | Lift: {lift:+.3f} {label}")

    print("\nTraining completato.")
    print("Avvia tennis_bot.py per il bot Telegram.")
    print("\nMercati mostrati nel report: winner, both_set, games_over, aces_over")
    print("Mercati nascosti (sotto baseline): sets_over, tiebreak")


if __name__ == "__main__":
    run_training()
