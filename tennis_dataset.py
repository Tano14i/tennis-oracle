"""
tennis_dataset.py
Download e preparazione dataset Jeff Sackmann (ATP + WTA).
Scarica automaticamente gli ultimi N anni e costruisce le statistiche rolling per giocatore.
"""

import os
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

ATP_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"

YEARS = list(range(2018, datetime.now().year + 1))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(force=False):
    """Scarica tutti gli anni ATP e WTA e li salva in data/."""
    dfs = []
    for year in YEARS:
        for tour, url_template in [("ATP", ATP_URL), ("WTA", WTA_URL)]:
            fname = os.path.join(DATA_DIR, f"{tour.lower()}_{year}.csv")
            if not os.path.exists(fname) or force:
                url = url_template.format(year=year)
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200 and len(r.content) > 1000:
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write(r.text)
                        print(f"Scaricato: {tour} {year}")
                    else:
                        continue
                except Exception as e:
                    print(f"Errore {tour} {year}: {e}")
                    continue
            try:
                df = pd.read_csv(fname, low_memory=False)
                df["tour"] = tour
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["score"].notna()]
    df = df[~df["score"].str.contains(r"W/O|RET|DEF|walkover", case=False, na=True)]
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Parser score
# ---------------------------------------------------------------------------

def parse_score(score):
    """Estrae statistiche dal punteggio testuale."""
    if not isinstance(score, str):
        return {}
    sets = score.strip().split()
    n_sets = 0
    total_games = 0
    winner_games = 0
    loser_games = 0
    tiebreaks = 0

    for s in sets:
        s_clean = s.split("(")[0]
        if "-" not in s_clean:
            continue
        parts = s_clean.split("-")
        try:
            w = int(parts[0])
            l = int(parts[1])
            n_sets += 1
            total_games += w + l
            winner_games += w
            loser_games += l
            if "(" in s:
                tiebreaks += 1
        except Exception:
            pass

    return {
        "n_sets": n_sets,
        "total_games": total_games,
        "winner_games": winner_games,
        "loser_games": loser_games,
        "tiebreaks": tiebreaks,
        "had_tiebreak": int(tiebreaks > 0),
        "both_won_set": int(n_sets == 3) if n_sets in (2, 3) else int(n_sets >= 3),
    }


def enrich_dataframe(df):
    """Aggiunge colonne target per ogni mercato."""
    parsed = df["score"].apply(parse_score)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    df = df[df["n_sets"] > 0].copy()

    # Targets
    df["target_winner"] = 1  # vincitore e sempre il winner nel dataset
    df["target_sets_over"] = (df["n_sets"] == 3).astype(int)  # BO3: 3 set
    df["target_both_set"] = df["both_won_set"].astype(int)
    df["target_games_over_225"] = (df["total_games"] > 22.5).astype(int)

    df["total_aces"] = df["w_ace"].fillna(0) + df["l_ace"].fillna(0)
    df["target_aces_over_105"] = (df["total_aces"] > 10.5).astype(int)
    df["target_tiebreak"] = df["had_tiebreak"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Statistiche rolling per giocatore
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 20


def _build_player_rows(df):
    """
    Ritorna un DataFrame con una riga per ogni (giocatore, partita),
    indipendentemente dal fatto che abbia vinto o perso.
    Elimina il selection bias di calcolare stats solo su winner_id o loser_id.
    """
    winner_rows = df[["tourney_date", "surface", "winner_id", "loser_id",
                       "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon"]].copy()
    winner_rows = winner_rows.rename(columns={
        "winner_id": "player_id", "loser_id": "opponent_id",
        "w_ace": "aces", "w_df": "df",
        "w_svpt": "svpt", "w_1stIn": "first_in",
        "w_1stWon": "first_won", "w_2ndWon": "second_won",
    })
    winner_rows["won"] = 1

    loser_rows = df[["tourney_date", "surface", "loser_id", "winner_id",
                      "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon"]].copy()
    loser_rows = loser_rows.rename(columns={
        "loser_id": "player_id", "winner_id": "opponent_id",
        "l_ace": "aces", "l_df": "df",
        "l_svpt": "svpt", "l_1stIn": "first_in",
        "l_1stWon": "first_won", "l_2ndWon": "second_won",
    })
    loser_rows["won"] = 0

    rows = pd.concat([winner_rows, loser_rows], ignore_index=True)
    rows = rows.sort_values(["player_id", "tourney_date"]).reset_index(drop=True)
    return rows


RECENT_FORM_WINDOW = 5


def build_player_stats_lookup(df):
    """
    Costruisce lookup statistiche per ogni giocatore senza selection bias.
    Le stats servizio sono calcolate sugli ultimi ROLLING_WINDOW match
    indipendentemente dal risultato. Aggiunge forma recente (ultimi 5) e streak.
    """
    rows = _build_player_rows(df)
    lookup = {}

    for pid, group in rows.groupby("player_id"):
        last_n = group.tail(ROLLING_WINDOW)
        last_5 = group.tail(RECENT_FORM_WINDOW)

        aces      = last_n["aces"].mean()
        df_rate   = last_n["df"].mean()
        first_in  = (last_n["first_in"] / last_n["svpt"].replace(0, np.nan)).mean()
        first_won = (last_n["first_won"] / last_n["first_in"].replace(0, np.nan)).mean()
        svpt_2nd  = (last_n["svpt"] - last_n["first_in"]).replace(0, np.nan)
        second_won = (last_n["second_won"] / svpt_2nd).mean()
        win_rate  = group["won"].mean()

        # Forma recente: win rate ultimi 5 match
        recent_wr = last_5["won"].mean() if len(last_5) > 0 else win_rate

        # Streak: conta vittorie/sconfitte consecutive dall'ultimo match
        results_seq = group["won"].tolist()
        streak = 0
        if results_seq:
            last_result = results_seq[-1]
            for r in reversed(results_seq):
                if r == last_result:
                    streak += 1
                else:
                    break
            if last_result == 0:
                streak = -streak  # negativo = sconfitte consecutive

        def _f(v): return round(float(v), 3) if not np.isnan(v) else 0.0

        lookup[pid] = {
            "p_aces_avg":   _f(aces),
            "p_df_avg":     _f(df_rate),
            "p_1st_in":     _f(first_in),
            "p_1st_won":    _f(first_won),
            "p_2nd_won":    _f(second_won),
            "p_win_rate":   _f(win_rate),
            "p_recent_wr":  _f(recent_wr),
            "p_streak":     int(streak),
            "p_matches":    len(group),
        }

    return lookup


# ---------------------------------------------------------------------------
# Surface stats per giocatore
# ---------------------------------------------------------------------------

def build_surface_stats(df):
    """Win rate per superficie per ogni giocatore (senza selection bias)."""
    rows = _build_player_rows(df)
    stats = {}

    for pid, group in rows.groupby("player_id"):
        s = {}
        for surf in ["Hard", "Clay", "Grass"]:
            sub = group[group["surface"] == surf]
            total = len(sub)
            s[f"wr_{surf.lower()}"] = round(sub["won"].mean(), 3) if total > 0 else 0.5
            s[f"n_{surf.lower()}"] = total
        stats[pid] = s
    return stats


# ---------------------------------------------------------------------------
# Head to head
# ---------------------------------------------------------------------------

def get_h2h(df, p1_id, p2_id):
    """Statistiche head-to-head tra due giocatori."""
    mask = (
        ((df["winner_id"] == p1_id) & (df["loser_id"] == p2_id)) |
        ((df["winner_id"] == p2_id) & (df["loser_id"] == p1_id))
    )
    h2h = df[mask].copy()
    if h2h.empty:
        return {"h2h_total": 0, "h2h_p1_wins": 0, "h2h_p1_win_rate": 0.5}

    p1_wins = (h2h["winner_id"] == p1_id).sum()
    return {
        "h2h_total": len(h2h),
        "h2h_p1_wins": int(p1_wins),
        "h2h_p1_win_rate": round(p1_wins / len(h2h), 3),
    }


def build_h2h_lookup(df):
    """
    Precalcola H2H cumulativo per ogni coppia (min_id, max_id) fino ad ogni partita.
    Ritorna un dict (winner_id, loser_id, tourney_date) -> h2h stats visti PRIMA di quella data.
    Usato durante il training per non avere data leakage.
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Conta wins cumulativi per coppia (p1 < p2 per normalizzare)
    # Struttura: {(a, b): {"a_wins": int, "b_wins": int}}
    h2h_state = {}
    lookup = {}  # index -> stats prima di questa riga

    for idx, row in df.iterrows():
        w = int(row["winner_id"]) if not pd.isna(row["winner_id"]) else -1
        l = int(row["loser_id"])  if not pd.isna(row["loser_id"])  else -1
        key = (min(w, l), max(w, l))

        state = h2h_state.get(key, {"a_wins": 0, "b_wins": 0})
        total = state["a_wins"] + state["b_wins"]

        # p1=winner, p2=loser per convenzione del training
        p1_wins = state["a_wins"] if w == key[0] else state["b_wins"]
        lookup[idx] = {
            "h2h_total": total,
            "h2h_p1_win_rate": round(p1_wins / max(1, total), 3) if total > 0 else 0.5,
        }

        # Aggiorna stato DOPO aver salvato (no leakage)
        if w == key[0]:
            h2h_state[key] = {"a_wins": state["a_wins"] + 1, "b_wins": state["b_wins"]}
        else:
            h2h_state[key] = {"a_wins": state["a_wins"], "b_wins": state["b_wins"] + 1}

    return lookup


def get_h2h_on_surface(df, p1_id, p2_id, surface):
    """H2H su una superficie specifica."""
    mask = (
        ((df["winner_id"] == p1_id) & (df["loser_id"] == p2_id)) |
        ((df["winner_id"] == p2_id) & (df["loser_id"] == p1_id))
    ) & (df["surface"] == surface)
    h2h = df[mask]
    if h2h.empty:
        return {"h2h_surf_total": 0, "h2h_surf_p1_win_rate": 0.5}
    p1_wins = (h2h["winner_id"] == p1_id).sum()
    return {
        "h2h_surf_total": len(h2h),
        "h2h_surf_p1_win_rate": round(p1_wins / len(h2h), 3),
    }


# ---------------------------------------------------------------------------
# Lookup nome -> ID giocatore
# ---------------------------------------------------------------------------

def build_name_lookup(df):
    """Dizionario nome normalizzato -> player_id."""
    lookup = {}
    for _, row in df[["winner_id", "winner_name"]].dropna().drop_duplicates().iterrows():
        name = str(row["winner_name"]).strip().lower()
        lookup[name] = int(row["winner_id"])
    for _, row in df[["loser_id", "loser_name"]].dropna().drop_duplicates().iterrows():
        name = str(row["loser_name"]).strip().lower()
        lookup[name] = int(row["loser_id"])
    return lookup


def find_player_id(name, lookup):
    """Cerca player ID per nome parziale."""
    name = name.strip().lower()
    # Match esatto
    if name in lookup:
        return lookup[name]
    # Match parziale (cognome)
    matches = [(k, v) for k, v in lookup.items() if name in k or k in name]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        # Preferisce match più lungo
        matches.sort(key=lambda x: -len(x[0]))
        return matches[0][1]
    return None


if __name__ == "__main__":
    print("Download dataset...")
    df = download_dataset()
    print(f"Partite scaricate: {len(df)}")
    df = enrich_dataframe(df)
    print(f"Partite valide: {len(df)}")
    print("Dataset pronto.")
