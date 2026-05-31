"""
tennis_odds.py
Integrazione The Odds API — value bet su mercati tennis.
https://the-odds-api.com
"""

import os
import requests
import re

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

TENNIS_SPORTS = [
    "tennis_atp_french_open",
    "tennis_wta_french_open",
    "tennis_atp_wimbledon",
    "tennis_wta_wimbledon",
    "tennis_atp_us_open",
    "tennis_wta_us_open",
    "tennis_atp_aus_open",
    "tennis_wta_aus_open",
    "tennis_atp",
    "tennis_wta",
]

PREFERRED_BOOKS = ["pinnacle", "betfair_ex_eu", "betfair_ex_uk", "unibet_eu",
                   "bet365", "williamhill", "1xbet", "bwin"]

MIN_EDGE = 0.04  # edge minimo per segnalare value bet (4%)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def get_tennis_odds(sport_key=None, regions="eu", markets="h2h,totals"):
    if not ODDS_API_KEY:
        return []
    sports = [sport_key] if sport_key else TENNIS_SPORTS
    all_odds = []
    for sport in sports:
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/sports/{sport}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": regions,
                        "markets": markets, "oddsFormat": "decimal"},
                timeout=10
            )
            if r.status_code == 200:
                all_odds.extend(r.json())
        except Exception:
            continue
    return all_odds


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _normalize_name(name):
    """Lowercase, rimuove punteggiatura, prende cognome."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    parts = name.split()
    return parts[-1] if parts else name


def _name_match(player_name, team_name):
    """True se il cognome del giocatore è nel nome del team (o viceversa)."""
    pn = _normalize_name(player_name)
    tn = _normalize_name(team_name)
    return pn in tn or tn in pn or pn[:4] == tn[:4]


def parse_match_odds(odds_data):
    """
    Estrae le migliori quote h2h e totals per ogni match.
    Ritorna lista di dict con odds dettagliate.
    """
    matches = []
    for match in odds_data:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        time = match.get("commence_time", "")
        sport = match.get("sport_key", "")

        best = {
            "home_h2h": 0.0, "away_h2h": 0.0, "book_h2h": "",
            "over_total": 0.0, "under_total": 0.0,
            "total_point": None, "book_totals": "",
        }

        books_sorted = sorted(
            match.get("bookmakers", []),
            key=lambda b: PREFERRED_BOOKS.index(b["key"]) if b["key"] in PREFERRED_BOOKS else 999
        )

        for bm in books_sorted:
            for mkt in bm.get("markets", []):
                outcomes = mkt.get("outcomes", [])

                if mkt["key"] == "h2h":
                    od = {o["name"]: o["price"] for o in outcomes}
                    h = od.get(home, 0)
                    a = od.get(away, 0)
                    if h > 1.0 and a > 1.0 and (best["book_h2h"] == "" or
                       bm["key"] in PREFERRED_BOOKS[:3]):
                        best["home_h2h"]  = round(h, 3)
                        best["away_h2h"]  = round(a, 3)
                        best["book_h2h"]  = bm["key"]

                elif mkt["key"] == "totals" and best["book_totals"] == "":
                    for o in outcomes:
                        pt = o.get("point")
                        if o["name"] == "Over" and o["price"] > 1.0:
                            best["over_total"]  = round(o["price"], 3)
                            best["total_point"] = pt
                        elif o["name"] == "Under" and o["price"] > 1.0:
                            best["under_total"] = round(o["price"], 3)
                    if best["over_total"] > 0:
                        best["book_totals"] = bm["key"]

        if best["home_h2h"] > 1.0:
            matches.append({
                "home_team":     home,
                "away_team":     away,
                "commence_time": time,
                "sport_key":     sport,
                **best,
            })

    return matches


# ---------------------------------------------------------------------------
# Value bet calculation
# ---------------------------------------------------------------------------

def compute_value_bet(ml_prob, odds, min_edge=MIN_EDGE):
    """Calcola edge e Kelly fraction per una singola quota."""
    if odds <= 1.0 or ml_prob <= 0:
        return {"has_value": False, "edge": 0.0, "ev": 0.0, "kelly": 0.0}

    implied  = 1.0 / odds
    edge     = ml_prob - implied
    ev       = ml_prob * odds - 1.0
    kelly    = max(0, (ml_prob * odds - 1) / (odds - 1))

    return {
        "has_value":    edge >= min_edge and ev > 0,
        "edge":         round(edge, 4),
        "ev":           round(ev, 4),
        "kelly":        round(kelly / 2, 4),   # half-Kelly
        "implied_prob": round(implied, 4),
        "ml_prob":      round(ml_prob, 4),
    }


def find_value_bets(ml_predictions, odds_matches, p1_name, p2_name):
    """
    Trova value bet per una partita confrontando prob ML con quote reali.
    Copre: winner (h2h) + totals (games over/under).
    """
    # Trova la partita nelle odds
    match_odds = None
    for m in odds_matches:
        h_match = _name_match(p1_name, m["home_team"])
        a_match = _name_match(p2_name, m["away_team"])
        if h_match and a_match:
            match_odds = m
            break
        # Prova scambiato
        h_match2 = _name_match(p2_name, m["home_team"])
        a_match2 = _name_match(p1_name, m["away_team"])
        if h_match2 and a_match2:
            match_odds = {
                **m,
                "home_team":    m["away_team"],
                "away_team":    m["home_team"],
                "home_h2h":     m["away_h2h"],
                "away_h2h":     m["home_h2h"],
            }
            break

    if not match_odds:
        return {}

    value_bets = {}
    winner_pred  = ml_predictions.get("winner", {})
    ml_prob_p1   = winner_pred.get("prob", 0.5)
    games_pred   = ml_predictions.get("games_over", {})
    ml_prob_over = games_pred.get("prob", 0.5)

    # --- Winner H2H ---
    for direction, ml_prob, odds_val, label in [
        ("p1", ml_prob_p1,       match_odds["home_h2h"], f"{p1_name} vince"),
        ("p2", 1 - ml_prob_p1,   match_odds["away_h2h"], f"{p2_name} vince"),
    ]:
        vb = compute_value_bet(ml_prob, odds_val)
        if vb["has_value"]:
            value_bets[f"winner_{direction}"] = {
                **vb,
                "market":    "Winner",
                "bet_label": label,
                "odds":      odds_val,
                "bookmaker": match_odds["book_h2h"],
            }

    # --- Totals (games over/under) ---
    if match_odds.get("total_point") and match_odds.get("over_total", 0) > 1.0:
        point = match_odds["total_point"]
        for direction, ml_prob, odds_val, label in [
            ("over",  ml_prob_over,       match_odds["over_total"],  f"Over {point}"),
            ("under", 1 - ml_prob_over,   match_odds["under_total"], f"Under {point}"),
        ]:
            vb = compute_value_bet(ml_prob, odds_val)
            if vb["has_value"]:
                value_bets[f"totals_{direction}"] = {
                    **vb,
                    "market":    f"Games Over/Under {point}",
                    "bet_label": label,
                    "odds":      odds_val,
                    "bookmaker": match_odds["book_totals"],
                    "point":     point,
                }

    # Includi sempre le quote trovate (anche senza value) per trasparenza
    if not value_bets and match_odds["home_h2h"] > 1.0:
        value_bets["_odds_only"] = {
            "has_value":  False,
            "home_team":  match_odds["home_team"],
            "away_team":  match_odds["away_team"],
            "home_odds":  match_odds["home_h2h"],
            "away_odds":  match_odds["away_h2h"],
            "bookmaker":  match_odds["book_h2h"],
            "ml_prob_p1": round(ml_prob_p1, 3),
            "message":    "Quote trovate ma nessun edge ≥4%",
        }

    return value_bets


def remaining_requests():
    if not ODDS_API_KEY:
        return None
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports",
                         params={"apiKey": ODDS_API_KEY}, timeout=5)
        return {
            "remaining": r.headers.get("x-requests-remaining"),
            "used":      r.headers.get("x-requests-used"),
        }
    except Exception:
        return None
