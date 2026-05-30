"""
tennis_odds.py
Integrazione The Odds API per calcolo value bet.

Free tier: 500 richieste/mese — https://the-odds-api.com
API key: aggiungila come variabile d'ambiente ODDS_API_KEY
         o impostala direttamente qui (non pushare mai la key su git)
"""

import os
import requests

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Sport keys per tennis su The Odds API
TENNIS_SPORTS = [
    "tennis_atp_french_open",
    "tennis_wta_french_open",
    "tennis_atp_wimbledon",
    "tennis_wta_wimbledon",
    "tennis_atp_us_open",
    "tennis_wta_us_open",
    "tennis_atp_aus_open",
    "tennis_wta_aus_open",
    "tennis_atp",   # ATP Tour generico
    "tennis_wta",   # WTA Tour generico
]

# Bookmaker preferiti (per ordine di affidabilità linee)
PREFERRED_BOOKS = ["pinnacle", "betfair_ex_eu", "betfair", "unibet", "bet365"]


def get_tennis_odds(sport_key=None, regions="eu", markets="h2h"):
    """
    Recupera quote per partite tennis.
    Ritorna lista di match con quote per ogni bookmaker.
    """
    if not ODDS_API_KEY:
        return []

    sports = [sport_key] if sport_key else TENNIS_SPORTS
    all_odds = []

    for sport in sports:
        try:
            url = f"{ODDS_API_BASE}/sports/{sport}/odds"
            params = {
                "apiKey":  ODDS_API_KEY,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                all_odds.extend(r.json())
            elif r.status_code == 422:
                continue  # sport non disponibile in questo momento
        except Exception:
            continue

    return all_odds


def parse_match_odds(odds_data):
    """
    Trasforma raw odds API in formato semplice:
    {
      "home_team": str,
      "away_team": str,
      "commence_time": str,
      "best_home_odds": float,
      "best_away_odds": float,
      "bookmaker": str,
    }
    """
    matches = []
    for match in odds_data:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        time = match.get("commence_time", "")

        best_home = 0.0
        best_away = 0.0
        best_book = ""

        bookmakers = match.get("bookmakers", [])
        # Ordina per bookmaker preferito
        bookmakers_sorted = sorted(
            bookmakers,
            key=lambda b: PREFERRED_BOOKS.index(b["key"]) if b["key"] in PREFERRED_BOOKS else 999
        )

        for bm in bookmakers_sorted:
            for mkt in bm.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                h = outcomes.get(home, 0)
                a = outcomes.get(away, 0)
                if h > best_home and a > best_away:
                    best_home = h
                    best_away = a
                    best_book = bm["key"]

        if best_home > 1.0 and best_away > 1.0:
            matches.append({
                "home_team":      home,
                "away_team":      away,
                "commence_time":  time,
                "best_home_odds": round(best_home, 3),
                "best_away_odds": round(best_away, 3),
                "bookmaker":      best_book,
            })

    return matches


def compute_value_bet(ml_prob_p1, best_odds_p1, min_edge=0.04):
    """
    Calcola se esiste un value bet.

    ml_prob_p1: probabilità ML per p1 (0-1)
    best_odds_p1: migliore quota decimale disponibile per p1
    min_edge: edge minimo richiesto (default 4%)

    Ritorna dict con:
      - has_value: bool
      - edge: float (ML prob - implied prob)
      - expected_value: float (EV per unità scommessa)
      - kelly: float (frazione Kelly suggerita)
    """
    if best_odds_p1 <= 1.0:
        return {"has_value": False, "edge": 0, "ev": 0, "kelly": 0}

    implied_prob = 1.0 / best_odds_p1
    edge = ml_prob_p1 - implied_prob
    ev   = ml_prob_p1 * best_odds_p1 - 1  # EV per unità scommessa

    # Kelly criterion (frazione della bankroll da scommettere)
    # Kelly = (prob * odds - 1) / (odds - 1)
    kelly = max(0, (ml_prob_p1 * best_odds_p1 - 1) / (best_odds_p1 - 1))
    # Usa half-Kelly per sicurezza
    kelly_half = round(kelly / 2, 3)

    return {
        "has_value":   edge >= min_edge and ev > 0,
        "edge":        round(edge, 4),
        "ev":          round(ev, 4),
        "kelly":       kelly_half,
        "implied_prob": round(implied_prob, 4),
        "ml_prob":      round(ml_prob_p1, 4),
    }


def find_value_bets(ml_predictions, odds_matches, p1_name, p2_name):
    """
    Dato un dict di predizioni ML e le quote disponibili,
    trova i value bet per questa partita.

    ml_predictions: dict market -> {prob, bet_label, bet_prob}
    odds_matches: lista da parse_match_odds()
    """
    # Cerca la partita nelle odds per nome
    match_odds = None
    p1_lower = p1_name.lower().split()[-1]  # cognome
    p2_lower = p2_name.lower().split()[-1]

    for m in odds_matches:
        h = m["home_team"].lower()
        a = m["away_team"].lower()
        if p1_lower in h and p2_lower in a:
            match_odds = m
            break
        if p2_lower in h and p1_lower in a:
            # Swappa home/away
            match_odds = {
                **m,
                "home_team":      m["away_team"],
                "away_team":      m["home_team"],
                "best_home_odds": m["best_away_odds"],
                "best_away_odds": m["best_home_odds"],
            }
            break

    if not match_odds:
        return {}

    value_bets = {}
    winner_pred = ml_predictions.get("winner", {})
    ml_prob_p1  = winner_pred.get("prob", 0.5)

    # Value bet su P1
    vb_p1 = compute_value_bet(ml_prob_p1, match_odds["best_home_odds"])
    if vb_p1["has_value"]:
        value_bets["winner_p1"] = {
            **vb_p1,
            "bet_label": f"{p1_name} vince",
            "odds":      match_odds["best_home_odds"],
            "bookmaker": match_odds["bookmaker"],
        }

    # Value bet su P2
    vb_p2 = compute_value_bet(1 - ml_prob_p1, match_odds["best_away_odds"])
    if vb_p2["has_value"]:
        value_bets["winner_p2"] = {
            **vb_p2,
            "bet_label": f"{p2_name} vince",
            "odds":      match_odds["best_away_odds"],
            "bookmaker": match_odds["bookmaker"],
        }

    return value_bets


def remaining_requests():
    """Controlla le richieste API rimanenti nel mese."""
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
