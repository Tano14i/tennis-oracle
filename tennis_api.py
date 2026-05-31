"""
tennis_api.py
Flask REST API per Tennis Oracle — chiamata dalla web app GitHub Pages.
"""

import os
import sys
import traceback
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tennis_dataset import find_player_id, get_h2h, get_h2h_on_surface
from tennis_models import predict_match, MODELS_DIR, RELIABLE_MARKETS
from tennis_analysis import (explain_winner, explain_sets, explain_games,
                              explain_aces, explain_tiebreak)
from tennis_odds import get_tennis_odds, parse_match_odds, find_value_bets, remaining_requests

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Caricamento modelli all'avvio (una volta sola)
# ---------------------------------------------------------------------------

print("Caricamento modelli ML...")
try:
    player_stats  = joblib.load(os.path.join(MODELS_DIR, "player_stats.pkl"))
    surface_stats = joblib.load(os.path.join(MODELS_DIR, "surface_stats.pkl"))
    name_lookup   = joblib.load(os.path.join(MODELS_DIR, "name_lookup.pkl"))
    df_history    = joblib.load(os.path.join(MODELS_DIR, "df_history.pkl"))
    print(f"Modelli caricati. Giocatori: {len(player_stats)}")
    MODELS_READY = True
except Exception as e:
    print(f"ERRORE caricamento modelli: {e}")
    MODELS_READY = False

# ---------------------------------------------------------------------------
# Generatore narrativa con Claude API + web search
# ---------------------------------------------------------------------------

def generate_narrative(p1_name, p2_name, tournament, surface, tour,
                       round_str, best_of, predictions, p1_stats_resp,
                       p2_stats_resp, h2h_resp, data_quality):
    """Genera narrativa giornalistica con Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        import requests as req

        winner_pred = predictions.get("winner", {})
        prob        = winner_pred.get("prob", 0.5)
        favorite    = p1_name if prob >= 0.5 else p2_name
        win_pct     = round(max(prob, 1 - prob) * 100)

        markets_objs = [p for p in predictions.values() if p.get("show")]

        h2h_total = h2h_resp.get("total", 0)
        h2h_str   = (f"{p1_name} {h2h_resp['p1_wins']}-{h2h_resp['p2_wins']} {p2_name}"
                     if h2h_total > 0 else "nessun precedente")

        surf_names = {"Hard": "cemento", "Clay": "terra battuta", "Grass": "erba"}
        surf_it    = surf_names.get(surface, surface)

        import datetime
        now = datetime.datetime.now()
        data_oggi = now.strftime("%d %B %Y")

        prompt = f"""Sei un analista sportivo tennis. Scrivi un'analisi in italiano stile Sisal/Snai.

REGOLE CRITICHE — RISPETTA SEMPRE:
1. Data di oggi: {data_oggi}. Usa SEMPRE questa data e questo anno ({now.year}). Mai anni passati.
2. Il dataset ML contiene SOLO match ATP/WTA ufficiali (non Challenger/ITF). Giocatori forti nei Challenger potrebbero avere statistiche ATP più basse del loro reale livello su una superficie.
3. I dati ML hanno precedenza su ogni tua conoscenza pregressa. Non contraddirli.
4. Per ogni affermazione usa la fonte: [ML] per dati ML, [CG] per conoscenza generale, [ND] per non documentato.
5. Se un dato ti sembra inaffidabile (es. win rate molto basso per un giocatore conosciuto), segnalalo con [dato ML limitato].

DATI ML (solo match ATP ufficiali):
Partita: {p1_name} vs {p2_name} — {tournament or tour} | {surf_it} | {round_str} | Best of {best_of}
Favorito: {favorite} {win_pct}% [ML]
H2H nel dataset ATP: {h2h_str} [ML]
{p1_name}: win rate ATP {p1_stats_resp['win_rate']:.0f}%, ultimi 5 {p1_stats_resp['recent_wr']:.0f}%, streak {p1_stats_resp['streak']:+d}, {surf_it} {p1_stats_resp['surf_wr']:.0f}% [ML]
{p2_name}: win rate ATP {p2_stats_resp['win_rate']:.0f}%, ultimi 5 {p2_stats_resp['recent_wr']:.0f}%, streak {p2_stats_resp['streak']:+d}, {surf_it} {p2_stats_resp['surf_wr']:.0f}% [ML]
Mercati: {" | ".join(f"{m['bet_label']} {m['bet_prob']:.0f}% conf.{m['confidence']}" for m in markets_objs)}

Formato ESATTO (mantieni emoji e struttura):

🎾 {p1_name} vs {p2_name} — [Torneo {now.year} Round]
🏆 Favorito: [nome] — [1 frase, fonte inline]
📊 Analisi:
[2-3 righe su forma, tattica, superficie con fonti inline tra parentesi quadre]
💰 Scommessa principale: [bet dal mercato ML più forte]
Perché: [2 righe con fonti]
Confidenza: [ALTA/MEDIA/BASSA]
💡 Scommessa alternativa: [bet diverso]
Perché: [1 riga con fonte]
⚠️ Attenzione: [1 rischio concreto con fonte]
📋 Fonti: ML (metriche usate) | CG: [info giocatori dalla tua conoscenza, o "nessuna verificata"] | ND: [affermazioni non documentate, o "nessuna"]

Massimo 350 parole. Tono vivace."""

        # Chiama Gemini REST API direttamente (evita problemi di versione SDK)
        model = "gemini-2.5-flash"
        url   = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                 f"{model}:generateContent?key={api_key}")
        body  = {"contents": [{"parts": [{"text": prompt}]}]}
        r     = req.post(url, json=body, timeout=30)
        r.raise_for_status()
        narrative = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return narrative if narrative else None

    except Exception as e:
        print(f"Narrative generation failed: {e}\n{traceback.format_exc()}")
        return f"[ERRORE NARRATIVA: {str(e)}]"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": MODELS_READY,
        "players": len(player_stats) if MODELS_READY else 0,
        "narrative_enabled": bool(os.environ.get("GEMINI_API_KEY"))
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        return _analyze_inner()
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def _analyze_inner():
    if not MODELS_READY:
        return jsonify({"error": "Modelli non caricati"}), 503

    data       = request.get_json(force=True)
    p1_name    = data.get("p1", "").strip()
    p2_name    = data.get("p2", "").strip()
    surface    = data.get("surface", "Hard")
    tour       = data.get("tour", "ATP")
    round_str  = data.get("round", "R32")
    tournament = data.get("tournament", "").lower()
    best_of    = int(data.get("best_of", 3))

    if not p1_name or not p2_name:
        return jsonify({"error": "Nomi giocatori mancanti"}), 400

    # Rileva Grand Slam
    GRAND_SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")
    is_slam = any(gs in tournament for gs in GRAND_SLAMS)
    if tour == "WTA":
        best_of = 3
    elif tour == "ATP" and is_slam:
        best_of = 5
    else:
        best_of = 3

    # Risolvi ID giocatori
    p1_id = find_player_id(p1_name, name_lookup) or -1
    p2_id = find_player_id(p2_name, name_lookup) or -2

    p1_found = p1_id != -1
    p2_found = p2_id != -2

    # Protezioni qualità dati
    p1_matches = player_stats.get(p1_id, {}).get("p_matches", 0)
    p2_matches = player_stats.get(p2_id, {}).get("p_matches", 0)
    MIN_MATCHES = 10
    data_quality = "alta"
    quality_warnings = []
    if not p1_found:
        quality_warnings.append(f"{p1_name} non trovato nel dataset storico")
        data_quality = "bassa"
    elif p1_matches < MIN_MATCHES:
        quality_warnings.append(f"{p1_name} ha solo {p1_matches} match nel dataset")
        data_quality = "media" if data_quality == "alta" else data_quality
    if not p2_found:
        quality_warnings.append(f"{p2_name} non trovato nel dataset storico")
        data_quality = "bassa"
    elif p2_matches < MIN_MATCHES:
        quality_warnings.append(f"{p2_name} ha solo {p2_matches} match nel dataset")
        data_quality = "media" if data_quality == "alta" else data_quality

    surf_key_check = f"n_{surface.lower()}"
    if p1_found and surface_stats.get(p1_id, {}).get(surf_key_check, 0) < 5:
        quality_warnings.append(f"{p1_name} ha pochi match su {surface}")
    if p2_found and surface_stats.get(p2_id, {}).get(surf_key_check, 0) < 5:
        quality_warnings.append(f"{p2_name} ha pochi match su {surface}")

    try:
        predictions, h2h, h2h_surf = predict_match(
            p1_id, p2_id, surface, "A", round_str, best_of,
            player_stats, surface_stats, df_history
        )
    except Exception as e:
        return jsonify({"error": f"Errore predizione: {str(e)}"}), 500

    # Statistiche giocatori
    p1_stats = player_stats.get(p1_id, {})
    p2_stats = player_stats.get(p2_id, {})
    p1_surf  = surface_stats.get(p1_id, {})
    p2_surf  = surface_stats.get(p2_id, {})
    surf_key = f"wr_{surface.lower()}"

    def pct(v): return round(float(v) * 100, 1)

    p1_stats_resp = {
        "win_rate":  pct(p1_stats.get("p_win_rate", 0.5)),
        "recent_wr": pct(p1_stats.get("p_recent_wr", p1_stats.get("p_win_rate", 0.5))),
        "streak":    p1_stats.get("p_streak", 0),
        "surf_wr":   pct(p1_surf.get(surf_key, 0.5)),
        "aces_avg":  round(p1_stats.get("p_aces_avg", 0), 1),
        "matches":   p1_stats.get("p_matches", 0),
    }
    p2_stats_resp = {
        "win_rate":  pct(p2_stats.get("p_win_rate", 0.5)),
        "recent_wr": pct(p2_stats.get("p_recent_wr", p2_stats.get("p_win_rate", 0.5))),
        "streak":    p2_stats.get("p_streak", 0),
        "surf_wr":   pct(p2_surf.get(surf_key, 0.5)),
        "aces_avg":  round(p2_stats.get("p_aces_avg", 0), 1),
        "matches":   p2_stats.get("p_matches", 0),
    }
    h2h_resp = {
        "total":   h2h.get("h2h_total", 0),
        "p1_wins": h2h.get("h2h_p1_wins", 0),
        "p2_wins": h2h.get("h2h_total", 0) - h2h.get("h2h_p1_wins", 0),
    }

    # Per BO5 (Grand Slam) usa soglia 38.5 invece di 22.5
    games_threshold = "38.5" if best_of == 5 else "22.5"
    aces_threshold  = "17.5" if best_of == 5 else "10.5"

    market_labels = {
        "winner":     {"title": "Winner",                                   "yes": f"{p1_name} vince",      "no": f"{p2_name} vince"},
        "both_set":   {"title": "Entrambi vincono un set",                  "yes": "Sì",                    "no": "No — vittoria netta"},
        "games_over": {"title": f"Games Over/Under {games_threshold}",      "yes": f"Over {games_threshold}","no": f"Under {games_threshold}"},
        "aces_over":  {"title": f"Aces Over/Under {aces_threshold}",        "yes": f"Over {aces_threshold}", "no": f"Under {aces_threshold}"},
        "sets_over":  {"title": "Set Over/Under 2.5",                       "yes": "Over 2.5 set",          "no": "Under 2.5 set"},
        "tiebreak":   {"title": "Tiebreak (almeno 1)",                      "yes": "Sì — almeno 1 tiebreak","no": "No tiebreak"},
    }

    h2h_surf = get_h2h_on_surface(df_history, p1_id, p2_id, surface)
    markets  = {}

    for market, pred in predictions.items():
        prob     = pred.get("prob", 0.5)
        prob_pct = pred.get("prob_pct", 50.0)
        conf     = pred.get("confidence", "bassa")
        # In BO5 (Grand Slam): mostra sets_over invece di games_over (soglie diverse)
        # games_over soglia 22.5 non ha senso in BO5 — nascosto nel frontend
        reliable = market in RELIABLE_MARKETS
        if best_of == 5 and market == "sets_over":
            reliable = True   # "Over 3.5 set" è valido e utile in Grand Slam
        labels   = market_labels.get(market, {"title": market, "yes": "Sì", "no": "No"})

        if prob >= 0.5:
            bet_label, bet_prob = labels["yes"], prob_pct
        else:
            bet_label, bet_prob = labels["no"], round((1 - prob) * 100, 1)

        # Degrada confidenza se dati scarsi
        if data_quality == "bassa" and conf != "alta":
            conf = "bassa"
        elif data_quality == "media" and conf == "alta" and abs(prob - 0.5) < 0.20:
            conf = "media"

        # Motivazioni brevi (bullet)
        if market == "winner":
            reasons = explain_winner(p1_name, p2_name, p1_stats, p2_stats,
                                     p1_surf, p2_surf, surface, h2h, h2h_surf, prob)
        elif market in ("sets_over", "both_set"):
            reasons = explain_sets(p1_name, p2_name, p1_stats, p2_stats, surface, best_of, prob)
        elif market == "games_over":
            reasons = explain_games(p1_name, p2_name, p1_stats, p2_stats, surface, prob)
        elif market == "aces_over":
            reasons = explain_aces(p1_name, p2_name, p1_stats, p2_stats, surface, prob, best_of)
        elif market == "tiebreak":
            reasons = explain_tiebreak(p1_name, p2_name, p1_stats, p2_stats, surface, prob)
        else:
            reasons = []

        alt_label = labels["no"] if prob >= 0.5 else labels["yes"]
        alt_prob  = round((1 - prob) * 100, 1) if prob >= 0.5 else prob_pct

        markets[market] = {
            "title":      labels["title"],
            "prob":       round(prob, 3),
            "prob_pct":   prob_pct,
            "bet_label":  bet_label,
            "bet_prob":   bet_prob,
            "alt_label":  alt_label,
            "alt_prob":   alt_prob,
            "confidence": conf,
            "reliable":   reliable,
            "show":       reliable,
            "reasons":    reasons[:3],
        }

    return jsonify({
        "p1_name":          p1_name,
        "p2_name":          p2_name,
        "surface":          surface,
        "tour":             tour,
        "round":            round_str,
        "best_of":          best_of,
        "is_slam":          is_slam,
        "p1_found":         p1_found,
        "p2_found":         p2_found,
        "p1_stats":         p1_stats_resp,
        "p2_stats":         p2_stats_resp,
        "h2h":              h2h_resp,
        "markets":          markets,
        "data_quality":     data_quality,
        "quality_warnings": quality_warnings,
        "tournament":       tournament,
        "dataset_note":     "Statistiche su match ATP/WTA + Challenger ufficiali.",
        "value_bets":       _get_value_bets(p1_name, p2_name, markets),
    })


# ---------------------------------------------------------------------------
# Cache quote (aggiornate ogni 10 minuti per risparmiare richieste API)
# ---------------------------------------------------------------------------

import time as _time
_odds_cache = {"data": [], "ts": 0}
_ODDS_TTL   = 600  # secondi

def _get_cached_odds():
    now = _time.time()
    if now - _odds_cache["ts"] > _ODDS_TTL:
        try:
            raw  = get_tennis_odds()
            _odds_cache["data"] = parse_match_odds(raw)
            _odds_cache["ts"]   = now
        except Exception as e:
            print(f"Odds fetch error: {e}")
    return _odds_cache["data"]

def _get_value_bets(p1_name, p2_name, markets):
    try:
        odds = _get_cached_odds()
        if not odds:
            return {}
        return find_value_bets(markets, odds, p1_name, p2_name)
    except Exception:
        return {}


@app.route("/odds_status")
def odds_status():
    info = remaining_requests()
    return jsonify({
        "odds_api_enabled": bool(os.environ.get("ODDS_API_KEY")),
        "cached_matches":   len(_odds_cache["data"]),
        "api_requests":     info,
    })


# ---------------------------------------------------------------------------
# Endpoint narrativa separato (chiamato dal frontend dopo i risultati ML)
# ---------------------------------------------------------------------------

@app.route("/narrative", methods=["POST"])
def narrative_endpoint():
    try:
        data       = request.get_json(force=True)
        p1_name    = data.get("p1_name", "")
        p2_name    = data.get("p2_name", "")
        tournament = data.get("tournament", "")
        surface    = data.get("surface", "Hard")
        tour       = data.get("tour", "ATP")
        round_str  = data.get("round", "R32")
        best_of    = int(data.get("best_of", 3))
        predictions = data.get("markets", {})
        p1_stats_resp = data.get("p1_stats", {})
        p2_stats_resp = data.get("p2_stats", {})
        h2h_resp      = data.get("h2h", {})
        data_quality  = data.get("data_quality", "media")

        narrative = generate_narrative(
            p1_name, p2_name, tournament, surface, tour, round_str, best_of,
            predictions, p1_stats_resp, p2_stats_resp, h2h_resp, data_quality
        )
        return jsonify({"narrative": narrative})
    except Exception as e:
        return jsonify({"narrative": None, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
