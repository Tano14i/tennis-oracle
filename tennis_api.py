"""
tennis_api.py
Flask REST API per Tennis Oracle — chiamata dalla web app GitHub Pages.
"""

import os
import sys
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tennis_dataset import find_player_id, get_h2h, get_h2h_on_surface
from tennis_models import predict_match, MODELS_DIR, RELIABLE_MARKETS

app = Flask(__name__)
CORS(app)  # permette chiamate da GitHub Pages

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
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_ready": MODELS_READY, "players": len(player_stats) if MODELS_READY else 0})


@app.route("/analyze", methods=["POST"])
def analyze():
    if not MODELS_READY:
        return jsonify({"error": "Modelli non caricati"}), 503

    data = request.get_json(force=True)
    p1_name = data.get("p1", "").strip()
    p2_name = data.get("p2", "").strip()
    surface = data.get("surface", "Hard")
    tour    = data.get("tour", "ATP")
    round_str = data.get("round", "R32")
    best_of = int(data.get("best_of", 3))

    if not p1_name or not p2_name:
        return jsonify({"error": "Nomi giocatori mancanti"}), 400

    # best_of: WTA sempre 3, ATP slam 5
    if tour == "WTA":
        best_of = 3
    elif round_str in ("F", "SF") and tour == "ATP":
        best_of = best_of  # lascia come passato

    # Risolvi ID giocatori
    p1_id = find_player_id(p1_name, name_lookup)
    p2_id = find_player_id(p2_name, name_lookup)

    # Se non trovati, usa ID fittizi (predizioni degraded ma funzionanti)
    p1_id = p1_id if p1_id else -1
    p2_id = p2_id if p2_id else -2

    p1_found = p1_id != -1
    p2_found = p2_id != -2

    try:
        predictions, h2h, h2h_surf = predict_match(
            p1_id, p2_id, surface, "A", round_str, best_of,
            player_stats, surface_stats, df_history
        )
    except Exception as e:
        return jsonify({"error": f"Errore predizione: {str(e)}"}), 500

    # Statistiche giocatori per il frontend
    p1_stats = player_stats.get(p1_id, {})
    p2_stats = player_stats.get(p2_id, {})
    p1_surf  = surface_stats.get(p1_id, {})
    p2_surf  = surface_stats.get(p2_id, {})
    surf_key = f"wr_{surface.lower()}"

    def pct(v): return round(float(v) * 100, 1)

    response = {
        "p1_name": p1_name,
        "p2_name": p2_name,
        "surface": surface,
        "tour": tour,
        "round": round_str,
        "best_of": best_of,
        "p1_found": p1_found,
        "p2_found": p2_found,
        "p1_stats": {
            "win_rate":    pct(p1_stats.get("p_win_rate", 0.5)),
            "recent_wr":   pct(p1_stats.get("p_recent_wr", p1_stats.get("p_win_rate", 0.5))),
            "streak":      p1_stats.get("p_streak", 0),
            "surf_wr":     pct(p1_surf.get(surf_key, 0.5)),
            "aces_avg":    round(p1_stats.get("p_aces_avg", 0), 1),
            "matches":     p1_stats.get("p_matches", 0),
        },
        "p2_stats": {
            "win_rate":    pct(p2_stats.get("p_win_rate", 0.5)),
            "recent_wr":   pct(p2_stats.get("p_recent_wr", p2_stats.get("p_win_rate", 0.5))),
            "streak":      p2_stats.get("p_streak", 0),
            "surf_wr":     pct(p2_surf.get(surf_key, 0.5)),
            "aces_avg":    round(p2_stats.get("p_aces_avg", 0), 1),
            "matches":     p2_stats.get("p_matches", 0),
        },
        "h2h": {
            "total":       h2h.get("h2h_total", 0),
            "p1_wins":     h2h.get("h2h_p1_wins", 0),
            "p2_wins":     h2h.get("h2h_total", 0) - h2h.get("h2h_p1_wins", 0),
        },
        "markets": {},
    }

    market_labels = {
        "winner":     {"title": "Winner",           "yes": f"{p1_name} vince",              "no": f"{p2_name} vince"},
        "both_set":   {"title": "Entrambi vincono un set", "yes": "Sì",                     "no": "No — vittoria netta"},
        "games_over": {"title": "Games Over/Under 22.5",   "yes": "Over 22.5",              "no": "Under 22.5"},
        "aces_over":  {"title": "Aces Over/Under 10.5",    "yes": "Over 10.5",              "no": "Under 10.5"},
        "sets_over":  {"title": "Set Over/Under 2.5",      "yes": "Over 2.5 set",           "no": "Under 2.5 set"},
        "tiebreak":   {"title": "Tiebreak (almeno 1)",     "yes": "Sì — almeno 1 tiebreak", "no": "No tiebreak"},
    }

    for market, pred in predictions.items():
        prob     = pred.get("prob", 0.5)
        prob_pct = pred.get("prob_pct", 50.0)
        conf     = pred.get("confidence", "bassa")
        reliable = market in RELIABLE_MARKETS
        labels   = market_labels.get(market, {"title": market, "yes": "Sì", "no": "No"})

        if prob >= 0.5:
            bet_label = labels["yes"]
            bet_prob  = prob_pct
        else:
            bet_label = labels["no"]
            bet_prob  = round((1 - prob) * 100, 1)

        response["markets"][market] = {
            "title":     labels["title"],
            "prob":      round(prob, 3),
            "prob_pct":  prob_pct,
            "bet_label": bet_label,
            "bet_prob":  bet_prob,
            "confidence": conf,
            "reliable":  reliable,
            "show":      reliable,
        }

    return jsonify(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
