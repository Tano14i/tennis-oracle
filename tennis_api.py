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
    """Genera narrativa giornalistica con Claude API + web search."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        winner_pred = predictions.get("winner", {})
        prob        = winner_pred.get("prob", 0.5)
        favorite    = p1_name if prob >= 0.5 else p2_name
        underdog    = p2_name if prob >= 0.5 else p1_name
        win_pct     = round(max(prob, 1 - prob) * 100)

        # Mercati rilevanti
        markets_lines = []
        markets_lines_objs = []
        for mk, pred in predictions.items():
            if pred.get("show"):
                markets_lines.append(
                    f"  - {pred['title']}: {pred['bet_label']} ({pred['bet_prob']:.0f}%)"
                    f" — confidenza {pred['confidence']}"
                )
                markets_lines_objs.append(pred)

        # H2H
        h2h_total = h2h_resp.get("total", 0)
        h2h_str   = (f"{p1_name} {h2h_resp['p1_wins']}-{h2h_resp['p2_wins']} {p2_name}"
                     if h2h_total > 0 else "nessun precedente")

        surf_names = {"Hard": "cemento", "Clay": "terra battuta", "Grass": "erba"}
        surf_it    = surf_names.get(surface, surface)

        prompt = f"""Sei un analista sportivo tennis. Genera un'analisi in italiano basata sui dati ML forniti.

PARTITA: {p1_name} vs {p2_name} | {tournament or tour} | {surf_it} | {round_str} | Bo{best_of}
FAVORITO ML: {favorite} {win_pct}% | H2H: {h2h_str}
{p1_name}: wr {p1_stats_resp['win_rate']:.0f}% recente {p1_stats_resp['recent_wr']:.0f}% streak {p1_stats_resp['streak']:+d}
{p2_name}: wr {p2_stats_resp['win_rate']:.0f}% recente {p2_stats_resp['recent_wr']:.0f}% streak {p2_stats_resp['streak']:+d}
MERCATI: {" | ".join(f"{m['bet_label']} {m['bet_prob']:.0f}% ({m['confidence']})" for m in markets_lines_objs)}

Formato ESATTO (mantieni emoji e struttura):
🎾 {p1_name} vs {p2_name} — [torneo round]
🏆 Favorito: [nome] — [frase vivace 1 riga sul perché]
📊 Analisi: [2-3 righe su forma, superfice, tattica, H2H]
💰 Scommessa principale: [bet]
Perché: [2 righe ragionamento]
Confidenza: [ALTA/MEDIA/BASSA]
💡 Scommessa alternativa: [bet secondario]
Perché: [1 riga]
⚠️ Attenzione: [1 variabile di rischio]"""

        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        narrative = ""
        for block in resp.content:
            if hasattr(block, "text"):
                narrative += block.text

        return narrative.strip() if narrative.strip() else None

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
        "narrative_enabled": bool(os.environ.get("ANTHROPIC_API_KEY"))
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

    market_labels = {
        "winner":     {"title": "Winner",                    "yes": f"{p1_name} vince",      "no": f"{p2_name} vince"},
        "both_set":   {"title": "Entrambi vincono un set",   "yes": "Sì",                    "no": "No — vittoria netta"},
        "games_over": {"title": "Games Over/Under 22.5",     "yes": "Over 22.5",             "no": "Under 22.5"},
        "aces_over":  {"title": "Aces Over/Under 10.5",      "yes": "Over 10.5",             "no": "Under 10.5"},
        "sets_over":  {"title": "Set Over/Under 2.5",        "yes": "Over 2.5 set",          "no": "Under 2.5 set"},
        "tiebreak":   {"title": "Tiebreak (almeno 1)",       "yes": "Sì — almeno 1 tiebreak","no": "No tiebreak"},
    }

    h2h_surf = get_h2h_on_surface(df_history, p1_id, p2_id, surface)
    markets  = {}

    for market, pred in predictions.items():
        prob     = pred.get("prob", 0.5)
        prob_pct = pred.get("prob_pct", 50.0)
        conf     = pred.get("confidence", "bassa")
        reliable = market in RELIABLE_MARKETS
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
            reasons = explain_aces(p1_name, p2_name, p1_stats, p2_stats, surface, prob)
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

    # Narrativa giornalistica (Claude API + web search)
    narrative = generate_narrative(
        p1_name, p2_name, tournament, surface, tour, round_str, best_of,
        markets, p1_stats_resp, p2_stats_resp, h2h_resp, data_quality
    )

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
        "narrative":        narrative,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
