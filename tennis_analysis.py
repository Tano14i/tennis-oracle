"""
tennis_analysis.py
Genera il report completo con 7 mercati + motivazione per ogni partita.
"""

import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Soglie NO BET
# ---------------------------------------------------------------------------

NO_BET_THRESHOLD = 0.55   # sotto questa confidenza -> no bet
MIN_MATCHES_REQUIRED = 10  # minimo match storici per analisi affidabile


# ---------------------------------------------------------------------------
# Motivazioni per mercato
# ---------------------------------------------------------------------------

def explain_winner(p1_name, p2_name, p1_stats, p2_stats, p1_surf, p2_surf,
                   surface, h2h, h2h_surf, prob_p1):
    """Genera motivazione per il mercato Winner."""
    reasons = []

    r1 = p1_stats.get("p_rank", 200)
    r2 = p2_stats.get("p_rank", 200)
    wr1 = p1_stats.get("p_win_rate", 0.5)
    wr2 = p2_stats.get("p_win_rate", 0.5)
    surf_key = f"wr_{surface.lower()}"
    s1 = p1_surf.get(surf_key, 0.5)
    s2 = p2_surf.get(surf_key, 0.5)

    # Ranking
    if r1 < r2:
        reasons.append(f"{p1_name} ranking superiore (#{int(r1)} vs #{int(r2)})")
    else:
        reasons.append(f"{p2_name} ranking superiore (#{int(r2)} vs #{int(r1)})")

    # Win rate superficie
    surf_diff = s1 - s2
    if abs(surf_diff) >= 0.08:
        better = p1_name if surf_diff > 0 else p2_name
        worse = p2_name if surf_diff > 0 else p1_name
        reasons.append(f"{better} migliore su {surface} ({max(s1,s2)*100:.0f}% vs {min(s1,s2)*100:.0f}%)")

    # H2H
    h2h_total = h2h.get("h2h_total", 0)
    if h2h_total >= 3:
        h2h_rate = h2h.get("h2h_p1_win_rate", 0.5)
        if h2h_rate >= 0.6:
            reasons.append(f"H2H favorevole a {p1_name} ({int(h2h_rate*h2h_total)}-{h2h_total-int(h2h_rate*h2h_total)} totale)")
        elif h2h_rate <= 0.4:
            reasons.append(f"H2H sfavorevole a {p1_name} ({int(h2h_rate*h2h_total)}-{h2h_total-int(h2h_rate*h2h_total)} totale)")

    # H2H su superficie
    h2h_surf_total = h2h_surf.get("h2h_surf_total", 0)
    if h2h_surf_total >= 2:
        h2h_surf_rate = h2h_surf.get("h2h_surf_p1_win_rate", 0.5)
        if abs(h2h_surf_rate - 0.5) >= 0.25:
            better = p1_name if h2h_surf_rate > 0.5 else p2_name
            reasons.append(f"H2H su {surface}: {better} avanti ({h2h_surf_total} precedenti)")

    return reasons


def explain_sets(p1_name, p2_name, p1_stats, p2_stats, surface, best_of, prob_over):
    """Genera motivazione per Set Over/Under."""
    reasons = []

    wr1 = p1_stats.get("p_win_rate", 0.5)
    wr2 = p2_stats.get("p_win_rate", 0.5)
    rank_diff = abs(p1_stats.get("p_rank", 200) - p2_stats.get("p_rank", 200))

    # Differenza di livello
    if rank_diff > 50:
        reasons.append(f"Grande differenza di ranking ({int(rank_diff)} posizioni) favorisce match diretto")
    elif rank_diff < 20:
        reasons.append(f"Giocatori di livello simile, alta probabilità di lotta")

    # Superficie
    if surface == "Clay":
        reasons.append("Su terra i match tendono ad essere più lunghi e combattuti")
    elif surface == "Grass":
        reasons.append("Su erba i match spesso si decidono in fretta (servizio dominante)")

    # Win rate simile = più set
    wr_diff = abs(wr1 - wr2)
    if wr_diff < 0.05:
        reasons.append(f"Win rate simili ({wr1*100:.0f}% vs {wr2*100:.0f}%) — match equilibrato")

    return reasons


def explain_games(p1_name, p2_name, p1_stats, p2_stats, surface, prob_over):
    """Genera motivazione per Games Over/Under."""
    reasons = []

    s1_1st = p1_stats.get("p_1st_won", 0.70)
    s2_1st = p2_stats.get("p_1st_won", 0.70)
    s1_2nd = p1_stats.get("p_2nd_won", 0.50)
    s2_2nd = p2_stats.get("p_2nd_won", 0.50)

    avg_hold = (s1_1st + s2_1st + s1_2nd + s2_2nd) / 4

    if avg_hold > 0.72:
        reasons.append(f"Entrambi forti al servizio — alto hold rate atteso (più game)")
    elif avg_hold < 0.62:
        reasons.append(f"Servizi vulnerabili — attesi molti break (meno game)")

    if surface == "Grass":
        reasons.append("Erba favorisce il servizio: meno break, meno game totali")
    elif surface == "Clay":
        reasons.append("Terra favorisce lo scambio: più game, più break")

    return reasons


def explain_aces(p1_name, p2_name, p1_stats, p2_stats, surface, prob_over):
    """Genera motivazione per Aces Over/Under."""
    reasons = []

    a1 = p1_stats.get("p_aces_avg", 5.0)
    a2 = p2_stats.get("p_aces_avg", 5.0)
    total = a1 + a2

    reasons.append(f"Media aces: {p1_name} {a1:.1f}/match, {p2_name} {a2:.1f}/match (totale atteso {total:.1f})")

    ht1 = p1_stats.get("p_ht", 185)
    ht2 = p2_stats.get("p_ht", 185)
    if max(ht1, ht2) >= 196:
        reasons.append(f"Giocatore alto presente — servizio potente favorisce gli aces")

    if surface == "Grass":
        reasons.append("Erba massimizza gli aces (+30% rispetto alla media su altre superfici)")
    elif surface == "Clay":
        reasons.append("Terra riduce il rimbalzo del servizio (-40% aces rispetto all'erba)")

    return reasons


def explain_tiebreak(p1_name, p2_name, p1_stats, p2_stats, surface, prob):
    """Genera motivazione per Tiebreak Yes/No."""
    reasons = []

    s1_1st_in = p1_stats.get("p_1st_in", 0.60)
    s2_1st_in = p2_stats.get("p_1st_in", 0.60)

    if (s1_1st_in + s2_1st_in) / 2 > 0.65:
        reasons.append("Alta percentuale di prime palle in — servizi solidi aumentano prob. tiebreak")

    if surface == "Grass":
        reasons.append(f"Erba: tiebreak storicamente nel 37% dei match")
    elif surface == "Hard":
        reasons.append(f"Cemento: tiebreak nel 32% dei match")
    elif surface == "Clay":
        reasons.append(f"Terra: meno tiebreak (28%) — i break sono più frequenti")

    return reasons


# ---------------------------------------------------------------------------
# Generatore report completo
# ---------------------------------------------------------------------------

def generate_match_report(p1_name, p2_name, p1_id, p2_id,
                           tournament, surface, tour, round_str, best_of,
                           player_stats, surface_stats, df_history,
                           predictions, h2h, h2h_surf):
    """Genera il report completo per una partita con tutti i 7 mercati."""

    p1_stats = player_stats.get(p1_id, {})
    p2_stats = player_stats.get(p2_id, {})
    p1_surf_stats = surface_stats.get(p1_id, {})
    p2_surf_stats = surface_stats.get(p2_id, {})

    p1_matches = p1_stats.get("p_matches", 0)
    p2_matches = p2_stats.get("p_matches", 0)
    data_ok = min(p1_matches, p2_matches) >= MIN_MATCHES_REQUIRED

    lines = []

    # Header
    lines.append(f"🎾 <b>{p1_name.upper()} vs {p2_name.upper()}</b>")
    lines.append(f"🏆 {tour} — {tournament}")
    surf_icon = {"Hard": "🔵", "Clay": "🟤", "Grass": "🟢"}.get(surface, "⚪")
    lines.append(f"{surf_icon} {surface} | {round_str} | Best of {best_of}")
    lines.append(f"{'─' * 32}")

    if not data_ok:
        lines.append(f"⚠️ Dati storici insufficienti per {p1_name if p1_matches < MIN_MATCHES_REQUIRED else p2_name}")
        lines.append(f"Le analisi sono orientative.")
        lines.append("")

    # Ranking e forma
    r1 = int(p1_stats.get("p_rank", 999))
    r2 = int(p2_stats.get("p_rank", 999))
    wr1 = p1_stats.get("p_win_rate", 0.5)
    wr2 = p2_stats.get("p_win_rate", 0.5)
    if r1 < 900 or r2 < 900:
        lines.append(f"📊 <b>Ranking:</b> {p1_name} #{r1} | {p2_name} #{r2}")
    lines.append(f"📈 <b>Win rate recente:</b> {p1_name} {wr1*100:.0f}% | {p2_name} {wr2*100:.0f}%")

    h2h_total = h2h.get("h2h_total", 0)
    if h2h_total > 0:
        h2h_p1 = h2h.get("h2h_p1_wins", 0)
        lines.append(f"⚔️ <b>H2H:</b> {p1_name} {h2h_p1}-{h2h_total-h2h_p1} {p2_name}")
    else:
        lines.append(f"⚔️ <b>H2H:</b> Nessun precedente trovato")
    lines.append(f"{'─' * 32}")

    # Mercati
    market_configs = [
        {
            "key": "winner",
            "title": "1. 🏆 WINNER",
            "label_yes": f"{p1_name} vince",
            "label_no": f"{p2_name} vince",
        },
        {
            "key": "sets_over",
            "title": f"2. 📊 SET OVER/UNDER ({'2.5' if best_of == 3 else '3.5'})",
            "label_yes": f"Over {'2.5' if best_of == 3 else '3.5'} set",
            "label_no": f"Under {'2.5' if best_of == 3 else '3.5'} set",
        },
        {
            "key": "both_set",
            "title": "3. 🎯 ENTRAMBI VINCONO UN SET",
            "label_yes": "Sì — entrambi vincono almeno 1 set",
            "label_no": "No — vittoria senza cedere set",
        },
        {
            "key": "games_over",
            "title": "4. 🎮 GAMES OVER/UNDER 22.5",
            "label_yes": "Over 22.5 games totali",
            "label_no": "Under 22.5 games totali",
        },
        {
            "key": "aces_over",
            "title": "5. 💥 ACES OVER/UNDER 10.5",
            "label_yes": "Over 10.5 aces totali",
            "label_no": "Under 10.5 aces totali",
        },
        {
            "key": "tiebreak",
            "title": "6. ⚡ TIEBREAK (almeno 1)",
            "label_yes": "Sì — almeno 1 tiebreak",
            "label_no": "No — nessun tiebreak",
        },
    ]

    for cfg in market_configs:
        key = cfg["key"]
        pred = predictions.get(key, {})
        prob = pred.get("prob", 0.5)
        prob_pct = pred.get("prob_pct", 50.0)
        conf = pred.get("confidence", "bassa")

        lines.append(f"\n<b>{cfg['title']}</b>")

        if not data_ok and conf == "bassa":
            lines.append(f"  ⚠️ <b>NO BET</b> — dati insufficienti")
            lines.append(f"  Probabilità indicativa: {prob_pct:.0f}%")
            continue

        if prob >= 0.5:
            bet_label = cfg["label_yes"]
            bet_prob = prob_pct
        else:
            bet_label = cfg["label_no"]
            bet_prob = round((1 - prob) * 100, 1)

        # No bet se confidenza bassa
        if conf == "bassa" or abs(prob - 0.5) < 0.05:
            lines.append(f"  ⚠️ <b>NO BET</b> — segnale debole ({bet_prob:.0f}% vs 50%)")
            lines.append(f"  Troppo equilibrato per raccomandare un'entrata")
        elif conf == "media":
            lines.append(f"  🟡 <b>BET CAUTO:</b> {bet_label}")
            lines.append(f"  Probabilità modello: <b>{bet_prob:.0f}%</b> | Confidenza: media")
        else:
            lines.append(f"  🟢 <b>BET CONSIGLIATO:</b> {bet_label}")
            lines.append(f"  Probabilità modello: <b>{bet_prob:.0f}%</b> | Confidenza: alta")

        # Motivazioni
        if key == "winner":
            reasons = explain_winner(p1_name, p2_name, p1_stats, p2_stats,
                                     p1_surf_stats, p2_surf_stats,
                                     surface, h2h, h2h_surf, prob)
        elif key == "sets_over":
            reasons = explain_sets(p1_name, p2_name, p1_stats, p2_stats, surface, best_of, prob)
        elif key == "both_set":
            reasons = explain_sets(p1_name, p2_name, p1_stats, p2_stats, surface, best_of, prob)
        elif key == "games_over":
            reasons = explain_games(p1_name, p2_name, p1_stats, p2_stats, surface, prob)
        elif key == "aces_over":
            reasons = explain_aces(p1_name, p2_name, p1_stats, p2_stats, surface, prob)
        elif key == "tiebreak":
            reasons = explain_tiebreak(p1_name, p2_name, p1_stats, p2_stats, surface, prob)
        else:
            reasons = []

        for r in reasons[:3]:
            lines.append(f"  • {r}")

    lines.append(f"\n{'─' * 32}")
    lines.append(f"⚙️ <i>Analisi basata su ultimi 20 match per superficie</i>")

    return "\n".join(lines)


def generate_summary_line(p1_name, p2_name, tournament, surface, predictions):
    """Genera una riga riassuntiva per il palinsesto del giorno."""
    bets = []
    for key, label in [
        ("winner", "W"),
        ("sets_over", "S"),
        ("games_over", "G"),
        ("aces_over", "A"),
        ("tiebreak", "TB"),
    ]:
        pred = predictions.get(key, {})
        if pred.get("confidence") in ("alta", "media"):
            prob = pred.get("prob", 0.5)
            direction = "+" if prob >= 0.5 else "-"
            bets.append(f"{label}{direction}")

    surf_icon = {"Hard": "🔵", "Clay": "🟤", "Grass": "🟢"}.get(surface, "⚪")
    bet_str = " ".join(bets) if bets else "—"
    return f"{surf_icon} {p1_name} vs {p2_name} | {bet_str}"
