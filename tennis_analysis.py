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

    wr1 = p1_stats.get("p_win_rate", 0.5)
    wr2 = p2_stats.get("p_win_rate", 0.5)
    surf_key = f"wr_{surface.lower()}"
    s1 = p1_surf.get(surf_key, 0.5)
    s2 = p2_surf.get(surf_key, 0.5)

    # Win rate generale
    wr_diff = wr1 - wr2
    if abs(wr_diff) >= 0.08:
        better = p1_name if wr_diff > 0 else p2_name
        worse  = p2_name if wr_diff > 0 else p1_name
        reasons.append(f"{better} win rate superiore ({max(wr1,wr2)*100:.0f}% vs {min(wr1,wr2)*100:.0f}% su ATP)")

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
    wr_diff = abs(wr1 - wr2)

    # Differenza di livello basata su win rate (p_rank non disponibile nel dataset)
    if wr_diff > 0.20:
        stronger = p1_name if wr1 > wr2 else p2_name
        reasons.append(f"Netto divario di livello: {stronger} domina su win rate ({max(wr1,wr2)*100:.0f}% vs {min(wr1,wr2)*100:.0f}%)")
    elif wr_diff < 0.06:
        reasons.append(f"Win rate simili ({wr1*100:.0f}% vs {wr2*100:.0f}%) — match equilibrato")

    going_over = prob_over >= 0.5

    # Superficie — coerente con la direzione e con best_of
    min_sets = 3 if best_of == 5 else 2
    threshold = "3.5" if best_of == 5 else "2.5"
    if surface == "Clay" and going_over:
        reasons.append(f"Terra: scambi lunghi aumentano probabilità di match lungo (Over {threshold})")
    elif surface == "Clay" and not going_over:
        reasons.append(f"Terra ma divario netto: vittoria rapida attesa ({min_sets}-0 o {min_sets}-1, Under {threshold})")
    elif surface == "Grass" and not going_over:
        reasons.append(f"Erba: servizio dominante favorisce match breve (Under {threshold})")
    elif surface == "Grass" and going_over:
        reasons.append(f"Erba con servizi vulnerabili: atteso match lungo (Over {threshold})")

    # Best of 5 — ragioni specifiche (minimo 3 set, quindi "over 3.5" = 4+ set)
    if best_of == 5 and going_over:
        reasons.append("Formato Bo5: con giocatori equilibrati attesi 4-5 set (Over 3.5)")
    elif best_of == 5 and not going_over:
        reasons.append("Formato Bo5 ma divario netto: vittoria 3-0 o 3-1 probabile (Under 3.5)")

    return reasons


def explain_games(p1_name, p2_name, p1_stats, p2_stats, surface, prob_over):
    """Genera motivazione per Games Over/Under 22.5 (solo BO3)."""
    reasons = []
    going_over = prob_over >= 0.5

    s1_1st = p1_stats.get("p_1st_won", 0.70)
    s2_1st = p2_stats.get("p_1st_won", 0.70)
    s1_2nd = p1_stats.get("p_2nd_won", 0.50)
    s2_2nd = p2_stats.get("p_2nd_won", 0.50)
    avg_hold = (s1_1st + s2_1st + s1_2nd + s2_2nd) / 4

    # Motivo servizio — coerente con la direzione
    if avg_hold > 0.72 and going_over:
        reasons.append(f"Entrambi forti al servizio → più hold, più game totali")
    elif avg_hold < 0.62 and not going_over:
        reasons.append(f"Servizi vulnerabili → molti break, set brevi")
    elif avg_hold > 0.72 and not going_over:
        reasons.append(f"Servizi solidi ma atteso match rapido per divario di livello")
    elif avg_hold < 0.62 and going_over:
        reasons.append(f"Servizi vulnerabili ma match equilibrato → attesi scambi lunghi")

    # Motivo superficie — coerente con la direzione
    if surface == "Grass" and not going_over:
        reasons.append("Erba favorisce servizio e punti rapidi: meno game totali")
    elif surface == "Grass" and going_over:
        reasons.append("Erba ma servizi non dominanti: attesi più scambi del solito")
    elif surface == "Clay" and going_over:
        reasons.append("Terra: scambi lunghi e più break favoriscono game totali elevati")
    elif surface == "Clay" and not going_over:
        reasons.append("Terra ma match con netto favorito: vittoria attesa in tre set rapidi")
    elif surface == "Hard":
        if going_over:
            reasons.append("Cemento: superficie equilibrata, match atteso combattuto")
        else:
            reasons.append("Cemento veloce: punti rapidi favoriscono Under")

    return reasons


def explain_aces(p1_name, p2_name, p1_stats, p2_stats, surface, prob_over, best_of=3):
    """Genera motivazione per Aces Over/Under."""
    reasons = []

    a1 = p1_stats.get("p_aces_avg", 5.0)
    a2 = p2_stats.get("p_aces_avg", 5.0)
    total_bo3 = a1 + a2

    # Aggiusta per BO5: moltiplica per il rapporto set medi attesi
    if best_of == 5:
        total_expected = round(total_bo3 * (5 / 3), 1)
        reasons.append(
            f"Media aces/match BO3: {p1_name} {a1:.1f}, {p2_name} {a2:.1f} "
            f"→ atteso ~{total_expected} aces in Bo5 (~{total_bo3:.1f}×5/3)"
        )
    else:
        reasons.append(
            f"Media aces: {p1_name} {a1:.1f}/match, {p2_name} {a2:.1f}/match "
            f"(totale atteso {total_bo3:.1f})"
        )

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
