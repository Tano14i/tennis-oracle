"""
tennis_ai_analysis.py
Genera analisi narrative in italiano via Claude API (claude-sonnet-4-20250514).
Sostituisce i template fissi con testo narrativo contestuale.
"""

import os
import json
import requests

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"


def build_analysis_prompt(
    p1_name, p2_name,
    tournament, surface, tour, round_str, best_of,
    p1_stats, p2_stats,
    p1_surf_stats, p2_surf_stats,
    h2h, h2h_surf,
    predictions,
):
    """Costruisce il prompt per Claude con tutti i dati disponibili."""

    surf_key = f"wr_{surface.lower()}"
    p1_rank = int(p1_stats.get("p_rank", 999))
    p2_rank = int(p2_stats.get("p_rank", 999))
    p1_wr = round(p1_stats.get("p_win_rate", 0.5) * 100, 1)
    p2_wr = round(p2_stats.get("p_win_rate", 0.5) * 100, 1)
    p1_surf_wr = round(p1_surf_stats.get(surf_key, 0.5) * 100, 1)
    p2_surf_wr = round(p2_surf_stats.get(surf_key, 0.5) * 100, 1)
    p1_aces = p1_stats.get("p_aces_avg", 0)
    p2_aces = p2_stats.get("p_aces_avg", 0)
    p1_1st_won = round(p1_stats.get("p_1st_won", 0) * 100, 1)
    p2_1st_won = round(p2_stats.get("p_1st_won", 0) * 100, 1)
    p1_matches = p1_stats.get("p_matches", 0)
    p2_matches = p2_stats.get("p_matches", 0)

    h2h_total = h2h.get("h2h_total", 0)
    h2h_p1_wins = h2h.get("h2h_p1_wins", 0)
    h2h_surf_total = h2h_surf.get("h2h_surf_total", 0)
    h2h_surf_rate = h2h_surf.get("h2h_surf_p1_win_rate", 0.5)

    # Predizioni ML
    winner_prob = round(predictions.get("winner", {}).get("prob", 0.5) * 100, 1)
    both_set_prob = round(predictions.get("both_set", {}).get("prob", 0.5) * 100, 1)
    games_over_prob = round(predictions.get("games_over", {}).get("prob", 0.5) * 100, 1)
    aces_over_prob = round(predictions.get("aces_over", {}).get("prob", 0.5) * 100, 1)

    prompt = f"""Sei un esperto analista tennis. Genera un'analisi pre-match in italiano per questo incontro.

PARTITA: {p1_name} vs {p2_name}
TORNEO: {tournament} ({tour})
SUPERFICIE: {surface}
ROUND: {round_str} | Best of {best_of}

DATI STATISTICI (ultimi 20 match):
{p1_name}:
- Ranking: #{p1_rank}
- Win rate generale: {p1_wr}% (su {p1_matches} match)
- Win rate su {surface}: {p1_surf_wr}%
- Ace/match: {p1_aces:.1f}
- % punti con 1a di servizio: {p1_1st_won}%

{p2_name}:
- Ranking: #{p2_rank}
- Win rate generale: {p2_wr}% (su {p2_matches} match)
- Win rate su {surface}: {p2_surf_wr}%
- Ace/match: {p2_aces:.1f}
- % punti con 1a di servizio: {p2_1st_won}%

HEAD TO HEAD:
- Totale incontri: {h2h_total}
- {p1_name} ha vinto: {h2h_p1_wins} ({h2h_total - h2h_p1_wins} per {p2_name})
- Su {surface}: {h2h_surf_total} precedenti (win rate {p1_name}: {round(h2h_surf_rate*100)}%)

PROBABILITÀ MODELLO ML:
- Winner {p1_name}: {winner_prob}%
- Entrambi vincono un set: {both_set_prob}%
- Over 22.5 games: {games_over_prob}%
- Over 10.5 ace totali: {aces_over_prob}%

ISTRUZIONI:
Genera un'analisi in formato JSON con questa struttura esatta:
{{
  "sommario": "Una frase che descrive il match in modo vivace (es: 'Una battaglia tra il veterano e il giovane rampante')",
  "favorito": "{p1_name} o {p2_name}",
  "analisi_winner": "2-3 frasi narrative sul perché uno dei due dovrebbe vincere, considerando forma, superficie, H2H. Sii specifico e umano.",
  "scommessa_principale": {{
    "mercato": "es: Winner, Entrambi vincono un set, Over aces, ecc.",
    "giocata": "la giocata consigliata in modo chiaro",
    "motivazione": "2-3 frasi narrative sul perché questa giocata ha valore, considerando i dati statistici",
    "confidenza": "ALTA / MEDIA / BASSA",
    "no_bet": false
  }},
  "scommessa_secondaria": {{
    "mercato": "mercato alternativo",
    "giocata": "giocata alternativa",
    "motivazione": "motivazione breve",
    "confidenza": "ALTA / MEDIA / BASSA",
    "no_bet": false
  }},
  "attenzione": "Un avviso o fattore di rischio da tenere in considerazione (es: infortuni, condizioni meteo, motivazione)"
}}

IMPORTANTE:
- Scrivi in italiano colloquiale ma professionale
- Sii specifico sui dati — cita win rate, H2H, superficie
- Se i dati sono insufficienti (meno di 10 match), indica no_bet: true
- Non consigliare bet su favoriti schiaccianti dove le quote non hanno valore (es: top 5 vs qualificata)
- Identifica il mercato con più valore reale, non solo quello con probabilità più alta
- Rispondi SOLO con il JSON, nessun testo prima o dopo"""

    return prompt


def generate_ai_analysis(
    p1_name, p2_name,
    tournament, surface, tour, round_str, best_of,
    p1_stats, p2_stats,
    p1_surf_stats, p2_surf_stats,
    h2h, h2h_surf,
    predictions,
    headers_anthropic,
):
    """Chiama Claude API e restituisce l'analisi narrativa."""

    prompt = build_analysis_prompt(
        p1_name, p2_name,
        tournament, surface, tour, round_str, best_of,
        p1_stats, p2_stats,
        p1_surf_stats, p2_surf_stats,
        h2h, h2h_surf,
        predictions,
    )

    try:
        response = requests.post(
            CLAUDE_API_URL,
            headers=headers_anthropic,
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        if response.status_code != 200:
            return None, f"API error {response.status_code}"

        content = response.json().get("content", [])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")

        # Pulizia JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        return data, None

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"Errore: {e}"


def format_ai_report(p1_name, p2_name, tournament, surface, tour, round_str, best_of, analysis):
    """Formatta il report narrativo per Telegram."""

    surf_icon = {"Hard": "🔵", "Clay": "🟤", "Grass": "🟢"}.get(surface, "⚪")
    lines = []

    lines.append(f"🎾 <b>{p1_name.upper()} vs {p2_name.upper()}</b>")
    lines.append(f"🏆 {tour} — {tournament}")
    lines.append(f"{surf_icon} {surface} | {round_str} | Best of {best_of}")
    lines.append("─" * 30)

    sommario = analysis.get("sommario", "")
    if sommario:
        lines.append(f"<i>{sommario}</i>")
        lines.append("")

    # Analisi winner
    favorito = analysis.get("favorito", "")
    analisi = analysis.get("analisi_winner", "")
    if analisi:
        lines.append(f"🏆 <b>Chi vince?</b> {favorito}")
        lines.append(analisi)
        lines.append("")

    # Scommessa principale
    bet1 = analysis.get("scommessa_principale", {})
    if bet1 and not bet1.get("no_bet"):
        conf = bet1.get("confidenza", "BASSA")
        conf_icon = {"ALTA": "🟢", "MEDIA": "🟡", "BASSA": "⚠️"}.get(conf, "⚠️")
        lines.append(f"💰 <b>BET PRINCIPALE</b> {conf_icon} {conf}")
        lines.append(f"<b>{bet1.get('mercato', '')}</b>: {bet1.get('giocata', '')}")
        lines.append(bet1.get("motivazione", ""))
        lines.append("")
    elif bet1 and bet1.get("no_bet"):
        lines.append(f"⚠️ <b>NO BET</b> — {bet1.get('motivazione', 'Segnale debole')}")
        lines.append("")

    # Scommessa secondaria
    bet2 = analysis.get("scommessa_secondaria", {})
    if bet2 and not bet2.get("no_bet"):
        conf2 = bet2.get("confidenza", "BASSA")
        conf_icon2 = {"ALTA": "🟢", "MEDIA": "🟡", "BASSA": "⚠️"}.get(conf2, "⚠️")
        lines.append(f"💡 <b>BET SECONDARIO</b> {conf_icon2} {conf2}")
        lines.append(f"<b>{bet2.get('mercato', '')}</b>: {bet2.get('giocata', '')}")
        lines.append(bet2.get("motivazione", ""))
        lines.append("")

    # Attenzione
    attenzione = analysis.get("attenzione", "")
    if attenzione:
        lines.append(f"⚠️ <i>{attenzione}</i>")

    lines.append("─" * 30)
    lines.append("⚙️ <i>Analisi ML + AI — Dataset Sackmann 2018-2026</i>")

    return "\n".join(lines)
