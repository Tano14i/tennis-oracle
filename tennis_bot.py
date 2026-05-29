"""
tennis_bot.py - Tennis Oracle Bot con menu interattivo
Flusso: /oggi -> lista partite con bottoni -> analisi singola on-demand
"""

import os
import sys
import time
import threading
import traceback
from datetime import datetime, timezone

import telebot
from telebot import types
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tennis_schedule import get_matches_today, filter_matches, group_by_tournament, detect_surface
from tennis_analysis import generate_match_report, generate_summary_line
from tennis_dataset import get_h2h, get_h2h_on_surface, find_player_id
from tennis_models import predict_match, MODELS_DIR

try:
    from tennis_ai_analysis import generate_ai_analysis, format_ai_report
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from tennis_config import BOT_TOKEN, CHANNEL_ID, ADMIN_ID
    ANTHROPIC_API_KEY = getattr(__import__("tennis_config"), "ANTHROPIC_API_KEY", "")
except ImportError:
    print("ERRORE: crea tennis_config.py con BOT_TOKEN, CHANNEL_ID, ADMIN_ID")
    sys.exit(1)

ANTHROPIC_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
} if ANTHROPIC_API_KEY else {}

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# ---------------------------------------------------------------------------
# Stato sessione per utente (menu partite)
# ---------------------------------------------------------------------------
# { chat_id: { "matches": [...], "page": 0, "date": "2026-05-26" } }
user_sessions = {}
PAGE_SIZE = 8  # partite per pagina

# ---------------------------------------------------------------------------
# Carica modelli
# ---------------------------------------------------------------------------
def load_all():
    required = ["player_stats.pkl", "surface_stats.pkl", "name_lookup.pkl",
                "df_history.pkl", "model_winner.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            return None, None, None, None
    player_stats  = joblib.load(os.path.join(MODELS_DIR, "player_stats.pkl"))
    surface_stats = joblib.load(os.path.join(MODELS_DIR, "surface_stats.pkl"))
    name_lookup   = joblib.load(os.path.join(MODELS_DIR, "name_lookup.pkl"))
    df_history    = joblib.load(os.path.join(MODELS_DIR, "df_history.pkl"))
    return player_stats, surface_stats, name_lookup, df_history

player_stats, surface_stats, name_lookup, df_history = load_all()
MODELS_LOADED = player_stats is not None
if MODELS_LOADED:
    print(f"Modelli caricati. Giocatori: {len(player_stats)}")
else:
    print("ATTENZIONE: modelli non trovati. Esegui tennis_trainer.py prima.")

# ---------------------------------------------------------------------------
# Analisi singola partita
# ---------------------------------------------------------------------------
def analyze_single_match(m):
    """Analizza una singola partita e restituisce il report."""
    if not MODELS_LOADED:
        return "⚠️ Modelli non caricati. Esegui tennis_trainer.py prima."

    p1_name    = m["p1"]
    p2_name    = m["p2"]
    surface    = m.get("surface", "Hard")
    tour       = m.get("tour", "ATP")
    tournament = m.get("tournament", "")
    round_str  = m.get("round", "R32")
    best_of    = 5 if tour == "ATP" and round_str in ("F", "SF", "QF") and "Grand Slam" in tournament else 3

    p1_id = find_player_id(p1_name, name_lookup)
    p2_id = find_player_id(p2_name, name_lookup)

    if not p1_id or not p2_id:
        missing = p1_name if not p1_id else p2_name
        return (
            f"🎾 <b>{p1_name.upper()} vs {p2_name.upper()}</b>\n"
            f"⚠️ {missing} non trovato nel database storico.\n"
            f"Analisi non disponibile."
        )

    try:
        predictions, h2h, h2h_surf = predict_match(
            p1_id, p2_id, surface,
            m.get("category", "A"), round_str, best_of,
            player_stats, surface_stats, df_history,
        )

        # AI analysis se disponibile
        if AI_AVAILABLE and ANTHROPIC_HEADERS:
            p1_stats_d = player_stats.get(p1_id, {})
            p2_stats_d = player_stats.get(p2_id, {})
            p1_surf_d  = surface_stats.get(p1_id, {})
            p2_surf_d  = surface_stats.get(p2_id, {})
            ai_data, ai_err = generate_ai_analysis(
                p1_name, p2_name,
                tournament, surface, tour, round_str, best_of,
                p1_stats_d, p2_stats_d,
                p1_surf_d, p2_surf_d,
                h2h, h2h_surf, predictions,
                ANTHROPIC_HEADERS,
            )
            if ai_data:
                return format_ai_report(
                    p1_name, p2_name, tournament, surface, tour, round_str, best_of, ai_data
                )

        # Fallback template
        return generate_match_report(
            p1_name, p2_name, p1_id, p2_id,
            tournament, surface, tour, round_str, best_of,
            player_stats, surface_stats, df_history,
            predictions, h2h, h2h_surf,
        )

    except Exception as e:
        return f"🎾 <b>{p1_name} vs {p2_name}</b>\n⚠️ Errore: {str(e)}"


# ---------------------------------------------------------------------------
# Menu partite con bottoni inline
# ---------------------------------------------------------------------------
def build_match_keyboard(matches, page=0, show_back=False):
    """Costruisce la tastiera inline con le partite della pagina corrente."""
    keyboard = types.InlineKeyboardMarkup(row_width=1)

    start = page * PAGE_SIZE
    end   = min(start + PAGE_SIZE, len(matches))
    page_matches = matches[start:end]

    for i, m in enumerate(page_matches):
        idx = start + i
        surf_icon = {"Hard": "🔵", "Clay": "🟤", "Grass": "🟢"}.get(m["surface"], "⚪")
        label = f"{surf_icon} {m['p1']} vs {m['p2']}"
        if m.get("time"):
            label += f" | {m['time']}"
        keyboard.add(types.InlineKeyboardButton(label, callback_data=f"match_{idx}"))

    # Navigazione pagine
    nav_buttons = []
    if page > 0:
        nav_buttons.append(types.InlineKeyboardButton("⬅️ Precedenti", callback_data=f"page_{page-1}"))
    if end < len(matches):
        nav_buttons.append(types.InlineKeyboardButton("Successive ➡️", callback_data=f"page_{page+1}"))
    if nav_buttons:
        keyboard.row(*nav_buttons)

    # Bottone aggiorna
    keyboard.add(types.InlineKeyboardButton("🔄 Aggiorna palinsesto", callback_data="refresh"))

    return keyboard


def send_match_menu(chat_id, matches, page=0, edit_msg_id=None):
    """Invia o aggiorna il menu partite."""
    today = datetime.now(timezone.utc).strftime("%A %d %B %Y")
    total = len(matches)

    start = page * PAGE_SIZE + 1
    end   = min((page + 1) * PAGE_SIZE, total)

    text = (
        f"🎾 <b>TENNIS ORACLE</b>\n"
        f"📅 {today}\n"
        f"{'─' * 28}\n"
        f"<b>{total} partite in programma</b>\n"
        f"Mostrando {start}-{end} di {total}\n\n"
        f"Seleziona una partita per l'analisi:"
    )

    keyboard = build_match_keyboard(matches, page)

    if edit_msg_id:
        try:
            bot.edit_message_text(
                text, chat_id, edit_msg_id,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        except Exception:
            bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=keyboard)
    else:
        bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=keyboard)


# ---------------------------------------------------------------------------
# Comandi principali
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    text = (
        "🎾 <b>Tennis Oracle</b>\n\n"
        "Analisi pre-match ATP | WTA | ITF | Challenger\n\n"
        "<b>Comandi:</b>\n"
        "/oggi — Palinsesto di oggi\n"
        "/atp — Solo ATP\n"
        "/wta — Solo WTA\n"
        "/analizza Sinner Alcaraz — Match specifico\n"
        "/status — Stato sistema\n"
    )
    bot.send_message(message.chat.id, text)


@bot.message_handler(commands=["oggi"])
def handle_oggi(message):
    _show_schedule(message.chat.id, tours=None)


@bot.message_handler(commands=["atp"])
def handle_atp(message):
    _show_schedule(message.chat.id, tours={"ATP"})


@bot.message_handler(commands=["wta"])
def handle_wta(message):
    _show_schedule(message.chat.id, tours={"WTA"})


@bot.message_handler(commands=["itf"])
def handle_itf(message):
    _show_schedule(message.chat.id, tours={"ITF", "Challenger"})


def _show_schedule(chat_id, tours=None):
    """Carica il palinsesto e mostra il menu."""
    loading_msg = bot.send_message(chat_id, "⏳ Carico il palinsesto...")
    try:
        matches = get_matches_today()
        if tours:
            matches = filter_matches(matches, tours)

        if not matches:
            bot.edit_message_text(
                "📭 Nessuna partita trovata per oggi.",
                chat_id, loading_msg.message_id
            )
            return

        # Salva sessione utente
        user_sessions[chat_id] = {
            "matches": matches,
            "page": 0,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        bot.delete_message(chat_id, loading_msg.message_id)
        send_match_menu(chat_id, matches, page=0)

    except Exception as e:
        bot.edit_message_text(
            f"⚠️ Errore: {str(e)}",
            chat_id, loading_msg.message_id
        )


# ---------------------------------------------------------------------------
# Callback bottoni inline
# ---------------------------------------------------------------------------
@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    chat_id = call.message.chat.id
    msg_id  = call.message.message_id
    data    = call.data

    session = user_sessions.get(chat_id)

    # Aggiorna palinsesto
    if data == "refresh":
        bot.answer_callback_query(call.id, "Aggiorno...")
        _show_schedule(chat_id, tours=None)
        return

    # Cambio pagina
    if data.startswith("page_"):
        page = int(data.split("_")[1])
        if session:
            session["page"] = page
            send_match_menu(chat_id, session["matches"], page=page, edit_msg_id=msg_id)
        bot.answer_callback_query(call.id)
        return

    # Selezione partita
    if data.startswith("match_"):
        idx = int(data.split("_")[1])
        if not session or idx >= len(session["matches"]):
            bot.answer_callback_query(call.id, "Sessione scaduta. Usa /oggi")
            return

        m = session["matches"][idx]
        bot.answer_callback_query(call.id, f"Analisi {m['p1']} vs {m['p2']}...")

        # Messaggio di caricamento
        loading = bot.send_message(
            chat_id,
            f"⏳ Analisi in corso: <b>{m['p1']} vs {m['p2']}</b>...",
            parse_mode="HTML"
        )

        report = analyze_single_match(m)

        bot.delete_message(chat_id, loading.message_id)

        # Invia report con bottone "torna al menu"
        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(types.InlineKeyboardButton(
            "↩️ Torna al palinsesto",
            callback_data=f"page_{session.get('page', 0)}"
        ))

        bot.send_message(
            chat_id, report,
            parse_mode="HTML",
            reply_markup=keyboard,
            disable_web_page_preview=True,
        )
        return

    bot.answer_callback_query(call.id)


# ---------------------------------------------------------------------------
# Comando analizza singolo match
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["analizza"])
def handle_analizza(message):
    if not MODELS_LOADED:
        bot.send_message(message.chat.id, "⚠️ Modelli non caricati.")
        return

    parts = message.text.split()[1:]
    if len(parts) < 2:
        bot.send_message(message.chat.id, "Uso: /analizza NomeGiocatore1 NomeGiocatore2 [Hard|Clay|Grass]")
        return

    surface = "Hard"
    if len(parts) >= 3 and parts[-1] in ("Hard", "Clay", "Grass"):
        surface = parts[-1]
        names = parts[:-1]
    else:
        names = parts

    p1_name = names[0]
    p2_name = names[1] if len(names) > 1 else ""

    p1_id = find_player_id(p1_name, name_lookup)
    p2_id = find_player_id(p2_name, name_lookup) if p2_name else None

    if not p1_id:
        bot.send_message(message.chat.id, f"⚠️ Giocatore non trovato: {p1_name}")
        return
    if not p2_id:
        bot.send_message(message.chat.id, f"⚠️ Giocatore non trovato: {p2_name}")
        return

    loading = bot.send_message(message.chat.id, f"⏳ Analisi {p1_name} vs {p2_name}...")

    m = {
        "p1": p1_name, "p2": p2_name,
        "surface": surface, "tour": "ATP",
        "tournament": "Match personalizzato",
        "round": "R32", "time": "", "category": "A",
    }
    report = analyze_single_match(m)
    bot.delete_message(message.chat.id, loading.message_id)
    bot.send_message(message.chat.id, report, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["status"])
def handle_status(message):
    models_ok = sum(1 for m in ["winner","sets_over","both_set","games_over","aces_over","tiebreak"]
                    if os.path.exists(os.path.join(MODELS_DIR, f"model_{m}.pkl")))
    ai_status = "✅ Attiva" if (AI_AVAILABLE and ANTHROPIC_HEADERS) else "⚠️ Non configurata (solo template)"
    text = (
        f"🎾 <b>Tennis Oracle — Status</b>\n"
        f"{'─' * 28}\n"
        f"Modelli ML: {models_ok}/6\n"
        f"Giocatori nel lookup: {len(player_stats) if MODELS_LOADED else 'N/A'}\n"
        f"Analisi AI: {ai_status}\n"
        f"{'─' * 28}\n"
        f"Usa /oggi per il palinsesto interattivo"
    )
    bot.send_message(message.chat.id, text)


# ---------------------------------------------------------------------------
# Scheduler automatico
# ---------------------------------------------------------------------------
def scheduler_loop():
    print("Scheduler avviato — invio automatico alle 08:00 UTC")
    sent_today = None
    while True:
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        if now.hour == 8 and now.minute == 0 and sent_today != today_str:
            try:
                matches = get_matches_today()
                if matches and CHANNEL_ID:
                    user_sessions[CHANNEL_ID] = {
                        "matches": matches,
                        "page": 0,
                        "date": today_str,
                    }
                    send_match_menu(CHANNEL_ID, matches, page=0)
                    sent_today = today_str
                    print(f"Palinsesto inviato: {len(matches)} partite")
            except Exception as e:
                print(f"Errore scheduler: {e}")
        time.sleep(30)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("TENNIS ORACLE BOT — Menu Interattivo")
    print("=" * 50)
    if not MODELS_LOADED:
        print("ATTENZIONE: Modelli non trovati. Esegui tennis_trainer.py prima.")
    ai_msg = "AI analysis: ATTIVA" if (AI_AVAILABLE and ANTHROPIC_HEADERS) else "AI analysis: NON CONFIGURATA"
    print(ai_msg)

    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()

    print("Bot in ascolto... Comandi: /oggi /atp /wta /analizza /status")
    bot.infinity_polling(timeout=60, long_polling_timeout=30)
