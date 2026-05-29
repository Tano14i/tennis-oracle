"""
tennis_config.py
Configurazione Tennis Oracle Bot.
"""

# Token bot Telegram (da @BotFather)
BOT_TOKEN = "8713818655:AAGRAuWbyGf1Vi-pGW6M0UZnqZJIbTh4IK0"

# ID canale o chat dove inviare le analisi
# Esempi:
#   Canale pubblico:  "@tennis_oracle_signals"
#   Chat privata:     123456789  (il tuo chat_id numerico)
#   Gruppo:           -123456789
CHANNEL_ID = 7638572013

# Il tuo chat_id Telegram (per ricevere notifiche di errore)
ADMIN_ID = 7638572013

# Ora invio automatico (UTC)
SCHEDULE_HOUR = 8
SCHEDULE_MINUTE = 0

# Tour da analizzare automaticamente ogni mattina
ACTIVE_TOURS = {"ATP", "WTA", "ITF", "Challenger"}

# Minimo partite storiche per analisi affidabile
MIN_MATCHES = 10

# Soglia confidenza per BET consigliato
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.58

# API key Anthropic per analisi narrative AI
# Ottienila su https://console.anthropic.com/
ANTHROPIC_API_KEY = ""  # <-- inserisci la tua API key qui
