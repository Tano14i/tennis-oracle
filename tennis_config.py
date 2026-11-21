"""Tennis Oracle Bot settings. Secrets must be supplied via environment variables."""
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "")
ADMIN_ID = os.environ.get("ADMIN_ID", "")

SCHEDULE_HOUR = int(os.environ.get("SCHEDULE_HOUR", "8"))
SCHEDULE_MINUTE = int(os.environ.get("SCHEDULE_MINUTE", "0"))
ACTIVE_TOURS = {"ATP", "WTA", "ITF", "Challenger"}
MIN_MATCHES = 10
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.58
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
