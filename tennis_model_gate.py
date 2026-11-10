"""Pure, testable reliability gate for the experimental Daily Picks endpoint."""
from datetime import date, datetime, timezone


def daily_model_gate(manifest, now=None, max_age_days=21):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must have a timezone")
    if not isinstance(manifest, dict):
        return False, "Modelli cronologici assenti: nuovo training richiesto."
    if manifest.get("feature_engine") != "temporal-v2":
        return False, "Modelli legacy non validati: nessuna selezione."
    if not {"winner", "games_over"}.issubset(set(manifest.get("trained_markets", []))):
        return False, "Modelli Winner e Games non pronti."
    try:
        latest = date.fromisoformat(manifest["training_data_latest_date"])
    except (KeyError, ValueError, TypeError):
        return False, "Data ultima partita storica non verificabile."
    age_days = (now.date() - latest).days
    if age_days < 0:
        return False, "Data storica futura rispetto al sistema: verifica l'orologio."
    if age_days > max_age_days:
        return False, "Dataset storico troppo vecchio: nessun pronostico pubblicato."
    return True, "Modelli cronologici presenti; probabilita ancora sperimentali."
