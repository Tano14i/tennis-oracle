"""Independent data-integrity audit for Valuebetennis seasonal CSV files.

Does NOT merge provider IDs with Sackmann and NEVER produces betting picks.
A reported completed match with a UTC start time in the future is a critical
provenance failure; the entire source batch is quarantined until investigated.
License and provenance: https://www.valuebetennis.com/en/donnees.htm (CC BY 4.0).
"""
from __future__ import annotations

import csv
import io
import math
from collections import Counter
from datetime import datetime, timezone

REQUIRED = {
    "match_id", "date", "genre", "surface", "joueur1", "joueur1_id",
    "joueur2", "joueur2_id", "vainqueur_id", "score",
    "cote1_ouverture", "cote2_ouverture",
    "cote1_cloture", "cote2_cloture",
}


def _price(text):
    if text is None or not str(text).strip():
        return None
    try:
        number = float(str(text).strip().replace(",", "."))
    except ValueError:
        return None
    return number if math.isfinite(number) and 1 < number < 1000 else None


def audit_valuebet_csv(csv_text, now=None, max_examples=5):
    """Audit snapshot *without trusting it* or ingesting any scoring fields.

    Returns an inspectable report; accepted=True only if all critical
    integrity checks pass. Caller must still check licensing and provenance.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")), delimiter=";")
    headers = set(reader.fieldnames or [])
    missing = sorted(REQUIRED - headers)
    report = {
        "source": "Valuebetennis (CC BY 4.0; publisher attribution required)",
        "checked_at_utc": now.astimezone(timezone.utc).isoformat(),
        "rows": 0,
        "missing_columns": missing,
        "invalid_date": 0,
        "future_settled": 0,
        "invalid_winner": 0,
        "duplicate_match_ids": 0,
        "invalid_odds_pairs": 0,
        "opening_odds_pairs": 0,
        "closing_odds_pairs": 0,
        "by_tour": {},
        "future_settled_examples": [],
        "accepted": False,
        "reason": None,
    }
    if missing:
        report["reason"] = "Missing required CSV fields"
        return report
    seen_ids = set()
    tours = Counter()
    future_examples = []
    for row in reader:
        report["rows"] += 1
        key = str(row.get("match_id") or "").strip()
        if not key or key in seen_ids:
            report["duplicate_match_ids"] += 1
        seen_ids.add(key)
        tours[(row.get("genre") or "").strip().lower()] += 1
        try:
            start = datetime.strptime(row["date"].strip(), "%Y-%m-%d %H:%M:%S")
            start = start.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, KeyError, AttributeError):
            report["invalid_date"] += 1
            continue
        winner = str(row.get("vainqueur_id") or "").strip()
        p1 = str(row.get("joueur1_id") or "").strip()
        p2 = str(row.get("joueur2_id") or "").strip()
        if not p1 or not p2 or p1 == p2 or winner not in (p1, p2):
            report["invalid_winner"] += 1
        has_result = bool(winner and str(row.get("score") or "").strip())
        if has_result and start > now:
            report["future_settled"] += 1
            if len(future_examples) < max_examples:
                future_examples.append({
                    "match_id": key, "date": row["date"],
                    "tournament": row.get("tournoi"),
                    "players": [row.get("joueur1"), row.get("joueur2")],
                    "score": row.get("score"),
                })
        for p1_field, p2_field, dest in [
            ("cote1_ouverture", "cote2_ouverture", "opening_odds_pairs"),
            ("cote1_cloture", "cote2_cloture", "closing_odds_pairs"),
        ]:
            one, two = _price(row.get(p1_field)), _price(row.get(p2_field))
            if one is not None and two is not None:
                report[dest] += 1
            elif bool(row.get(p1_field)) != bool(row.get(p2_field)) or (
                bool(row.get(p1_field)) and (one is None or two is None)
            ):
                report["invalid_odds_pairs"] += 1
    report["by_tour"] = dict(tours)
    report["future_settled_examples"] = future_examples
    issues = [k for k in (
        "invalid_date", "future_settled", "invalid_winner", "duplicate_match_ids"
    ) if report[k] > 0]
    if not report["rows"]:
        issues.append("empty_file")
    report["accepted"] = not issues
    report["reason"] = (
        "CRITICAL: " + ", ".join(issues) if issues else
        "Integrity checks passed; independent provider verification still required"
    )
    return report
