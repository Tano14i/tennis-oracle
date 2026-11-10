"""
Tennis Oracle 2.0 - experimental Daily Picks engine.

Builds four research tickets from pre-match, actually quoted outcomes:
single (>=1.40), 3x3 (~3.00), multiple (>=5), bomb (>=7).
No quote or uncertain player/match identity -> no recommendation.
These outputs are NOT validated predictions or evidence of value.
"""
from __future__ import annotations

import itertools
import math
import re
import unicodedata
from datetime import datetime, timezone

MIN_ODDS = 1.40
MIN_MATCHES = 20
MIN_SURFACE_MATCHES = 5
MAX_CANDIDATES = 18
TIERS = (
    ("single", "Singola", 1.40),
    ("three_by_three", "3×3", 2.75),
    ("multiple_5", "Multipla 5+", 5.00),
    ("bomb_7", "Bomba 7+", 7.00),
)


def _normalized(name):
    clean = unicodedata.normalize("NFKD", str(name or ""))
    clean = "".join(c for c in clean if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", clean.lower()))


def names_match(a, b):
    """Exact names or compatible first initial and surname; no arbitrary substrings."""
    aa, bb = _normalized(a).split(), _normalized(b).split()
    if not aa or not bb:
        return False
    if aa == bb:
        return True
    if len(aa[-1]) < 4 or aa[-1] != bb[-1]:
        return False
    if len(aa) == 1 or len(bb) == 1:
        return True
    first_a, first_b = aa[0], bb[0]
    return first_a == first_b or ((len(first_a) == 1 or len(first_b) == 1) and first_a[0] == first_b[0])


def strict_player_id(name, name_lookup):
    """Accept unambiguous identities only; refuse the legacy fuzzy-name fallback."""
    name_norm = _normalized(name)
    direct = {int(v) for k, v in name_lookup.items() if _normalized(k) == name_norm}
    if len(direct) == 1:
        return next(iter(direct))
    if direct:
        return None
    matches = {int(v) for k, v in name_lookup.items() if names_match(k, name)}
    return next(iter(matches)) if len(matches) == 1 else None


def _date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _valid_odds(value):
    try:
        number = float(value)
        return math.isfinite(number) and MIN_ODDS <= number <= 100
    except (TypeError, ValueError):
        return False


def _missing(reason):
    return {"status": "unavailable", "reason": reason, "legs": [], "combined_odds": None}


def empty_report(now, reason):
    timestamp = now.astimezone(timezone.utc).isoformat()
    return {
        "date_utc": timestamp[:10],
        "generated_at_utc": timestamp,
        "mode": "experimental_research_only",
        "verified_profitability": False,
        "disclaimer": "Modelli non validati fuori campione: indicazioni sperimentali, non garanzie.",
        "candidate_count": 0,
        "scanned_matches": 0,
        "products": {key: _missing(reason) for key, _, _ in TIERS},
        "notes": [reason],
    }


def _fixture_key(p1, p2, commence):
    return "|".join(sorted((_normalized(p1), _normalized(p2)))) + "|" + commence


def _ticket(legs):
    product = 1.0
    for leg in legs:
        product *= leg["odds"]
    quality = sum(leg["quality_score"] for leg in legs) / len(legs)
    return {
        "status": "experimental",
        "legs": list(legs),
        "combined_odds": round(product, 2),
        "quality_score": round(quality),
        "model_joint_prob_approx": round(math.prod(leg["model_probability"] for leg in legs), 4),
        "probability_note": "Prodotto delle probabilita modello; indipendenza solo approssimativa.",
    }


def _choose_combo(legs, sizes, low, target, banned_ids=(), high=None):
    """No two markets from the same match; keep distinct ticket signatures."""
    best = None
    best_rank = None
    for n in sizes:
        for choice in itertools.combinations(legs, n):
            if len({item["fixture_id"] for item in choice}) != n:
                continue
            signature = frozenset(item["id"] for item in choice)
            if signature in banned_ids:
                continue
            t = _ticket(choice)
            combined = t["combined_odds"]
            if combined < low or (high is not None and combined > high):
                continue
            # Robust choices first, then proximity to target, fewer legs as tie breaker.
            base = sum(x["ranking_score"] for x in choice) / n
            score = base - 0.10 * abs(math.log(combined / target)) - 0.008 * max(0, n - 3)
            rank = (round(score, 8), -abs(combined - target), -n, tuple(sorted(signature)))
            if best_rank is None or rank > best_rank:
                best, best_rank = t, rank
    return best


def _get_legs(match, price, result, p1_id, p2_id, stats, surface_stats):
    surface = match["surface"]
    s_key = "n_" + surface.lower()
    m1 = stats.get(p1_id, {}).get("p_matches", 0)
    m2 = stats.get(p2_id, {}).get("p_matches", 0)
    s1 = surface_stats.get(p1_id, {}).get(s_key, 0)
    s2 = surface_stats.get(p2_id, {}).get(s_key, 0)
    if min(m1, m2) < MIN_MATCHES or min(s1, s2) < MIN_SURFACE_MATCHES:
        return []
    quality = min(100, 55 + min(25, min(m1, m2) // 4) + min(20, min(s1, s2) * 2))
    fixture = _fixture_key(match["p1"], match["p2"], price["commence_time"])
    p1_is_home = names_match(match["p1"], price["home_team"])
    markets = []
    winner = result.get("winner", {})
    if winner.get("reliable"):
        p = winner.get("prob")
        if p is not None and 0 <= p <= 1:
            options = (
                (match["p1"] + " vince", float(p), price["home_h2h"] if p1_is_home else price["away_h2h"]),
                (match["p2"] + " vince", 1-float(p), price["away_h2h"] if p1_is_home else price["home_h2h"]),
            )
            markets.extend(("winner", label, prob, odd, price["book_h2h"]) for label, prob, odd in options)

    # Existing games model was trained ONLY for over/under 22.5 in BO3.
    games = result.get("games_over", {})
    try:
        line = float(price.get("total_point"))
    except (TypeError, ValueError):
        line = None
    if line == 22.5 and games.get("reliable"):
        p = games.get("prob")
        if p is not None and 0 <= p <= 1:
            markets.extend([
                ("games", "Over 22.5 game", float(p), price["over_total"], price["book_totals"]),
                ("games", "Under 22.5 game", 1-float(p), price["under_total"], price["book_totals"]),
            ])

    output = []
    for market, label, probability, odd, bookmaker in markets:
        if not _valid_odds(odd) or probability < 0.55 or not bookmaker:
            continue
        odd = float(odd)
        code = label.lower().replace(" ", "_")
        output.append({
            "id": fixture + "|" + code,
            "fixture_id": fixture,
            "p1": match["p1"], "p2": match["p2"],
            "tournament": match.get("tournament", ""),
            "tour": match.get("tour", ""),
            "surface": surface,
            "commence_time_utc": price["commence_time"],
            "market": market, "selection": label,
            "odds": round(odd, 3), "bookmaker": bookmaker,
            "model_probability": round(probability, 4),
            "quality_score": int(quality),
            "ranking_score": round(0.75 * probability + 0.25 * quality / 100, 6),
            "data_matches": [int(m1), int(m2)],
            "surface_matches": [int(s1), int(s2)],
        })
    return output


def build_daily_products(matches, prices, predict, resolve, player_stats, surface_stats, now=None):
    """
    predict(p1_id, p2_id, surface, level, round, best_of) -> per-market dictionaries.
    resolve(player_name) -> ID or None. Inputs must come from trusted fixtures
    and real pre-match odds; this routine makes NO external requests.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    report = empty_report(now, "Nessuna selezione verificabile disponibile.")
    report["scanned_matches"] = len(matches)
    legs = []
    seen = set()
    for match in matches:
        if match.get("tour") not in ("ATP", "WTA") or match.get("surface") not in ("Hard", "Clay", "Grass"):
            continue
        if match.get("state") not in ("pre", "scheduled"):
            continue
        p1, p2 = match.get("p1", ""), match.get("p2", "")
        if not p1 or not p2 or names_match(p1, p2):
            continue
        p1_id, p2_id = resolve(p1), resolve(p2)
        if not p1_id or not p2_id or p1_id == p2_id:
            continue
        for quote in prices:
            if not (names_match(p1, quote.get("home_team")) and names_match(p2, quote.get("away_team"))) and not (
                names_match(p2, quote.get("home_team")) and names_match(p1, quote.get("away_team"))
            ):
                continue
            start = _date(quote.get("commence_time"))
            if not start or start <= now or start.date() != now.astimezone(timezone.utc).date():
                continue
            # No BO5 from a model trained on BO3 game totals. BO5 markets need their own model.
            tournament = match.get("tournament", "").lower()
            if match.get("tour") == "ATP" and any(x in tournament for x in (
                "wimbledon", "roland garros", "french open", "us open", "australian open",
            )):
                continue
            key = _fixture_key(p1, p2, quote["commence_time"])
            if key in seen:
                continue
            seen.add(key)
            try:
                preds = predict(p1_id, p2_id, match["surface"], "A", match.get("round") or "R32", 3)
                legs.extend(_get_legs(match, quote, preds, p1_id, p2_id, player_stats, surface_stats))
            except (KeyError, TypeError, ValueError, ArithmeticError):
                continue
            break

    # Keep strongest market per fixture where ranking permits; still retain more
    # than one candidate per fixture so the combination optimizer can choose.
    ranked = sorted(legs, key=lambda x: (-x["ranking_score"], -x["quality_score"], x["id"]))
    shortlist = ranked[:MAX_CANDIDATES]
    report["candidate_count"] = len(legs)
    report["notes"] = []
    if not shortlist:
        report["notes"].append("Nessun incontro pre-match con quote >=1.40, modello e dati sufficienti.")
        return report

    report["products"]["single"] = _ticket([shortlist[0]])
    triple = _choose_combo(shortlist, (3,), 2.75, 3.0, high=3.45)
    report["products"]["three_by_three"] = triple or _missing("Nessuna terna attendibile vicino a quota 3.")
    multi = _choose_combo(shortlist, (3,4,5), 5.0, 5.5)
    report["products"]["multiple_5"] = multi or _missing("Non ci sono combinazioni idonee a quota 5+.")
    banned = (frozenset(x["id"] for x in multi["legs"]),) if multi else ()
    bomb = _choose_combo(shortlist, (3,4,5), 7.0, 7.7, banned_ids=banned)
    report["products"]["bomb_7"] = bomb or _missing("Non ci sono combinazioni idonee a quota 7+.")
    return report
