"""Prospective, research-only Tennis Oracle paper-snapshot runner.

Designed to run on the GitHub Actions clock, NOT from old odds/history.
The emitted JSON has an honest received timestamp before the match.
No wager placement. No closing prices masquerading as pre-match odds.

Optional bookmaker snapshots require BOTH the already-owned ODDS_API_KEY
and explicit ORACLE_PAPER_ODDS_ENABLE=1; absent either, odds remain null.
No additional services or paid plans are created by this program.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
import math
import subprocess

import pandas as pd

from tennis_elo import pre_match_elo, predict_from_state, elo_configuration
from tennis_hf_audit import NAMES, download as download_hf, _date as parse_utc
from tennis_hf_crosswalk import (
    run as build_verified_crosswalk,
    parse_csv, historical_signatures, source_signature,
)
from tennis_hf_research_backtest import prepare_rows
from tennis_schedule import get_matches_today
from tennis_daily_picks import names_match

MIN_LEAD_MINUTES = 20
MAX_HISTORY_AGE_DAYS = 21
MIN_PRICE = 1.40
MODEL_ID = "elo_g28_s36_w065_prior12_v1"
ALLOWED_STATES = {"pre", "scheduled"}
ALLOWED_SURFACES = {"Hard", "Clay", "Grass"}


def _clock(value):
    if value.tzinfo is None:
        raise ValueError("UTC-aware timestamp required for prospective snapshots")
    return value.astimezone(timezone.utc)


def source_lookup(by_tour, verified_mapping):
    """Stable unique ESPN full-name -> historical player ID matcher.

    Each HF source identifier must use one consistent surname+initials
    signature, AND it must be verified against historical fixture evidence.
    Any collision, duplicate player, ambiguous signature: refuse.
    """
    source_signatures = defaultdict(set)
    for tour in ("ATP", "WTA"):
        for row in by_tour.get(tour, []):
            for side in ("winner", "loser"):
                pid = str(row.get(side + "_id") or "").strip()
                signature = source_signature(row.get(side + "_name"))
                if signature and f"{tour}:{pid}" in verified_mapping:
                    source_signatures[(tour, pid)].add(signature)
    lookup = defaultdict(set)
    for (tour, sid), sigs in source_signatures.items():
        if len(sigs) != 1:
            continue
        numeric_id = int(verified_mapping[f"{tour}:{sid}"]["legacy_id"])
        lookup[(tour, next(iter(sigs)))].add(numeric_id)
    return lookup


def resolve_full_name(full_name, tour, lookup):
    matches = set()
    for signature in historical_signatures(full_name):
        matches.update(lookup.get((tour, signature), set()))
    return next(iter(matches)) if len(matches) == 1 else None


def match_start_utc(event, on_date):
    """ESPN adapter explicitly exposes a UTC clock but not an ISO date."""
    value = str(event.get("time") or "").strip()
    if not value.endswith(" UTC"):
        return None
    clock = value.removesuffix(" UTC")
    try:
        hm = datetime.strptime(clock, "%H:%M").time()
    except ValueError:
        return None
    return datetime.combine(on_date, hm, timezone.utc)


def _book_price(odds, match, chosen_name, at, start):
    """Strict paired H2H quote from a single bookmaker and matching fixture.

    Quote capture must be BEFORE scheduled start by at least 20 minutes;
    provider last_update must not be from the future or older than 90 min.
    Event time must agree within one hour of the ESPN fixture.
    """
    for item in odds:
        begins = parse_utc(item.get("commence_time"))
        if not begins or abs((begins-start).total_seconds()) > 3600:
            continue
        home = str(item.get("home_team") or "")
        away = str(item.get("away_team") or "")
        p1, p2 = match["p1"], match["p2"]
        if not ((names_match(p1, home) and names_match(p2, away)) or
                (names_match(p2, home) and names_match(p1, away))):
            continue
        if not (names_match(chosen_name, home) or names_match(chosen_name, away)):
            continue
        for book in item.get("bookmakers") or []:
            bkey = str(book.get("key") or "").strip()
            for market in book.get("markets") or []:
                if market.get("key") != "h2h":
                    continue
                updated = parse_utc(market.get("last_update") or book.get("last_update"))
                if (updated is None or updated > at or
                    (at-updated).total_seconds() > 90*60):
                    continue
                prices = {}
                for out in market.get("outcomes") or []:
                    try:
                        price = float(out.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(price) and 1.0 < price < 1000:
                        prices[str(out.get("name") or "")] = price
                if home not in prices or away not in prices or not bkey:
                    continue
                chosen = home if names_match(chosen_name, home) else away
                if prices[chosen] < MIN_PRICE:
                    return {"odds_decimal": prices[chosen],
                            "market_status": "below_min_1_40",
                            "bookmaker": bkey, "provider_last_update_utc": updated.isoformat()}
                return {"odds_decimal": prices[chosen], "market_status": "priced",
                        "bookmaker": bkey, "provider_last_update_utc": updated.isoformat()}
    return {"odds_decimal": None, "market_status": "missing_eligible_h2h_quote",
            "bookmaker": None, "provider_last_update_utc": None}


def build_snapshot(events, historic, by_tour, mapping, now, odds_records=None):
    """Pure, testable snapshot builder. Never scores outcomes or settles bets."""
    now = _clock(now)
    now_day = now.date()
    cutoff = pd.Timestamp(now_day)
    past = historic.loc[pd.to_datetime(historic["tourney_date"]) < cutoff].copy()
    if past.empty:
        raise ValueError("No historical records strictly before snapshot UTC date")
    # Elo state is calculated strictly from matches on preceding UTC dates.
    _, state = pre_match_elo(past, return_state=True)
    if pd.Timestamp(state["last_historical_date"]).date() >= now_day:
        raise AssertionError("Same-day/future results contaminated Elo state")
    players = source_lookup(by_tour, mapping)
    skipped = Counter()
    predictions = []
    for event in events:
        tour = str(event.get("tour") or "")
        surf = str(event.get("surface") or "")
        if event.get("state") not in ALLOWED_STATES:
            skipped["not_scheduled"] += 1
            continue
        if tour not in ("ATP", "WTA") or surf not in ALLOWED_SURFACES:
            skipped["unsupported_tour_or_surface"] += 1
            continue
        start = match_start_utc(event, now_day)
        if start is None or start <= now + timedelta(minutes=MIN_LEAD_MINUTES):
            skipped["late_or_unknown_start"] += 1
            continue
        one = resolve_full_name(event.get("p1"), tour, players)
        two = resolve_full_name(event.get("p2"), tour, players)
        if one is None or two is None or one == two:
            skipped["identity_unverified_or_ambiguous"] += 1
            continue
        likelihood = predict_from_state(state, tour, surf, one, two)
        p = likelihood["p1_probability"]
        chosen = event["p1"] if p >= .5 else event["p2"]
        fixture_id = str(event.get("id") or "").strip()
        if not fixture_id:
            skipped["missing_fixture_id"] += 1
            continue
        market = _book_price(
            odds_records or [], event, chosen, now, start
        )
        confidence = max(p, 1-p)
        predictions.append({
            "fixture_id": fixture_id,
            "tour": tour, "surface": surf, "surface_source": "ESPN_tournament_heuristic",
            "tournament": event.get("tournament"), "p1": event["p1"],
            "p2": event["p2"], "p1_historical_id": one,
            "p2_historical_id": two, "scheduled_start_utc": start.isoformat(),
            "captured_at_utc": now.isoformat(),
            "history_last_date": state["last_historical_date"],
            "model_id": MODEL_ID, "p1_probability": round(p, 6),
            "picked_winner_name": chosen,
            "picked_winner_probability": round(confidence, 6),
            "p1_prior_matches": likelihood["p1_matches_before"],
            "p2_prior_matches": likelihood["p2_matches_before"],
            "quality_flags": {
                "surface_not_independently_verified": True,
                "cold_start": min(likelihood["p1_matches_before"],
                                  likelihood["p2_matches_before"]) < 10,
                "market_quote_unavailable": market["odds_decimal"] is None,
            },
            **market,
            "paper_only": True, "wager_submitted": False,
        })
    return {
        "schema_version": 1, "mode": "PROSPECTIVE_RESEARCH_PAPER_ONLY",
        "captured_at_utc": now.isoformat(), "model_id": MODEL_ID,
        "model_git_revision": os.environ.get("ORACLE_PAPER_MODEL_SHA", ""),
        "source_history_strictly_before_date": now_day.isoformat(),
        "ratings_last_historical_date": state["last_historical_date"],
        "source_identity_mapping_count": len(mapping),
        "fixture_source": "ESPN scoreboard, date-local UTC only",
        "surface_quality": "Tournament-name heuristic, not independently verified",
        "quote_mode": "OPTIONAL_EXPLICIT_EXISTING_KEY_ONLY",
        "odds_missing_is_not_a_value_bet": True,
        "no_wager_execution": True,
        "events_seen": len(events), "predictions_count": len(predictions),
        "priced_predictions": sum(x["market_status"] == "priced" for x in predictions),
        "skipped": dict(skipped), "predictions": predictions,
    }


def run(now=None):
    now = _clock(now or datetime.now(timezone.utc))
    crosswalk = build_verified_crosswalk(now=now)
    with open("hf_verified_crosswalk_research.json", encoding="utf-8") as f:
        mapping = json.load(f)["mapping"]
    by_tour = {"ATP": [], "WTA": []}
    latest = None
    for filename in NAMES:
        raw, checksum = download_hf(filename)
        if checksum != crosswalk["source_sha256_by_filename"][filename]:
            raise ValueError("External data changed between audit and prospective scoring")
        tour = "ATP" if filename.startswith("atp_") else "WTA"
        rows = parse_csv(raw)
        by_tour[tour].extend(rows)
        for row in rows:
            when = parse_utc(row.get("start_time_utc"))
            if when and when < now and (latest is None or when > latest):
                latest = when
    if not latest or not (0 <= (now-latest).days <= MAX_HISTORY_AGE_DAYS):
        raise ValueError("Last finished historical match is unavailable or stale")
    data, _historical_closing_prices_never_used, quality = prepare_rows(
        by_tour, mapping, now
    )
    matches = get_matches_today()
    odds = []
    if os.environ.get("ORACLE_PAPER_ODDS_ENABLE") == "1":
        if not os.environ.get("ODDS_API_KEY"):
            raise ValueError("Optional odds explicitly enabled but no authorized API key")
        # Operator opt-in only. Two supported sport groups at most.
        from tennis_odds import get_tennis_odds
        tours = {m.get("tour") for m in matches}
        for tour in ("ATP", "WTA"):
            if tour in tours:
                odds.extend(get_tennis_odds(
                    sport_key=("tennis_atp" if tour == "ATP" else "tennis_wta"),
                    regions="eu", markets="h2h"
                ))
    result = build_snapshot(matches, data, by_tour, mapping, now, odds)
    result["verified_data_freshest_result_utc"] = latest.isoformat()
    result["research_data_records"] = len(data)
    result["source_sha256"] = crosswalk["source_sha256_by_filename"]
    result["historical_identity_qc"] = {
        "verified_player_count": len(mapping),
        "independent_match_overlaps": crosswalk["independent_exact_fixture_overlaps"],
    }
    return result


def main():
    at = datetime.now(timezone.utc)
    try:
        report = run(at)
    except Exception as error:
        report = {
            "schema_version": 1, "mode": "PROSPECTIVE_RESEARCH_PAPER_ONLY",
            "captured_at_utc": at.isoformat(), "model_id": MODEL_ID,
            "status": "BLOCKED_NO_PREDICTIONS",
            "reason": type(error).__name__ + ": " + str(error)[:240],
            "predictions": [], "predictions_count": 0, "priced_predictions": 0,
            "no_wager_execution": True,
        }
    path = os.environ.get("ORACLE_PAPER_OUTPUT", "paper_snapshot.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({
        "captured_at_utc": report["captured_at_utc"],
        "status": report.get("status", "PAPER_SNAPSHOT"),
        "predictions_count": report["predictions_count"],
        "priced_predictions": report["priced_predictions"],
        "events_seen": report.get("events_seen"),
        "skipped": report.get("skipped"),
        "reason": report.get("reason"),
    }, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
