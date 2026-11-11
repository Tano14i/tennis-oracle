"""Independent result settlement for append-only Tennis Oracle paper trials.

Only post-match ESPN scoreboard winner flags may settle a forecast.
Lack of clear verified result -> pending, NEVER inferred from price or Elo.

Settlements are separate immutable files, not edits to predictions.
No actual bets; unit P&L is only available when a real pre-match
bookmaker quote was recorded in the original snapshot.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tennis_daily_picks import names_match
from tennis_paper_ledger import accepted_prediction, safe_token, utc

BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"


def _competitor_name(record):
    athlete = record.get("athlete") or {}
    if isinstance(athlete, dict):
        return str(athlete.get("displayName") or athlete.get("fullName") or "")
    return str(record.get("displayName") or record.get("name") or "")


def finished_winner_by_id(payload):
    """Extract only unambiguous winner for completed ESPN competitions."""
    result = {}
    for event in payload.get("events") or []:
        for group in event.get("groupings") or []:
            for comp in group.get("competitions") or []:
                mid = str(comp.get("id") or "")
                status = (comp.get("status") or {}).get("type") or {}
                if not mid or status.get("state") != "post":
                    continue
                competitors = comp.get("competitors") or []
                if len(competitors) != 2:
                    continue
                wins = [record.get("winner") is True for record in competitors]
                if wins.count(True) != 1:
                    continue
                winner = _competitor_name(competitors[wins.index(True)])
                loser = _competitor_name(competitors[wins.index(False)])
                if not winner or not loser or names_match(winner, loser):
                    continue
                # No winner is inferred if different data returned for same ID.
                if mid in result and result[mid] != winner:
                    result[mid] = None
                else:
                    result[mid] = winner
    return {mid:winner for mid,winner in result.items() if winner}


def _fetch(tour, day):
    tour_slug = {"ATP":"atp","WTA":"wta"}[tour]
    url = f"{BASE}/{tour_slug}/scoreboard?dates={day.strftime('%Y%m%d')}"
    req = urllib.request.Request(url, headers={
        "User-Agent":"TennisOraclePaperSettlement/1.0",
        "Accept":"application/json",
    })
    with urllib.request.urlopen(req, timeout=20) as response:
        data=response.read(3_000_001)
    if len(data) > 3_000_000:
        raise ValueError("ESPN result response exceeds safe bound")
    return json.loads(data)


def match_result(prediction, winner):
    """Resolve only if winner matches exactly one of the originally frozen names."""
    if not winner:
        return None
    p1 = names_match(prediction["p1"], winner)
    p2 = names_match(prediction["p2"], winner)
    if p1 == p2:
        return None
    selected = prediction["picked_winner_name"]
    selected_win = (p1 and names_match(selected,prediction["p1"])) or (
        p2 and names_match(selected,prediction["p2"])
    )
    quoted = (
        prediction.get("market_status") == "priced" and
        isinstance(prediction.get("odds_decimal"), (int,float)) and
        prediction["odds_decimal"] >= 1.4
    )
    return {
        "actual_winner": prediction["p1"] if p1 else prediction["p2"],
        "predicted_correct": bool(selected_win),
        "picked_winner_probability": prediction["picked_winner_probability"],
        "brier": (prediction["picked_winner_probability"] - int(selected_win))**2,
        "quoted_paper_unit_net": (
            (prediction["odds_decimal"] - 1.0 if selected_win else -1.0)
            if quoted else None
        ),
        "quoted_decimal_odds": prediction.get("odds_decimal") if quoted else None,
    }


def settle_ledger(root, now, fetch=_fetch):
    """Write one immutable settlement per first snapshot; report unresolved."""
    now=utc(now.isoformat()) if isinstance(now, datetime) else utc(now)
    root=Path(root)
    originals=list((root/"paper_predictions").glob("*/*.json"))
    pending=[]
    for path in originals:
        try:
            packet=json.loads(path.read_text(encoding="utf-8"))
            pred=packet["prediction"]
            if not accepted_prediction(pred):
                continue
            start=utc(pred["scheduled_start_utc"])
            if start + timedelta(hours=3) > now or (now - start).days > 14:
                continue
            settled=root/"paper_settlements"/start.date().isoformat()/path.name
            if settled.exists():
                continue
            pending.append((pred,start,settled))
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):
            continue
    cache={}
    errors={}
    settled_now=0
    still_pending=0
    for pred,start,path in pending:
        tour=pred["tour"]
        key=(tour,start.date())
        if key not in cache:
            try:
                cache[key]=finished_winner_by_id(fetch(tour,start.date()))
            except Exception as error:
                cache[key]={}
                errors[f"{tour}:{start.date()}"]=type(error).__name__
        actual=cache[key].get(str(pred["fixture_id"]))
        result=match_result(pred,actual)
        if result is None:
            still_pending+=1
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        output={
            "settlement_schema":1,
            "source":"ESPN_SCOREBOARD_POST_WITH_WINNER_FLAG",
            "source_match_id":pred["fixture_id"],
            "tour":tour,
            "fixture_date":start.date().isoformat(),
            "matched_as_of_utc":now.isoformat(),
            "original_snapshot_utc":pred["captured_at_utc"],
            "model_id":pred["model_id"],
            "no_real_wager":True,
            **result,
        }
        path.write_text(json.dumps(output,indent=2,ensure_ascii=False),
                        encoding="utf-8")
        settled_now+=1

    all_files=list((root/"paper_settlements").glob("*/*.json"))
    records=[]
    for file in all_files:
        try:
            records.append(json.loads(file.read_text(encoding="utf-8")))
        except (ValueError,KeyError):
            continue
    n=len(records)
    priced=[r["quoted_paper_unit_net"] for r in records
            if r.get("quoted_paper_unit_net") is not None]
    return {
        "run_at_utc":now.isoformat(), "source":"ESPN",
        "pending_examined":len(pending),"settled_this_run":settled_now,
        "still_pending":still_pending,"api_errors_by_tour_day":errors,
        "first_snapshots_total":len(originals),
        "settled_total":n,
        "winner_accuracy":round(
            sum(bool(r["predicted_correct"]) for r in records)/n,5) if n else None,
        "brier_mean":round(sum(r["brier"] for r in records)/n,5) if n else None,
        "priced_settled_count":len(priced),
        "quoted_paper_roi_pct":round(
            sum(priced)/len(priced)*100,3) if priced else None,
        "roi_method":"Hypothetical flat 1-unit bets using captured prematch odds only; not actual betting",
        "not_financial_advice":True,"no_real_wager":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",required=True)
    p.add_argument("--run-id",required=True)
    args=p.parse_args()
    now=datetime.now(timezone.utc)
    report=settle_ledger(args.ledger,now)
    folder=Path(args.ledger)/"paper_settlement_runs"/now.date().isoformat()
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/(safe_token(args.run_id)+".json")
    if target.exists():
        raise ValueError("This settlement run ID already exists")
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
