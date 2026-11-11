"""Append-only paper research ledger on a dedicated Git branch.

Never overwrites a first snapshot for the same ESPN fixture ID/date/tour.
A paper forecast with no authenticated quoted price stays UNPRICED.
This code does not trade, move funds, settle a match or use provider secrets.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc(text):
    try:
        dt=datetime.fromisoformat(str(text).replace("Z","+00:00"))
    except (ValueError, TypeError):
        raise ValueError("Invalid timestamp")
    if dt.tzinfo is None:
        raise ValueError("Snapshot timestamp lacks timezone")
    return dt.astimezone(timezone.utc)


def safe_token(value):
    """Stable Git-safe ASCII path component; never trust fixture IDs as paths."""
    val=re.sub(r"[^a-zA-Z0-9_-]","_",str(value))
    if not val or len(val)>75:
        raise ValueError("Invalid or excessively long token")
    return val


def accepted_prediction(record):
    """Fail closed on any pre-match, price, or player identity anomaly."""
    if record.get("paper_only") is not True or record.get("wager_submitted") is not False:
        return False
    start, captured=utc(record["scheduled_start_utc"]),utc(record["captured_at_utc"])
    if (start-captured).total_seconds() < 20*60:
        return False
    if not record.get("fixture_id") or record.get("tour") not in ("ATP","WTA"):
        return False
    if record.get("surface") not in ("Hard","Grass","Clay"):
        return False
    if record.get("p1_historical_id")==record.get("p2_historical_id"):
        return False
    if not record.get("picked_winner_name") or record.get("picked_winner_name") not in (
        record.get("p1"),record.get("p2")
    ):
        return False
    p=record.get("picked_winner_probability")
    if not isinstance(p,(int,float)) or not math.isfinite(p) or not .5<=p<=1:
        return False
    status=record.get("market_status")
    price=record.get("odds_decimal")
    if status=="priced":
        if (not isinstance(price,(int,float)) or not math.isfinite(price) or
            not 1.4<=price<1000 or not record.get("bookmaker")):
            return False
        updated=utc(record.get("provider_last_update_utc"))
        if updated>captured or (captured-updated)>timedelta(minutes=90):
            return False
    elif status=="missing_eligible_h2h_quote":
        if price is not None or record.get("bookmaker"):
            return False
    elif status=="below_min_1_40":
        if (not isinstance(price,(int,float)) or
            not math.isfinite(price) or not 1<price<1.4 or not record.get("bookmaker")):
            return False
        updated=utc(record.get("provider_last_update_utc"))
        if updated>captured or (captured-updated)>timedelta(minutes=90):
            return False
    else:
        return False
    return True


def persist(snapshot, root, run_id):
    """Append one attempt report, and the FIRST valid capture per fixture.

    Idempotent on re-runs, guarded against out-of-order attempts.
    Expected workflow uses one concurrency group, so pushes cannot race.
    """
    if snapshot.get("mode")!="PROSPECTIVE_RESEARCH_PAPER_ONLY":
        raise ValueError("Not a prospective paper snapshot")
    root=Path(root)
    when=utc(snapshot["captured_at_utc"])
    attempt_dir=root/"paper_attempts"/when.date().isoformat()
    attempt_dir.mkdir(parents=True,exist_ok=True)
    run=safe_token(run_id)
    attempt=attempt_dir/(run+".json")
    if attempt.exists():
        raise ValueError("Duplicate run ID; historical attempts cannot be overwritten")
    attempt.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False),encoding="utf-8")
    new_count=0
    prior_count=0
    rejected=0
    for prediction in snapshot.get("predictions") or []:
        if not accepted_prediction(prediction):
            rejected+=1
            continue
        start=utc(prediction["scheduled_start_utc"])
        folder=root/"paper_predictions"/start.date().isoformat()
        folder.mkdir(parents=True,exist_ok=True)
        slug=safe_token(prediction["tour"]+"_"+str(prediction["fixture_id"]))
        path=folder/(slug+".json")
        if path.exists():
            prior_count+=1
            continue
        data={
            "mode":"PROSPECTIVE_PAPER_ONLY", "ledger_schema":1,
            "first_capture_run":run,
            "source_model_version":snapshot.get("model_id"),
            "model_git_revision":snapshot.get("model_git_revision"),
            "prediction":prediction,
            "settlement_status":"pending",
            "no_real_wager":True,
        }
        path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
        new_count+=1
    return {"attempt":str(attempt.relative_to(root)),
            "new_first_snapshots":new_count, "previously_captured":prior_count,
            "invalid_predictions_rejected":rejected,
            "priced_new_or_existing":sum(
                p.get("market_status")=="priced"
                for p in snapshot.get("predictions") or []
            )}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",default="paper_snapshot.json")
    p.add_argument("--ledger",required=True)
    p.add_argument("--run-id",required=True)
    args=p.parse_args()
    with open(args.snapshot,encoding="utf-8") as stream:
        data=json.load(stream)
    print(json.dumps(persist(data,args.ledger,args.run_id),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
