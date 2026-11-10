"""SQLite paper ledger for Oracle Daily Picks.

Stores the first published ticket per UTC day/category, preserving the
pre-match timestamp and quoted selections. Settlement is manual, attributed
and never described as independently verified match data.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS oracle_tickets (
  ticket_id TEXT PRIMARY KEY,
  date_utc TEXT NOT NULL,
  category TEXT NOT NULL,
  generated_at_utc TEXT NOT NULL,
  combined_odds REAL NOT NULL,
  legs_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  payout_units REAL,
  settlement_source TEXT,
  settled_at_utc TEXT,
  UNIQUE(date_utc, category)
)
"""


def _db(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    db.commit()
    return db


def record_report(db_path, report):
    """Idempotently record first available ticket per UTC date/category."""
    if not db_path or report.get("mode") != "experimental_research_only":
        return 0
    count = 0
    with _db(db_path) as db:
        for category, ticket in report.get("products", {}).items():
            if ticket.get("status") != "experimental":
                continue
            legs = ticket.get("legs", [])
            if not legs or len({leg.get("fixture_id") for leg in legs}) != len(legs):
                continue
            if any(not leg.get("bookmaker") or not leg.get("odds") for leg in legs):
                continue
            day = report["date_utc"]
            cur = db.execute(
                """INSERT OR IGNORE INTO oracle_tickets
                   (ticket_id,date_utc,category,generated_at_utc,combined_odds,legs_json)
                   VALUES(?,?,?,?,?,?)""",
                (day + ":" + category, day, category, report["generated_at_utc"],
                 ticket["combined_odds"], json.dumps(legs, sort_keys=True, ensure_ascii=False))
            )
            count += cur.rowcount
    return count


def read_history(db_path, limit=100):
    """Returns paper performance, not real-money betting returns."""
    if not db_path or not Path(db_path).exists():
        return {"tracking_enabled": bool(db_path), "tickets": [], "summary": {"settled": 0, "wins": 0, "roi_units": None}}
    limit = max(1, min(1000, int(limit)))
    with _db(db_path) as db:
        items = db.execute(
            "SELECT * FROM oracle_tickets ORDER BY date_utc DESC, category LIMIT ?", (limit,)
        ).fetchall()
        stats = db.execute(
            """SELECT COUNT(*) AS n, SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN status IN ('won','lost') THEN payout_units ELSE 0 END) AS payout
               FROM oracle_tickets WHERE status IN ('won','lost')"""
        ).fetchone()
    n = int(stats["n"] or 0)
    wins = int(stats["wins"] or 0)
    payout = float(stats["payout"] or 0)
    return {
        "tracking_enabled": True,
        "tickets": [
            {"ticket_id": x["ticket_id"], "date_utc": x["date_utc"],
             "category": x["category"], "generated_at_utc": x["generated_at_utc"],
             "combined_odds": x["combined_odds"], "legs": json.loads(x["legs_json"]),
             "status": x["status"], "payout_units": x["payout_units"],
             "settlement_source": x["settlement_source"],
             "settled_at_utc": x["settled_at_utc"]}
            for x in items
        ],
        "summary": {
            "settled": n, "wins": wins,
            "roi_units": round(payout - n, 3) if n else None,
            "roi_pct_flat": round((payout - n) / n * 100, 2) if n else None,
            "method": "Paper simulation at fixed 1 unit/ticket; manual settlement.",
        },
    }


def settle_ticket(db_path, ticket_id, outcomes):
    """CLI-only: outcomes must be one of win/loss/void for each leg."""
    outcomes = [str(x).lower().strip() for x in outcomes]
    if not outcomes or any(x not in {"win", "loss", "void"} for x in outcomes):
        raise ValueError("Use win/loss/void for every leg")
    with _db(db_path) as db:
        row = db.execute("SELECT * FROM oracle_tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if row is None:
            raise ValueError("Ticket not found")
        if row["status"] != "pending":
            raise ValueError("Ticket already settled")
        legs = json.loads(row["legs_json"])
        if len(legs) != len(outcomes):
            raise ValueError("Outcome count must match ticket legs")
        won = all(x != "loss" for x in outcomes)
        payout = 1.0
        if won:
            for leg, result in zip(legs, outcomes):
                payout *= float(leg["odds"]) if result == "win" else 1.0
        else:
            payout = 0.0
        status = "won" if won else "lost"
        db.execute(
            """UPDATE oracle_tickets
               SET status=?,payout_units=?,settlement_source='manual',
                   settled_at_utc=? WHERE ticket_id=?""",
            (status, round(payout, 4), datetime.now(timezone.utc).isoformat(), ticket_id)
        )
    return {"ticket_id": ticket_id, "status": status, "payout_units": round(payout, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual paper ledger settlement; never actual bet execution")
    parser.add_argument("command", choices=["history", "settle"])
    parser.add_argument("--db", required=True, help="SQLite ledger path")
    parser.add_argument("--id", help="Ticket ID, YYYY-MM-DD:single etc")
    parser.add_argument("--outcomes", help="Comma-separated win/loss/void per leg")
    args = parser.parse_args()
    if args.command == "history":
        print(json.dumps(read_history(args.db), indent=2, ensure_ascii=False))
    else:
        if not args.id or not args.outcomes:
            parser.error("--id and --outcomes are required for settle")
        print(json.dumps(settle_ticket(args.db, args.id, args.outcomes.split(",")), indent=2))
