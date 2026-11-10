"""Offline tests for idempotent ticket snapshots and manual settlement."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tennis_daily_picks import build_daily_products
from tennis_daily_tracker import read_history, record_report, settle_ticket
from test_daily_picks import candidates


class TrackerTests(unittest.TestCase):
    def test_first_quote_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "tracker.sqlite")
            report, *_ = candidates(6)
            self.assertEqual(record_report(db, report), 4)
            history = read_history(db)
            self.assertEqual(len(history["tickets"]), 4)
            first = next(x for x in history["tickets"] if x["category"] == "single")
            modified = dict(report)
            changed = dict(report["products"]["single"])
            changed["combined_odds"] = 99
            modified["products"] = {**report["products"], "single": changed}
            self.assertEqual(record_report(db, modified), 0)
            updated = read_history(db)
            s = next(x for x in updated["tickets"] if x["category"] == "single")
            self.assertEqual(s["combined_odds"], first["combined_odds"])

    def test_manual_paper_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "tracker.sqlite")
            report, *_ = candidates(6)
            record_report(db, report)
            ticket = next(x for x in read_history(db)["tickets"] if x["category"] == "single")
            done = settle_ticket(db, ticket["ticket_id"], ["win"])
            self.assertEqual(done["status"], "won")
            self.assertGreater(done["payout_units"], 1)
            stats = read_history(db)["summary"]
            self.assertEqual(stats["settled"], 1)
            self.assertEqual(stats["wins"], 1)
            with self.assertRaises(ValueError):
                settle_ticket(db, ticket["ticket_id"], ["loss"])

    def test_wrong_outcome_count_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "tracker.sqlite")
            report, *_ = candidates(6)
            record_report(db, report)
            multi = next(x for x in read_history(db)["tickets"] if x["category"] == "multiple_5")
            with self.assertRaises(ValueError):
                settle_ticket(db, multi["ticket_id"], ["win"])
            self.assertEqual(read_history(db)["summary"]["settled"], 0)


if __name__ == "__main__":
    unittest.main()
