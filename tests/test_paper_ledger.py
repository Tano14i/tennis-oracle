"""Prospective first-snapshot ledger tests; no API or money movements."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tennis_paper_ledger import accepted_prediction, persist

CAPTURE="2026-11-11T06:00:00+00:00"
START="2026-11-11T17:00:00+00:00"

def pred(price=None,status="missing_eligible_h2h_quote"):
    return {
        "fixture_id":"123", "tour":"ATP", "surface":"Hard",
        "p1":"Jannik Sinner", "p2":"Carlos Alcaraz",
        "p1_historical_id":101,"p2_historical_id":202,
        "picked_winner_name":"Jannik Sinner",
        "picked_winner_probability":.62,"p1_probability":.62,
        "scheduled_start_utc":START,"captured_at_utc":CAPTURE,
        "market_status":status,"odds_decimal":price,
        "bookmaker":"examplebook" if price else None,
        "provider_last_update_utc":CAPTURE if price else None,
        "paper_only":True,"wager_submitted":False,
    }

def snapshot(proposal):
    return {"mode":"PROSPECTIVE_RESEARCH_PAPER_ONLY",
            "captured_at_utc":CAPTURE,"model_id":"frozen-v1",
            "predictions":[proposal]}

class LedgerTests(unittest.TestCase):
    def test_reject_postmatch_and_fake_prices(self):
        self.assertFalse(accepted_prediction({**pred(),"captured_at_utc":START}))
        self.assertFalse(accepted_prediction({**pred(),"market_status":"priced"}))
        self.assertFalse(accepted_prediction({**pred(),"wager_submitted":True}))
        self.assertFalse(accepted_prediction({**pred(price=1.30,status="priced")}))
        self.assertFalse(accepted_prediction({**pred(),"p1_historical_id":202}))
        self.assertFalse(accepted_prediction({**pred(),"fixture_id":"../../bad", "tour":"ITF"}))

    def test_valid_paper_without_odds(self):
        self.assertTrue(accepted_prediction(pred()))
        self.assertTrue(accepted_prediction(pred(1.55,"priced")))

    def test_keep_first_snapshot_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=persist(snapshot(pred()),tmp,"run-one")
            self.assertEqual(first["new_first_snapshots"],1)
            target=Path(tmp)/"paper_predictions"/"2026-11-11"/"ATP_123.json"
            raw=target.read_text()
            after=persist(snapshot(pred(1.8,"priced")),tmp,"run-two")
            self.assertEqual(after["new_first_snapshots"],0)
            self.assertEqual(after["previously_captured"],1)
            self.assertEqual(target.read_text(),raw)
            self.assertIsNone(json.loads(raw)["prediction"]["odds_decimal"])

    def test_attempts_are_immutable_even_without_picks(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist(snapshot({**pred(),"wager_submitted":True}),tmp,"run-1")
            path=Path(tmp)/"paper_attempts"/"2026-11-11"/"run-1.json"
            self.assertTrue(path.exists())
            self.assertFalse(list((Path(tmp)/"paper_predictions").glob("**/*.json"))
                             if (Path(tmp)/"paper_predictions").exists() else True)
            with self.assertRaises(ValueError):
                persist(snapshot(pred()),tmp,"run-1")

    def test_price_provenance_checks(self):
        late={**pred(1.55,"priced"),"provider_last_update_utc":"2026-11-11T07:00:00+00:00"}
        self.assertFalse(accepted_prediction(late))
        stale={**pred(1.55,"priced"),"provider_last_update_utc":"2026-11-10T02:00:00+00:00"}
        self.assertFalse(accepted_prediction(stale))

if __name__ == "__main__":
    unittest.main()
