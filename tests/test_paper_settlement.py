"""Offline simulated result settlement, never inferred from predicted outcome."""
import tempfile
import unittest
from datetime import datetime,timezone,timedelta

from tennis_paper_settlement import (
    finished_winner_by_id, match_result, settle_ledger
)
from tennis_paper_ledger import persist

CAPTURE="2026-11-11T06:00:00+00:00"
START="2026-11-11T17:00:00+00:00"
NOW=datetime(2026,11,12,8,tzinfo=timezone.utc)
PRED={
    "fixture_id":"match-101","tour":"ATP","surface":"Hard",
    "p1":"Jannik Sinner","p2":"Carlos Alcaraz",
    "p1_historical_id":101,"p2_historical_id":202,
    "picked_winner_name":"Jannik Sinner",
    "picked_winner_probability":.62,"p1_probability":.62,
    "scheduled_start_utc":START,"captured_at_utc":CAPTURE,
    "market_status":"priced","odds_decimal":1.55,
    "bookmaker":"mockbook","provider_last_update_utc":CAPTURE,
    "model_id":"frozen-v1","paper_only":True,"wager_submitted":False,
}

def event(state="post",winner="Jannik Sinner"):
    return {"events":[{"groupings":[{"competitions":[{
        "id":"match-101","status":{"type":{"state":state}},
        "competitors":[
            {"athlete":{"displayName":"Jannik Sinner"},"winner":winner=="Jannik Sinner"},
            {"athlete":{"displayName":"Carlos Alcaraz"},"winner":winner=="Carlos Alcaraz"},
        ],
    }]}]}]}

class PaperSettlementTests(unittest.TestCase):
    def test_only_final_winner_flag_settles(self):
        self.assertEqual(finished_winner_by_id(event()),{"match-101":"Jannik Sinner"})
        self.assertEqual(finished_winner_by_id(event(state="in")),{})

    def test_no_guess_without_winner_flag(self):
        missing=event()
        for c in missing["events"][0]["groupings"][0]["competitions"][0]["competitors"]:
            c.pop("winner")
        self.assertEqual(finished_winner_by_id(missing),{})

    def test_win_and_loss_paper_units(self):
        self.assertAlmostEqual(match_result(PRED,"Jannik Sinner")["quoted_paper_unit_net"],.55)
        self.assertAlmostEqual(match_result(PRED,"Carlos Alcaraz")["quoted_paper_unit_net"],-1)
        no_price={**PRED,"market_status":"missing_eligible_h2h_quote",
                  "odds_decimal":None,"bookmaker":None,
                  "provider_last_update_utc":None}
        self.assertIsNone(match_result(no_price,"Jannik Sinner")["quoted_paper_unit_net"])

    def test_unrecognized_identity_never_settled(self):
        self.assertIsNone(match_result(PRED,"Unknown Player"))

    def test_immutable_single_settlement_and_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot={"mode":"PROSPECTIVE_RESEARCH_PAPER_ONLY","captured_at_utc":CAPTURE,
                      "predictions":[PRED],"model_id":"frozen-v1"}
            persist(snapshot,tmp,"first-run")
            seen=[]
            def fake_fetch(tour,day):
                seen.append((tour,day.isoformat()))
                return event()
            result=settle_ledger(tmp,NOW,fetch=fake_fetch)
            self.assertEqual(result["settled_this_run"],1)
            self.assertEqual(result["winner_accuracy"],1)
            self.assertEqual(result["priced_settled_count"],1)
            self.assertAlmostEqual(result["quoted_paper_roi_pct"],55)
            again=settle_ledger(tmp,NOW+timedelta(hours=1),fetch=fake_fetch)
            self.assertEqual(again["settled_this_run"],0)
            self.assertEqual(again["settled_total"],1)
            self.assertEqual(len(seen),1)

    def test_missing_result_remains_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot={"mode":"PROSPECTIVE_RESEARCH_PAPER_ONLY","captured_at_utc":CAPTURE,
                      "predictions":[PRED],"model_id":"frozen-v1"}
            persist(snapshot,tmp,"first-run")
            result=settle_ledger(tmp,NOW,fetch=lambda tour,day: event(state="in"))
            self.assertEqual(result["settled_this_run"],0)
            self.assertEqual(result["still_pending"],1)
            self.assertIsNone(result["quoted_paper_roi_pct"])

if __name__=="__main__":
    unittest.main()
