"""Offline prospective paper snapshot tests; external results and odds mocked."""
import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
from tennis_paper_trial import (
    source_lookup, resolve_full_name, match_start_utc,
    build_snapshot, _book_price, MIN_LEAD_MINUTES,
)

NOW = datetime(2026,11,11,7,0,tzinfo=timezone.utc)
MATCH = {
    "id":"espn-99", "state":"pre", "tour":"ATP", "surface":"Hard",
    "p1":"Jannik Sinner", "p2":"Carlos Alcaraz",
    "time":"17:00 UTC", "tournament":"ATP Finals",
}
HISTORY = pd.DataFrame([
    {"source_id":"ATP:one", "tour":"ATP", "surface":"Hard",
     "winner_id":101,"loser_id":202,
     "tourney_date":"2026-11-08","score":"6-4 6-3"}
])
SOURCES={"ATP":[
    {"winner_id":"srcA","winner_name":"Sinner J.",
     "loser_id":"srcB","loser_name":"Alcaraz C."},
]}
MAPPING={
    "ATP:srcA":{"legacy_id":101},
    "ATP:srcB":{"legacy_id":202},
}

class ProspectivePaperTests(unittest.TestCase):
    def test_identity_lookup_exact_and_verified(self):
        lookup=source_lookup(SOURCES,MAPPING)
        self.assertEqual(resolve_full_name("Jannik Sinner","ATP",lookup),101)
        self.assertIsNone(resolve_full_name("Jannik Sinner","WTA",lookup))
        self.assertIsNone(resolve_full_name("Unknown Name","ATP",lookup))

    def test_time_strict_utc(self):
        self.assertEqual(match_start_utc(MATCH,NOW.date()),
                         datetime(2026,11,11,17,tzinfo=timezone.utc))
        self.assertIsNone(match_start_utc({**MATCH,"time":"17:00"},NOW.date()))

    def test_original_frozen_picks_have_no_invented_odds(self):
        result=build_snapshot([MATCH], HISTORY, SOURCES, MAPPING, NOW)
        self.assertEqual(result["predictions_count"],1)
        p=result["predictions"][0]
        self.assertTrue(p["captured_at_utc"] < p["scheduled_start_utc"])
        self.assertGreater(p["p1_probability"],.5)
        self.assertEqual(p["picked_winner_name"],"Jannik Sinner")
        self.assertIsNone(p["odds_decimal"])
        self.assertEqual(p["market_status"],"missing_eligible_h2h_quote")
        self.assertFalse(p["wager_submitted"])

    def test_same_day_results_never_change_picks(self):
        future=pd.DataFrame([{
            "source_id":"ATP:future", "tour":"ATP", "surface":"Hard",
            "winner_id":202,"loser_id":101,
            "tourney_date":"2026-11-11","score":"6-4 6-3"
        }])
        p1=build_snapshot([MATCH],HISTORY,SOURCES,MAPPING,NOW)["predictions"][0]
        p2=build_snapshot([MATCH],pd.concat([HISTORY,future],ignore_index=True),SOURCES,MAPPING,NOW)["predictions"][0]
        self.assertEqual(p1["p1_probability"],p2["p1_probability"])

    def test_started_match_must_be_omitted(self):
        event={**MATCH,"time":"06:55 UTC"}
        result=build_snapshot([event],HISTORY,SOURCES,MAPPING,NOW)
        self.assertEqual(result["predictions_count"],0)
        self.assertEqual(result["skipped"]["late_or_unknown_start"],1)

    def test_completed_match_must_be_omitted(self):
        event={**MATCH,"state":"post"}
        result=build_snapshot([event],HISTORY,SOURCES,MAPPING,NOW)
        self.assertEqual(result["predictions_count"],0)

    def test_unverified_player_prevents_snapshot(self):
        event={**MATCH,"p2":"Unknown Opponent"}
        result=build_snapshot([event],HISTORY,SOURCES,MAPPING,NOW)
        self.assertEqual(result["predictions_count"],0)

    def test_quoted_market_must_be_earlier_and_paired(self):
        record={
            "commence_time":"2026-11-11T17:00:00Z",
            "home_team":"Jannik Sinner","away_team":"Carlos Alcaraz",
            "bookmakers":[{
                "key":"verifiedbook","last_update":"2026-11-11T06:50:00Z",
                "markets":[{
                    "key":"h2h","last_update":"2026-11-11T06:50:00Z",
                    "outcomes":[{"name":"Jannik Sinner","price":1.60},
                                {"name":"Carlos Alcaraz","price":2.35}],
                }]
            }]
        }
        q=_book_price([record],MATCH,"Jannik Sinner",NOW,
                      datetime(2026,11,11,17,tzinfo=timezone.utc))
        self.assertEqual(q["odds_decimal"],1.6)
        self.assertEqual(q["market_status"],"priced")
        stale={**record,"bookmakers":[{
            **record["bookmakers"][0],
            "markets":[{
                **record["bookmakers"][0]["markets"][0],
                "last_update":"2026-11-10T04:00:00Z"
            }]
        }]}
        q2=_book_price([stale],MATCH,"Jannik Sinner",NOW,
                       datetime(2026,11,11,17,tzinfo=timezone.utc))
        self.assertIsNone(q2["odds_decimal"])

    def test_quote_under_140_not_eligible(self):
        record={
            "commence_time":"2026-11-11T17:00:00Z",
            "home_team":"Jannik Sinner","away_team":"Carlos Alcaraz",
            "bookmakers":[{
                "key":"book","last_update":"2026-11-11T06:59:00Z",
                "markets":[{
                    "key":"h2h","outcomes":[{"name":"Jannik Sinner","price":1.30},
                                          {"name":"Carlos Alcaraz","price":3.40}],
                }]
            }]
        }
        found=_book_price([record],MATCH,"Jannik Sinner",NOW,
                          datetime(2026,11,11,17,tzinfo=timezone.utc))
        self.assertEqual(found["market_status"],"below_min_1_40")

if __name__=="__main__":
    unittest.main()
