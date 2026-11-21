"""Offline identity reconciliation tests; no external data download."""
import unittest
from datetime import datetime, timezone
from tennis_hf_crosswalk import (
    source_signature, historical_signatures, build_crosswalk, score_key
)

NOW = datetime(2026, 11, 10, 19, tzinfo=timezone.utc)

def hf(w="Sinner J.", l="Alcaraz C.", wid="new1", lid="new2",
       date="2025-03-19T15:00:00Z", score="6-4 6-3"):
    return {"winner_name": w, "loser_name": l, "winner_id": wid,
            "loser_id": lid, "start_time_utc": date, "score": score}

def old(w="Jannik Sinner", l="Carlos Alcaraz", wid="101", lid="202",
        tournament_date="20250317", score="6-4 6-3"):
    return {"winner_name": w, "loser_name": l, "winner_id": wid,
            "loser_id": lid, "tourney_date": tournament_date, "score": score}

def run(hf_rows, old_rows, tour="ATP"):
    return build_crosswalk({tour: hf_rows}, {tour: old_rows}, now=NOW)

class CrosswalkTests(unittest.TestCase):
    def test_surname_and_multi_initials(self):
        self.assertEqual(source_signature("De Minaur A."), ("de minaur", "a"))
        self.assertEqual(source_signature("Cerundolo J. M."), ("cerundolo", "jm"))
        self.assertIn(("de minaur","a"),historical_signatures("Alex de Minaur"))
        self.assertIn(("cerundolo","jm"),historical_signatures("Juan Manuel Cerundolo"))
        self.assertIsNone(source_signature("Alcaraz"))  # surname alone is unsafe

    def test_overlap_window_uses_tournament_start_not_exact_day(self):
        result=run([hf()], [old()])
        self.assertEqual(result["independent_exact_fixture_overlaps"],1)
        self.assertEqual(result["overlap_verified_players"],2)
        self.assertEqual(result["verified_crosswalk"]["ATP:new1"]["legacy_id"],101)

    def test_name_only_does_not_verify(self):
        result=run([hf(score="6-0 6-0")],[old()])
        self.assertEqual(result["overlap_verified_players"],0)
        self.assertEqual(result["name_only_unverified_players"],2)

    def test_homonyms_quarantined(self):
        game=old()
        other=old(wid="303",w="James Sinner", l="Fritz Taylor",
                  lid="404",score="6-2 6-3")
        result=run([hf()],[game,other])
        self.assertNotIn("ATP:new1",result["verified_crosswalk"])
        self.assertEqual(result["name_candidates_after_one_to_one"],1)

    def test_two_provider_ids_same_historical_id_rejected(self):
        new_matches=[hf(),hf(wid="another_sinner",date="2025-03-21T10:00:00Z",
                             score="6-4 6-3")]
        result=run(new_matches,[old()])
        self.assertNotIn("ATP:new1",result["verified_crosswalk"])
        self.assertNotIn("ATP:another_sinner",result["verified_crosswalk"])
        self.assertGreater(result["name_conflicts_removed"],0)

    def test_score_orientation_and_window_must_match(self):
        result=run([hf(date="2025-04-22T15:00:00Z")],[old()])
        self.assertEqual(result["independent_exact_fixture_overlaps"],0)
        second=run([hf()],[old(w="Carlos Alcaraz",l="Jannik Sinner",wid="202",lid="101")])
        self.assertEqual(second["independent_exact_fixture_overlaps"],0)

    def test_tour_ids_are_isolated(self):
        result=build_crosswalk(
            {"ATP":[hf()],"WTA":[hf(w="Sinner J.",l="Alcaraz C.")]},
            {"ATP":[old()],"WTA":[old(wid="7",lid="8")]},now=NOW)
        self.assertEqual(result["overlap_verified_players"],4)
        self.assertEqual(result["verified_crosswalk"]["WTA:new1"]["legacy_id"],7)
        self.assertEqual(result["verified_crosswalk"]["ATP:new1"]["legacy_id"],101)

    def test_retirement_is_not_match_evidence(self):
        self.assertIsNone(score_key("6-4 2-1 RET"))
        result=run([hf(score="6-4 2-1 RET")],[old(score="6-4 2-1 RET")])
        self.assertEqual(result["independent_exact_fixture_overlaps"],0)

    def test_newer_unmatched_2026_kept_out_of_verified(self):
        prior=hf()
        latest=hf(date="2026-11-08T15:00:00Z",score="6-1 6-2")
        result=run([prior,latest],[old()])
        self.assertEqual(result["after_2026_06_01"]["ATP"]["matches"],1)
        self.assertEqual(result["after_2026_06_01"]["ATP"]["both_overlap_verified"],1)

if __name__=="__main__":
    unittest.main()
