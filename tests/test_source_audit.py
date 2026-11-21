import unittest
from datetime import datetime, timezone, timedelta
from tennis_source_audit import audit_valuebet_csv

HEAD = "match_id;date;genre;surface;joueur1;joueur1_id;joueur2;joueur2_id;vainqueur_id;score;cote1_ouverture;cote2_ouverture;cote1_cloture;cote2_cloture\n"
NOW = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)


def line(id="1", date="2026-11-09 16:00:00", winner="100", p1="100", p2="200", score="6-3 6-4", odd1="1.60", odd2="2.25"):
    return f"{id};{date};atp;dur;First Player;{p1};Second Player;{p2};{winner};{score};{odd1};{odd2};1.50;2.40\n"


class AuditTests(unittest.TestCase):
    def test_legitimate_past_match_with_pair_odds(self):
        r = audit_valuebet_csv(HEAD + line(), now=NOW)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["closing_odds_pairs"], 1)
        self.assertEqual(r["by_tour"]["atp"], 1)

    def test_future_completed_result_quarantined(self):
        r = audit_valuebet_csv(HEAD + line(date="2026-11-10 18:30:00"), now=NOW)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["future_settled"], 1)
        self.assertEqual(len(r["future_settled_examples"]), 1)

    def test_wrong_winner_and_duplicate_match_id(self):
        r = audit_valuebet_csv(HEAD + line(winner="404") + line(), now=NOW)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["invalid_winner"], 1)
        self.assertEqual(r["duplicate_match_ids"], 1)

    def test_missing_columns_rejected(self):
        r = audit_valuebet_csv("date;score\n2026-11-09;6-3\n", now=NOW)
        self.assertFalse(r["accepted"])
        self.assertIn("match_id", r["missing_columns"])

    def test_invalid_odds_not_accepted_as_prices(self):
        r = audit_valuebet_csv(HEAD + line(odd1="1.0", odd2="2.0"), now=NOW)
        self.assertEqual(r["opening_odds_pairs"], 0)
        self.assertGreater(r["invalid_odds_pairs"], 0)

    def test_source_uses_utc_not_naive_clock(self):
        r = audit_valuebet_csv(HEAD + line(date="2026-11-10 07:59:59"), now=NOW)
        self.assertTrue(r["accepted"])

    def test_requires_timezone(self):
        with self.assertRaises(ValueError):
            audit_valuebet_csv(HEAD + line(), now=datetime(2026,11,10,8,0))


if __name__ == "__main__":
    unittest.main()
