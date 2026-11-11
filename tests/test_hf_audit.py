import unittest
from datetime import datetime,timezone
from tennis_hf_audit import audit_csv,download
NOW=datetime(2026,11,10,8,tzinfo=timezone.utc)
H=("match_id,start_time_utc,winner_id,loser_id,winner_name,loser_name,score,"
   "winner_close_prob,winner_close_fair_odds,winner_best_close_odds,loser_best_close_odds\n")
def row(i,date="2026-11-08T14:00:00Z",opp="bbbb0000",prob=".6"):
    return f"{i},{date},aaaa0000,{opp},Alice,Bob,6-3 6-4,{prob},1.667,1.8,2.1\n"
class SourceTests(unittest.TestCase):
    def test_good(self):
        r=audit_csv(H+"".join(row(i) for i in range(120)),NOW)
        self.assertTrue(r["accepted"])
    def test_future_result(self):
        s=H+"".join(row(i) for i in range(119))+row(119,"2026-11-11T14:00:00Z")
        self.assertEqual(audit_csv(s,NOW)["future_settled"],1)
        self.assertFalse(audit_csv(s,NOW)["accepted"])
    def test_duplicate_and_wrong_players(self):
        s=H+"".join(row(1,opp="aaaa0000") for i in range(120))
        r=audit_csv(s,NOW)
        self.assertGreater(r["duplicates"],0)
        self.assertGreater(r["invalid_players"],0)
        self.assertFalse(r["accepted"])
    def test_invalid_probability(self):
        s=H+"".join(row(i,prob=".99") for i in range(120))
        self.assertFalse(audit_csv(s,NOW)["accepted"])
    def test_missing_columns(self):
        self.assertFalse(audit_csv("match_id,winner_id\n1,a\n",NOW)["accepted"])
    def test_only_allowlisted_urls(self):
        with self.assertRaises(ValueError):
            download("evil.csv")
