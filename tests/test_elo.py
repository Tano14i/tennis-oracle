"""Elo's pre-match probabilities may never use same-day or future outcomes."""
import unittest
import pandas as pd
from tennis_elo import pre_match_elo, win_probability, elo_configuration


def match(idx, day, wid=10, lid=20, surface="Hard", tour="ATP"):
    return {"source_id": str(idx), "tourney_date": day, "winner_id": wid,
            "loser_id": lid, "surface": surface, "tour": tour}


class HistoricalEloTests(unittest.TestCase):
    def test_two_new_players_have_equal_pre_match_probability(self):
        result = pre_match_elo(pd.DataFrame([match(1, "2026-01-01")]))
        self.assertAlmostEqual(result.iloc[0]["elo_p1_probability"], .5)
        self.assertEqual(result.iloc[0]["y_p1_wins"], 1)

    def test_opponent_order_does_not_reveal_winner(self):
        a = pre_match_elo(pd.DataFrame([match(1, "2026-01-01", wid=20, lid=10)]))
        self.assertAlmostEqual(a.iloc[0]["elo_p1_probability"], .5)
        self.assertEqual(a.iloc[0]["y_p1_wins"], 0)

    def test_day_two_observes_day_one(self):
        data = pd.DataFrame([
            match(1, "2026-01-01"),
            match(2, "2026-01-02"),
        ])
        p = pre_match_elo(data)
        self.assertGreater(p.iloc[1]["elo_p1_probability"], .5)
        self.assertEqual(p.iloc[1]["p1_global_matches_before"], 1)

    def test_same_day_order_independent(self):
        a = match(1, "2026-01-01", wid=10, lid=20)
        b = match(2, "2026-01-01", wid=20, lid=10)
        c = match(3, "2026-01-02", wid=10, lid=20)
        p = pre_match_elo(pd.DataFrame([a,b,c])).set_index("source_id")
        q = pre_match_elo(pd.DataFrame([b,a,c])).set_index("source_id")
        self.assertAlmostEqual(p.loc["1", "elo_p1_probability"], .5)
        self.assertAlmostEqual(p.loc["2", "elo_p1_probability"], .5)
        self.assertAlmostEqual(p.loc["3", "elo_p1_probability"], q.loc["3", "elo_p1_probability"])

    def test_no_future_leakage_even_when_rows_shuffled(self):
        early = pd.DataFrame([
            match(1, "2026-01-01"),
            match(2, "2026-01-03", wid=20, lid=10),
        ])
        future = pd.concat([
            pd.DataFrame([match(3, "2027-03-01", wid=20, lid=10)]),
            early
        ],ignore_index=True)
        a=pre_match_elo(early).set_index("source_id")
        b=pre_match_elo(future).set_index("source_id")
        for mid in ["1","2"]:
            self.assertAlmostEqual(a.loc[mid,"elo_p1_probability"], b.loc[mid,"elo_p1_probability"])

    def test_surface_independence(self):
        data=pd.DataFrame([
            match(1, "2026-01-01", surface="Hard"),
            match(2, "2026-01-02", surface="Hard"),
            match(3, "2026-01-03", surface="Clay"),
        ])
        result=pre_match_elo(data)
        self.assertEqual(result.iloc[2]["p1_surface_matches_before"],0)
        self.assertEqual(result.iloc[2]["elo_surface_weight"],0)
        self.assertGreater(result.iloc[2]["elo_global_probability"],.5)

    def test_tour_identity_separation(self):
        data=pd.DataFrame([
            match(1, "2026-01-01", tour="ATP"),
            match(2, "2026-01-02", tour="WTA"),
        ])
        result=pre_match_elo(data)
        self.assertAlmostEqual(result.iloc[1]["elo_p1_probability"], .5)

    def test_missing_surface_fails_closed(self):
        with self.assertRaises(ValueError):
            pre_match_elo(pd.DataFrame([match(1, "2026-01-01", surface="Unknown")]))

    def test_reject_duplicate_match_id(self):
        with self.assertRaises(ValueError):
            pre_match_elo(pd.DataFrame([match(1,"2026-01-01"),match(1,"2026-01-02")]))

    def test_fixed_parameters_stated(self):
        self.assertFalse(elo_configuration()["same_day_results_used_for_prematch"])
        self.assertFalse(elo_configuration()["parameters_tuned_on_2026_test"])

    def test_probability_symmetric(self):
        self.assertAlmostEqual(win_probability(1600,1500)+win_probability(1500,1600),1)

if __name__ == "__main__":
    unittest.main()
