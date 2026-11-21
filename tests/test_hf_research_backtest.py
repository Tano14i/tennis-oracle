"""No future closing odds in model features; controlled offline shape checks."""
import unittest
from datetime import datetime, timezone
from tennis_hf_research_backtest import (
    prepare_rows, evaluate_probabilities, fit_and_evaluate, INFER_FIELDS
)

NOW = datetime(2026, 11, 10, 19, tzinfo=timezone.utc)

def hf(i, when="2026-11-08T15:00:00Z", winner="srcA", loser="srcB"):
    return {
        "match_id": f"t{i}", "start_time_utc": when,
        "winner_id": winner, "loser_id": loser,
        "winner_name": "A", "loser_name": "B",
        "score": "6-3 6-4", "best_of": "3", "surface": "Hard",
        "tourney_date": "20261030",
        "winner_rank": "10", "loser_rank": "30",
        "winner_close_prob": ".81",
        "winner_best_close_odds": "1.20", "loser_best_close_odds": "4.5",
        "w_ace": "8", "l_ace": "3", "w_svpt": "70", "l_svpt": "70",
    }

MAP = {
    "ATP:srcA": {"tour": "ATP", "source_id": "srcA", "legacy_id": 5},
    "ATP:srcB": {"tour": "ATP", "source_id": "srcB", "legacy_id": 10},
}

class BenchmarkTests(unittest.TestCase):
    def test_close_data_never_enters_features(self):
        df, closings, counts = prepare_rows({"ATP": [hf(1)]}, MAP, NOW)
        self.assertEqual(counts["eligible"], 1)
        self.assertEqual(df.iloc[0]["tourney_date"].date().isoformat(), "2026-11-08")
        self.assertNotIn("winner_close_prob", df.columns)
        self.assertNotIn("winner_best_close_odds", df.columns)
        self.assertEqual(closings["ATP:t1"], .81)
        self.assertEqual(tuple(df.columns), INFER_FIELDS)

    def test_unverified_identity_dropped(self):
        df, _, counts = prepare_rows({"ATP": [hf(1), hf(2, loser="unverified")]}, MAP, NOW)
        self.assertEqual(len(df), 1)
        self.assertEqual(counts["excluded_unmapped"], 1)

    def test_future_scored_match_dropped(self):
        with self.assertRaises(ValueError):
            prepare_rows({"ATP": [hf(1, when="2026-11-11T15:00:00Z")]}, MAP, NOW)

    def test_evaluation_truth_orientation(self):
        scores = evaluate_probabilities([1, 0], [.8, .2])
        self.assertEqual(scores["accuracy"], 1)
        self.assertAlmostEqual(scores["brier"], .04)

    def test_not_enough_data_fails_before_training(self):
        df, closing, _ = prepare_rows({"ATP": [hf(1)]}, MAP, NOW)
        with self.assertRaises(ValueError):
            fit_and_evaluate(df, closing, n_estimators=2)

if __name__=="__main__":
    unittest.main()
