"""Small end-to-end chronological model fit (offline synthetic data)."""
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tennis_temporal import build_temporal_frame, temporal_model_sets, chronological_masks


def _game(i):
    date = (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
    winner = 10 if i % 3 else 20
    loser = 20 if winner == 10 else 10
    return {
        "tourney_date": date, "tour": "ATP",
        "winner_id": winner, "loser_id": loser, "surface": "Hard",
        "best_of": 3, "winner_rank": 10 + (i % 23), "loser_rank": 50 + (i % 17),
        "winner_rank_points": 950, "loser_rank_points": 250,
        "winner_age": 27, "loser_age": 26, "winner_ht": 190, "loser_ht": 180,
        "tourney_level": "A", "round": "R32", "w_ace": i % 14,
        "l_ace": i % 7, "w_df": 2, "l_df": 3,
        "w_svpt": 80, "l_svpt": 80,
        "w_1stIn": 48, "l_1stIn": 50,
        "w_1stWon": 36, "l_1stWon": 32,
        "w_2ndWon": 18, "l_2ndWon": 15,
        "target_games_over_225": int(i % 4 < 2),
        "target_aces_over_105": int(i % 14 > 6),
        "target_sets_over": int(i % 3 == 0),
        "target_both_set": int(i % 3 == 0),
        "target_tiebreak": int(i % 5 == 0),
    }


class TrainerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_temporal_frame(pd.DataFrame(_game(i) for i in range(330)))
        cls.train, cls.cal, cls.test, _ = chronological_masks(cls.pack, "2023-11-15", .28)

    def test_chronological_split_fit_and_predict(self):
        models, report = temporal_model_sets(self.pack, self.train, self.cal, n_estimators=8)
        self.assertIn("winner", models)
        self.assertIn("games_over", models)
        self.assertTrue(self.test.sum() > 0)
        p = models["winner"].predict_proba(self.pack["X_winner"].loc[self.test])[:,1]
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(report["winner"]["train_samples"], 200)
        self.assertGreater(report["winner"]["calibration_samples"], 80)

    def test_overlapping_intervals_refused(self):
        with self.assertRaises(ValueError):
            temporal_model_sets(self.pack, self.train, self.train, n_estimators=4)


if __name__ == "__main__":
    unittest.main()
