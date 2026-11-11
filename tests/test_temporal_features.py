"""Temporal feature tests: same-day outcomes cannot leak into another fixture."""
import unittest
from datetime import datetime
import pandas as pd
from tennis_temporal import build_temporal_frame, chronological_masks


def row(day, winner, loser, ace, tour="ATP"):
    return {
        "tourney_date": day, "tour": tour,
        "winner_id": winner, "loser_id": loser, "surface": "Hard",
        "winner_rank": 20, "loser_rank": 100,
        "winner_rank_points": 1000, "loser_rank_points": 100,
        "winner_age": 25, "loser_age": 27,
        "winner_ht": 190, "loser_ht": 180,
        "best_of": 3, "tourney_level": "A", "round": "R32",
        "w_ace": ace, "l_ace": 2,
        "w_df": 1, "l_df": 2,
        "w_svpt": 100, "l_svpt": 90,
        "w_1stIn": 60, "l_1stIn": 55,
        "w_1stWon": 40, "l_1stWon": 30,
        "w_2ndWon": 20, "l_2ndWon": 10,
        "target_games_over_225": 1, "target_aces_over_105": int(ace+2>10.5),
        "target_sets_over": 1, "target_both_set": 1, "target_tiebreak": 0,
    }


class TemporalFeaturesTests(unittest.TestCase):
    def test_current_and_same_day_aces_not_used(self):
        games = [
            row("2025-01-01", 10, 20, 50),
            row("2025-01-01", 10, 30, 80),
            row("2025-01-02", 10, 40, 90),
        ]
        pack = build_temporal_frame(pd.DataFrame(games))
        x = pack["X_base"]
        self.assertEqual(x.loc[0, "p1_aces_avg"], 5.0)
        self.assertEqual(x.loc[1, "p1_aces_avg"], 5.0)
        self.assertEqual(x.loc[2, "p1_aces_avg"], 65.0)
        self.assertEqual(x.loc[0, "h2h_total"], 0)

    def test_future_results_do_not_change_past_features(self):
        before = [row("2025-01-01", 10, 20, 8), row("2025-01-02", 10, 30, 9)]
        after = before + [row("2025-02-01", 20, 10, 90)]
        left = build_temporal_frame(pd.DataFrame(before))["X_winner"]
        right = build_temporal_frame(pd.DataFrame(after))["X_winner"]
        pd.testing.assert_frame_equal(left, right.iloc[:2].reset_index(drop=True))

    def test_winner_orientation_is_not_always_one(self):
        games = [row("2025-01-01", 10, 20, 7), row("2025-01-02", 30, 20, 8)]
        pack = build_temporal_frame(pd.DataFrame(games))
        self.assertEqual(pack["y_winner"].tolist(), [1, 0])

    def test_cross_tour_form_does_not_mix(self):
        games = [row("2025-01-01", 10, 20, 80, "ATP"), row("2025-01-02", 10, 20, 7, "WTA")]
        pack = build_temporal_frame(pd.DataFrame(games))
        self.assertEqual(pack["X_base"].iloc[1]["p1_aces_avg"], 5.0)
        self.assertNotIn(10, pack["player_stats"])  # ambiguous raw ID is dropped

    def test_train_calibration_window_temporal(self):
        games = [row(f"2025-01-{i:02d}", (10 if i%2 else 30), 20, 8) for i in range(1,16)]
        p = build_temporal_frame(pd.DataFrame(games))
        train, cal, test, date = chronological_masks(p, "2025-01-13")
        self.assertTrue(train.any() and cal.any() and test.any())
        self.assertFalse((train & cal).any())
        self.assertLess(p["dates"][train].max(), p["dates"][cal].min())


if __name__ == "__main__":
    unittest.main()
