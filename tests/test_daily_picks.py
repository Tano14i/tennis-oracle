"""Fast, offline unit tests for the experimental selection engine."""
import unittest
from datetime import datetime, timezone
from tennis_daily_picks import build_daily_products, strict_player_id, names_match


NOW = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)


def fixture(i, state="pre", surface="Hard"):
    return {"p1": "Player" + str(i) + " Alpha" + str(i),
            "p2": "Player" + str(i) + " Beta" + str(i),
            "tour": "ATP", "state": state,
            "surface": surface, "round": "R32", "tournament": "Test Cup"}


def quote(m, odd=1.50, line=22.5, time="2026-11-10T15:00:00Z"):
    return {"home_team": m["p1"], "away_team": m["p2"],
            "commence_time": time, "home_h2h": odd, "away_h2h": 2.8,
            "book_h2h": "testbook", "over_total": 1.8, "under_total": 1.9,
            "total_point": line, "book_totals": "testbook"}


def candidates(n=6, odds=1.5, line=22.5):
    matches = [fixture(i) for i in range(n)]
    prices = [quote(m, odd=odds, line=line) for m in matches]
    stats = {}
    surf = {}
    ids = {}
    for i, m in enumerate(matches):
        for j, name in enumerate((m["p1"], m["p2"])):
            pid = i * 2 + j + 1
            ids[name] = pid
            stats[pid] = {"p_matches": 100}
            surf[pid] = {"n_hard": 15}
    def predict(*args):
        return {"winner": {"prob": 0.71, "reliable": True},
                "games_over": {"prob": 0.60, "reliable": True}}
    report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
    return report, matches, prices, stats, surf, ids, predict


class PicksTests(unittest.TestCase):
    def test_four_products(self):
        report, *_ = candidates()
        self.assertEqual(report["mode"], "experimental_research_only")
        for key in ("single", "three_by_three", "multiple_5", "bomb_7"):
            self.assertEqual(report["products"][key]["status"], "experimental", key)
        self.assertEqual(len(report["products"]["three_by_three"]["legs"]), 3)
        self.assertGreaterEqual(report["products"]["single"]["combined_odds"], 1.4)
        self.assertGreaterEqual(report["products"]["multiple_5"]["combined_odds"], 5)
        self.assertGreaterEqual(report["products"]["bomb_7"]["combined_odds"], 7)
        for product in report["products"].values():
            self.assertEqual(len({p["fixture_id"] for p in product["legs"]}), len(product["legs"]))

    def test_no_odds_no_tickets(self):
        _, matches, _, stats, surf, ids, predict = candidates()
        report = build_daily_products(matches, [], predict, ids.get, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)
        self.assertTrue(all(x["status"] == "unavailable" for x in report["products"].values()))

    def test_no_started_matches(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        for x in prices:
            x["commence_time"] = "2026-11-10T07:00:00Z"
        report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)

    def test_no_scheduled_state(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        for x in matches:
            x["state"] = "in"
        report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)

    def test_model_line_mismatch(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        for p in prices:
            p["total_point"] = 24.5
        report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
        # Game market must not be added when bookmaker offers different line.
        self.assertEqual(report["candidate_count"], len(matches))
        self.assertTrue(all(x["market"] == "winner" for x in report["products"]["multiple_5"]["legs"]))

    def test_3x3_does_not_force_large_combined_price(self):
        report, *_ = candidates(3, odds=1.8)
        self.assertEqual(report["products"]["three_by_three"]["status"], "unavailable")

    def test_no_false_match_with_wrong_bookmaker_identity(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        for p in prices:
            p["home_team"] = "Different Person"
        report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)

    def test_low_sample_rejected(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        for s in stats.values():
            s["p_matches"] = 4
        report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)

    def test_unmatched_player_rejected(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        report = build_daily_products(matches, prices, predict, lambda name: None, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)

    def test_unverified_odd_rejected(self):
        _, matches, prices, stats, surf, ids, predict = candidates()
        for p in prices:
            p["home_h2h"] = 0
            p["away_h2h"] = 0
            p["over_total"] = 0
            p["under_total"] = 0
        report = build_daily_products(matches, prices, predict, ids.get, stats, surf, now=NOW)
        self.assertEqual(report["candidate_count"], 0)

    def test_strict_lookup_refuses_ambiguous_surname(self):
        lookup = {"jannik sinner": 11, "john sinner": 22, "carlos alcaraz": 33}
        self.assertIsNone(strict_player_id("Sinner", lookup))
        self.assertIsNone(strict_player_id("J. Sinner", lookup))
        self.assertEqual(strict_player_id("Jannik Sinner", lookup), 11)
        self.assertTrue(names_match("Jannik Sinner", "J. Sinner"))
        self.assertFalse(names_match("Jannik Sinner", "John Sinner"))


if __name__ == "__main__":
    unittest.main()
