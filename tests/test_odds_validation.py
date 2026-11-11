"""Offline coherence and identity tests for market prices."""
import unittest
from tennis_odds import parse_match_odds, find_value_bets


def event(outcomes_h2h, outcomes_totals, name="Sinner"):
    return [{"id": "match1", "home_team": "Jannik Sinner", "away_team": "Carlos Alcaraz",
             "commence_time": "2026-11-10T15:00:00Z", "sport_key": "tennis_atp",
             "bookmakers": [
               {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": outcomes_h2h},
                                             {"key": "totals", "outcomes": outcomes_totals}]}
             ]}]


class OddsTests(unittest.TestCase):
    def test_refuse_mixed_lines(self):
        odds = parse_match_odds(event([
            {"name": "Jannik Sinner", "price": 1.6},
            {"name": "Carlos Alcaraz", "price": 2.4}], [
            {"name": "Over", "price": 1.8, "point": 22.5},
            {"name": "Under", "price": 2.0, "point": 23.5}]))
        self.assertIsNone(odds[0]["total_point"])
        self.assertEqual(odds[0]["over_total"], 0.0)

    def test_same_market_and_book(self):
        odds = parse_match_odds(event([
            {"name": "Jannik Sinner", "price": 1.6},
            {"name": "Carlos Alcaraz", "price": 2.4}], [
            {"name": "Over", "price": 1.8, "point": 22.5},
            {"name": "Under", "price": 2.0, "point": 22.5}]))
        self.assertEqual(odds[0]["total_point"], 22.5)
        self.assertEqual(odds[0]["book_totals"], "pinnacle")
        self.assertEqual(odds[0]["under_total"], 2.0)

    def test_no_false_value_bet_on_wrong_line(self):
        prices = parse_match_odds(event([
            {"name": "Jannik Sinner", "price": 1.6},
            {"name": "Carlos Alcaraz", "price": 2.4}], [
            {"name": "Over", "price": 1.8, "point": 25.5},
            {"name": "Under", "price": 2.0, "point": 25.5}]))
        picks = find_value_bets({"winner":{"prob":.7,"reliable":False},
                                 "games_over":{"prob":.8,"reliable":True}},
                                prices,"Jannik Sinner","Carlos Alcaraz")
        self.assertFalse(any(x.get("has_value") for x in picks.values()))

    def test_name_mismatch_refused(self):
        prices = parse_match_odds(event([
            {"name": "Jannik Sinner", "price": 1.6},
            {"name": "Carlos Alcaraz", "price": 2.4}], []))
        picks = find_value_bets({"winner":{"prob":.7,"reliable":True}},
                                prices,"John Sinner","Carlos Alcaraz")
        self.assertEqual(picks, {})


if __name__ == "__main__":
    unittest.main()
