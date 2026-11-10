"""Gate refuses stale or legacy model artifacts. No network dependencies."""
import unittest
from datetime import datetime, timezone
from tennis_model_gate import daily_model_gate

TODAY = datetime(2026, 11, 10, 8, tzinfo=timezone.utc)
FRESH = {"feature_engine": "temporal-v2",
         "training_data_latest_date": "2026-11-09",
         "trained_markets": ["winner", "games_over", "aces_over"]}


class ModelGateTests(unittest.TestCase):
    def test_new_model_with_fresh_data_can_generate_research(self):
        ok, _ = daily_model_gate(FRESH, TODAY)
        self.assertTrue(ok)

    def test_stale_archive_refused(self):
        ok, _ = daily_model_gate({**FRESH, "training_data_latest_date": "2026-06-01"}, TODAY)
        self.assertFalse(ok)

    def test_legacy_and_missing_refused(self):
        self.assertFalse(daily_model_gate(None, TODAY)[0])
        self.assertFalse(daily_model_gate({**FRESH, "feature_engine": "legacy"}, TODAY)[0])
        self.assertFalse(daily_model_gate({**FRESH, "trained_markets": ["winner"]}, TODAY)[0])

    def test_future_data_refused(self):
        self.assertFalse(daily_model_gate({**FRESH, "training_data_latest_date": "2026-11-11"}, TODAY)[0])


if __name__ == "__main__":
    unittest.main()
