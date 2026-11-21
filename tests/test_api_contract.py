"""API smoke checks that never contact real tennis or bookmaker services."""
import unittest
from unittest.mock import patch

import tennis_api


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.client = tennis_api.app.test_client()

    def test_daily_picks_fails_closed_without_models(self):
        with patch.object(tennis_api, "MODELS_READY", False):
            response = self.client.get("/daily_picks")
        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "experimental_research_only")
        for ticket in payload["products"].values():
            self.assertEqual(ticket["status"], "unavailable")

    def test_health_explains_readiness(self):
        with patch.object(tennis_api, "MODELS_READY", False):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["models_ready"])
        self.assertFalse(payload["daily_picks_ready"])
        self.assertTrue(isinstance(payload["daily_picks_note"], str))


if __name__ == "__main__":
    unittest.main()
