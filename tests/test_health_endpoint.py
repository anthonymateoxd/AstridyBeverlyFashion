from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


class HealthEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")
        cls.client = cls.A.app.test_client()

    def test_health_reports_connected_camera_without_sensitive_details(self):
        with (
            patch.object(self.A, "fetch_one", return_value={"ok": 1}),
            patch.object(self.A, "latest_camera_frame", object()),
            patch.object(
                self.A,
                "latest_frame_timestamp_monotonic",
                100.0,
            ),
            patch.object(
                self.A.time,
                "monotonic",
                return_value=101.0,
            ),
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["database"], "ok")
        self.assertEqual(payload["camera"], "connected")
        self.assertNotIn("detail", payload)
        self.assertNotIn("camera_source", payload)
        self.assertNotIn("detection_mode", payload)

    def test_health_is_degraded_when_camera_has_no_recent_frame(self):
        with (
            patch.object(self.A, "fetch_one", return_value={"ok": 1}),
            patch.object(self.A, "latest_camera_frame", None),
            patch.object(
                self.A,
                "latest_frame_timestamp_monotonic",
                None,
            ),
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["database"], "ok")
        self.assertEqual(payload["camera"], "disconnected")

    def test_health_database_failure_is_generic_and_returns_503(self):
        with patch.object(
            self.A,
            "fetch_one",
            side_effect=RuntimeError("sensitive-database-detail"),
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(
            payload,
            {
                "status": "error",
                "database": "unavailable",
            },
        )
        self.assertNotIn(
            "sensitive-database-detail",
            response.get_data(as_text=True),
        )


if __name__ == "__main__":
    unittest.main()
