"""Contrato de salud compatible con AWS: disponibilidad web separada de cámara e IA."""
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

    def _consultar(self, *, connected):
        with (
            patch.object(
                self.A,
                "fetch_one",
                side_effect=[{"c": 539}, None],
            ) as fetch_one,
            patch.object(self.A, "fetch_all", return_value=[]),
            patch.object(
                self.A,
                "summarize_patchcore_readiness",
                return_value={
                    "detection_mode": "patchcore",
                    "model_ready": False,
                    "ready_patchcore_models": 0,
                },
            ),
            patch.object(
                self.A,
                "latest_camera_frame",
                object() if connected else None,
            ),
            patch.object(
                self.A,
                "latest_frame_timestamp_monotonic",
                100.0 if connected else None,
            ),
            patch.object(self.A.time, "monotonic", return_value=101.0),
        ):
            response = self.client.get("/health")
        self.assertEqual(fetch_one.call_count, 2)
        return response

    def test_health_reports_patchcore_and_connected_camera(self):
        response = self._consultar(connected=True)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["database"], self.A.DB_NAME)
        self.assertEqual(payload["camera"], "connected")
        self.assertEqual(payload["detection_mode"], "patchcore")
        self.assertEqual(payload["inspections"], 539)
        self.assertNotIn("detail", payload)
        self.assertNotIn("rtsp://", response.get_data(as_text=True).lower())

    def test_camera_disconnected_does_not_mean_web_failure(self):
        response = self._consultar(connected=False)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["camera"], "disconnected")
        self.assertEqual(payload["detection_mode"], "patchcore")
        self.assertFalse(payload["model_ready"])

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
