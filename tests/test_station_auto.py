from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


class AutomaticStationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")

    def setUp(self):
        A = self.A
        self.saved = {
            "AUTO_INSPECTION_ENABLED": A.AUTO_INSPECTION_ENABLED,
            "AUTO_LAST_RESULT": A.AUTO_LAST_RESULT,
            "AUTO_LAST_ERROR": A.AUTO_LAST_ERROR,
            "AUTO_LAST_CAPTURE_TIME": A.AUTO_LAST_CAPTURE_TIME,
            "AUTO_COOLDOWN_SECONDS": A.AUTO_COOLDOWN_SECONDS,
            "AUTO_GARMENT_ENTER_COVERAGE": A.AUTO_GARMENT_ENTER_COVERAGE,
            "AUTO_GARMENT_CAPTURE_COVERAGE": A.AUTO_GARMENT_CAPTURE_COVERAGE,
            "AUTO_GARMENT_EXIT_COVERAGE": A.AUTO_GARMENT_EXIT_COVERAGE,
            "AUTO_GARMENT_CONFIRM_FRAMES": A.AUTO_GARMENT_CONFIRM_FRAMES,
            "AUTO_GARMENT_EXIT_FRAMES": A.AUTO_GARMENT_EXIT_FRAMES,
            "AUTO_GARMENT_MAX_TRACK_FRAMES": A.AUTO_GARMENT_MAX_TRACK_FRAMES,
            "FRAME_SELECTION_BENCHMARK": A.FRAME_SELECTION_BENCHMARK,
        }
        A.AUTO_INSPECTION_ENABLED = True
        A.AUTO_LAST_RESULT = None
        A.AUTO_LAST_ERROR = None
        A.AUTO_LAST_CAPTURE_TIME = 0.0
        A.AUTO_COOLDOWN_SECONDS = 0.0
        A.AUTO_GARMENT_ENTER_COVERAGE = 0.28
        A.AUTO_GARMENT_CAPTURE_COVERAGE = 0.33
        A.AUTO_GARMENT_EXIT_COVERAGE = 0.12
        A.AUTO_GARMENT_CONFIRM_FRAMES = 3
        A.AUTO_GARMENT_EXIT_FRAMES = 3
        A.AUTO_GARMENT_MAX_TRACK_FRAMES = 16
        A.FRAME_SELECTION_BENCHMARK = False

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(self.A, key, value)

    def _run_worker(self, coverages, stop_after_registrations):
        A = self.A
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        sequence = iter(range(1, len(coverages) + 1))
        coverage_iter = iter(coverages)
        registrations = []

        def latest_frame():
            try:
                seq = next(sequence)
            except StopIteration:
                A.AUTO_INSPECTION_ENABLED = False
                return None, None, -1
            return frame.copy(), float(seq), seq

        def coverage(_frame):
            return next(coverage_iter)

        def register(selected_frame, **kwargs):
            registrations.append((selected_frame.copy(), kwargs))
            if len(registrations) >= stop_after_registrations:
                A.AUTO_INSPECTION_ENABLED = False
            return {
                "code": f"TEST-{len(registrations)}",
                "status": "Aprobado",
                "defect_type": "Sin defecto",
                "garment_token": kwargs.get("garment_token"),
                "ai_decision": "NORMAL",
                "confidence": 1.0,
                "batch_id": 1,
                "batch_position": len(registrations),
            }

        with (
            patch.object(A, "is_ai_capture_mode_active", return_value=False),
            patch.object(A, "get_latest_camera_frame", side_effect=latest_frame),
            patch.object(A, "compute_roi_coverage", side_effect=coverage),
            patch.object(A, "register_inspection_from_frame", side_effect=register),
            patch.object(A, "next_garment_token", side_effect=lambda: f"token-{len(registrations) + 1}"),
            patch.object(A, "timeline_mark"),
            patch.object(A, "reset_timeline"),
            patch.object(A, "log_timeline"),
            patch.object(A, "log_camera_latency"),
            patch.object(A.time, "sleep", return_value=None),
        ):
            A.auto_inspection_worker()

        return registrations

    def test_garment_at_real_production_coverage_triggers_inspection(self):
        registrations = self._run_worker(
            [0.30, 0.34, 0.36, 0.34],
            stop_after_registrations=1,
        )

        self.assertEqual(len(registrations), 1)
        self.assertEqual(registrations[0][1]["source"], "auto")
        self.assertIsNone(self.A.AUTO_LAST_ERROR)
        self.assertIsNotNone(self.A.AUTO_LAST_RESULT)

    def test_same_garment_is_not_registered_twice_before_exit(self):
        registrations = self._run_worker(
            [
                0.30, 0.34, 0.36, 0.34,
                0.36, 0.35,
                0.05, 0.04, 0.03,
                0.30, 0.34, 0.36, 0.34,
            ],
            stop_after_registrations=2,
        )

        self.assertEqual(len(registrations), 2)
        self.assertNotEqual(
            registrations[0][1]["garment_token"],
            registrations[1][1]["garment_token"],
        )


class DeploymentEnvironmentTests(unittest.TestCase):
    def test_automatic_and_guided_runtime_settings_are_forwarded(self):
        compose = (ROOT / "docker-compose.yaml").read_text(
            encoding="utf-8"
        )
        required = {
            "AUTO_GARMENT_CAPTURE_COVERAGE",
            "AUTO_GARMENT_ENTER_COVERAGE",
            "AUTO_GARMENT_EXIT_COVERAGE",
            "AUTO_GARMENT_CONFIRM_FRAMES",
            "AUTO_GARMENT_EXIT_FRAMES",
            "AUTO_GARMENT_MAX_TRACK_FRAMES",
            "AI_GUIDED_PRESENT_COVERAGE",
            "AI_GUIDED_EXIT_COVERAGE",
            "AI_GUIDED_MANUAL_COOLDOWN_SECONDS",
            "CAMERA_RETRY_SECONDS",
            "CAMERA_OPEN_TIMEOUT_MS",
            "CAMERA_READ_TIMEOUT_MS",
        }
        missing = sorted(
            key for key in required
            if f"{key}:" not in compose
        )
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
