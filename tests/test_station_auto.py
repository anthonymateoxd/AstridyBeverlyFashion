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
            "AUTO_THREAD": A.AUTO_THREAD,
            "MANUAL_INSPECTION_IN_PROGRESS": A.MANUAL_INSPECTION_IN_PROGRESS,
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
        A.AUTO_THREAD = None
        A.MANUAL_INSPECTION_IN_PROGRESS = False
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

    def test_manual_inspection_is_blocked_while_automatic_mode_is_active(self):
        A = self.A
        client = A.app.test_client()

        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = A.ROLE_ADMIN

        user = {
            "id": 1,
            "username": "audit-admin",
            "role": A.ROLE_ADMIN,
            "active": 1,
        }

        A.AUTO_INSPECTION_ENABLED = True

        with (
            patch.object(A, "fetch_one", return_value=user),
            patch.object(A, "read_camera_frame") as read_camera_frame,
            patch.object(A, "register_inspection_from_frame") as register,
        ):
            response = client.post("/api/station/manual")

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertIn("modo automático", payload["message"].lower())
        read_camera_frame.assert_not_called()
        register.assert_not_called()

    def test_start_does_not_create_second_worker_when_already_active(self):
        A = self.A
        client = A.app.test_client()

        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = A.ROLE_ADMIN

        user = {
            "id": 1,
            "username": "audit-admin",
            "role": A.ROLE_ADMIN,
            "active": 1,
        }
        batch = {
            "id": 1,
            "processed_quantity": 0,
            "planned_quantity": 5,
        }

        class FakeThread:
            def __init__(self, *args, **kwargs):
                self.started = False

            def start(self):
                self.started = True

            def is_alive(self):
                return self.started

        A.AUTO_INSPECTION_ENABLED = False
        A.AUTO_THREAD = None

        with (
            patch.object(A, "fetch_one", return_value=user),
            patch.object(A, "get_active_batch", return_value=batch),
            patch.object(A, "is_ai_capture_mode_active", return_value=False),
            patch.object(A, "CAMERA_CONNECTED", True),
            patch.object(A.threading, "Thread", wraps=FakeThread) as thread_cls,
        ):
            first = client.post("/api/station/auto/start")
            second = client.post("/api/station/auto/start")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(first.get_json()["ok"])
        self.assertTrue(second.get_json()["ok"])
        self.assertEqual(thread_cls.call_count, 1)

    def test_start_waits_for_previous_worker_to_finish(self):
        A = self.A
        client = A.app.test_client()

        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = A.ROLE_ADMIN

        user = {
            "id": 1,
            "username": "audit-admin",
            "role": A.ROLE_ADMIN,
            "active": 1,
        }
        batch = {
            "id": 1,
            "processed_quantity": 0,
            "planned_quantity": 5,
        }

        class StillStoppingThread:
            def is_alive(self):
                return True

        A.AUTO_INSPECTION_ENABLED = False
        A.AUTO_THREAD = StillStoppingThread()

        with (
            patch.object(A, "fetch_one", return_value=user),
            patch.object(A, "get_active_batch", return_value=batch),
            patch.object(A, "is_ai_capture_mode_active", return_value=False),
            patch.object(A, "CAMERA_CONNECTED", True),
            patch.object(A.threading, "Thread") as thread_cls,
        ):
            response = client.post("/api/station/auto/start")

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertIn("deteniendo", payload["message"].lower())
        thread_cls.assert_not_called()

    def test_start_is_blocked_while_manual_inspection_is_in_progress(self):
        A = self.A
        client = A.app.test_client()

        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = A.ROLE_ADMIN

        user = {
            "id": 1,
            "username": "audit-admin",
            "role": A.ROLE_ADMIN,
            "active": 1,
        }
        batch = {
            "id": 1,
            "processed_quantity": 0,
            "planned_quantity": 5,
        }

        A.AUTO_INSPECTION_ENABLED = False
        A.AUTO_THREAD = None
        A.MANUAL_INSPECTION_IN_PROGRESS = True

        with (
            patch.object(A, "fetch_one", return_value=user),
            patch.object(A, "get_active_batch", return_value=batch),
            patch.object(A, "is_ai_capture_mode_active", return_value=False),
            patch.object(A, "CAMERA_CONNECTED", True),
            patch.object(A.threading, "Thread") as thread_cls,
        ):
            response = client.post("/api/station/auto/start")

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertIn("manual", payload["message"].lower())
        thread_cls.assert_not_called()


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

    def test_runtime_image_directories_are_persistent(self):
        compose = (ROOT / "docker-compose.yaml").read_text(
            encoding="utf-8"
        )
        required_mounts = {
            "./static/captures:/app/static/captures",
            "./static/results:/app/static/results",
            "./static/garment_models:/app/static/garment_models",
            "./ai_artifacts:/app/ai_artifacts",
        }

        missing = sorted(
            mount for mount in required_mounts
            if mount not in compose
        )
        self.assertEqual(missing, [])

    def test_container_runtime_versions_are_pinned(self):
        dockerfile = (ROOT / "Dockerfile").read_text(
            encoding="utf-8"
        )
        requirements = (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "FROM python:3.11.17-slim-trixie",
            dockerfile,
        )
        for dependency in (
            "anomalib==2.5.0",
            "torch==2.14.0",
            "torchvision==0.29.0",
            "timm==1.0.29",
            "gunicorn==23.0.0",
        ):
            self.assertIn(dependency, requirements)

    def test_container_uses_single_worker_production_wsgi_server(self):
        dockerfile = (ROOT / "Dockerfile").read_text(
            encoding="utf-8"
        )
        gunicorn_config = (ROOT / "gunicorn.conf.py").read_text(
            encoding="utf-8"
        )
        wsgi = (ROOT / "wsgi.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]',
            dockerfile,
        )
        self.assertIn("workers = 1", gunicorn_config)
        self.assertIn("worker_class = \"gthread\"", gunicorn_config)
        self.assertIn("ensure_camera_capture_worker()", wsgi)
        self.assertIn("init_db()", wsgi)

    def test_app_has_container_healthcheck(self):
        compose = (ROOT / "docker-compose.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("healthcheck:", compose)
        self.assertIn(
            "http://127.0.0.1:5000/health",
            compose,
        )


if __name__ == "__main__":
    unittest.main()
