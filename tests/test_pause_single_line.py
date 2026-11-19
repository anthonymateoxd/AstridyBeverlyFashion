"""Pruebas de pausas sin modificar los datos del proyecto."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class CapturePauseTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domain = importlib.import_module("ai_domain")

    def test_allowed_transitions(self):
        self.domain.validate_capture_status_transition("ABIERTA", "PAUSADA")
        self.domain.validate_capture_status_transition("PAUSADA", "ABIERTA")
        self.domain.validate_capture_status_transition("PAUSADA", "CANCELADA")

    def test_completed_cannot_be_reopened(self):
        for original in ("COMPLETADA", "CANCELADA"):
            with self.subTest(original=original):
                with self.assertRaises(self.domain.AIDomainError):
                    self.domain.validate_capture_status_transition(original, "ABIERTA")

    def test_pause_keeps_finished_at_empty(self):
        cursor = Mock()
        cursor.fetchone.return_value = {
            "id": 7, "garment_model_id": 66, "status": "ABIERTA"
        }
        result = self.domain.transition_capture_session_status(
            cursor, 7, "PAUSADA"
        )
        self.assertEqual(result["status"], "PAUSADA")
        calls = cursor.execute.call_args_list
        self.assertIn("finished_at = CASE", calls[-1].args[0])
        self.assertEqual(calls[-1].args[1], ("PAUSADA", "PAUSADA", 7))

    def test_resume_keeps_original_session_id(self):
        cursor = Mock()
        # GET_LOCK -> open session query -> paused row -> transition -> RELEASE_LOCK
        answers = iter([
            {"got": 1},
            None,
            {
                "id": 29, "garment_model_id": 66,
                "target_ai_model_id": None, "started_at": "2026-10-16",
            },
            {"id": 29, "garment_model_id": 66, "status": "PAUSADA"},
            {"released": 1},
        ])
        cursor.fetchone.side_effect = lambda: next(answers)
        result = self.domain.start_ai_capture_session(
            cursor, 66, 9, config={"target_images": 40}
        )
        self.assertTrue(result["resumed"])
        self.assertEqual(result["id"], 29)
        sql = "\n".join(str(call.args[0]) for call in cursor.execute.call_args_list)
        self.assertNotIn("INSERT INTO ai_capture_sessions", sql)

    def test_resume_rejects_other_ai_version(self):
        cursor = Mock()
        answers = iter([
            {"got": 1},
            None,
            {
                "id": 29, "garment_model_id": 66,
                "target_ai_model_id": 123, "started_at": "2026-10-16",
            },
            {"released": 1},
        ])
        cursor.fetchone.side_effect = lambda: next(answers)
        with self.assertRaises(self.domain.AIDomainError):
            self.domain.start_ai_capture_session(
                cursor, 66, 9, target_ai_model_id=124,
                config={"target_images": 40},
            )


class LotPauseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = importlib.import_module("app")

    def setUp(self):
        self.client = self.a.app.test_client()
        with self.client.session_transaction() as session:
            session.update(user_id=9, role="ADMIN", username="test-admin")
        self.user = {
            "id": 9, "username": "test-admin", "role": "ADMIN", "active": 1
        }

    @staticmethod
    def conn_with(cursor):
        connection = Mock()
        connection.in_transaction = False
        connection.cursor.return_value = cursor
        return connection

    def test_pause_does_not_delete_inspections(self):
        cursor = Mock()
        cursor.fetchone.return_value = {
            "id": 87, "code": "LOT-87", "status": "EN_INSPECCION"
        }
        cursor.rowcount = 1
        conn = self.conn_with(cursor)
        with (
            patch.object(self.a, "fetch_one", return_value=self.user),
            patch.object(self.a, "db", return_value=conn),
            patch.object(self.a, "AUTO_THREAD", None),
            patch.object(self.a, "MANUAL_INSPECTION_IN_PROGRESS", False),
        ):
            result = self.client.post("/lotes/87/pausar")
        self.assertEqual(result.status_code, 302)
        conn.commit.assert_called_once()
        sql = "\n".join(call.args[0] for call in cursor.execute.call_args_list)
        self.assertIn("SET status='PAUSADO'", sql)
        self.assertNotIn("DELETE FROM inspections", sql)
        self.assertNotIn("INSERT INTO inspections", sql)

    def test_resume_rejects_open_capture(self):
        cursor = Mock()
        cursor.fetchone.side_effect = [
            {"acquired": 1},
            {
                "id": 87, "code": "LOT-87", "status": "PAUSADO",
                "model_active": 1, "ai_active": 1, "ai_status": "ACTIVO",
            },
            None,
            {"id": 5},
            {"released": 1},
        ]
        conn = self.conn_with(cursor)
        with (
            patch.object(self.a, "fetch_one", return_value=self.user),
            patch.object(self.a, "db", return_value=conn),
            patch.object(self.a, "AUTO_THREAD", None),
            patch.object(self.a, "AUTO_INSPECTION_ENABLED", False),
            patch.object(self.a, "MANUAL_INSPECTION_IN_PROGRESS", False),
        ):
            result = self.client.post("/lotes/87/reanudar")
        self.assertEqual(result.status_code, 302)
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()
        sql = "\n".join(call.args[0] for call in cursor.execute.call_args_list)
        self.assertNotIn("UPDATE batches SET status='EN_INSPECCION'", sql)

    def test_resume_retains_batch(self):
        cursor = Mock()
        cursor.fetchone.side_effect = [
            {"acquired": 1},
            {
                "id": 87, "code": "LOT-87", "status": "PAUSADO",
                "model_active": 1, "ai_active": 1, "ai_status": "ACTIVO",
            },
            None, None, {"released": 1},
        ]
        cursor.rowcount = 1
        conn = self.conn_with(cursor)
        with (
            patch.object(self.a, "fetch_one", return_value=self.user),
            patch.object(self.a, "db", return_value=conn),
            patch.object(self.a, "AUTO_THREAD", None),
            patch.object(self.a, "AUTO_INSPECTION_ENABLED", False),
            patch.object(self.a, "MANUAL_INSPECTION_IN_PROGRESS", False),
        ):
            result = self.client.post("/lotes/87/reanudar")
        self.assertEqual(result.status_code, 302)
        self.assertIn("/estacion", result.location)
        conn.commit.assert_called_once()
        sql = "\n".join(call.args[0] for call in cursor.execute.call_args_list)
        self.assertIn("UPDATE batches SET status='EN_INSPECCION'", sql)


if __name__ == "__main__":
    unittest.main()
