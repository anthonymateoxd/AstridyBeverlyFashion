"""Pruebas unitarias del flujo corto temporal de demostración."""

import os
import unittest
from unittest import mock

import ai_training


class _FakeCursor:
    def __init__(self):
        self.rowcount = 0
        self.closed = False
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "UPDATE garment_ai_models" in sql:
            self.rowcount = 1

    def fetchone(self):
        return {
            "id": 77,
            "garment_model_id": 12,
            "version": "v2",
            "status": "ENTRENADO",
            "active": 0,
            "threshold_final": None,
            "threshold_frozen_at": None,
            "threshold_provenance": None,
        }

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self):
        self.cursor_instance = _FakeCursor()
        self.in_transaction = False
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self, dictionary=False):
        return self.cursor_instance

    def start_transaction(self):
        self.in_transaction = True

    def commit(self):
        self.committed = True
        self.in_transaction = False

    def rollback(self):
        self.rolled_back = True
        self.in_transaction = False

    def close(self):
        self.closed = True


class DemoFastTrackUnitTests(unittest.TestCase):
    def test_flag_is_disabled_by_default(self):
        self.assertFalse(ai_training.demo_fast_track_enabled({}))
        self.assertTrue(
            ai_training.demo_fast_track_enabled(
                {"AI_DEMO_FAST_TRACK": "true"}
            )
        )

    def test_threshold_is_explicit_and_validated(self):
        self.assertEqual(
            ai_training.demo_fast_track_threshold(
                {"AI_DEMO_FAST_TRACK_THRESHOLD": "55.20"}
            ),
            55.20,
        )
        with self.assertRaises(ai_training.AIDomainError):
            ai_training.demo_fast_track_threshold(
                {"AI_DEMO_FAST_TRACK_THRESHOLD": "101"}
            )

    def test_activation_walks_normal_states_and_records_audit(self):
        connection = _FakeConnection()
        transitions = []
        events = []

        def fake_transition(cur, ai_model_id, status, actor_id, **kwargs):
            transitions.append(status)
            return {"id": ai_model_id, "status": status}

        def fake_event(cur, event_type, **kwargs):
            events.append(event_type)
            return {"event_type": event_type}

        with mock.patch.dict(
            os.environ,
            {"AI_DEMO_FAST_TRACK_THRESHOLD": "55.20"},
            clear=False,
        ), mock.patch.object(
            ai_training,
            "transition_ai_model_status",
            side_effect=fake_transition,
        ), mock.patch.object(
            ai_training,
            "record_ai_event",
            side_effect=fake_event,
        ):
            result = ai_training._activate_demo_fast_track(
                lambda: connection,
                ai_model_id=77,
            )

        self.assertEqual(
            transitions,
            ["VALIDACION", "VALIDADO", "ACTIVO"],
        )
        self.assertEqual(
            events,
            ["THRESHOLD_FROZEN", "VALIDATED", "ACTIVATED"],
        )
        self.assertEqual(result["status"], "ACTIVO")
        self.assertEqual(result["threshold"], 55.20)
        self.assertTrue(connection.committed)
        self.assertFalse(connection.rolled_back)
        self.assertTrue(connection.closed)

        updates = [
            (sql, params)
            for sql, params in connection.cursor_instance.executed
            if "UPDATE garment_ai_models" in sql
        ]
        self.assertEqual(len(updates), 1)
        self.assertEqual(float(updates[0][1][0]), 55.20)
        self.assertTrue(
            str(updates[0][1][1]).startswith("MODO DEMOSTRACION")
        )


if __name__ == "__main__":
    unittest.main()
