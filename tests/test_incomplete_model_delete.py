from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Cursor:
    def __init__(self, model):
        self.model = dict(model)
        self.rowcount = 0
        self.update_called = False
        self.update_params = None
        self.closed = False

    def execute(self, sql, params=()):
        normalized = " ".join(str(sql).split())

        if "FROM garment_models gm" in normalized and "FOR UPDATE" in normalized:
            return

        if normalized.startswith("UPDATE garment_models"):
            self.update_called = True
            self.update_params = params
            self.rowcount = 1

    def fetchone(self):
        return dict(self.model)

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.in_transaction = False
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self, dictionary=False):
        return self.cursor_obj

    def start_transaction(self):
        self.in_transaction = True

    def commit(self):
        self.commits += 1
        self.in_transaction = False

    def rollback(self):
        self.rollbacks += 1
        self.in_transaction = False

    def close(self):
        self.closed = True


class GarmentModelDeleteIncompleteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")

    def _admin_user(self):
        return {
            "id": 1,
            "username": "audit-admin",
            "role": self.A.ROLE_ADMIN,
            "active": 1,
        }

    def _model(self, **overrides):
        model = {
            "id": 77,
            "code": "BLUSA-077",
            "name": "Modelo incompleto",
            "active": 1,
            "has_batches": 0,
            "has_active_ai": 0,
            "ai_training_now": 0,
        }
        model.update(overrides)
        return model

    def _client_with_admin_session(self):
        client = self.A.app.test_client()
        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = self.A.ROLE_ADMIN
        return client

    def test_admin_can_soft_delete_unused_incomplete_model(self):
        cursor = _Cursor(self._model())
        conn = _Connection(cursor)
        client = self._client_with_admin_session()

        with (
            patch.object(self.A, "fetch_one", return_value=self._admin_user()),
            patch.object(self.A, "db", return_value=conn),
        ):
            response = client.post(
                "/modelos-prenda/77/eliminar-incompleto",
            )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(cursor.update_called)
        self.assertEqual(cursor.update_params[1], 77)
        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.rollbacks, 0)
        self.assertTrue(cursor.closed)
        self.assertTrue(conn.closed)

    def test_model_used_by_batch_cannot_be_deleted(self):
        cursor = _Cursor(self._model(has_batches=1))
        conn = _Connection(cursor)
        client = self._client_with_admin_session()

        with (
            patch.object(self.A, "fetch_one", return_value=self._admin_user()),
            patch.object(self.A, "db", return_value=conn),
        ):
            response = client.post(
                "/modelos-prenda/77/eliminar-incompleto",
            )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(cursor.update_called)
        self.assertEqual(conn.commits, 0)
        self.assertEqual(conn.rollbacks, 1)

    def test_model_with_active_ai_cannot_be_deleted(self):
        cursor = _Cursor(self._model(has_active_ai=1))
        conn = _Connection(cursor)
        client = self._client_with_admin_session()

        with (
            patch.object(self.A, "fetch_one", return_value=self._admin_user()),
            patch.object(self.A, "db", return_value=conn),
        ):
            response = client.post(
                "/modelos-prenda/77/eliminar-incompleto",
            )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(cursor.update_called)
        self.assertEqual(conn.commits, 0)
        self.assertEqual(conn.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()
