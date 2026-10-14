from __future__ import annotations

import importlib
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


def plan(*, has_goal=True, remaining=40, suggested=40):
    return {
        "date": date(2026, 10, 7),
        "has_goal": has_goal,
        "remaining_garments": remaining,
        "suggested_quantity": suggested,
    }


class _StartBatchCursor:
    def __init__(self, batch):
        self.results = iter([
            {"acquired": 1},
            batch,
            {"released": 1},
        ])
        self.closed = False
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return next(self.results)

    def close(self):
        self.closed = True


class _StartBatchConnection:
    def __init__(self, batch):
        self.cursor_obj = _StartBatchCursor(batch)
        self.in_transaction = False
        self.rollbacks = 0
        self.commits = 0
        self.closed = False

    def cursor(self, dictionary=False):
        return self.cursor_obj

    def start_transaction(self):
        self.in_transaction = True

    def rollback(self):
        self.rollbacks += 1
        self.in_transaction = False

    def commit(self):
        self.commits += 1
        self.in_transaction = False

    def close(self):
        self.closed = True


class BatchRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")

    def setUp(self):
        self.client = self.A.app.test_client()
        self.user = {
            "id": 1,
            "username": "audit-admin",
            "role": self.A.ROLE_ADMIN,
            "active": 1,
        }
        with self.client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = self.A.ROLE_ADMIN

    def test_batch_quantity_guard_runs_before_database_insert(self):
        with (
            patch.object(
                self.A,
                "get_today_production_plan",
                return_value=plan(
                    has_goal=True,
                    remaining=40,
                    suggested=40,
                ),
            ),
            patch.object(self.A, "fetch_all", return_value=[]),
            patch.object(self.A, "get_inspection_lines", return_value=[]),
            patch.object(
                self.A,
                "fetch_one",
                return_value=self.user,
            ) as fetch_one,
            patch.object(self.A, "db") as connect_db,
        ):
            response = self.client.post(
                "/lotes",
                data={
                    "garment_model_id": "1735",
                    "inspection_line_id": "1",
                    "planned_quantity": "30",
                    "quantity_override": "0",
                },
            )

        self.assertEqual(response.status_code, 302)
        fetch_one.assert_called_once()
        connect_db.assert_not_called()

    def test_goal_completely_planned_rejects_without_database(self):
        with (
            patch.object(
                self.A, "get_today_production_plan",
                return_value=plan(has_goal=True, remaining=0, suggested=0),
            ),
            patch.object(self.A, "fetch_all", return_value=[]),
            patch.object(self.A, "get_inspection_lines", return_value=[]),
            patch.object(self.A, "fetch_one", return_value=self.user) as auth,
            patch.object(self.A, "db") as connect_db,
        ):
            response = self.client.post(
                "/lotes",
                data={
                    "garment_model_id": "1735",
                    "inspection_line_id": "1",
                    "planned_quantity": "10",
                    "quantity_override": "0",
                },
            )

        self.assertEqual(response.status_code, 302)
        auth.assert_called_once()
        connect_db.assert_not_called()

    def test_explicit_override_skips_goal_guard_but_still_checks_line(self):
        with (
            patch.object(
                self.A, "get_today_production_plan",
                return_value=plan(has_goal=True, remaining=40, suggested=40),
            ),
            patch.object(self.A, "fetch_all", return_value=[]),
            patch.object(self.A, "get_inspection_lines", return_value=[]),
            patch.object(
                self.A, "fetch_one",
                side_effect=[self.user, None],
            ) as fetch_one,
            patch.object(self.A, "db") as connect_db,
        ):
            response = self.client.post(
                "/lotes",
                data={
                    "garment_model_id": "1735",
                    "inspection_line_id": "1",
                    "planned_quantity": "30",
                    "quantity_override": "1",
                },
            )

        self.assertEqual(response.status_code, 302)
        # Al superar la guarda, se consulta la línea (ficticia, no válida).
        self.assertEqual(fetch_one.call_count, 2)
        connect_db.assert_not_called()

    def test_no_goal_skips_guard_but_still_checks_line(self):
        with (
            patch.object(
                self.A, "get_today_production_plan",
                return_value=plan(has_goal=False),
            ),
            patch.object(self.A, "fetch_all", return_value=[]),
            patch.object(self.A, "get_inspection_lines", return_value=[]),
            patch.object(
                self.A, "fetch_one",
                side_effect=[self.user, None],
            ) as fetch_one,
            patch.object(self.A, "db") as connect_db,
        ):
            response = self.client.post(
                "/lotes",
                data={
                    "garment_model_id": "1735",
                    "inspection_line_id": "1",
                    "planned_quantity": "30",
                    "quantity_override": "0",
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(fetch_one.call_count, 2)
        connect_db.assert_not_called()

    def test_batch_model_query_requires_production_ready_ai(self):
        captured = []

        def fetch_all(sql, params=()):
            captured.append(" ".join(sql.split()))
            return []

        with (
            patch.object(
                self.A,
                "get_today_production_plan",
                return_value=plan(has_goal=False),
            ),
            patch.object(self.A, "fetch_all", side_effect=fetch_all),
            patch.object(self.A, "get_inspection_lines", return_value=[]),
            patch.object(self.A, "get_batches", return_value=[]),
            patch.object(self.A, "get_active_batch", return_value=None),
            patch.object(self.A, "render_template", return_value="ok"),
            patch.object(
                self.A,
                "fetch_one",
                return_value=self.user,
            ),
        ):
            response = self.client.get("/lotes")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(captured)
        sql = captured[0]
        self.assertIn("NULLIF(TRIM(ai.checkpoint_path), '') IS NOT NULL", sql)
        self.assertIn("ai.threshold_final IS NOT NULL", sql)
        self.assertIn("ai.threshold_frozen_at IS NOT NULL", sql)

    def test_start_batch_rejects_ai_without_runtime_artifacts(self):
        batch = {
            "id": 9001,
            "code": "LOT-TEST",
            "status": "PREPARACION",
            "garment_model_id": 1735,
            "ai_model_id": 2538,
            "garment_model_code": "BLUSA-TEST",
            "garment_model_name": "Blusa test",
            "garment_model_status": "APROBADO",
            "garment_model_active": 1,
            "linked_ai_id": 2538,
            "ai_garment_model_id": 1735,
            "ai_version": "v1",
            "ai_model_type": "PatchCore",
            "ai_status": "ACTIVO",
            "ai_active": 1,
            "ai_checkpoint_path": None,
            "ai_threshold_final": None,
            "ai_threshold_frozen_at": None,
        }
        conn = _StartBatchConnection(batch)

        with (
            patch.object(self.A, "db", return_value=conn),
            patch.object(
                self.A,
                "fetch_one",
                return_value=self.user,
            ),
        ):
            response = self.client.post("/lotes/9001/iniciar")

        self.assertEqual(response.status_code, 302)
        self.assertGreaterEqual(conn.rollbacks, 1)
        self.assertEqual(conn.commits, 0)
        self.assertTrue(conn.cursor_obj.closed)
        self.assertTrue(conn.closed)


if __name__ == "__main__":
    unittest.main()
