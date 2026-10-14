from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


class _Cursor:
    def __init__(self, *, fail_after=None):
        self.lastrowid = 321
        self.calls = 0
        self.fail_after = fail_after
        self.closed = False

    def execute(self, sql, params=()):
        self.calls += 1
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("forced insert failure")

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


class GarmentModelAtomicityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")

    def test_reference_files_are_removed_if_database_insert_fails(self):
        cursor = _Cursor(fail_after=2)

        with tempfile.TemporaryDirectory() as tmp, self.A.app.test_request_context("/"):
            self.A.session["user_id"] = 1

            with patch.object(self.A.app, "root_path", tmp):
                with self.assertRaisesRegex(RuntimeError, "forced insert failure"):
                    self.A.save_garment_reference_images(
                        321,
                        [
                            (".jpg", b"first"),
                            (".jpg", b"second"),
                        ],
                        cur=cursor,
                    )

                directory = (
                    Path(tmp)
                    / "static"
                    / "garment_models"
                    / "321"
                )
                self.assertFalse(directory.exists())

    def test_create_route_rolls_back_model_when_reference_persistence_fails(self):
        cursor = _Cursor()
        conn = _Connection(cursor)
        client = self.A.app.test_client()
        user = {
            "id": 1,
            "username": "audit-admin",
            "role": self.A.ROLE_ADMIN,
            "active": 1,
        }

        with client.session_transaction() as session:
            session["user_id"] = 1
            session["username"] = "audit-admin"
            session["role"] = self.A.ROLE_ADMIN

        with (
            patch.object(self.A, "fetch_one", return_value=user),
            patch.object(
                self.A,
                "generate_garment_model_code",
                return_value="BLUSA-999",
            ),
            patch.object(
                self.A,
                "prepare_garment_reference_images",
                return_value=[(".jpg", b"image")],
            ),
            patch.object(self.A, "db", return_value=conn),
            patch.object(
                self.A,
                "save_garment_reference_images",
                side_effect=RuntimeError("disk failed"),
            ),
        ):
            response = client.post(
                "/modelos-prenda/nuevo",
                data={
                    "name": "Modelo prueba",
                    "color": "Azul",
                    "description": "Prueba atomica",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(conn.commits, 0)
        self.assertEqual(conn.rollbacks, 1)
        self.assertTrue(cursor.closed)
        self.assertTrue(conn.closed)


if __name__ == "__main__":
    unittest.main()
