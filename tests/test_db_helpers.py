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


class _Cursor:
    def __init__(self, *, row=None, rows=None, error=None):
        self.row = row
        self.rows = rows or []
        self.error = error
        self.closed = False
        self.lastrowid = 77

    def execute(self, sql, params=()):
        if self.error is not None:
            raise self.error

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.closed = False
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, dictionary=False):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class DatabaseHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")

    def test_fetch_one_closes_resources_when_query_fails(self):
        cursor = _Cursor(error=RuntimeError("query failed"))
        conn = _Connection(cursor)

        with patch.object(self.A, "db", return_value=conn):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                self.A.fetch_one("SELECT broken")

        self.assertTrue(cursor.closed)
        self.assertTrue(conn.closed)

    def test_fetch_all_closes_resources_when_query_fails(self):
        cursor = _Cursor(error=RuntimeError("query failed"))
        conn = _Connection(cursor)

        with patch.object(self.A, "db", return_value=conn):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                self.A.fetch_all("SELECT broken")

        self.assertTrue(cursor.closed)
        self.assertTrue(conn.closed)

    def test_execute_rolls_back_and_closes_when_query_fails(self):
        cursor = _Cursor(error=RuntimeError("write failed"))
        conn = _Connection(cursor)

        with patch.object(self.A, "db", return_value=conn):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.A.execute("UPDATE broken")

        self.assertEqual(conn.commits, 0)
        self.assertEqual(conn.rollbacks, 1)
        self.assertTrue(cursor.closed)
        self.assertTrue(conn.closed)

    def test_execute_commits_and_returns_last_id(self):
        cursor = _Cursor()
        conn = _Connection(cursor)

        with patch.object(self.A, "db", return_value=conn):
            last_id = self.A.execute("INSERT ok")

        self.assertEqual(last_id, 77)
        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.rollbacks, 0)
        self.assertTrue(cursor.closed)
        self.assertTrue(conn.closed)


if __name__ == "__main__":
    unittest.main()
