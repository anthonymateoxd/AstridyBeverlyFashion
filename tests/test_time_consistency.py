from __future__ import annotations

import importlib
import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


class TimeConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = importlib.import_module("app")

    def test_utc_now_db_uses_utc_independent_of_host_timezone(self):
        before = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None)
        value = self.A.utc_now_db()
        after = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None)

        self.assertIsInstance(value, datetime)
        self.assertIsNone(value.tzinfo)
        self.assertLessEqual(before, value)
        self.assertLessEqual(value, after)

    def test_database_writes_do_not_use_local_naive_now(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertNotIn("datetime.utcnow()", source)
        self.assertNotIn(
            'datetime.now().strftime("%Y-%m-%d %H:%M:%S")',
            source,
        )


if __name__ == "__main__":
    unittest.main()
