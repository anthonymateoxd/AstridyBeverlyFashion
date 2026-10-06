import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as A


class _Cursor:
    def __init__(self, row):
        self.row = row
        self.closed = False

    def execute(self, sql, params=()):
        self.sql = sql

    def fetchone(self):
        return self.row

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, row):
        self.cursor_obj = _Cursor(row)
        self.closed = False

    def cursor(self, dictionary=False):
        return self.cursor_obj

    def close(self):
        self.closed = True


class ProductionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint_rel = (
            "garment_1735/models/ai_model_2538/checkpoint/model.ckpt"
        )
        checkpoint = self.root / self.checkpoint_rel
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")

        config = (
            self.root
            / "garment_1735/models/ai_model_2538/training/config.json"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps({
                "config": {
                    "validation_preprocessing_profile": "PER_IMAGE"
                }
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def _row(self, **changes):
        row = {
            "batch_id": 46,
            "garment_model_id": 1735,
            "ai_model_id": 2538,
            "ai_garment_model_id": 1735,
            "ai_version": "v1",
            "ai_status": "ACTIVO",
            "ai_active": 1,
            "checkpoint_path": self.checkpoint_rel,
            "checkpoint_hash": "a" * 64,
            "input_size": "256",
            "threshold_final": 44.41,
            "threshold_frozen_at": "2026-10-06 04:42:44",
            "dataset_id": 858,
        }
        row.update(changes)
        return row

    def test_runtime_uses_active_batch_version_threshold_and_profile(self):
        conn = _Connection(self._row())
        inspector = object()
        with patch.object(A, "db", return_value=conn), \
             patch.object(A, "get_ai_artifacts_root", return_value=self.root), \
             patch.object(A.ai_validation, "get_model_inspector", return_value=inspector) as get_inspector:
            runtime = A.resolve_active_production_ai_runtime()

        self.assertEqual(runtime["batch_id"], 46)
        self.assertEqual(runtime["ai_model_id"], 2538)
        self.assertEqual(runtime["garment_model_id"], 1735)
        self.assertEqual(runtime["version"], "v1")
        self.assertEqual(runtime["threshold"], 44.41)
        self.assertEqual(runtime["input_size"], 256)
        self.assertEqual(runtime["preprocessing_profile"], "PER_IMAGE")
        self.assertIs(runtime["inspector"], inspector)
        bundle = get_inspector.call_args.args[0]
        self.assertEqual(bundle["checkpoint_path"], self.checkpoint_rel)
        self.assertEqual(bundle["threshold_final"], 44.41)
        self.assertTrue(conn.cursor_obj.closed)
        self.assertTrue(conn.closed)

    def test_runtime_rejects_ai_from_another_garment_model(self):
        conn = _Connection(self._row(ai_garment_model_id=9999))
        with patch.object(A, "db", return_value=conn), \
             patch.object(A, "get_ai_artifacts_root", return_value=self.root), \
             patch.object(A.ai_validation, "get_model_inspector") as get_inspector:
            with self.assertRaisesRegex(RuntimeError, "ACTIVA válida"):
                A.resolve_active_production_ai_runtime()
        get_inspector.assert_not_called()

    def test_runtime_rejects_unfrozen_threshold(self):
        conn = _Connection(self._row(threshold_frozen_at=None))
        with patch.object(A, "db", return_value=conn), \
             patch.object(A, "get_ai_artifacts_root", return_value=self.root), \
             patch.object(A.ai_validation, "get_model_inspector") as get_inspector:
            with self.assertRaisesRegex(RuntimeError, "threshold definitivo congelado"):
                A.resolve_active_production_ai_runtime()
        get_inspector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
