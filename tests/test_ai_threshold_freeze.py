import json
import unittest
from unittest.mock import patch

import ai_validation
from ai_domain import AIDomainError


class FreezeCursor:
    def __init__(self, *, threshold_final=None):
        self.model = {
            "id": 2538, "garment_model_id": 1735, "version": "v1",
            "status": "VALIDACION", "active": 0, "notes": None,
            "threshold_final": threshold_final,
            "threshold_frozen_at": "2026-10-06 04:00:00" if threshold_final is not None else None,
            "threshold_frozen_by": 7 if threshold_final is not None else None,
            "threshold_provenance": "Candidato EVALUADO" if threshold_final is not None else None,
        }
        self.session = {
            "id": 922, "status": "EVALUADA", "threshold_candidate": 44.410,
            "metrics_json": json.dumps({"best_threshold": 44.41}),
        }
        self.sql = []
        self.params = []
        self.result = None
        self.rows = []
        self.rowcount = 1

    def execute(self, sql, params=()):
        text = " ".join(sql.lower().split())
        self.sql.append(text)
        self.params.append(params)
        if "from garment_ai_models" in text:
            self.result = dict(self.model)
        elif "from ai_validation_sessions" in text:
            self.result = dict(self.session)
        elif "from ai_dataset_images" in text:
            self.rows = []
            self.result = self.rows
        elif "from ai_validation_cases" in text:
            self.rows = [
                {"id": i, "category": "NORMAL" if i <= 10 else "MANCHA",
                 "image_sha256": f"{i:064x}", "image_path": f"originals/{i}.jpg",
                 "anomaly_score": 40.0 if i <= 10 else 50.0}
                for i in range(1, 21)
            ]
            self.result = self.rows
        else:
            self.result = None

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.rows


class ThresholdFreezeTests(unittest.TestCase):
    def test_freeze_persists_only_evaluated_candidate_without_activating(self):
        cur = FreezeCursor()
        with patch.object(ai_validation, "verify_validation_bank_snapshot", return_value={
            "valid": True, "case_count": 20, "snapshot_sha256": "a" * 64,
        }) as bank, patch.object(ai_validation, "record_ai_event") as audit:
            result = ai_validation.freeze_validation_threshold(
                cur,
                ai_model_id=2538,
                threshold=44.41,
                actor_id=7,
                provenance="Threshold candidato de la sesión EVALUADA; banco verificado.",
            )
        self.assertEqual(result["threshold_final"], 44.41)
        self.assertEqual(result["active"], 0)
        self.assertTrue(any("threshold_final = %s" in sql for sql in cur.sql))
        self.assertFalse(any("active =" in sql for sql in cur.sql))
        self.assertFalse(any("checkpoint" in sql for sql in cur.sql))
        self.assertFalse(any(
            action in sql and "ai_validation_cases" in sql
            for sql in cur.sql for action in ("update", "insert into", "delete from")
        ))
        bank.assert_called_once()
        self.assertEqual(audit.call_args.args[1], "THRESHOLD_FROZEN")
        self.assertEqual(audit.call_args.kwargs["payload"]["threshold_final"], 44.41)

    def test_non_candidate_threshold_is_rejected(self):
        cur = FreezeCursor()
        with patch.object(ai_validation, "verify_validation_bank_snapshot") as bank:
            with self.assertRaisesRegex(AIDomainError, "Solo puede congelarse el threshold candidato"):
                ai_validation.freeze_validation_threshold(
                    cur, ai_model_id=2538, threshold=50.0, actor_id=7,
                    provenance="not the candidate",
                )
        bank.assert_not_called()

    def test_invalid_bank_blocks_freeze(self):
        cur = FreezeCursor()
        with patch.object(ai_validation, "verify_validation_bank_snapshot", side_effect=AIDomainError("Banco inválido")):
            with self.assertRaisesRegex(AIDomainError, "Banco inválido"):
                ai_validation.freeze_validation_threshold(
                    cur, ai_model_id=2538, threshold=44.41, actor_id=7,
                    provenance="candidato",
                )
        self.assertFalse(any("update garment_ai_models" in sql for sql in cur.sql))

    def test_second_same_freeze_is_idempotent_but_cannot_replace_value(self):
        cur = FreezeCursor(threshold_final=44.41)
        same = ai_validation.freeze_validation_threshold(
            cur, ai_model_id=2538, threshold=44.41, actor_id=8,
            provenance="repeat",
        )
        self.assertTrue(same["already_frozen"])
        with self.assertRaisesRegex(AIDomainError, "ya está congelado"):
            ai_validation.freeze_validation_threshold(
                cur, ai_model_id=2538, threshold=45.0, actor_id=8,
                provenance="replacement",
            )
        self.assertFalse(any("update garment_ai_models" in sql for sql in cur.sql))


if __name__ == "__main__":
    unittest.main()
