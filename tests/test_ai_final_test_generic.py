import json
import unittest
from unittest.mock import patch

import ai_validation
from ai_domain import AIDomainError


class GenericFinalCursor:
    def __init__(self):
        self.result = None
        self.rows = []
        self.lastrowid = 0
        self.final_session = None
        self.sql = []

    def execute(self, sql, params=()):
        text = " ".join(sql.lower().split())
        self.sql.append(text)
        self.result = None
        self.rows = []
        if text.startswith("select id from ai_final_test_sessions"):
            self.result = {"id": self.final_session["id"]} if self.final_session else None
        elif "from ai_validation_sessions" in text and text.startswith("select"):
            self.result = {
                "id": 922,
                "status": "CERRADA",
                "threshold_candidate": 44.41,
                "metrics_json": json.dumps({"best_threshold": 44.41, "threshold_candidate_metrics": {"threshold": 44.41}, "calibration_case_count": 20, "scored": 20}),
                "created_at": None,
                "evaluated_at": None,
                "closed_at": None,
            }
        elif text.startswith("insert into ai_final_test_sessions"):
            self.lastrowid = 77
            self.final_session = {
                "id": 77,
                "garment_model_id": int(params[0]),
                "ai_model_id": int(params[1]),
                "cohort": "FINAL_TEST",
                "threshold_fixed": float(params[2]),
                "preprocessing_profile": params[3],
                "status": "ABIERTA",
                "metrics_json": None,
                "created_by": params[4],
                "created_at": "2026-10-06 00:00:00",
                "evaluated_at": None,
            }
        elif "from ai_final_test_sessions where ai_model_id" in text and text.startswith("select"):
            self.result = dict(self.final_session) if self.final_session else None
        elif "from ai_final_test_cases where final_test_session_id" in text:
            self.rows = []

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.rows


class GenericFinalTestTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "id": 2538,
            "garment_model_id": 1735,
            "version": "v1",
            "status": "VALIDADO",
            "active": 0,
            "threshold_final": 44.41,
            "threshold_frozen_at": "2026-10-06 00:00:00",
            "threshold_frozen_by": 1,
            "threshold_provenance": "validación cerrada",
            "notes": None,
            "dataset_id": 858,
        }

    def test_validated_frozen_model_can_start_empty_final_test(self):
        cur = GenericFinalCursor()
        with patch.object(ai_validation, "_fetch_validation_model", return_value=self.model), \
             patch.object(ai_validation, "load_model_bundle", return_value={"preprocessing_profile": "PER_IMAGE"}), \
             patch.object(ai_validation, "list_validation_cases", return_value=[{"id": i} for i in range(20)]), \
             patch.object(ai_validation, "verify_validation_bank_snapshot", return_value={"valid": True, "case_count": 20}), \
             patch.object(ai_validation, "record_ai_event"):
            state = ai_validation.start_final_test(cur, ai_model_id=2538, actor_id=1)
        self.assertTrue(state["started"])
        self.assertEqual(state["threshold_fixed"], 44.41)
        self.assertEqual(state["preprocessing_profile"], "PER_IMAGE")
        self.assertEqual(state["counts"]["total"], 0)
        self.assertEqual(state["ai_model_id"], 2538)
        self.assertEqual(state["garment_model_id"], 1735)

    def test_generic_final_test_requires_frozen_threshold(self):
        model = dict(self.model, threshold_final=None, threshold_frozen_at=None)
        with patch.object(ai_validation, "_fetch_validation_model", return_value=model), \
             patch.object(ai_validation, "load_model_bundle", return_value={"preprocessing_profile": "PER_IMAGE"}):
            contract = ai_validation._final_test_contract(GenericFinalCursor(), ai_model_id=2538)
        self.assertFalse(contract["can_start"])
        self.assertIn("threshold", contract["start_block_reason"].lower())

    def test_generic_final_test_requires_inactive_model(self):
        model = dict(self.model, active=1)
        with patch.object(ai_validation, "_fetch_validation_model", return_value=model), \
             patch.object(ai_validation, "load_model_bundle", return_value={"preprocessing_profile": "PER_IMAGE"}):
            contract = ai_validation._final_test_contract(GenericFinalCursor(), ai_model_id=2538)
        self.assertFalse(contract["can_start"])
        self.assertIn("activar", contract["start_block_reason"].lower())


if __name__ == "__main__":
    unittest.main()
