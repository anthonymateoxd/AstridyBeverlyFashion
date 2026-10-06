import unittest
from pathlib import Path
from unittest.mock import patch

import ai_validation
from ai_domain import AIDomainError


class FinalTestCursor:
    def __init__(self, *, collision=None, populated=False, with_cases=False):
        self.sql = []
        self.result = None
        self.rows = []
        self.collision = collision
        self.populated = populated
        self.with_cases = with_cases
        self.lastrowid = 1

    def execute(self, sql, params=()):
        text = " ".join(sql.lower().split())
        self.sql.append(text)
        if "from ai_final_test_sessions" in text and text.startswith("select"):
            if self.populated:
                self.result = {
                    "id": 7, "garment_model_id": 1082, "ai_model_id": 1649,
                    "cohort": "FINAL_TEST", "threshold_fixed": 47.32,
                    "preprocessing_profile": "FULL_ROI", "status": "ABIERTA",
                    "metrics_json": None, "created_by": 2,
                    "created_at": "2026-10-05 12:00:00", "evaluated_at": None,
                }
            else:
                self.result = None
        elif "from ai_final_test_cases" in text and text.startswith("select"):
            if "where image_sha256" in text and self.collision == "final":
                self.result = {"id": 1}
            elif "where final_test_session_id" in text:
                self.rows = self._metric_cases() if self.with_cases else []
                self.result = self.rows
            else:
                self.rows = []
                self.result = None
        elif "from ai_training_images" in text:
            self.result = {"id": 22} if self.collision == "training" else None
        elif "from ai_validation_cases" in text:
            self.result = {"id": 33} if self.collision == "calibration" else None
        elif text.startswith("insert into ai_final_test_sessions"):
            self.lastrowid = 9
            self.populated = True
            self.result = None
        else:
            self.result = None

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.rows

    @staticmethod
    def _metric_cases():
        return (
            [{"category": "NORMAL", "anomaly_score": 40.0} for _ in range(10)]
            + [{"category": "MANCHA", "anomaly_score": 55.0} for _ in range(10)]
        )


class FinalTestIsolationTests(unittest.TestCase):
    def test_new_final_test_state_is_empty_and_has_locked_contract(self):
        state = ai_validation.get_final_test_state(FinalTestCursor())
        self.assertFalse(state["started"])
        self.assertEqual(state["cases"], [])
        self.assertEqual(state["counts"], {"BUENA": 0, "MANCHA": 0, "total": 0})
        self.assertEqual(state["threshold_fixed"], 47.32)
        self.assertEqual(state["preprocessing_profile"], "FULL_ROI")

    def test_historical_cases_are_not_used_to_initialize_or_count_final_test(self):
        cur = FinalTestCursor()
        state = ai_validation.get_final_test_state(cur)
        self.assertEqual(state["counts"]["total"], 0)
        self.assertFalse(any("ai_validation_cases" in sql for sql in cur.sql))

    def test_start_creates_a_new_empty_session_with_audit_contract(self):
        cur = FinalTestCursor()
        model = {
            "id": 1649, "garment_model_id": 1082, "version": "v1",
            "status": "ENTRENADO", "active": 0, "threshold_final": None,
            "notes": None,
        }
        with patch.object(ai_validation, "_fetch_validation_model", return_value=model), \
             patch.object(ai_validation, "load_model_bundle", return_value={
                 "preprocessing_profile": "FULL_ROI", "threshold_final": None,
             }):
            state = ai_validation.start_final_test(cur, ai_model_id=1649, actor_id=4)
        self.assertTrue(state["started"])
        self.assertEqual(state["counts"]["total"], 0)
        self.assertEqual(state["threshold_fixed"], 47.32)
        self.assertTrue(any("insert into ai_final_test_sessions" in sql for sql in cur.sql))

    def test_active_model_is_refused_at_final_test_start(self):
        model = {
            "id": 1649, "garment_model_id": 1082, "version": "v1",
            "status": "VALIDACION", "active": 1, "threshold_final": None,
            "notes": None,
        }
        with patch.object(ai_validation, "_fetch_validation_model", return_value=model), \
             patch.object(ai_validation, "load_model_bundle") as load_bundle:
            with self.assertRaisesRegex(AIDomainError, "activa"):
                ai_validation.start_final_test(FinalTestCursor(), ai_model_id=1649)
        load_bundle.assert_not_called()

    def test_training_hash_collision_is_rejected_before_inference(self):
        cur = FinalTestCursor(collision="training", populated=True)
        with patch.object(ai_validation, "predict_with_ai_model") as predict:
            with self.assertRaisesRegex(AIDomainError, "TRAINING"):
                ai_validation.register_final_test_case(
                    cur, image_bytes=b"unique-frame", category="NORMAL", actor_id=1
                )
        predict.assert_not_called()

    def test_calibration_hash_collision_is_rejected_before_inference(self):
        cur = FinalTestCursor(collision="calibration", populated=True)
        with patch.object(ai_validation, "predict_with_ai_model") as predict:
            with self.assertRaisesRegex(AIDomainError, "VALIDATION/calibración"):
                ai_validation.register_final_test_case(
                    cur, image_bytes=b"unique-frame", category="NORMAL", actor_id=1
                )
        predict.assert_not_called()

    def test_duplicate_final_hash_is_rejected_before_inference(self):
        cur = FinalTestCursor(collision="final", populated=True)
        with patch.object(ai_validation, "predict_with_ai_model") as predict:
            with self.assertRaisesRegex(AIDomainError, "duplicada"):
                ai_validation.register_final_test_case(
                    cur, image_bytes=b"unique-frame", category="NORMAL", actor_id=1
                )
        predict.assert_not_called()

    def test_capture_infers_with_full_roi_and_stores_fixed_threshold(self):
        cur = FinalTestCursor(populated=True)
        prediction = {
            "score_percent": 48.5,
            "anomaly_map": None,
            "preprocessing_profile": "FULL_ROI",
        }
        with patch.object(ai_validation, "predict_with_ai_model", return_value=prediction) as predict, \
             patch.object(ai_validation, "get_ai_artifacts_root", return_value=Path("C:/tmp/final-test")), \
             patch.object(ai_validation, "atomic_write_bytes"):
            result = ai_validation.register_final_test_case(
                cur, image_bytes=b"new-camera-image", category="MANCHA", actor_id=3
            )
        self.assertEqual(result["threshold_used"], 47.32)
        self.assertTrue(result["correct"])
        self.assertEqual(predict.call_args.kwargs["preprocessing_profile"], "FULL_ROI")
        self.assertTrue(any("insert into ai_final_test_cases" in sql for sql in cur.sql))

    def test_evaluation_uses_only_final_test_cases_and_fixed_threshold(self):
        cur = FinalTestCursor(populated=True, with_cases=True)
        metrics = ai_validation.evaluate_final_test(cur, actor_id=5)
        self.assertEqual(metrics["threshold"], 47.32)
        self.assertEqual(metrics["tp"], 10)
        self.assertEqual(metrics["tn"], 10)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[10, 0], [0, 10]])
        self.assertEqual(metrics["candidate_thresholds"], [])
        self.assertFalse(metrics["recalibration_performed"])
        self.assertTrue(any("from ai_final_test_cases where final_test_session_id" in sql for sql in cur.sql))
        self.assertFalse(any("from ai_validation_cases" in sql for sql in cur.sql))
        self.assertFalse(any("update garment_ai_models" in sql for sql in cur.sql))
        self.assertFalse(any("ai_training_images" in sql for sql in cur.sql))
        self.assertFalse(any("checkpoint" in sql for sql in cur.sql))

    def test_final_evaluation_never_updates_threshold_or_activates_model(self):
        cur = FinalTestCursor(populated=True, with_cases=True)
        ai_validation.evaluate_final_test(cur, actor_id=5)
        self.assertFalse(any("update garment_ai_models" in sql for sql in cur.sql))
        self.assertTrue(any("update ai_final_test_sessions" in sql for sql in cur.sql))

    def test_final_test_fixed_values_are_not_mutable_by_request(self):
        self.assertEqual(ai_validation.FINAL_TEST_THRESHOLD, 47.32)
        self.assertEqual(ai_validation.FINAL_TEST_PREPROCESSING, "FULL_ROI")
        self.assertEqual(ai_validation.FINAL_TEST_TARGET_PER_CLASS, 10)


if __name__ == "__main__":
    unittest.main()
