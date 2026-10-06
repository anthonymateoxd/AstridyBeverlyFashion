import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_domain
import ai_training
from ai_domain import (
    AIDomainError,
    compute_manifest_hash,
    normal_augmentation_summary,
    prepare_normal_v2_version,
    validate_normal_v2_capture_confirmation,
)


class PrepareV2Cursor:
    def __init__(self, root, *, validation_collision=False, final_collision=False, bad_manifest_category=False):
        self.root = Path(root)
        self.sql = []
        self.result = None
        self.rows = []
        self.lastrowid = 2600
        self.inserted_model = False
        self.source_rows = []
        self.validation_collision = validation_collision
        self.final_collision = final_collision
        for image_id in range(27):
            content = f"v1-good-training-image-{image_id}".encode()
            digest = hashlib.sha256(content).hexdigest()
            capture_rel = f"garment_1082/capture_sessions/session_1829/accepted/frame_{image_id}.jpg"
            materialized_rel = f"garment_1082/datasets/dataset_521/images/{image_id}.jpg"
            for rel in (capture_rel, materialized_rel):
                path = self.root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            self.source_rows.append({
                "id": image_id + 1,
                "sha256": digest,
                "image_path": capture_rel,
                "status": "ACEPTADA",
                "garment_model_id": 1082,
                "capture_session_id": 1829,
                "capture_status": "COMPLETADA",
            })
        manifest_entries = [
            {
                "image_id": row["id"],
                "path": f"garment_1082/datasets/dataset_521/images/{i}.jpg",
                "sha256": row["sha256"],
                "size": (self.root / f"garment_1082/datasets/dataset_521/images/{i}.jpg").stat().st_size,
                "category": "MANCHA" if bad_manifest_category and i == 0 else "NORMAL",
                "capture_session_id": 1829,
            }
            for i, row in enumerate(self.source_rows)
        ]
        content_hash = compute_manifest_hash([
            {"id": item["image_id"], "sha256": item["sha256"]}
            for item in manifest_entries
        ])
        manifest = {
            "dataset_id": 521,
            "garment_model_id": 1082,
            "version": "d2",
            "image_count": 27,
            "content_hash": content_hash,
            "images": manifest_entries,
        }
        self.manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2
        ).encode()
        manifest_path = self.root / "garment_1082/datasets/dataset_521/manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(self.manifest_bytes)
        self.checkpoint_bytes = b"immutable-v1-checkpoint"
        checkpoint = self.root / "garment_1082/models/ai_model_1649/checkpoint/model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(self.checkpoint_bytes)
        self.checkpoint_hash = hashlib.sha256(self.checkpoint_bytes).hexdigest()

    def execute(self, sql, params=()):
        text = " ".join(sql.lower().split())
        self.sql.append((text, params))
        if "from garment_models" in text and "for update" in text:
            self.result = {"id": 1082, "code": "BLUSA-762", "status": "APROBADO", "active": 1}
        elif "from garment_ai_models where id=%s for update" in text:
            self.result = {
                "id": 1649, "garment_model_id": 1082, "version": "v1",
                "model_type": "PatchCore", "status": "VALIDACION", "active": 0,
                "dataset_id": 521, "checkpoint_path": "garment_1082/models/ai_model_1649/checkpoint/model.ckpt",
                "checkpoint_hash": self.checkpoint_hash, "threshold_final": None,
            }
        elif "from ai_datasets where id=%s for update" in text:
            self.result = {
                "id": 521, "version": "d2", "status": "CERRADO", "image_count": 27,
                "manifest_path": "garment_1082/datasets/dataset_521/manifest.json",
                "manifest_hash": hashlib.sha256(self.manifest_bytes).hexdigest(),
            }
        elif "from ai_dataset_images di" in text and "join ai_capture_sessions cs" in text:
            self.rows = [dict(row) for row in self.source_rows]
            self.result = self.rows
        elif "from ai_validation_cases" in text:
            digest = self.source_rows[0]["sha256"] if self.validation_collision else "f" * 64
            self.rows = [{"image_sha256": digest}]
            self.result = self.rows
        elif "from ai_final_test_sessions" in text:
            self.result = {
                "id": 1, "category": "FINAL_TEST", "threshold_fixed": 47.32,
                "preprocessing_profile": "FULL_ROI", "status": "EVALUADA",
            }
        elif "from ai_final_test_cases" in text and "group by category" in text:
            self.rows = [{"category": "NORMAL", "total": 10}, {"category": "MANCHA", "total": 10}]
            self.result = self.rows
        elif "from ai_final_test_cases" in text:
            digest = self.source_rows[0]["sha256"] if self.final_collision else "e" * 64
            self.rows = [{"image_sha256": digest}]
            self.result = self.rows
        elif "from ai_training_images ti" in text and "join ai_capture_sessions cs" in text:
            self.rows = [{"id": row["id"]} for row in self.source_rows]
            self.result = self.rows
        elif "from ai_capture_sessions" in text and "status='abierta'" in text:
            self.result = None
        elif "from ai_jobs j" in text:
            self.result = None
        elif "from garment_ai_models" in text and "where garment_model_id=%s order by id for update" in text:
            self.rows = [{"id": 1649, "version": "v1", "status": "VALIDACION", "active": 0}]
            self.result = self.rows
        elif text.startswith("insert into garment_ai_models"):
            self.inserted_model = True
            self.result = None
            self.lastrowid = 2070
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.rows


class NormalV2PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, cur):
        return prepare_normal_v2_version(
            cur,
            garment_model_id=1082,
            parent_ai_model_id=1649,
            actor_id=9,
            artifacts_root=self.root,
            min_new_images=30,
        )

    def test_prepares_inactive_v2_with_exactly_27_normal_training_images_and_no_job(self):
        cur = PrepareV2Cursor(self.root)
        with patch.object(ai_domain, "_lock_garment_model"), \
             patch.object(ai_domain, "create_ai_dataset", return_value={"id": 522}), \
             patch.object(ai_domain, "add_images_to_dataset", return_value=27) as add_images, \
             patch.object(ai_domain, "record_ai_event") as audit:
            result = self.prepare(cur)
        self.assertEqual(result["version"], "v2")
        self.assertEqual(result["status"], "PREPARACION")
        self.assertEqual(result["active"], 0)
        self.assertEqual(result["inherited_good_images"], 27)
        self.assertEqual(result["minimum_new_good_images"], 30)
        self.assertEqual(result["recommended_new_good_images"], 40)
        self.assertFalse(result["training_started"])
        self.assertFalse(result["checkpoint_created"])
        self.assertTrue(cur.inserted_model)
        self.assertEqual(add_images.call_args.args[2], list(range(1, 28)))
        self.assertEqual(audit.call_args.kwargs["payload"]["source_dataset_id"], 521)
        self.assertFalse(any("insert into ai_jobs" in sql for sql, _ in cur.sql))
        self.assertFalse(any("update garment_ai_models" in sql for sql, _ in cur.sql))
        self.assertFalse(any("update ai_validation_cases" in sql for sql, _ in cur.sql))
        self.assertFalse(any("update ai_final_test_cases" in sql for sql, _ in cur.sql))
        self.assertEqual(cur.checkpoint_bytes, b"immutable-v1-checkpoint")

    def test_validation_hash_collision_blocks_creation_without_mutating_v1(self):
        cur = PrepareV2Cursor(self.root, validation_collision=True)
        with patch.object(ai_domain, "_lock_garment_model"), \
             patch.object(ai_domain, "create_ai_dataset") as create_dataset:
            with self.assertRaisesRegex(AIDomainError, "VALIDATION"):
                self.prepare(cur)
        self.assertFalse(cur.inserted_model)
        create_dataset.assert_not_called()

    def test_final_test_hash_collision_blocks_creation(self):
        cur = PrepareV2Cursor(self.root, final_collision=True)
        with patch.object(ai_domain, "_lock_garment_model"), \
             patch.object(ai_domain, "create_ai_dataset") as create_dataset:
            with self.assertRaisesRegex(AIDomainError, "FINAL_TEST"):
                self.prepare(cur)
        self.assertFalse(cur.inserted_model)
        create_dataset.assert_not_called()

    def test_manifest_must_be_normal_only(self):
        cur = PrepareV2Cursor(self.root, bad_manifest_category=True)
        with patch.object(ai_domain, "_lock_garment_model"), \
             patch.object(ai_domain, "create_ai_dataset") as create_dataset:
            with self.assertRaisesRegex(AIDomainError, "normal"):
                self.prepare(cur)
        self.assertFalse(cur.inserted_model)
        create_dataset.assert_not_called()

    def test_v2_training_artifact_paths_are_not_v1_paths(self):
        v1 = ai_training.model_artifact_paths(1082, 1649)
        v2 = ai_training.model_artifact_paths(1082, 2070)
        self.assertNotEqual(v1["training_dir"], v2["training_dir"])
        self.assertNotEqual(v1["checkpoint_dir"], v2["checkpoint_dir"])
        self.assertIn("ai_model_2070", str(v2["training_dir"]))


class NormalV2NewImageMinimumTests(unittest.TestCase):
    class SummaryCursor:
        def __init__(self, new_count):
            self.results = iter([
                {"id": 2070, "garment_model_id": 1082, "version": "v2",
                 "status": "PREPARACION", "parent_ai_model_id": 1649,
                 "source_dataset_id": 521, "dataset_id": 522,
                 "normal_augmentation_min_new_images": 30},
                {"total": 27},
                {"total": new_count},
                {"total": new_count},
                {"total": 27 + new_count},
            ])

        def execute(self, sql, params=()):
            pass

        def fetchone(self):
            return next(self.results)

    def test_29_new_images_do_not_meet_minimum(self):
        state = normal_augmentation_summary(self.SummaryCursor(29), 2070)
        self.assertEqual(state["inherited_good_images"], 27)
        self.assertEqual(state["new_good_images"], 29)
        self.assertEqual(state["new_good_images_completed"], 29)
        self.assertEqual(state["total_available"], 56)
        self.assertFalse(state["can_train"])

    def test_30_new_images_meet_minimum_without_counting_inherited_images(self):
        state = normal_augmentation_summary(self.SummaryCursor(30), 2070)
        self.assertEqual(state["inherited_good_images"], 27)
        self.assertEqual(state["new_good_images_completed"], 30)
        self.assertEqual(state["minimum_new_good_images"], 30)
        self.assertTrue(state["can_train"])

    def test_40_new_images_meet_recommended_goal(self):
        state = normal_augmentation_summary(self.SummaryCursor(40), 2070)
        self.assertEqual(state["recommended_new_good_images"], 40)
        self.assertEqual(state["new_good_images"], 40)
        self.assertEqual(state["total_available"], 67)
        self.assertTrue(state["can_train"])


class NormalV2TrainingIsolationTests(unittest.TestCase):
    class IsolationCursor:
        def __init__(self, *, validation_collision=False, final_collision=False):
            self.result = None
            self.rows = []
            self.source = [
                {"id": i, "sha256": hashlib.sha256(f"seed-{i}".encode()).hexdigest(), "status": "ACEPTADA", "garment_model_id": 1082}
                for i in range(1, 28)
            ]
            self.new = [{
                "id": 300, "sha256": hashlib.sha256(b"new-v2-good").hexdigest(),
                "status": "ACEPTADA", "garment_model_id": 1082,
            }]
            self.validation_hash = self.new[0]["sha256"] if validation_collision else "d" * 64
            self.final_hash = self.new[0]["sha256"] if final_collision else "e" * 64

        def execute(self, sql, params=()):
            text = " ".join(sql.lower().split())
            if "from garment_ai_models where id=%s for update" in text:
                self.result = {
                    "source_dataset_id": 521, "parent_ai_model_id": 1649,
                    "version": "v2", "status": "PREPARACION",
                }
            elif "from ai_dataset_images di" in text:
                self.rows = self.source
                self.result = self.rows
            elif "where cs.target_ai_model_id=%s" in text:
                self.rows = self.new
                self.result = self.rows
            elif "where ti.garment_model_id" in text and "target_ai_model_id" not in text:
                self.rows = self.source + self.new
                self.result = self.rows
            elif "from ai_validation_cases" in text:
                self.rows = [{"image_sha256": self.validation_hash}]
                self.result = self.rows
            elif "from ai_final_test_cases" in text:
                self.rows = [{"image_sha256": self.final_hash}]
                self.result = self.rows
            else:
                raise AssertionError(f"Unexpected SQL: {sql}")

        def fetchone(self):
            return self.result

        def fetchall(self):
            return self.rows

    def test_training_snapshot_contains_seed_plus_new_good_and_no_history_hashes(self):
        cur = self.IsolationCursor()
        result = ai_training.verify_normal_expansion_capture_isolation(
            cur, ai_model_id=2070, garment_model_id=1082
        )
        self.assertEqual(len(result["inherited_ids"]), 27)
        self.assertEqual(len(result["new_ids"]), 1)
        self.assertEqual(len(result["training_hashes"] & result["validation_hashes"]), 0)
        self.assertEqual(len(result["training_hashes"] & result["final_test_hashes"]), 0)

    def test_new_capture_matching_validation_mancha_is_rejected(self):
        cur = self.IsolationCursor(validation_collision=True)
        with self.assertRaisesRegex(AIDomainError, "VALIDATION"):
            ai_training.verify_normal_expansion_capture_isolation(
                cur, ai_model_id=2070, garment_model_id=1082
            )

    def test_new_capture_matching_final_test_mancha_is_rejected(self):
        cur = self.IsolationCursor(final_collision=True)
        with self.assertRaisesRegex(AIDomainError, "FINAL_TEST"):
            ai_training.verify_normal_expansion_capture_isolation(
                cur, ai_model_id=2070, garment_model_id=1082
            )

    def test_only_explicit_buena_can_be_accepted_for_v2_training(self):
        validate_normal_v2_capture_confirmation(
            version="v2", decision="accept", human_category="BUENA"
        )
        with self.assertRaisesRegex(AIDomainError, "MANCHA/DEFECTUOSA"):
            validate_normal_v2_capture_confirmation(
                version="v2", decision="accept", human_category="MANCHA"
            )
        with self.assertRaisesRegex(AIDomainError, "Confirme explícitamente BUENA"):
            validate_normal_v2_capture_confirmation(
                version="v2", decision="accept", human_category=None
            )


if __name__ == "__main__":
    unittest.main()
