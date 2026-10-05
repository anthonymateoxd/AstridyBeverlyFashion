import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_domain
from ai_domain import AIDomainError, compute_manifest_hash, normal_augmentation_summary, prepare_normal_augmentation_version


class PrepareCursor:
    def __init__(self, root, rows, validation_hashes=()):
        self.root = Path(root)
        self.rows = rows
        self.validation_hashes = list(validation_hashes)
        self.result = None
        self.lastrowid = 1053
        self.inserted_model = False

    def execute(self, sql, params=()):
        text = " ".join(sql.lower().split())
        if "from garment_ai_models where id =" in text and "for update" in text:
            self.result = {
                "id": 1052, "garment_model_id": 35, "version": "v2",
                "model_type": "PatchCore", "status": "VALIDACION",
                "active": 0, "dataset_id": 47,
                "threshold_final": 50.0,
                "threshold_frozen_at": "2026-10-05 16:01:18",
                "checkpoint_hash": "a" * 64,
            }
        elif "from ai_datasets where id =" in text and "for update" in text:
            self.result = {
                "id": 47, "garment_model_id": 35, "version": "d1",
                "status": "CERRADO", "image_count": 52,
                "manifest_path": "garment_35/datasets/dataset_47/manifest.json",
                "manifest_hash": compute_manifest_hash(self.rows),
            }
        elif "join ai_training_images ti" in text and "where di.dataset_id" in text:
            self.result = self.rows
        elif "from ai_validation_cases" in text:
            self.result = [{"image_sha256": digest} for digest in self.validation_hashes]
        elif "from ai_training_images ti" in text and "join ai_capture_sessions cs" in text:
            self.result = [{"id": row["id"], "sha256": row["sha256"]} for row in self.rows]
        elif "from ai_capture_sessions" in text and "status = 'abierta'" in text:
            self.result = None
        elif "from ai_jobs j" in text:
            self.result = None
        elif "from garment_ai_models" in text and "where garment_model_id" in text:
            self.result = [
                {"id": 155, "version": "v1", "status": "ENTRENADO", "notes": None,
                 "parent_ai_model_id": None, "source_dataset_id": None},
                {"id": 1052, "version": "v2", "status": "VALIDACION", "notes": None,
                 "parent_ai_model_id": None, "source_dataset_id": None},
            ]
        elif text.startswith("insert into garment_ai_models"):
            self.inserted_model = True
            self.result = None
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result


class QueueCursor:
    def __init__(self, results):
        self.results = iter(results)

    def execute(self, sql, params=()):
        pass

    def fetchone(self):
        return next(self.results)


class Phase3CPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for image_id in range(1, 53):
            content = f"good-training-frame-{image_id}".encode()
            digest = hashlib.sha256(content).hexdigest()
            rel = f"garment_35/capture_sessions/session_558/accepted/frame_{image_id}.jpg"
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.rows.append({
                "id": image_id,
                "sha256": digest,
                "image_path": rel,
                "status": "ACEPTADA",
                "garment_model_id": 35,
            })
        digest = compute_manifest_hash(self.rows)
        manifest = {
            "image_count": 52,
            "manifest_hash": digest,
            "images": [{"image_id": row["id"], "sha256": row["sha256"]} for row in self.rows],
        }
        manifest_path = self.root / "garment_35/datasets/dataset_47/manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_v3_links_only_the_52_verified_good_images_without_training(self):
        cur = PrepareCursor(self.root, self.rows)
        with patch.object(ai_domain, "_lock_garment_model"), \
             patch.object(ai_domain, "create_ai_dataset", return_value={"id": 48}), \
             patch.object(ai_domain, "add_images_to_dataset", return_value=52) as add, \
             patch.object(ai_domain, "record_ai_event") as audit:
            result = prepare_normal_augmentation_version(
                cur, garment_model_id=35, parent_ai_model_id=1052,
                actor_id=8, artifacts_root=self.root,
            )
        self.assertEqual(result["version"], "v3")
        self.assertEqual(result["status"], "PREPARACION")
        self.assertEqual(result["inherited_good_images"], 52)
        self.assertEqual(result["new_good_images"], 0)
        self.assertFalse(result["training_started"])
        self.assertTrue(cur.inserted_model)
        self.assertEqual(len(add.call_args.args[2]), 52)
        self.assertEqual(audit.call_args.kwargs["payload"]["source_dataset_id"], 47)

    def test_validation_hash_collision_blocks_v3_preparation(self):
        colliding = self.rows[0]["sha256"]
        cur = PrepareCursor(self.root, self.rows, validation_hashes=[colliding])
        with patch.object(ai_domain, "_lock_garment_model"), \
             patch.object(ai_domain, "create_ai_dataset") as create:
            with self.assertRaisesRegex(AIDomainError, "Contaminación"):
                prepare_normal_augmentation_version(
                    cur, garment_model_id=35, parent_ai_model_id=1052,
                    actor_id=8, artifacts_root=self.root,
                )
        self.assertFalse(cur.inserted_model)
        create.assert_not_called()

    def test_training_is_gated_on_new_completed_good_images_not_the_52_inherited(self):
        cur = QueueCursor([
            {"id": 1053, "garment_model_id": 35, "version": "v3",
             "status": "PREPARACION", "parent_ai_model_id": 1052,
             "source_dataset_id": 47, "dataset_id": 48,
             "normal_augmentation_min_new_images": 20},
            {"total": 52},
            {"total": 19},
            {"total": 19},
            {"total": 52},
        ])
        summary = normal_augmentation_summary(cur, 1053)
        self.assertEqual(summary["inherited_good_images"], 52)
        self.assertEqual(summary["new_good_images"], 19)
        self.assertEqual(summary["total_available"], 71)
        self.assertEqual(summary["new_good_images_completed"], 19)
        self.assertFalse(summary["can_train"])


if __name__ == "__main__":
    unittest.main()
