import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from ai_domain import atomic_write_bytes, compute_manifest_hash
from ai_validation import export_validation_bank


class BankCursor:
    def __init__(self, root, cases, training, model=None):
        self.root = Path(root)
        self.cases = cases
        self.training = training
        self.model = model or {
            "id": 1052, "garment_model_id": 35, "version": "v2",
            "dataset_id": 47, "active": 0, "status": "VALIDACION",
        }
        self.result = None

    def execute(self, sql, params=()):
        normalized = " ".join(sql.lower().split())
        if "from garment_ai_models" in normalized:
            self.result = self.model
        elif "from ai_validation_cases" in normalized:
            self.result = self.cases
        elif "from ai_dataset_images" in normalized:
            self.result = self.training
        elif "from ai_datasets" in normalized:
            self.result = {
                "manifest_path": "garment_35/datasets/dataset_47/manifest.json",
                "manifest_hash": self.manifest_hash,
                "image_count": len(self.training),
            }
        else:
            raise AssertionError(sql)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result


class ValidationBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        training_hash = hashlib.sha256(b"training-good").hexdigest()
        manifest_hash = compute_manifest_hash([{"id": 1, "sha256": training_hash}])
        manifest = (json.dumps({"image_count": 1, "manifest_hash": manifest_hash,
                                "images": [{"image_id": 1, "sha256": training_hash}]}) + "\n").encode()
        path = self.root / "garment_35/datasets/dataset_47/manifest.json"
        atomic_write_bytes(path, manifest)
        self.training_hash = training_hash
        self.training = [{"id": 1, "sha256": self.training_hash,
                          "image_path": "garment_35/training/good.jpg"}]
        atomic_write_bytes(self.root / self.training[0]["image_path"], b"training-good")
        self.cursor = BankCursor(self.root, [], self.training)
        self.cursor.manifest_hash = manifest_hash

    def tearDown(self):
        self.temp.cleanup()

    def case(self, case_id, category, session, data=None, original_path=None):
        data = data if data is not None else f"case-{case_id}".encode()
        rel = original_path or f"validations/garment_35/model_1052/normal/case_{case_id}/original.jpg"
        atomic_write_bytes(self.root / rel, data)
        return {
            "id": case_id, "validation_session_id": session,
            "ai_model_id": 1052, "garment_model_id": 35,
            "category": category, "image_path": rel,
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "anomaly_score": 42.5, "heatmap_path": None,
            "comparison_path": None, "created_at": datetime(2026, 1, 2),
        }

    def export_bank(self):
        return export_validation_bank(self.cursor, ai_model_id=1052,
                                      artifacts_root=self.root)

    def test_exports_current_classification_across_sessions_and_empty_category(self):
        self.cursor.cases = [self.case(1, "NORMAL", 61),
                             self.case(2, "MANCHA", 62,
                                       original_path="validations/garment_35/model_1052/normal/case_2/original.jpg"),
                             self.case(3, "AGUJERO", 64)]
        result = self.export_bank()
        bank = Path(result["path"])
        manifest = json.loads((bank / "manifest.json").read_text("utf-8"))
        self.assertEqual(result["counts"], {"buenas": 1, "manchas": 1, "agujeros": 1, "total": 3})
        self.assertEqual(manifest["source_validation_sessions"], [61, 62, 64])
        self.assertEqual(manifest["cases"][1]["validation_session_id"], 62)
        self.assertEqual(manifest["cases"][1]["tipo_defecto"], "MANCHA")
        self.assertEqual(len(list((bank / "buenas").glob("*"))), 1)
        self.assertEqual(len(list((bank / "manchas").glob("*"))), 1)
        self.assertEqual(len(list((bank / "agujeros").glob("*"))), 1)

    def test_zero_holes_and_repeated_export_are_idempotent(self):
        self.cursor.cases = [self.case(1, "NORMAL", 61)]
        first = self.export_bank()
        manifest_first = Path(first["manifest_path"]).read_bytes()
        files_first = sorted(str(p.relative_to(first["path"])) for p in Path(first["path"]).rglob("*") if p.is_file())
        second = self.export_bank()
        files_second = sorted(str(p.relative_to(second["path"])) for p in Path(second["path"]).rglob("*") if p.is_file())
        manifest = json.loads(Path(second["manifest_path"]).read_text("utf-8"))
        self.assertEqual(first["counts"]["agujeros"], 0)
        self.assertEqual(files_first, files_second)
        self.assertEqual(manifest["counts"]["total"], 1)
        self.assertEqual(hashlib.sha256(manifest_first).hexdigest(), first["manifest_sha256"])
        self.assertEqual(len(list((Path(second["path"]) / "buenas").glob("*"))), 1)

    def test_training_validation_hash_collision_blocks_export(self):
        data = b"training-good"
        self.cursor.cases = [self.case(1, "NORMAL", 61, data=data)]
        with self.assertRaisesRegex(ValueError, "Contaminación TRAIN/VALIDATION"):
            self.export_bank()
        self.assertFalse((self.root / "garment_35/validation_datasets/ai_model_1052_v2/manifest.json").exists())

    def test_duplicate_validation_hashes_block_without_losing_case_ids(self):
        self.cursor.cases = [self.case(1, "NORMAL", 61, data=b"same"),
                             self.case(2, "MANCHA", 62, data=b"same")]
        with self.assertRaisesRegex(ValueError, "case_ids=\[1, 2\]"):
            self.export_bank()


if __name__ == "__main__":
    unittest.main()
