"""Tests FASE 3B — validación controlada de la versión entrenada.

Cubre:
- dominio: categorías, rutas del banco de validación, scores y métricas;
- aislamiento: predict(ai_model_id=...) resuelve el artefacto de ESA
  versión y jamás el checkpoint productivo (PATCHCORE_CKPT);
- separación: la imagen de validación no entra a ai_training_images ni
  al dataset, y la validación no altera los artefactos de entrenamiento;
- API: sesión, casos, evaluación (métricas + candidatos de umbral) y
  cierre VALIDADO — nunca ACTIVO;
- UI y permisos de la pantalla «Validar modelo».

NO entrena PatchCore de verdad, NO activa la versión, NO hace commit.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from ai_domain import (  # noqa: E402
    AIDomainError,
    VALIDATION_CATEGORIES,
    VALIDATION_CATEGORY_DIRS,
    VALIDATION_CLASSIFICATION_LABELS,
    VALIDATION_DEFECT_TYPE_LABELS,
    VALIDATION_DEFECT_TYPES,
    VALIDATION_ESTADO_LABELS,
    VALIDATION_ESTADOS,
    VALIDATION_BINARY_NOTICE,
    VALIDATION_IMAGE_NOTICE,
    VALIDATION_RESULT_INCORRECT,
    VALIDATION_RESULT_CORRECT,
    VALIDATION_RESULT_PENDING,
    VALIDATION_SESSION_STATUS_ABIERTA,
    VALIDATION_SESSION_STATUS_CERRADA,
    VALIDATION_SESSION_STATUS_EVALUADA,
    annotate_technical_invalidation,
    category_from_estado,
    defect_type_from_category,
    estado_real_from_category,
    get_ai_artifacts_root,
    normalize_validation_category,
    resolve_validation_category,
    validation_bank_root,
    validation_case_relative_dir,
    validation_case_relative_path,
    validation_classification,
    validate_validation_case_result,
)
import ai_validation  # noqa: E402

try:
    from tests.test_ai_phase3a import (  # noqa: E402
        DB_AVAILABLE,
        _TrainingDbCase,
    )
except ImportError:  # pragma: no cover - ejecución con discover -s tests
    from test_ai_phase3a import DB_AVAILABLE, _TrainingDbCase  # noqa: E402


# Secuencia global: cada imagen de validación es única (nunca colisiona
# con las sembradas para entrenar).
_VALIDATION_IMAGE_SEQUENCE = itertools.count()


# ============================================================
# DOMINIO (sin base de datos)
# ============================================================


class ValidationDomainTests(unittest.TestCase):
    """Categorías, banco de imágenes, scores, métricas y heatmap."""

    def test_category_normalization_and_aliases(self):
        self.assertEqual(normalize_validation_category("normal"), "NORMAL")
        self.assertEqual(normalize_validation_category(" Mancha "), "MANCHA")
        self.assertEqual(normalize_validation_category("hueco"), "AGUJERO")
        self.assertEqual(normalize_validation_category("AGUJEROS"), "AGUJERO")
        self.assertEqual(normalize_validation_category("manchas"), "MANCHA")
        self.assertEqual(normalize_validation_category("sin_defecto"), "NORMAL")
        self.assertEqual(normalize_validation_category("ok"), "NORMAL")

    def test_invalid_category_is_rejected(self):
        with self.assertRaises(AIDomainError) as raised:
            normalize_validation_category("RASGADO")

        self.assertIn("Categoría inválida", str(raised.exception))

    def test_categories_have_labels_and_directories(self):
        self.assertEqual(VALIDATION_CATEGORIES, ("NORMAL", "MANCHA", "AGUJERO"))

        for category in VALIDATION_CATEGORIES:
            self.assertIn(category, VALIDATION_CATEGORY_DIRS)
            self.assertTrue(VALIDATION_CATEGORY_DIRS[category])

    def test_validation_bank_layout_is_separate_from_training(self):
        relative = validation_case_relative_dir(35, 1052, "MANCHA", 7)

        self.assertEqual(
            relative,
            "validations/garment_35/model_1052/manchas/case_7",
        )
        self.assertTrue(relative.startswith("validations/"))
        self.assertNotIn("datasets/", relative)
        self.assertNotIn("checkpoints/", relative)
        self.assertNotIn("capture_sessions/", relative)

        self.assertEqual(
            validation_case_relative_dir(35, 1052, "AGUJERO", 2),
            "validations/garment_35/model_1052/aguajeros/case_2",
        )
        self.assertEqual(
            validation_case_relative_path(
                35, 1052, "NORMAL", 9, "original.png"
            ),
            "validations/garment_35/model_1052/normal/case_9/original.png",
        )
        self.assertEqual(
            validation_bank_root(Path("/tmp/artifacts")),
            Path("/tmp/artifacts") / "validations",
        )

    def test_validation_bank_layout_rejects_bad_arguments(self):
        with self.assertRaises(AIDomainError):
            validation_case_relative_dir(35, 1052, "RASGADO", 1)

        with self.assertRaises(AIDomainError):
            validation_case_relative_dir(0, 1052, "NORMAL", 1)

        with self.assertRaises(AIDomainError):
            validation_case_relative_dir(35, 1052, "NORMAL", -1)

    def test_score_is_normalized_to_percent_scale(self):
        self.assertEqual(ai_validation.normalize_score_percent(0.42), 42.0)
        self.assertEqual(ai_validation.normalize_score_percent(1), 100.0)
        self.assertEqual(ai_validation.normalize_score_percent(1.0), 100.0)
        self.assertEqual(ai_validation.normalize_score_percent(1.5), 1.5)
        self.assertEqual(ai_validation.normalize_score_percent(42), 42.0)
        self.assertEqual(ai_validation.normalize_score_percent(150), 100.0)
        self.assertEqual(ai_validation.normalize_score_percent(-5), 0.0)
        self.assertIsNone(ai_validation.normalize_score_percent(None))
        self.assertIsNone(ai_validation.normalize_score_percent("no-numerico"))
        self.assertIsNone(ai_validation.normalize_score_percent(float("nan")))

    def test_case_result_stays_pending_without_threshold(self):
        self.assertIsNone(
            validate_validation_case_result("NORMAL", 12.5, None)
        )
        self.assertIsNone(
            validate_validation_case_result("MANCHA", 88.0, None)
        )

        self.assertEqual(
            validate_validation_case_result("NORMAL", 40.0, 50.0),
            VALIDATION_RESULT_CORRECT,
        )
        self.assertEqual(
            validate_validation_case_result("NORMAL", 80.0, 50.0),
            VALIDATION_RESULT_INCORRECT,
        )
        self.assertEqual(
            validate_validation_case_result("MANCHA", 88.0, 50.0),
            VALIDATION_RESULT_CORRECT,
        )
        self.assertEqual(
            validate_validation_case_result("AGUJERO", 10.0, 50.0),
            VALIDATION_RESULT_INCORRECT,
        )

    # --------------------------------------------------------
    # FASE 3B.2 — dominio binario BUENA / DEFECTUOSA
    # --------------------------------------------------------

    def test_binary_state_derives_from_historical_categories(self):
        self.assertEqual(estado_real_from_category("NORMAL"), "BUENA")
        self.assertIsNone(defect_type_from_category("NORMAL"))

        self.assertEqual(estado_real_from_category("MANCHA"), "DEFECTUOSA")
        self.assertEqual(defect_type_from_category("MANCHA"), "MANCHA")

        self.assertEqual(estado_real_from_category("AGUJERO"), "DEFECTUOSA")
        self.assertEqual(defect_type_from_category("AGUJERO"), "AGUJERO")

    def test_binary_labels_and_notice(self):
        self.assertEqual(VALIDATION_ESTADOS, ("BUENA", "DEFECTUOSA"))
        self.assertEqual(VALIDATION_ESTADO_LABELS["BUENA"], "Buena")
        self.assertEqual(VALIDATION_ESTADO_LABELS["DEFECTUOSA"], "Defectuosa")
        self.assertEqual(
            VALIDATION_DEFECT_TYPES,
            ("MANCHA", "AGUJERO"),
        )
        self.assertEqual(
            VALIDATION_CLASSIFICATION_LABELS["NORMAL"], "Buena"
        )
        self.assertEqual(
            VALIDATION_CLASSIFICATION_LABELS["MANCHA"],
            "Defectuosa · Mancha",
        )
        self.assertEqual(
            VALIDATION_CLASSIFICATION_LABELS["AGUJERO"],
            "Defectuosa · Agujero",
        )
        self.assertIn("PatchCore evalúa si la prenda", VALIDATION_BINARY_NOTICE)
        self.assertIn(
            "tipo de defecto lo indica la persona", VALIDATION_BINARY_NOTICE
        )
        self.assertNotEqual(
            VALIDATION_DEFECT_TYPE_LABELS["MANCHA"],
            VALIDATION_DEFECT_TYPE_LABELS["AGUJERO"],
        )

    def test_binary_resolution_rules(self):
        self.assertEqual(
            resolve_validation_category(estado_real="BUENA"), "NORMAL"
        )
        self.assertEqual(
            resolve_validation_category(
                estado_real="defectuosa", tipo_defecto="mancha"
            ),
            "MANCHA",
        )
        self.assertEqual(
            resolve_validation_category(
                estado_real="DEFECTUOSA", tipo_defecto="Agujero"
            ),
            "AGUJERO",
        )

        with self.assertRaises(AIDomainError):
            resolve_validation_category(estado_real="DEFECTUOSA")

        with self.assertRaises(AIDomainError):
            resolve_validation_category(
                estado_real="BUENA", tipo_defecto="MANCHA"
            )

        with self.assertRaises(AIDomainError):
            resolve_validation_category(estado_real="DEFECTUOSA",
                                        tipo_defecto="RASGADO")

        with self.assertRaises(AIDomainError):
            resolve_validation_category()

        with self.assertRaises(AIDomainError):
            category_from_estado("BUENA", "AGUJERO")

        # El formato histórico sigue funcionando (compatibilidad).
        self.assertEqual(
            resolve_validation_category(category="NORMAL"), "NORMAL"
        )
        self.assertEqual(
            resolve_validation_category(category="AGUJERO"), "AGUJERO"
        )

        with self.assertRaises(AIDomainError):
            resolve_validation_category(
                category="NORMAL", tipo_defecto="MANCHA"
            )

        with self.assertRaises(AIDomainError):
            resolve_validation_category(
                category="MANCHA", tipo_defecto="AGUJERO"
            )

    def test_classification_payload_for_the_ui(self):
        buena = validation_classification("NORMAL")
        self.assertEqual(buena["estado_real"], "BUENA")
        self.assertIsNone(buena["tipo_defecto"])
        self.assertIsNone(buena["tipo_defecto_label"])
        self.assertEqual(buena["classification"], "Buena")

        mancha = validation_classification("MANCHA")
        self.assertEqual(mancha["estado_real"], "DEFECTUOSA")
        self.assertEqual(mancha["tipo_defecto"], "MANCHA")
        self.assertEqual(mancha["tipo_defecto_label"], "Mancha")
        self.assertEqual(mancha["classification"], "Defectuosa · Mancha")

        agujero = validation_classification("AGUJERO")
        self.assertEqual(agujero["tipo_defecto"], "AGUJERO")
        self.assertEqual(agujero["classification"], "Defectuosa · Agujero")

    @staticmethod
    def _synthetic_cases():
        return [
            {"category": "NORMAL", "anomaly_score": 10.0},
            {"category": "NORMAL", "anomaly_score": 20.0},
            {"category": "MANCHA", "anomaly_score": 80.0},
            {"category": "AGUJERO", "anomaly_score": 90.0},
            {"category": "MANCHA", "anomaly_score": 30.0},
        ]

    def test_metrics_without_threshold_stay_pending(self):
        metrics = ai_validation.compute_validation_metrics(
            self._synthetic_cases()
        )

        self.assertEqual(metrics["total"], 5)
        self.assertEqual(metrics["scored"], 5)
        self.assertEqual(metrics["pending_scores"], 0)
        self.assertTrue(metrics["pending_calibration"])
        self.assertIsNone(metrics["threshold"])
        self.assertIsNone(metrics["confusion"])
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["best_threshold"])
        self.assertEqual(metrics["status"], VALIDATION_RESULT_PENDING)
        self.assertEqual(metrics["candidates"], [])

        by_category = metrics["by_category"]
        self.assertEqual(by_category["NORMAL"]["count"], 2)
        self.assertEqual(by_category["MANCHA"]["count"], 2)
        self.assertEqual(by_category["AGUJERO"]["count"], 1)

        distribution = by_category["NORMAL"]["distribution"]
        self.assertEqual(distribution["min"], 10.0)
        self.assertEqual(distribution["max"], 20.0)
        self.assertEqual(distribution["mean"], 15.0)
        self.assertEqual(distribution["median"], 15.0)

    def test_metrics_with_threshold_report_confusion(self):
        metrics = ai_validation.compute_validation_metrics(
            self._synthetic_cases(),
            threshold=50.0,
            candidates=[30.0, 50.0, 70.0],
        )

        self.assertEqual(metrics["threshold"], 50.0)
        self.assertFalse(metrics["pending_calibration"])
        self.assertEqual(metrics["confusion"], {"tp": 2, "fp": 0, "fn": 1, "tn": 2})
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.6667)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["f1"], 0.8)
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertEqual(metrics["status"], "EVALUADO")

    def test_metrics_are_binary_and_keep_the_defect_breakdown(self):
        cases = self._synthetic_cases()

        pending = ai_validation.compute_validation_metrics(cases)

        self.assertEqual(
            pending["estado_counts"], {"BUENA": 2, "DEFECTUOSA": 3}
        )
        self.assertEqual(pending["by_estado"]["BUENA"]["count"], 2)
        self.assertEqual(pending["by_estado"]["DEFECTUOSA"]["count"], 3)
        self.assertEqual(pending["defect_counts"], {"MANCHA": 2, "AGUJERO": 1})

        # Sin umbral el desglose sólo cuenta casos (sin detectadas).
        self.assertIsNone(pending["defects"]["threshold"])
        self.assertEqual(pending["defects"]["MANCHA"]["evaluated"], 2)
        self.assertIsNone(pending["defects"]["MANCHA"]["detected"])

        evaluated = ai_validation.compute_validation_metrics(
            cases, threshold=50.0, candidates=[30.0, 50.0]
        )

        self.assertEqual(
            evaluated["confusion"], {"tp": 2, "fp": 0, "fn": 1, "tn": 2}
        )
        self.assertEqual(evaluated["threshold_reference"], 50.0)
        self.assertEqual(evaluated["threshold_source"], "applied")
        self.assertEqual(evaluated["defects"]["threshold"], 50.0)
        self.assertEqual(
            evaluated["defects"]["MANCHA"],
            {"evaluated": 2, "detected": 1, "missed": 1},
        )
        self.assertEqual(
            evaluated["defects"]["AGUJERO"],
            {"evaluated": 1, "detected": 1, "missed": 0},
        )

        # Con sólo candidatos se conserva el umbral aplicado = None y
        # las métricas binarias quedan en threshold_candidate_metrics.
        candidate = ai_validation.compute_validation_metrics(
            cases, candidates=[50.0]
        )
        self.assertIsNone(candidate["threshold"])
        self.assertIsNone(candidate["confusion"])
        self.assertEqual(candidate["threshold_source"], "candidate")
        self.assertEqual(
            candidate["threshold_candidate_metrics"]["fp"], 0
        )
        self.assertEqual(
            candidate["threshold_candidate_metrics"]["fn"], 1
        )

    def test_threshold_candidates_are_ranked_by_f1(self):
        cases = self._synthetic_cases()
        evaluated = ai_validation.evaluate_threshold_candidates(
            cases,
            [30.0, 50.0, 70.0],
        )

        self.assertEqual(len(evaluated), 3)
        self.assertEqual(evaluated[0]["threshold"], 30.0)
        self.assertEqual(evaluated[0]["f1"], 1.0)
        self.assertTrue(
            all(
                evaluated[index]["f1"] >= evaluated[index + 1]["f1"]
                for index in range(len(evaluated) - 1)
            )
        )

        candidates = ai_validation.default_threshold_candidates(cases)

        self.assertIn(5, candidates)
        self.assertIn(15, candidates)   # punto medio 10/20
        self.assertIn(55, candidates)   # punto medio 30/80
        self.assertEqual(candidates, sorted(set(candidates)))

        metrics = ai_validation.compute_validation_metrics(
            cases,
            candidates=candidates,
        )
        self.assertIn(
            metrics["best_threshold"],
            [item["threshold"] for item in metrics["candidates"]],
        )

    def test_binary_metrics_ignores_rows_without_score(self):
        metrics = ai_validation.binary_metrics(
            [
                {"category": "NORMAL", "anomaly_score": None},
                {"category": "MANCHA", "anomaly_score": 80.0},
            ],
            50.0,
        )

        self.assertEqual(metrics["evaluated"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_heatmap_and_comparison_shapes(self):
        try:
            import cv2
            import numpy as np
        except Exception:  # pragma: no cover - entorno sin cv2
            self.skipTest("cv2/numpy no disponibles")

        self.assertIsNone(ai_validation.render_heatmap(None))
        self.assertIsNone(ai_validation.render_comparison(None, None))

        values = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
        heatmap = ai_validation.render_heatmap(
            values,
            reference_shape=(64, 64, 3),
        )

        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape, (64, 64, 3))

        constant = ai_validation.render_heatmap(np.ones((8, 8)))
        self.assertIsNotNone(constant)

        original = np.full((64, 64, 3), 200, dtype=np.uint8)
        comparison = ai_validation.render_comparison(original, heatmap)

        self.assertIsNotNone(comparison)
        self.assertEqual(comparison.shape, (64, 128, 3))

        png = ai_validation._encode_png(comparison)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIsNone(ai_validation._encode_png(None))
        self.assertEqual(cv2.imdecode(np.frombuffer(png, np.uint8), 1).shape, (64, 128, 3))


# ============================================================
# PERSISTENCIA / API / UI (base de datos)
# ============================================================


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de validación FASE 3B omitidos.",
)
class ValidationDbTests(_TrainingDbCase):
    """Sesiones/casos, inferencia aislada, métricas y permisos."""

    ADMIN_USERNAME = "ph3b_admin"
    MM_USERNAME = "ph3b_mm"
    QM_USERNAME = "ph3b_qm"

    MODEL_CODE = "TEST-PH3B-VALID"
    FREEZE_CODE = "TEST-PH3B-FREEZE"
    POST_FREEZE_CODE = "TEST-PH3B-POST-FREEZE"
    INVALIDATED_CODE = "TEST-PH3B-INVAL"
    MODELS = (MODEL_CODE, INVALIDATED_CODE, FREEZE_CODE, POST_FREEZE_CODE)

    USERS = (
        (ADMIN_USERNAME, "ADMIN"),
        (MM_USERNAME, "MODEL_MANAGER"),
        (QM_USERNAME, "QUALITY_MANAGER"),
    )

    class _FakeTrainer:
        """Trainer falso: escribe un checkpoint trivial (sin anomalib)."""

        def train(self, ctx):
            ctx["report"](40, "EMBEDDINGS", "Representaciones de validación")

            checkpoint = Path(ctx["checkpoint_dir"]) / "model.ckpt"
            checkpoint.write_bytes(b"FAKE-CHECKPOINT-3B-" + b"0" * 512)

            return {
                "checkpoint": str(checkpoint),
                "memory_bank": {
                    "shape": [11, 1536],
                    "dtype": "torch.float32",
                },
                "image_size": [256, 256],
                "accelerator": "cpu",
            }

    # --------------------------------------------------------
    # ciclo de vida
    # --------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        actor = cls.user_ids[cls.ADMIN_USERNAME]
        manager = cls.user_ids[cls.MM_USERNAME]

        for code in cls.MODELS:
            model_id = cls.model_ids[code]

            # Ejecuciones anteriores interrumpidas pudieron dejar
            # versiones/datasets: se purga antes de reentrenar.
            cls._delete_model(model_id)
            cls._seed_accepted(model_id, actor)

            # El MODEL_MANAGER «creador» del modelo puede validarlo
            # (can_manage_garment_model exige created_by == usuario).
            cls.cur.execute(
                "UPDATE garment_models SET created_by = %s WHERE id = %s",
                (manager, model_id),
            )
            cls.conn.commit()

            cls._train_model(model_id, actor)

        cls._invalidate_latest(cls.model_ids[cls.INVALIDATED_CODE], actor)

    @classmethod
    def tearDownClass(cls):
        for code in cls.MODELS:
            model_id = cls.model_ids.get(code)

            if model_id:
                cls._clear_validation(model_id)

        ai_validation.clear_model_inspectors()
        super().tearDownClass()

    def setUp(self):
        self._login(self.ADMIN_USERNAME)
        self._cancel_stray_jobs()

        for code in self.MODELS:
            self._clear_validation(self.model_ids[code])

        ai_validation.clear_model_inspectors()

    # --------------------------------------------------------
    # utilidades de clase
    # --------------------------------------------------------

    @classmethod
    def _cancel_stray_jobs(cls):
        cls.cur.execute(
            """
            UPDATE ai_jobs
            SET status = 'CANCELADO', finished_at = NOW()
            WHERE status = 'PENDIENTE'
            """
        )
        cls.conn.commit()

    @classmethod
    def _train_model(cls, model_id, actor_id):
        """Encola y ejecuta un TRAINING real con el trainer falso."""
        import ai_worker
        from ai_training import execute_training_job

        cls._cancel_stray_jobs()

        with cls.client.session_transaction() as sess:
            sess.clear()
            sess["user_id"] = actor_id

        response = cls.client.post(
            "/api/ai/training/start",
            json={"garment_model_id": model_id},
        )

        if response.status_code != 201:
            raise AssertionError(
                f"No se pudo encolar el entrenamiento de {model_id}: "
                f"{response.status_code} "
                f"{response.get_data(as_text=True)}"
            )

        payload = json.loads(response.get_data(as_text=True))
        job_id = int(payload["job_id"])

        claimed = ai_worker.claim_job(cls._connect, "ph3b-worker")

        if claimed is None:
            raise AssertionError("Ningún job de entrenamiento fue reclamado.")

        result = execute_training_job(
            job_id,
            connect=cls._connect,
            trainer=cls._FakeTrainer(),
        )

        if not result.get("ok"):
            raise AssertionError(f"El entrenamiento falló: {result}")

        cls.conn.commit()

        cls.cur.execute(
            """
            SELECT id, status, checkpoint_path
            FROM garment_ai_models
            WHERE garment_model_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_id,),
        )
        row = cls.cur.fetchone()

        if row is None or row["status"] != "ENTRENADO":
            raise AssertionError(f"Versión no entrenada: {row}")

        return int(row["id"])

    @classmethod
    def _invalidate_latest(cls, model_id, actor_id):
        cls.cur.execute(
            """
            SELECT id FROM garment_ai_models
            WHERE garment_model_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_id,),
        )
        row = cls.cur.fetchone()

        annotate_technical_invalidation(
            cls.cur,
            int(row["id"]),
            "dataset de validación de prueba inválido",
            actor_id=actor_id,
        )
        cls.conn.commit()

    @classmethod
    def _clear_validation(cls, model_id):
        """Vuelve la versión a ENTRENADO y purga sesiones/casos/eventos."""
        cls.cur.execute(
            """
            DELETE c FROM ai_validation_cases c
            JOIN garment_ai_models m ON m.id = c.ai_model_id
            WHERE m.garment_model_id = %s
            """,
            (model_id,),
        )
        cls.cur.execute(
            """
            DELETE s FROM ai_validation_sessions s
            JOIN garment_ai_models m ON m.id = s.ai_model_id
            WHERE m.garment_model_id = %s
            """,
            (model_id,),
        )
        cls.cur.execute(
            """
            DELETE e FROM ai_events e
            JOIN garment_ai_models m ON m.id = e.ai_model_id
            WHERE m.garment_model_id = %s
              AND e.event_type IN (
                    'VALIDATION_STARTED',
                    'VALIDATION_CASE_REGISTERED',
                    'VALIDATION_CASE_CATEGORY_CHANGED',
                    'VALIDATION_CASE_DELETED',
                    'VALIDATION_EVALUATED',
                    'VALIDATED'
                )
            """,
            (model_id,),
        )
        cls.cur.execute(
            """
            UPDATE garment_ai_models
            SET status = 'ENTRENADO',
                active = 0,
                validated_by = NULL,
                validated_at = NULL
            WHERE garment_model_id = %s
              AND status IN ('ENTRENADO', 'VALIDACION', 'VALIDADO')
            """,
            (model_id,),
        )
        cls.conn.commit()

    # --------------------------------------------------------
    # utilidades de instancia
    # --------------------------------------------------------

    def _ai_row(self, ai_model_id):
        self.conn.commit()
        self.cur.execute(
            "SELECT * FROM garment_ai_models WHERE id = %s",
            (ai_model_id,),
        )
        return self.cur.fetchone()

    def _session_row(self, ai_model_id):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT id, garment_model_id, ai_model_id, status,
                   threshold_candidate, metrics_json, created_by,
                   created_at, evaluated_at, closed_at
            FROM ai_validation_sessions
            WHERE ai_model_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (ai_model_id,),
        )
        return self.cur.fetchone()

    def _case_count(self, ai_model_id):
        self.conn.commit()
        self.cur.execute(
            "SELECT COUNT(*) AS total FROM ai_validation_cases "
            "WHERE ai_model_id = %s",
            (ai_model_id,),
        )
        return int(self.cur.fetchone()["total"] or 0)

    def _event_count(self, ai_model_id, event_type):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT COUNT(*) AS total FROM ai_events
            WHERE ai_model_id = %s AND event_type = %s
            """,
            (ai_model_id, event_type),
        )
        return int(self.cur.fetchone()["total"] or 0)

    def _counts(self, model_id):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM ai_training_images
                 WHERE garment_model_id = %s) AS training_images,
                (SELECT COUNT(*) FROM ai_capture_sessions
                 WHERE garment_model_id = %s) AS capture_sessions,
                (SELECT COUNT(*) FROM ai_datasets
                 WHERE garment_model_id = %s) AS datasets,
                (SELECT COUNT(*) FROM ai_dataset_images di
                 JOIN ai_datasets d ON d.id = di.dataset_id
                 WHERE d.garment_model_id = %s) AS dataset_images,
                (SELECT COUNT(*) FROM ai_jobs j
                 JOIN garment_ai_models m ON m.id = j.ai_model_id
                 WHERE m.garment_model_id = %s) AS jobs
            """,
            (model_id, model_id, model_id, model_id, model_id),
        )
        return self.cur.fetchone()

    @staticmethod
    def _snapshot(directory):
        base = Path(directory)

        if not base.is_dir():
            return {}

        return {
            str(path.relative_to(base)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(base.rglob("*"))
            if path.is_file()
        }

    def _model_artifacts_dir(self, model_id, ai_model_id):
        return (
            Path(get_ai_artifacts_root())
            / f"garment_{model_id}"
            / "models"
            / f"ai_model_{ai_model_id}"
        )

    @staticmethod
    def _new_validation_image(tag=""):
        """Imagen NUEVA (nunca igual a las sembradas para entrenar)."""
        import cv2
        import numpy as np

        seed = int.from_bytes(
            hashlib.sha256(
                f"{tag}-{next(_VALIDATION_IMAGE_SEQUENCE)}".encode()
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)

        frame = np.full((480, 640, 3), 235, dtype=np.uint8)
        frame[80:400, 100:540] = (205, 215, 228)
        frame[420:460, 60:140] = rng.integers(0, 255, (40, 80, 3), dtype=np.uint8)

        ok, buffer = cv2.imencode(".png", frame)

        if not ok:
            raise RuntimeError("No se pudo generar la imagen de validación.")

        return buffer.tobytes()

    @staticmethod
    def _fake_predict(score=42.0, is_anomaly=None, anomaly_map=None):
        """Sustituye la inferencia real conservando la firma."""
        calls = []

        def fake(ai_model_id, image_path, *, cur=None,
                 inspector_factory=None, preprocess=True):
            calls.append({
                "ai_model_id": ai_model_id,
                "image_path": str(image_path),
                "preprocess": preprocess,
            })
            return {
                "ai_model_id": ai_model_id,
                "score": score,
                "score_percent": score,
                "is_anomaly": (
                    bool(is_anomaly)
                    if is_anomaly is not None
                    else float(score) >= 55.0
                ),
                "anomaly_map": anomaly_map,
                "source": "test_fake",
            }

        fake.calls = calls
        return fake

    def _post(self, path, payload, username=None):
        self._login(username or self.ADMIN_USERNAME)
        return self.client.post(path, json=payload)

    def _post_case(self, model_id, category, *, score=42.0, is_anomaly=None,
                   anomaly_map=None, observation="", image_bytes=None,
                   username=None, ai_model_id=None):
        payload = {
            "garment_model_id": model_id,
            "category": category,
            "observation": observation,
        }

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        if image_bytes is None:
            image_bytes = self._new_validation_image(category)

        payload["image_base64"] = base64.b64encode(image_bytes).decode("ascii")

        fake = self._fake_predict(
            score=score,
            is_anomaly=is_anomaly,
            anomaly_map=anomaly_map,
        )

        with patch.object(ai_validation, "predict_with_ai_model", fake):
            response = self._post(
                "/api/ai/validation/case",
                payload,
                username=username,
            )

        return response, fake

    def _start_session(self, model_id, ai_model_id=None, username=None):
        payload = {"garment_model_id": model_id}

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        return self._post("/api/ai/validation/session/start", payload, username)

    def _evaluate(self, model_id, ai_model_id=None, thresholds=None,
                  username=None):
        payload = {"garment_model_id": model_id}

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        if thresholds is not None:
            payload["thresholds"] = thresholds

        return self._post("/api/ai/validation/evaluate", payload, username)

    def _complete(self, model_id, ai_model_id=None, username=None):
        payload = {"garment_model_id": model_id}

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        return self._post("/api/ai/validation/complete", payload, username)

    # --------------------------------------------------------
    # carga aislada de la versión
    # --------------------------------------------------------

    def test_v2_is_loaded_by_ai_model_id_not_by_global_checkpoint(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        previous = os.environ.get("PATCHCORE_CKPT")
        os.environ["PATCHCORE_CKPT"] = "/opt/producto/no_debe_leerse.ckpt"

        try:
            conn = self._connect()
            cur = conn.cursor(dictionary=True)

            try:
                bundle = ai_validation.load_model_bundle(cur, ai_model_id)
            finally:
                cur.close()
                conn.close()
        finally:
            if previous is None:
                os.environ.pop("PATCHCORE_CKPT", None)
            else:
                os.environ["PATCHCORE_CKPT"] = previous

        expected = f"garment_{model_id}/models/ai_model_{ai_model_id}/"

        self.assertTrue(
            bundle["checkpoint_path"].startswith(expected),
            bundle["checkpoint_path"],
        )
        self.assertNotIn("no_debe_leerse", bundle["checkpoint_abs"])
        self.assertNotIn("ai_model_155", bundle["checkpoint_path"])
        self.assertEqual(bundle["status"], "ENTRENADO")
        self.assertEqual(bundle["input_size"], 256)
        self.assertEqual(bundle["active"], 0)

        received = {}

        class RecordingInspector:
            def __init__(self, checkpoint_path, image_size=256):
                received["checkpoint"] = str(checkpoint_path)
                received["image_size"] = int(image_size)

            def inspect(self, _image_path):
                return {
                    "score": 0.42,
                    "is_anomaly": False,
                    "anomaly_map": None,
                }

        image_path = Path(self.tmp_root) / "ph3b_aislada.png"
        image_path.write_bytes(self._new_validation_image("aislada"))

        result = ai_validation.predict_with_ai_model(
            ai_model_id,
            str(image_path),
            inspector_factory=RecordingInspector,
            preprocess=False,
        )

        self.assertEqual(result["source"], "ai_model_artifacts")
        self.assertEqual(result["score_percent"], 42.0)
        self.assertEqual(result["ai_model_id"], ai_model_id)
        self.assertIn(f"ai_model_{ai_model_id}", received["checkpoint"])
        self.assertNotIn("no_debe_leerse", received["checkpoint"])
        self.assertEqual(received["image_size"], 256)

    def _ai_row_by_model(self, model_id):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT * FROM garment_ai_models
            WHERE garment_model_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_id,),
        )
        return self.cur.fetchone()

    # --------------------------------------------------------
    # separación training / validación
    # --------------------------------------------------------

    def test_validation_never_touches_production_or_training(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        model_before = self._ai_row(ai_model_id)
        counts_before = self._counts(model_id)
        artifacts_before = self._snapshot(
            self._model_artifacts_dir(model_id, ai_model_id)
        )
        threshold_before = self.A.PATCHCORE_SCORE_THRESHOLD
        checkpoint_before = self.A.PATCHCORE_CKPT
        inspector_before = getattr(self.A, "patchcore_inspector", None)

        response, _ = self._post_case(model_id, "MANCHA", score=61.5)
        self.assertEqual(response.status_code, 201, self._json(response))

        model_after = self._ai_row(ai_model_id)
        changed = {
            key
            for key in model_before
            if model_before[key] != model_after.get(key)
        }

        self.assertTrue(
            changed.issubset({"status", "validated_by", "validated_at"}),
            changed,
        )
        self.assertEqual(model_after["status"], "VALIDACION")
        self.assertEqual(int(model_after["active"] or 0), 0)
        self.assertEqual(
            model_after["checkpoint_path"],
            model_before["checkpoint_path"],
        )
        self.assertEqual(
            model_after["checkpoint_hash"],
            model_before["checkpoint_hash"],
        )
        self.assertEqual(
            model_after["dataset_id"],
            model_before["dataset_id"],
        )
        self.assertEqual(model_after["notes"], model_before["notes"])

        self.assertEqual(self._counts(model_id), counts_before)
        self.assertEqual(
            self._snapshot(self._model_artifacts_dir(model_id, ai_model_id)),
            artifacts_before,
        )
        self.assertEqual(self.A.PATCHCORE_SCORE_THRESHOLD, threshold_before)
        self.assertEqual(self.A.PATCHCORE_CKPT, checkpoint_before)
        self.assertIs(getattr(self.A, "patchcore_inspector", None),
                      inspector_before)

    def test_validation_image_never_enters_the_training_dataset(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        counts_before = self._counts(model_id)
        image_bytes = self._new_validation_image("separacion")

        response, _ = self._post_case(
            model_id,
            "NORMAL",
            image_bytes=image_bytes,
        )
        self.assertEqual(response.status_code, 201, self._json(response))

        case = self._json(response)["case"]
        self.assertTrue(
            case["image_path"].startswith("validations/"),
            case["image_path"],
        )
        self.assertNotIn("capture_sessions/", case["image_path"])
        self.assertEqual(case["image_sha256"], hashlib.sha256(image_bytes).hexdigest())

        self.assertEqual(self._counts(model_id), counts_before)

        self.conn.commit()
        self.cur.execute(
            "SELECT COUNT(*) AS total FROM ai_training_images "
            "WHERE sha256 = %s",
            (case["image_sha256"],),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 0)

        stored = Path(get_ai_artifacts_root()) / case["image_path"]
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.read_bytes(), image_bytes)

    def test_training_image_hash_is_rejected(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        self.conn.commit()
        self.cur.execute(
            """
            SELECT image_path FROM ai_training_images
            WHERE garment_model_id = %s AND status = 'ACEPTADA'
            ORDER BY id
            LIMIT 1
            """,
            (model_id,),
        )
        training_image = self.cur.fetchone()["image_path"]
        training_bytes = (
            Path(get_ai_artifacts_root()) / training_image
        ).read_bytes()

        response, _ = self._post_case(
            model_id,
            "NORMAL",
            image_bytes=training_bytes,
        )

        self.assertEqual(response.status_code, 409, self._json(response))
        body = self._json(response)
        self.assertIn("dataset de entrenamiento", body["error"])
        self.assertEqual(self._case_count(ai_model_id), 0)
        self.assertEqual(self._ai_row(ai_model_id)["status"], "ENTRENADO")
        self.assertIsNone(self._session_row(ai_model_id))

    # --------------------------------------------------------
    # sesión y casos
    # --------------------------------------------------------

    def test_start_session_moves_version_to_validacion(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response = self._start_session(model_id, ai_model_id)
        self.assertEqual(response.status_code, 201, self._json(response))

        body = self._json(response)
        self.assertTrue(body["ok"])
        self.assertEqual(body["session"]["status"],
                         VALIDATION_SESSION_STATUS_ABIERTA)
        self.assertEqual(body["validation"]["model_status"], "VALIDACION")
        self.assertTrue(body["validation"]["can_capture"])

        model = self._ai_row(ai_model_id)
        self.assertEqual(model["status"], "VALIDACION")
        self.assertEqual(int(model["active"] or 0), 0)
        self.assertEqual(self._event_count(ai_model_id, "VALIDATION_STARTED"), 1)

        # Reutiliza la sesión abierta en lugar de duplicarla.
        again = self._start_session(model_id, ai_model_id)
        self.assertEqual(again.status_code, 201, self._json(again))
        self.assertEqual(
            self._json(again)["session"]["id"],
            body["session"]["id"],
        )

    def test_all_categories_are_accepted_and_invalid_is_rejected(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        for category in VALIDATION_CATEGORIES:
            response, _ = self._post_case(model_id, category, score=33.0)
            self.assertEqual(
                response.status_code,
                201,
                f"{category}: {self._json(response)}",
            )
            self.assertEqual(self._json(response)["case"]["category"], category)

        response, _ = self._post_case(model_id, "RASGADO")
        self.assertEqual(response.status_code, 409, self._json(response))
        self.assertIn("Categoría inválida", self._json(response)["error"])
        self.assertEqual(self._case_count(ai_model_id), len(VALIDATION_CATEGORIES))

    def test_raw_score_is_persisted_without_threshold(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, fake = self._post_case(
            model_id,
            "MANCHA",
            score=47.25,
            is_anomaly=True,
            observation="mancha borde inferior",
        )

        self.assertEqual(response.status_code, 201, self._json(response))
        body = self._json(response)
        case = body["case"]

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["ai_model_id"], ai_model_id)
        self.assertTrue(fake.calls[0]["preprocess"])

        self.assertEqual(case["ai_model_id"], ai_model_id)
        self.assertEqual(case["anomaly_score"], 47.25)
        self.assertIsNone(case["threshold_used"])
        self.assertEqual(case["result"], VALIDATION_RESULT_PENDING)
        self.assertEqual(case["result_label"], "Pendiente de calibración")
        self.assertEqual(case["prediction"], "ANOMALIA")
        self.assertIn(VALIDATION_IMAGE_NOTICE, body["message"])
        self.assertIn(VALIDATION_IMAGE_NOTICE, case["notice"])
        self.assertIsNone(case["heatmap_path"])

        self.conn.commit()
        self.cur.execute(
            """
            SELECT anomaly_score, threshold_used, prediction, result,
                   observation, image_path
            FROM ai_validation_cases WHERE id = %s
            """,
            (case["id"],),
        )
        stored = self.cur.fetchone()

        self.assertEqual(float(stored["anomaly_score"]), 47.25)
        self.assertIsNone(stored["threshold_used"])
        self.assertEqual(stored["prediction"], "ANOMALIA")
        self.assertEqual(stored["result"], VALIDATION_RESULT_PENDING)
        self.assertEqual(stored["observation"], "mancha borde inferior")
        self.assertTrue(stored["image_path"].startswith("validations/"))

        state = body["validation"]
        self.assertTrue(state["pending_calibration"])
        self.assertEqual(state["counts"]["MANCHA"], 1)
        self.assertEqual(state["total_cases"], 1)
        self.assertIsNone(state["metrics"])

    def test_heatmap_and_comparison_artifacts_are_stored(self):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy no disponible")

        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        anomaly_map = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)

        response, _ = self._post_case(
            model_id,
            "AGUJERO",
            score=88.0,
            anomaly_map=anomaly_map,
        )

        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]

        self.assertTrue(case["heatmap_path"], case)
        self.assertTrue(case["comparison_path"], case)

        prefix = f"validations/garment_{model_id}/model_{ai_model_id}/"
        self.assertTrue(case["heatmap_path"].startswith(prefix))
        self.assertTrue(case["comparison_path"].startswith(prefix))

        root = Path(get_ai_artifacts_root())
        for relative in (case["image_path"], case["heatmap_path"],
                         case["comparison_path"]):
            self.assertTrue((root / relative).is_file(), relative)

    def test_case_asset_endpoints_serve_the_artifacts(self):
        model_id = self.model_ids[self.MODEL_CODE]

        response, _ = self._post_case(
            model_id,
            "NORMAL",
            score=12.0,
            anomaly_map=__import__("numpy").linspace(0, 1, 64).reshape(8, 8),
        )
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]

        asset = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/imagen"
            f"?case_id={case['id']}&kind=heatmap"
        )
        self.assertEqual(asset.status_code, 200)
        self.assertEqual(asset.mimetype, "image/png")

        asset = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/caso/"
            f"{case['id']}/comparison"
        )
        self.assertEqual(asset.status_code, 200)

        missing = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/imagen"
            f"?case_id={case['id']}&kind=desconocido"
        )
        self.assertEqual(missing.status_code, 404)

        other = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/imagen"
            "?case_id=99999999&kind=original"
        )
        self.assertEqual(other.status_code, 404)

    # --------------------------------------------------------
    # métricas y thresholds
    # --------------------------------------------------------

    def test_evaluation_reports_metrics_without_changing_model_status(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        for category, score in (("NORMAL", 12.0), ("MANCHA", 88.0),
                                ("AGUJERO", 35.0)):
            response, _ = self._post_case(model_id, category, score=score)
            self.assertEqual(response.status_code, 201, self._json(response))

        model_before = self._ai_row(ai_model_id)

        response = self._evaluate(
            model_id,
            ai_model_id,
            thresholds=[30, 50, 70],
        )

        self.assertEqual(response.status_code, 200, self._json(response))
        metrics = self._json(response)["result"]["metrics"]

        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["counts"], {
            "NORMAL": 1,
            "MANCHA": 1,
            "AGUJERO": 1,
        })
        self.assertEqual(metrics["status"], VALIDATION_RESULT_PENDING)
        self.assertTrue(metrics["pending_calibration"])
        self.assertIsNone(metrics["confusion"])
        self.assertIsNone(metrics["threshold"])
        self.assertEqual(len(metrics["candidates"]), 3)
        self.assertIn(
            metrics["best_threshold"],
            [item["threshold"] for item in metrics["candidates"]],
        )

        session = self._session_row(ai_model_id)
        self.assertEqual(session["status"], VALIDATION_SESSION_STATUS_EVALUADA)
        self.assertIsNotNone(session["metrics_json"])
        self.assertIsNotNone(session["evaluated_at"])
        self.assertEqual(
            float(session["threshold_candidate"]),
            float(metrics["best_threshold"]),
        )

        model_after = self._ai_row(ai_model_id)
        self.assertEqual(model_after["status"], model_before["status"])
        self.assertEqual(model_after["status"], "VALIDACION")
        self.assertEqual(int(model_after["active"] or 0), 0)
        self.assertEqual(
            model_after["checkpoint_path"],
            model_before["checkpoint_path"],
        )
        self.assertEqual(
            self._event_count(ai_model_id, "VALIDATION_EVALUATED"),
            1,
        )
        self.assertEqual(self._event_count(ai_model_id, "ACTIVATED"), 0)

    def test_evaluation_without_cases_is_rejected(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response = self._evaluate(model_id, ai_model_id)

        self.assertEqual(response.status_code, 409, self._json(response))
        self.assertIn(
            "Todavía no existe una sesión",
            self._json(response)["error"],
        )
        self.assertEqual(self._ai_row(ai_model_id)["status"], "ENTRENADO")

        started = self._start_session(model_id, ai_model_id)
        self.assertEqual(started.status_code, 201, self._json(started))

        response = self._evaluate(model_id, ai_model_id)

        self.assertEqual(response.status_code, 409, self._json(response))
        self.assertIn(
            "Todavía no hay casos",
            self._json(response)["error"],
        )
        self.assertEqual(self._ai_row(ai_model_id)["status"], "VALIDACION")

    # --------------------------------------------------------
    # cierre: VALIDADO, nunca ACTIVO
    # --------------------------------------------------------

    def test_complete_requires_evaluation_and_never_activates(self):
        model_id = self.model_ids[self.FREEZE_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        without_session = self._complete(model_id, ai_model_id)
        self.assertEqual(without_session.status_code, 409)
        self.assertIn(
            "Todavía no existe una sesión",
            self._json(without_session)["error"],
        )

        started = self._start_session(model_id, ai_model_id)
        self.assertEqual(started.status_code, 201, self._json(started))

        without_evaluation = self._complete(model_id, ai_model_id)
        self.assertEqual(without_evaluation.status_code, 409)
        self.assertIn(
            "Evalúe la validación",
            self._json(without_evaluation)["error"],
        )

        case, _ = self._post_case(model_id, "MANCHA", score=70.0)
        self.assertEqual(case.status_code, 201, self._json(case))

        evaluated = self._evaluate(model_id, ai_model_id)
        self.assertEqual(evaluated.status_code, 200, self._json(evaluated))

        not_frozen = self._complete(model_id, ai_model_id)
        self.assertEqual(not_frozen.status_code, 409, self._json(not_frozen))
        self.assertIn("congele el threshold", self._json(not_frozen)["error"])
        frozen = self._freeze_threshold(model_id, ai_model_id, 50.00)
        self.assertEqual(frozen.status_code, 200, self._json(frozen))
        self.assertEqual(
            self._freeze_threshold(model_id, ai_model_id, 50.00).status_code,
            200,
        )
        changed = self._freeze_threshold(model_id, ai_model_id, 51.00)
        self.assertEqual(changed.status_code, 409, self._json(changed))
        with self.assertRaises(Exception):
            self.cur.execute(
                "UPDATE garment_ai_models SET threshold_final = 51.00 WHERE id = %s",
                (ai_model_id,),
            )
        self.conn.rollback()

        completed = self._complete(model_id, ai_model_id)
        self.assertEqual(completed.status_code, 200, self._json(completed))

        body = self._json(completed)
        self.assertFalse(body["result"]["already_validated"])
        self.assertEqual(body["result"]["status"], "VALIDADO")
        self.assertIn("FASE 3C", body["message"])

        model = self._ai_row(ai_model_id)
        self.assertEqual(model["status"], "VALIDADO")
        self.assertEqual(int(model["active"] or 0), 0)
        self.assertEqual(model["validated_by"],
                         self.user_ids[self.ADMIN_USERNAME])
        self.assertIsNotNone(model["validated_at"])
        self.assertIsNone(model["activated_by"])
        self.assertIsNone(model["activated_at"])

        session = self._session_row(ai_model_id)
        self.assertEqual(session["status"], VALIDATION_SESSION_STATUS_CERRADA)
        self.assertIsNotNone(session["closed_at"])

        self.assertEqual(self._event_count(ai_model_id, "VALIDATED"), 1)
        self.assertEqual(self._event_count(ai_model_id, "ACTIVATED"), 0)
        self.assertEqual(self._event_count(ai_model_id, "THRESHOLD_FROZEN"), 1)

        again = self._complete(model_id, ai_model_id)
        self.assertEqual(again.status_code, 200, self._json(again))
        self.assertTrue(self._json(again)["result"]["already_validated"])

        restart = self._start_session(model_id, ai_model_id)
        self.assertEqual(restart.status_code, 409, self._json(restart))

    def test_post_freeze_cases_are_labeled_and_scored_at_frozen_threshold(self):
        model_id = self.model_ids[self.POST_FREEZE_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        self.assertEqual(self._start_session(model_id, ai_model_id).status_code, 201)
        baseline, _ = self._post_case(
            model_id, "NORMAL", score=30.0, ai_model_id=ai_model_id
        )
        self.assertEqual(baseline.status_code, 201, self._json(baseline))
        evaluated = self._evaluate(model_id, ai_model_id)
        self.assertEqual(evaluated.status_code, 200, self._json(evaluated))
        frozen = self._freeze_threshold(model_id, ai_model_id, 50.00)
        self.assertEqual(frozen.status_code, 200, self._json(frozen))
        response, _ = self._post_case(
            model_id, "MANCHA", score=50.0, is_anomaly=False,
            ai_model_id=ai_model_id,
        )
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]
        self.assertEqual(case["threshold_used"], 50.0)
        self.assertEqual(case["prediction"], "ANOMALIA")
        self.assertEqual(case["result"], "CORRECTO")
        self.assertEqual(case["validation_cohort"], "POST_FREEZE_FINAL_TEST")
        self.conn.commit()
        self.cur.execute(
            "SELECT threshold_final, status, active FROM garment_ai_models WHERE id = %s",
            (ai_model_id,),
        )
        frozen_model = self.cur.fetchone()
        self.assertEqual(float(frozen_model["threshold_final"]), 50.0)
        self.assertEqual(frozen_model["status"], "VALIDACION")
        self.assertEqual(int(frozen_model["active"] or 0), 0)

    # --------------------------------------------------------
    # versión invalidada
    # --------------------------------------------------------

    def test_invalidated_version_cannot_be_validated(self):
        model_id = self.model_ids[self.INVALIDATED_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response = self._start_session(model_id, ai_model_id)

        self.assertEqual(response.status_code, 404, self._json(response))
        self.assertIn("NO APTO PARA ACTIVACIÓN", self._json(response)["error"])
        self.assertEqual(self._ai_row(ai_model_id)["status"], "ENTRENADO")

        case, _ = self._post_case(model_id, "NORMAL", ai_model_id=ai_model_id)
        self.assertEqual(case.status_code, 404, self._json(case))

        conn = self._connect()
        cur = conn.cursor(dictionary=True)

        try:
            with self.assertRaises(AIDomainError):
                ai_validation.load_model_bundle(cur, ai_model_id)
        finally:
            cur.close()
            conn.close()

        page = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/{ai_model_id}"
        )
        self.assertEqual(page.status_code, 302)
        self.assertEqual(self._case_count(ai_model_id), 0)

    def test_training_status_exposes_the_validation_entry_point(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        self._login(self.ADMIN_USERNAME)
        response = self.client.get(
            f"/api/ai/training/status?garment_model_id={model_id}"
        )

        self.assertEqual(response.status_code, 200, self._json(response))
        training = self._json(response)["training"]

        self.assertTrue(training["validation_available"], training)
        self.assertEqual(training["validation_ai_model_id"], ai_model_id)
        self.assertEqual(
            training["validation_url"],
            f"/modelos-prenda/{model_id}/validacion-ia/{ai_model_id}",
        )

    # --------------------------------------------------------
    # UI y permisos
    # --------------------------------------------------------

    def test_validation_page_ui_and_permissions(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        url = f"/modelos-prenda/{model_id}/validacion-ia/{ai_model_id}"

        self._login(self.ADMIN_USERNAME)
        page = self.client.get(url)

        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)

        self.assertIn("CAPTURAR IMAGEN DE VALIDACIÓN", html)
        self.assertIn("Evaluar validación", html)
        self.assertIn("Marcar como VALIDADA", html)
        self.assertIn("Métricas de validación", html)
        self.assertIn(self.MODEL_CODE, html)
        self.assertIn(VALIDATION_IMAGE_NOTICE, html)
        self.assertIn("PENDIENTE DE VALIDACIÓN", html)

        frame = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/frame"
        )
        self.assertIn(frame.status_code, (200, 204))

        detail = self.client.get(f"/modelos-prenda/{model_id}")
        self.assertEqual(detail.status_code, 200)
        detail_html = detail.get_data(as_text=True)
        self.assertIn("Continuar a validación", detail_html)
        self.assertIn(
            f"/validacion-ia/{ai_model_id}",
            detail_html,
        )

        self._login(self.MM_USERNAME)
        self.assertEqual(self.client.get(url).status_code, 200)

        self._login(self.QM_USERNAME)
        self.assertEqual(self.client.get(url).status_code, 302)

        blocked, _ = self._post_case(
            model_id,
            "NORMAL",
            username=self.QM_USERNAME,
        )
        self.assertEqual(blocked.status_code, 403)

        forbidden = self._complete(
            model_id,
            ai_model_id,
            username=self.MM_USERNAME,
        )
        self.assertEqual(forbidden.status_code, 403)

        with self.client.session_transaction() as sess:
            sess.clear()

        self.assertEqual(self.client.get(url).status_code, 302)

    def test_frame_endpoint_without_camera_returns_no_content(self):
        model_id = self.model_ids[self.MODEL_CODE]

        self._login(self.ADMIN_USERNAME)
        response = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/frame"
        )

        self.assertIn(response.status_code, (200, 204))

        if response.status_code == 200:
            self.assertEqual(response.mimetype, "image/jpeg")


# ============================================================
# FASE 3B.1 — CORRECCIÓN DEL GROUND TRUTH Y BORRADO DE CASOS
# ============================================================


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de FASE 3B.1 omitidos.",
)
class ValidationCaseEditDbTests(_TrainingDbCase):
    """Editar la etiqueta humana o eliminar un caso, sin re-inferir."""

    ADMIN_USERNAME = "ph3e_admin"
    MM_USERNAME = "ph3e_mm"
    QM_USERNAME = "ph3e_qm"

    MODEL_CODE = "TEST-PH3E-EDIT"
    FREEZE_PAGE_CODE = "TEST-PH3E-FREEZE-PAGE"
    FREEZE_EDIT_CODE = "TEST-PH3E-FREEZE-EDIT"
    MODELS = (MODEL_CODE, FREEZE_PAGE_CODE, FREEZE_EDIT_CODE)

    USERS = (
        (ADMIN_USERNAME, "ADMIN"),
        (MM_USERNAME, "MODEL_MANAGER"),
        (QM_USERNAME, "QUALITY_MANAGER"),
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        actor = cls.user_ids[cls.ADMIN_USERNAME]
        manager = cls.user_ids[cls.MM_USERNAME]

        for code in cls.MODELS:
            model_id = cls.model_ids[code]

            cls._delete_model(model_id)
            cls._seed_accepted(model_id, actor)

            # El MODEL_MANAGER «creador» puede gestionar el modelo.
            cls.cur.execute(
                "UPDATE garment_models SET created_by = %s WHERE id = %s",
                (manager, model_id),
            )
            cls.conn.commit()

            cls._train_model(model_id, actor)

    @classmethod
    def tearDownClass(cls):
        for code in cls.MODELS:
            model_id = cls.model_ids.get(code)

            if model_id:
                cls._clear_validation(model_id)

        ai_validation.clear_model_inspectors()
        super().tearDownClass()

    # --------------------------------------------------------
    # utilidades
    # --------------------------------------------------------

    def _case_row(self, case_id):
        self.conn.commit()
        self.cur.execute(
            "SELECT * FROM ai_validation_cases WHERE id = %s",
            (case_id,),
        )
        return self.cur.fetchone()

    def _update_case(self, model_id, case_id, category, *,
                     ai_model_id=None, username=None):
        payload = {
            "garment_model_id": model_id,
            "case_id": case_id,
            "category": category,
        }

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        return self._post(
            "/api/ai/validation/case/update",
            payload,
            username,
        )

    def _delete_case(self, model_id, case_id, *,
                     ai_model_id=None, username=None):
        payload = {
            "garment_model_id": model_id,
            "case_id": case_id,
        }

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        return self._post(
            "/api/ai/validation/case/delete",
            payload,
            username,
        )

    @staticmethod
    def _file_digests(relative_paths):
        root = Path(get_ai_artifacts_root())
        return {
            relative: hashlib.sha256(
                (root / relative).read_bytes()
            ).hexdigest()
            for relative in relative_paths
            if (root / relative).is_file()
        }

    @staticmethod
    def _candidate(metrics, threshold=50.0):
        """Fila de métricas de un threshold candidato concreto."""
        for item in metrics.get("candidates") or []:
            if float(item["threshold"]) == float(threshold):
                return item

        raise AssertionError(
            f"No evaluó el threshold {threshold}: {metrics}"
        )

    def _latest_event(self, ai_model_id, event_type):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT actor_id, payload_json, created_at
            FROM ai_events
            WHERE ai_model_id = %s AND event_type = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (ai_model_id, event_type),
        )
        row = self.cur.fetchone()

        if row is None:
            return None

        return {
            "actor_id": row["actor_id"],
            "created_at": row["created_at"],
            "payload": (
                json.loads(row["payload_json"])
                if row["payload_json"]
                else None
            ),
        }

    # --------------------------------------------------------
    # 1-2. la edición preserva score, hash y artefactos
    # --------------------------------------------------------

    def test_category_edit_preserves_score_and_never_reinfers(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "NORMAL", score=47.25)
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]
        before = self._case_row(case["id"])

        with patch.object(
            ai_validation,
            "predict_with_ai_model",
            side_effect=AssertionError("la edición no debe inferir"),
        ):
            edited = self._update_case(
                model_id,
                case["id"],
                "MANCHA",
                ai_model_id=ai_model_id,
            )

        self.assertEqual(edited.status_code, 200, self._json(edited))
        body = self._json(edited)
        self.assertTrue(body["ok"])
        self.assertTrue(body["result"]["changed"])
        self.assertEqual(body["result"]["case"]["category"], "MANCHA")
        self.assertEqual(body["validation"]["counts"]["MANCHA"], 1)
        self.assertEqual(body["validation"]["counts"]["NORMAL"], 0)

        after = self._case_row(case["id"])
        self.assertEqual(after["category"], "MANCHA")

        # Todo lo demás del caso queda idéntico (incluye el score
        # predicho por el modelo y la fecha de registro).
        for key in set(before) - {"category"}:
            self.assertEqual(before[key], after.get(key), key)

        self.assertEqual(float(after["anomaly_score"]), 47.25)
        self.assertEqual(after["result"], VALIDATION_RESULT_PENDING)
        self.assertEqual(after["prediction"], "NORMAL")
        self.assertEqual(
            after["image_sha256"],
            hashlib.sha256(
                (Path(get_ai_artifacts_root()) / after["image_path"]).read_bytes()
            ).hexdigest(),
        )

        event = self._latest_event(
            ai_model_id, "VALIDATION_CASE_CATEGORY_CHANGED"
        )
        self.assertIsNotNone(event)
        self.assertEqual(
            event["actor_id"],
            self.user_ids[self.ADMIN_USERNAME],
        )
        self.assertIsNotNone(event["created_at"])
        self.assertEqual(event["payload"]["validation_case_id"], case["id"])
        self.assertEqual(event["payload"]["before_category"], "NORMAL")
        self.assertEqual(event["payload"]["after_category"], "MANCHA")
        self.assertEqual(event["payload"]["image_sha256"],
                         before["image_sha256"])
        self.assertAlmostEqual(
            float(event["payload"]["anomaly_score"]), 47.25
        )

    def test_category_edit_preserves_heatmap_and_comparison(self):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy no disponible")

        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        anomaly_map = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)

        response, _ = self._post_case(
            model_id,
            "MANCHA",
            score=88.0,
            is_anomaly=True,
            anomaly_map=anomaly_map,
        )
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]

        paths = [case["image_path"], case["heatmap_path"],
                 case["comparison_path"]]
        self.assertTrue(all(paths), case)
        before_files = self._file_digests(paths)
        self.assertEqual(len(before_files), 3)

        edited = self._update_case(
            model_id, case["id"], "AGUJERO", ai_model_id=ai_model_id
        )
        self.assertEqual(edited.status_code, 200, self._json(edited))

        after = self._case_row(case["id"])
        self.assertEqual(after["category"], "AGUJERO")

        for key in ("image_path", "heatmap_path", "comparison_path",
                    "image_sha256", "anomaly_score", "created_at",
                    "ai_model_id", "prediction"):
            self.assertEqual(after[key], case.get(key, after[key]), key)

        self.assertEqual(after["heatmap_path"], case["heatmap_path"])
        self.assertEqual(after["comparison_path"], case["comparison_path"])
        self.assertEqual(self._file_digests(paths), before_files)

    # --------------------------------------------------------
    # 3-4. categoría inválida y permisos
    # --------------------------------------------------------

    def test_invalid_category_is_rejected(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "NORMAL", score=21.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        rejected = self._update_case(
            model_id, case_id, "RASGADO", ai_model_id=ai_model_id
        )

        self.assertEqual(rejected.status_code, 400, self._json(rejected))
        self.assertIn(
            "Categoría inválida", self._json(rejected)["error"]
        )
        self.assertEqual(self._case_row(case_id)["category"], "NORMAL")

    def test_edit_and_delete_require_permissions(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "NORMAL", score=15.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        blocked = self._update_case(
            model_id, case_id, "MANCHA", username=self.QM_USERNAME
        )
        self.assertEqual(blocked.status_code, 403, self._json(blocked))

        blocked_delete = self._delete_case(
            model_id, case_id, username=self.QM_USERNAME
        )
        self.assertEqual(
            blocked_delete.status_code, 403, self._json(blocked_delete)
        )
        self.assertEqual(self._case_count(ai_model_id), 1)

        with self.client.session_transaction() as sess:
            sess.clear()

        anonymous = self.client.post(
            "/api/ai/validation/case/update",
            json={
                "garment_model_id": model_id,
                "case_id": case_id,
                "category": "MANCHA",
            },
        )
        self.assertIn(anonymous.status_code, (302, 401))

        allowed = self._update_case(
            model_id, case_id, "MANCHA", username=self.MM_USERNAME
        )
        self.assertEqual(allowed.status_code, 200, self._json(allowed))

        missing = self._update_case(model_id, 99999999, "MANCHA")
        self.assertEqual(missing.status_code, 409, self._json(missing))
        self.assertIn("no existe", self._json(missing)["error"])

    # --------------------------------------------------------
    # 5-6. la edición invalida las métricas de la sesión
    # --------------------------------------------------------

    def test_edit_invalidates_previous_session_metrics(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        for category, score in (("NORMAL", 12.0), ("MANCHA", 88.0)):
            response, _ = self._post_case(model_id, category, score=score)
            self.assertEqual(response.status_code, 201, self._json(response))

        case_id = self._json(response)["case"]["id"]

        evaluated = self._evaluate(model_id, ai_model_id, thresholds=[50])
        self.assertEqual(evaluated.status_code, 200, self._json(evaluated))

        session = self._session_row(ai_model_id)
        self.assertEqual(
            session["status"], VALIDATION_SESSION_STATUS_EVALUADA
        )
        self.assertIsNotNone(session["metrics_json"])
        self.assertIsNotNone(session["threshold_candidate"])
        self.assertIsNotNone(session["evaluated_at"])

        state = self._json(evaluated)["validation"]
        self.assertIsNotNone(state["metrics"])
        self.assertTrue(state["can_complete"])

        edited = self._update_case(
            model_id, case_id, "AGUJERO", ai_model_id=ai_model_id
        )
        self.assertEqual(edited.status_code, 200, self._json(edited))
        self.assertTrue(self._json(edited)["result"]["metrics_invalidated"])

        session = self._session_row(ai_model_id)
        self.assertEqual(
            session["status"], VALIDATION_SESSION_STATUS_ABIERTA
        )
        self.assertIsNone(session["metrics_json"])
        self.assertIsNone(session["threshold_candidate"])
        self.assertIsNone(session["evaluated_at"])

        state = self._json(edited)["validation"]
        self.assertIsNone(state["metrics"])
        self.assertTrue(state["can_evaluate"])
        self.assertFalse(state["can_complete"])

        # La sesión vuelve a exigir «Evaluar validación».
        reevaluate = self._evaluate(model_id, ai_model_id, thresholds=[50])
        self.assertEqual(
            reevaluate.status_code, 200, self._json(reevaluate)
        )
        self.assertIsNotNone(self._session_row(ai_model_id)["metrics_json"])

    def test_recalculated_metrics_use_the_new_category(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        normal, _ = self._post_case(model_id, "NORMAL", score=20.0)
        self.assertEqual(normal.status_code, 201, self._json(normal))
        normal_case = self._json(normal)["case"]

        response, _ = self._post_case(model_id, "MANCHA", score=80.0,
                                      is_anomaly=True)
        self.assertEqual(response.status_code, 201, self._json(response))

        evaluated = self._evaluate(model_id, ai_model_id, thresholds=[50])
        metrics = self._json(evaluated)["result"]["metrics"]

        self.assertEqual(
            metrics["counts"], {"NORMAL": 1, "MANCHA": 1, "AGUJERO": 0}
        )
        candidate = self._candidate(metrics)
        self.assertEqual(candidate["tp"], 1)
        self.assertEqual(candidate["tn"], 1)
        self.assertEqual(candidate["fn"], 0)

        # La etiqueta real cambia: el mismo score pasa a ser FN.
        edited = self._update_case(
            model_id, normal_case["id"], "MANCHA", ai_model_id=ai_model_id
        )
        self.assertEqual(edited.status_code, 200, self._json(edited))

        reevaluate = self._evaluate(model_id, ai_model_id, thresholds=[50])
        metrics = self._json(reevaluate)["result"]["metrics"]

        self.assertEqual(
            metrics["counts"], {"NORMAL": 0, "MANCHA": 2, "AGUJERO": 0}
        )
        candidate = self._candidate(metrics)
        self.assertEqual(candidate["tp"], 1)
        self.assertEqual(candidate["fn"], 1)
        self.assertEqual(candidate["tn"], 0)
        self.assertEqual(candidate["fp"], 0)

    # --------------------------------------------------------
    # 7-8. training intacto y versión sin activar
    # --------------------------------------------------------

    def test_training_is_not_touched_by_a_category_edit(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        counts_before = self._counts(model_id)
        artifacts_before = self._snapshot(
            self._model_artifacts_dir(model_id, ai_model_id)
        )
        threshold_before = self.A.PATCHCORE_SCORE_THRESHOLD
        checkpoint_before = self.A.PATCHCORE_CKPT

        response, _ = self._post_case(model_id, "NORMAL", score=33.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        edited = self._update_case(
            model_id, case_id, "AGUJERO", ai_model_id=ai_model_id
        )
        self.assertEqual(edited.status_code, 200, self._json(edited))

        self.assertEqual(self._counts(model_id), counts_before)
        self.assertEqual(
            self._snapshot(self._model_artifacts_dir(model_id, ai_model_id)),
            artifacts_before,
        )
        self.assertEqual(self.A.PATCHCORE_SCORE_THRESHOLD, threshold_before)
        self.assertEqual(self.A.PATCHCORE_CKPT, checkpoint_before)

        deleted = self._delete_case(
            model_id, case_id, ai_model_id=ai_model_id
        )
        self.assertEqual(deleted.status_code, 200, self._json(deleted))

        self.assertEqual(self._counts(model_id), counts_before)
        self.assertEqual(
            self._snapshot(self._model_artifacts_dir(model_id, ai_model_id)),
            artifacts_before,
        )

    def test_version_stays_unactivated_after_editing(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "NORMAL", score=33.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        edited = self._update_case(
            model_id, case_id, "MANCHA", ai_model_id=ai_model_id
        )
        self.assertEqual(edited.status_code, 200, self._json(edited))

        model = self._ai_row(ai_model_id)
        self.assertEqual(model["status"], "VALIDACION")
        self.assertEqual(int(model["active"] or 0), 0)
        self.assertIsNone(model["activated_by"])
        self.assertIsNone(model["activated_at"])
        self.assertIsNone(model["validated_by"])
        self.assertIsNone(model["validated_at"])

        self.assertEqual(self._event_count(ai_model_id, "ACTIVATED"), 0)
        self.assertEqual(self._event_count(ai_model_id, "VALIDATED"), 0)
        self.assertEqual(
            self._event_count(ai_model_id, "VALIDATION_CASE_REGISTERED"), 1
        )

        # Ninguna versión de este modelo quedó activa.
        self.conn.commit()
        self.cur.execute(
            """
            SELECT COUNT(*) AS total FROM garment_ai_models
            WHERE garment_model_id = %s AND active = 1
            """,
            (model_id,),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 0)

    # --------------------------------------------------------
    # eliminación de un caso
    # --------------------------------------------------------

    def test_delete_case_removes_row_and_artifacts(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        counts_before = self._counts(model_id)

        first, _ = self._post_case(model_id, "NORMAL", score=18.0)
        self.assertEqual(first.status_code, 201, self._json(first))
        first_case = self._json(first)["case"]

        second, _ = self._post_case(model_id, "MANCHA", score=77.0,
                                    is_anomaly=True)
        self.assertEqual(second.status_code, 201, self._json(second))
        second_case = self._json(second)["case"]

        evaluated = self._evaluate(model_id, ai_model_id, thresholds=[50])
        self.assertEqual(evaluated.status_code, 200, self._json(evaluated))

        removed_paths = [first_case["image_path"]]
        kept_paths = [second_case["image_path"]]
        self.assertEqual(len(self._file_digests(removed_paths)), 1)
        self.assertEqual(len(self._file_digests(kept_paths)), 1)

        deleted = self._delete_case(
            model_id, first_case["id"], ai_model_id=ai_model_id
        )
        self.assertEqual(deleted.status_code, 200, self._json(deleted))
        body = self._json(deleted)

        self.assertTrue(body["result"]["deleted"])
        self.assertTrue(body["result"]["metrics_invalidated"])
        self.assertEqual(body["result"]["case"]["category"], "NORMAL")
        self.assertEqual(body["validation"]["total_cases"], 1)

        self.assertIsNone(self._case_row(first_case["id"]))
        self.assertIsNotNone(self._case_row(second_case["id"]))
        self.assertEqual(self._case_count(ai_model_id), 1)

        self.assertEqual(self._file_digests(removed_paths), {})
        self.assertEqual(len(self._file_digests(kept_paths)), 1)

        session = self._session_row(ai_model_id)
        self.assertEqual(
            session["status"], VALIDATION_SESSION_STATUS_ABIERTA
        )
        self.assertIsNone(session["metrics_json"])
        self.assertIsNone(session["threshold_candidate"])

        event = self._latest_event(ai_model_id, "VALIDATION_CASE_DELETED")
        self.assertIsNotNone(event)
        self.assertEqual(
            event["actor_id"], self.user_ids[self.ADMIN_USERNAME]
        )
        self.assertEqual(
            event["payload"]["validation_case_id"], first_case["id"]
        )
        self.assertEqual(event["payload"]["category"], "NORMAL")
        self.assertIn(first_case["image_path"],
                      event["payload"]["files_removed"])

        self.assertEqual(self._counts(model_id), counts_before)

    def test_validation_page_shows_the_case_actions(self):
        model_id = self.model_ids[self.FREEZE_PAGE_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "NORMAL", score=12.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        self._login(self.ADMIN_USERNAME)
        page = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/{ai_model_id}"
        )
        self.assertEqual(page.status_code, 200, page.status_code)
        html = page.get_data(as_text=True)

        self.assertIn("Editar clasificación real", html)
        self.assertIn("Eliminar caso de validación", html)
        self.assertIn(f"data-case-id=\"{case_id}\"", html)
        self.assertIn(f"data-case-estado=\"{case_id}\"", html)
        self.assertIn(f"data-case-tipo=\"{case_id}\"", html)
        self.assertIn("can_edit_cases", html)
        self.assertIn('"canManage": true', html)

        # Tras cerrar la validación los casos dejan de ser editables.
        self.assertEqual(
            self._evaluate(model_id, ai_model_id).status_code, 200
        )
        self.assertEqual(
            self._freeze_threshold(model_id, ai_model_id).status_code, 200
        )
        self.assertEqual(self._complete(model_id, ai_model_id).status_code, 200)

        page = self.client.get(
            f"/modelos-prenda/{model_id}/validacion-ia/{ai_model_id}"
        )
        html = page.get_data(as_text=True)
        self.assertNotIn(f"data-case-id=\"{case_id}\"", html)
        self.assertIn('"can_edit_cases": false', html)

    def test_edit_is_blocked_after_the_validation_closes(self):
        model_id = self.model_ids[self.FREEZE_EDIT_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "NORMAL", score=24.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        self.assertEqual(
            self._evaluate(model_id, ai_model_id).status_code, 200
        )
        self.assertEqual(
            self._freeze_threshold(model_id, ai_model_id).status_code, 200
        )
        self.assertEqual(self._complete(model_id, ai_model_id).status_code, 200)

        session = self._session_row(ai_model_id)
        self.assertEqual(
            session["status"], VALIDATION_SESSION_STATUS_CERRADA
        )

        blocked = self._update_case(
            model_id, case_id, "MANCHA", ai_model_id=ai_model_id
        )
        self.assertEqual(blocked.status_code, 409, self._json(blocked))
        self.assertIn("ya fue cerrada", self._json(blocked)["error"])

        blocked_delete = self._delete_case(
            model_id, case_id, ai_model_id=ai_model_id
        )
        self.assertEqual(
            blocked_delete.status_code, 409, self._json(blocked_delete)
        )

        self.assertEqual(self._case_row(case_id)["category"], "NORMAL")
        self.assertEqual(self._case_count(ai_model_id), 1)


# Reutiliza el andamiaje de FASE 3B (entrenamiento falso, limpieza de
# sesión/casos, helpers de API) sin volver a ejecutar sus tests.
for _helper_name in (
    "_FakeTrainer",
    "setUp",
    "_cancel_stray_jobs",
    "_train_model",
    "_invalidate_latest",
    "_clear_validation",
    "_ai_row",
    "_ai_row_by_model",
    "_session_row",
    "_case_count",
    "_event_count",
    "_counts",
    "_snapshot",
    "_model_artifacts_dir",
    "_new_validation_image",
    "_fake_predict",
    "_post",
    "_post_case",
    "_start_session",
    "_evaluate",
    "_complete",
):
    setattr(
        ValidationCaseEditDbTests,
        _helper_name,
        vars(ValidationDbTests)[_helper_name],
    )


# ============================================================
# FASE 3B.2 — VALIDACIÓN BINARIA BUENA / DEFECTUOSA
# ============================================================


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de FASE 3B.2 omitidos.",
)
class ValidationBinaryDbTests(_TrainingDbCase):
    """Captura/edición binaria BUENA vs DEFECTUOSA + desglose."""

    ADMIN_USERNAME = "ph3b2_admin"
    MM_USERNAME = "ph3b2_mm"
    QM_USERNAME = "ph3b2_qm"

    MODEL_CODE = "TEST-PH3B2-BIN"
    MODELS = (MODEL_CODE,)

    USERS = (
        (ADMIN_USERNAME, "ADMIN"),
        (MM_USERNAME, "MODEL_MANAGER"),
        (QM_USERNAME, "QUALITY_MANAGER"),
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        actor = cls.user_ids[cls.ADMIN_USERNAME]
        manager = cls.user_ids[cls.MM_USERNAME]

        for code in cls.MODELS:
            model_id = cls.model_ids[code]

            cls._delete_model(model_id)
            cls._seed_accepted(model_id, actor)

            cls.cur.execute(
                "UPDATE garment_models SET created_by = %s WHERE id = %s",
                (manager, model_id),
            )
            cls.conn.commit()

            cls._train_model(model_id, actor)

    @classmethod
    def tearDownClass(cls):
        for code in cls.MODELS:
            model_id = cls.model_ids.get(code)

            if model_id:
                cls._clear_validation(model_id)

        ai_validation.clear_model_inspectors()
        super().tearDownClass()

    # --------------------------------------------------------
    # utilidades
    # --------------------------------------------------------

    def _post_binary_case(
        self,
        model_id,
        estado_real,
        tipo_defecto=None,
        *,
        score=42.0,
        is_anomaly=None,
        anomaly_map=None,
        observation="",
        username=None,
        ai_model_id=None,
        category=None,
    ):
        payload = {
            "garment_model_id": model_id,
            "estado_real": estado_real,
            "observation": observation,
        }

        if tipo_defecto is not None:
            payload["tipo_defecto"] = tipo_defecto

        if category is not None:
            payload["category"] = category

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        payload["image_base64"] = base64.b64encode(
            self._new_validation_image(
                f"{estado_real}-{tipo_defecto}-{category}"
            )
        ).decode("ascii")

        fake = self._fake_predict(
            score=score,
            is_anomaly=is_anomaly,
            anomaly_map=anomaly_map,
        )

        with patch.object(ai_validation, "predict_with_ai_model", fake):
            response = self._post(
                "/api/ai/validation/case",
                payload,
                username=username,
            )

        return response, fake

    def _update_case_binary(
        self, model_id, case_id, estado_real, tipo_defecto=None,
        *, ai_model_id=None, username=None,
    ):
        payload = {
            "garment_model_id": model_id,
            "case_id": case_id,
            "estado_real": estado_real,
        }

        if tipo_defecto is not None:
            payload["tipo_defecto"] = tipo_defecto

        if ai_model_id is not None:
            payload["ai_model_id"] = ai_model_id

        return self._post(
            "/api/ai/validation/case/update",
            payload,
            username,
        )

    def _case_row(self, case_id):
        self.conn.commit()
        self.cur.execute(
            "SELECT * FROM ai_validation_cases WHERE id = %s",
            (case_id,),
        )
        return self.cur.fetchone()

    @staticmethod
    def _file_digests(relative_paths):
        root = Path(get_ai_artifacts_root())
        return {
            relative: hashlib.sha256(
                (root / relative).read_bytes()
            ).hexdigest()
            for relative in relative_paths
            if (root / relative).is_file()
        }

    @staticmethod
    def _candidate(metrics, threshold=50.0):
        for item in metrics.get("candidates") or []:
            if float(item["threshold"]) == float(threshold):
                return item

        raise AssertionError(
            f"No evaluó el threshold {threshold}: {metrics}"
        )

    def _latest_event(self, ai_model_id, event_type):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT actor_id, payload_json, created_at
            FROM ai_events
            WHERE ai_model_id = %s AND event_type = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (ai_model_id, event_type),
        )
        row = self.cur.fetchone()

        if row is None:
            return None

        return {
            "actor_id": row["actor_id"],
            "created_at": row["created_at"],
            "payload": (
                json.loads(row["payload_json"])
                if row["payload_json"]
                else None
            ),
        }

    # --------------------------------------------------------
    # 1-4. captura binaria
    # --------------------------------------------------------

    def test_buena_is_stored_as_normal_without_defect_type(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_binary_case(
            model_id, "BUENA", score=41.5
        )
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]

        self.assertEqual(case["category"], "NORMAL")
        self.assertEqual(case["estado_real"], "BUENA")
        self.assertIsNone(case["tipo_defecto"])
        self.assertIsNone(case["tipo_defecto_label"])
        self.assertEqual(case["classification"], "Buena")
        self.assertEqual(case["anomaly_score"], 41.5)

        stored = self._case_row(case["id"])
        self.assertEqual(stored["category"], "NORMAL")

        state = self._json(response)["validation"]
        self.assertEqual(state["counts_estado"]["BUENA"], 1)
        self.assertEqual(state["counts_estado"]["DEFECTUOSA"], 0)
        self.assertEqual(state["counts"]["NORMAL"], 1)
        self.assertEqual(state["binary_notice"], VALIDATION_BINARY_NOTICE)

    def test_defectuosa_requires_a_defect_type(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        missing, _ = self._post_binary_case(model_id, "DEFECTUOSA")
        self.assertEqual(missing.status_code, 409, self._json(missing))
        self.assertIn("tipo de defecto", self._json(missing)["error"])

        with_type, _ = self._post_binary_case(
            model_id, "BUENA", "MANCHA"
        )
        self.assertEqual(with_type.status_code, 409, self._json(with_type))
        self.assertIn(
            "no puede indicar tipo", self._json(with_type)["error"]
        )

        self.assertEqual(self._case_count(ai_model_id), 0)

    def test_defectuosa_stores_mancha_or_agujero(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        for tipo, score in (("MANCHA", 88.0), ("AGUJERO", 61.5)):
            response, _ = self._post_binary_case(
                model_id,
                "DEFECTUOSA",
                tipo,
                score=score,
                is_anomaly=True,
            )
            self.assertEqual(
                response.status_code, 201, self._json(response)
            )
            case = self._json(response)["case"]

            self.assertEqual(case["category"], tipo)
            self.assertEqual(case["estado_real"], "DEFECTUOSA")
            self.assertEqual(case["tipo_defecto"], tipo)
            self.assertEqual(
                case["tipo_defecto_label"],
                VALIDATION_DEFECT_TYPE_LABELS[tipo],
            )
            self.assertEqual(
                case["classification"],
                VALIDATION_CLASSIFICATION_LABELS[tipo],
            )

        state = self._json(response)["validation"]
        self.assertEqual(state["counts_estado"]["DEFECTUOSA"], 2)
        self.assertEqual(state["counts_defects"], {"MANCHA": 1, "AGUJERO": 1})

    def test_legacy_category_payload_still_works(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_case(model_id, "MANCHA", score=70.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]
        self.assertEqual(case["estado_real"], "DEFECTUOSA")
        self.assertEqual(case["tipo_defecto"], "MANCHA")

        mismatch, _ = self._post_binary_case(
            model_id, "BUENA", category="MANCHA"
        )
        self.assertEqual(mismatch.status_code, 409, self._json(mismatch))

        self.assertEqual(self._case_count(ai_model_id), 1)

    # --------------------------------------------------------
    # 6-7. métricas binarias + desglose Mancha/Agujero
    # --------------------------------------------------------

    def test_evaluation_uses_buena_vs_defectuosa(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        for estado, tipo, score in (
            ("BUENA", None, 40.0),
            ("DEFECTUOSA", "MANCHA", 80.0),
            ("DEFECTUOSA", "AGUJERO", 30.0),
        ):
            response, _ = self._post_binary_case(
                model_id,
                estado,
                tipo,
                score=score,
                is_anomaly=estado == "DEFECTUOSA",
            )
            self.assertEqual(
                response.status_code, 201, self._json(response)
            )

        evaluated = self._evaluate(
            model_id, ai_model_id, thresholds=[50.0]
        )
        self.assertEqual(evaluated.status_code, 200, self._json(evaluated))
        metrics = self._json(evaluated)["result"]["metrics"]

        self.assertEqual(
            metrics["estado_counts"], {"BUENA": 1, "DEFECTUOSA": 2}
        )
        self.assertEqual(
            metrics["counts_estado"], {"BUENA": 1, "DEFECTUOSA": 2}
        )
        self.assertEqual(
            metrics["counts"], {"NORMAL": 1, "MANCHA": 1, "AGUJERO": 1}
        )
        self.assertEqual(
            metrics["defect_counts"], {"MANCHA": 1, "AGUJERO": 1}
        )
        self.assertEqual(metrics["threshold"], None)
        self.assertEqual(metrics["confusion"], None)

        candidate = self._candidate(metrics, 50.0)
        self.assertEqual(candidate["tp"], 1)
        self.assertEqual(candidate["fn"], 1)
        self.assertEqual(candidate["fp"], 0)
        self.assertEqual(candidate["tn"], 1)

        binary = metrics["threshold_candidate_metrics"]
        self.assertEqual(binary["fp"], 0)
        self.assertEqual(binary["fn"], 1)
        self.assertEqual(metrics["threshold_source"], "candidate")

        self.assertEqual(metrics["defects"]["MANCHA"], {
            "evaluated": 1, "detected": 1, "missed": 0,
        })
        self.assertEqual(metrics["defects"]["AGUJERO"], {
            "evaluated": 1, "detected": 0, "missed": 1,
        })
        self.assertEqual(metrics["defects"]["threshold"], 50.0)

        session = self._session_row(ai_model_id)
        self.assertEqual(
            session["status"], VALIDATION_SESSION_STATUS_EVALUADA
        )
        self.assertEqual(
            float(session["threshold_candidate"]), 50.0
        )

    # --------------------------------------------------------
    # 9. editar estado/tipo sin re-inferir
    # --------------------------------------------------------

    def test_binary_edit_preserves_artifacts_and_invalidates(self):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy no disponible")

        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        anomaly_map = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)

        response, _ = self._post_binary_case(
            model_id,
            "BUENA",
            score=47.25,
            anomaly_map=anomaly_map,
        )
        self.assertEqual(response.status_code, 201, self._json(response))
        case = self._json(response)["case"]
        before = self._case_row(case["id"])

        paths = [
            before["image_path"],
            before["heatmap_path"],
            before["comparison_path"],
        ]
        self.assertTrue(all(paths), before)
        digests_before = self._file_digests(paths)
        self.assertEqual(len(digests_before), 3)

        self.assertEqual(
            self._evaluate(model_id, ai_model_id, thresholds=[50.0])
            .status_code,
            200,
        )

        with patch.object(
            ai_validation,
            "predict_with_ai_model",
            side_effect=AssertionError("la edición no debe inferir"),
        ):
            edited = self._update_case_binary(
                model_id,
                case["id"],
                "DEFECTUOSA",
                "MANCHA",
                ai_model_id=ai_model_id,
            )

        self.assertEqual(edited.status_code, 200, self._json(edited))
        result = self._json(edited)["result"]
        self.assertTrue(result["changed"])
        self.assertTrue(result["metrics_invalidated"])
        self.assertEqual(result["case"]["category"], "MANCHA")
        self.assertEqual(result["case"]["estado_real"], "DEFECTUOSA")
        self.assertEqual(result["case"]["tipo_defecto"], "MANCHA")
        self.assertEqual(
            result["case"]["classification"], "Defectuosa · Mancha"
        )
        self.assertIn("Defectuosa · Mancha", result["message"])

        after = self._case_row(case["id"])
        self.assertEqual(after["category"], "MANCHA")

        for key in set(before) - {"category"}:
            self.assertEqual(before[key], after.get(key), key)

        self.assertEqual(float(after["anomaly_score"]), 47.25)
        self.assertEqual(self._file_digests(paths), digests_before)

        session = self._session_row(ai_model_id)
        self.assertEqual(
            session["status"], VALIDATION_SESSION_STATUS_ABIERTA
        )
        self.assertIsNone(session["metrics_json"])
        self.assertIsNone(session["threshold_candidate"])

        event = self._latest_event(
            ai_model_id, "VALIDATION_CASE_CATEGORY_CHANGED"
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["payload"]["before_estado_real"], "BUENA")
        self.assertEqual(event["payload"]["before_tipo_defecto"], None)
        self.assertEqual(
            event["payload"]["after_estado_real"], "DEFECTUOSA"
        )
        self.assertEqual(
            event["payload"]["after_tipo_defecto"], "MANCHA"
        )

        # Vuelve a validarse sin tocar el modelo ni el umbral.
        self.assertEqual(
            self._evaluate(model_id, ai_model_id, thresholds=[50.0])
            .status_code,
            200,
        )
        row = self._ai_row(ai_model_id)
        self.assertEqual(row["status"], "VALIDACION")
        self.assertEqual(int(row["active"] or 0), 0)

    def test_defectuosa_without_type_is_rejected_on_edit(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        response, _ = self._post_binary_case(model_id, "BUENA", score=22.0)
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        missing = self._update_case_binary(
            model_id, case_id, "DEFECTUOSA", ai_model_id=ai_model_id
        )
        self.assertEqual(missing.status_code, 400, self._json(missing))
        self.assertIn("tipo de defecto", self._json(missing)["error"])

        with_type = self._update_case_binary(
            model_id, case_id, "BUENA", "AGUJERO",
            ai_model_id=ai_model_id,
        )
        self.assertEqual(with_type.status_code, 400, self._json(with_type))

        self.assertEqual(self._case_row(case_id)["category"], "NORMAL")

    # --------------------------------------------------------
    # 10. tabla y miniaturas usan la clasificación persistida
    # --------------------------------------------------------

    def test_page_shows_binary_classification_and_thumbnails(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])
        url = f"/modelos-prenda/{model_id}/validacion-ia/{ai_model_id}"

        response, _ = self._post_binary_case(
            model_id, "DEFECTUOSA", "MANCHA", score=91.0, is_anomaly=True
        )
        self.assertEqual(response.status_code, 201, self._json(response))
        case_id = self._json(response)["case"]["id"]

        self._login(self.ADMIN_USERNAME)
        page = self.client.get(url)
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)

        self.assertIn("Estado real de la prenda", html)
        self.assertIn(VALIDATION_BINARY_NOTICE, html)
        self.assertIn("Defectuosa · Mancha", html)
        self.assertIn(f"data-case-estado=\"{case_id}\"", html)
        self.assertIn(f"data-case-tipo=\"{case_id}\"", html)

        thumbs = (
            html.split('id="valThumbs"', 1)[-1].split("</section>", 1)[0]
        )
        self.assertIn("Defectuosa · Mancha", thumbs)

        # Editar el caso refresca la miniatura con la nueva etiqueta.
        edited = self._update_case_binary(
            model_id, case_id, "BUENA", ai_model_id=ai_model_id
        )
        self.assertEqual(edited.status_code, 200, self._json(edited))

        page = self.client.get(url)
        html = page.get_data(as_text=True)
        thumbs = (
            html.split('id="valThumbs"', 1)[-1].split("</section>", 1)[0]
        )

        self.assertIn("Buena", thumbs)
        self.assertNotIn("Defectuosa · Mancha", thumbs)

    # --------------------------------------------------------
    # 12. training y versión intactos
    # --------------------------------------------------------

    def test_training_and_version_are_untouched(self):
        model_id = self.model_ids[self.MODEL_CODE]
        ai_model_id = int(self._ai_row_by_model(model_id)["id"])

        counts_before = self._counts(model_id)
        artifacts_before = self._snapshot(
            self._model_artifacts_dir(model_id, ai_model_id)
        )
        model_before = self._ai_row(ai_model_id)

        for estado, tipo in (
            ("BUENA", None),
            ("DEFECTUOSA", "MANCHA"),
            ("DEFECTUOSA", "AGUJERO"),
        ):
            response, _ = self._post_binary_case(
                model_id, estado, tipo, score=55.0
            )
            self.assertEqual(
                response.status_code, 201, self._json(response)
            )

        self.assertEqual(self._counts(model_id), counts_before)
        self.assertEqual(
            self._snapshot(
                self._model_artifacts_dir(model_id, ai_model_id)
            ),
            artifacts_before,
        )

        model_after = self._ai_row(ai_model_id)
        # Registrar el primer caso abre la sesión (ENTRENADO ->
        # VALIDACIÓN): la versión nunca queda ACTIVA ni VALIDADA.
        self.assertEqual(model_after["status"], "VALIDACION")
        self.assertEqual(int(model_after["active"] or 0), 0)
        self.assertEqual(int(model_before["active"] or 0), 0)
        self.assertEqual(
            model_after["checkpoint_path"],
            model_before["checkpoint_path"],
        )
        self.assertEqual(
            self._event_count(ai_model_id, "ACTIVATED"), 0
        )

        self.conn.commit()
        self.cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM ai_training_images t
            WHERE t.garment_model_id = %s
              AND t.sha256 IN (
                  SELECT c.image_sha256
                  FROM ai_validation_cases c
                  WHERE c.ai_model_id = %s
              )
            """,
            (model_id, ai_model_id),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 0)


# Reutiliza el andamiaje de FASE 3B (entrenamiento falso, limpieza de
# sesión/casos, helpers de API) sin volver a ejecutar sus tests.
for _helper_name in (
    "_FakeTrainer",
    "setUp",
    "_cancel_stray_jobs",
    "_train_model",
    "_invalidate_latest",
    "_clear_validation",
    "_ai_row",
    "_ai_row_by_model",
    "_session_row",
    "_case_count",
    "_event_count",
    "_counts",
    "_snapshot",
    "_model_artifacts_dir",
    "_new_validation_image",
    "_fake_predict",
    "_post",
    "_post_case",
    "_start_session",
    "_evaluate",
    "_complete",
):
    setattr(
        ValidationBinaryDbTests,
        _helper_name,
        vars(ValidationDbTests)[_helper_name],
    )
