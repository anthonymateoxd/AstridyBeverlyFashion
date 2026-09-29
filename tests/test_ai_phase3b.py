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
    VALIDATION_IMAGE_NOTICE,
    VALIDATION_RESULT_INCORRECT,
    VALIDATION_RESULT_CORRECT,
    VALIDATION_RESULT_PENDING,
    VALIDATION_SESSION_STATUS_ABIERTA,
    VALIDATION_SESSION_STATUS_CERRADA,
    VALIDATION_SESSION_STATUS_EVALUADA,
    annotate_technical_invalidation,
    get_ai_artifacts_root,
    normalize_validation_category,
    validation_bank_root,
    validation_case_relative_dir,
    validation_case_relative_path,
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
    INVALIDATED_CODE = "TEST-PH3B-INVAL"
    MODELS = (MODEL_CODE, INVALIDATED_CODE)

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
        model_id = self.model_ids[self.MODEL_CODE]
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

        again = self._complete(model_id, ai_model_id)
        self.assertEqual(again.status_code, 200, self._json(again))
        self.assertTrue(self._json(again)["result"]["already_validated"])

        restart = self._start_session(model_id, ai_model_id)
        self.assertEqual(restart.status_code, 409, self._json(restart))

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
