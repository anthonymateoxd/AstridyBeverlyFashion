"""Tests FASE 3A — entrenamiento PatchCore genérico, persistente y asíncrono.

Cubre:
- configuración efectiva del entrenamiento (defaults históricos + env);
- rutas de artefactos, log con redacción de secretos y mensajes humanos;
- API de entrenamiento (start/status/cancel) con permisos y barreras;
- ejecución real del job por el worker con un trainer FALSO;
- huérfanos (worker caído) y reintento con versión nueva;
- guardas: nunca se entrena en la petición ni se toca PATCHCORE_CKPT.

NO entrena PatchCore de verdad, NO modifica la inferencia productiva,
NO hace commit ni push.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from ai_domain import (  # noqa: E402
    AI_MODEL_TECHNICAL_INVALIDATION_PREFIX,
    AIDomainError,
    annotate_technical_invalidation,
    capture_image_relative_path,
    evaluate_retrain_availability,
    get_ai_artifacts_root,
    get_ai_capture_config,
    is_technically_invalidated,
    next_version_label,
    transition_ai_model_status,
)
from ai_training import (  # noqa: E402
    STAGE_PROTOCOL,
    STAGING_QUALITY_GATE_MESSAGE,
    TRAINING_CONFIG_DEFAULTS,
    TRAINING_UI_BLOCKED,
    TRAINING_UI_COMPLETED,
    TRAINING_UI_FAILED,
    TRAINING_UI_HISTORICAL,
    TRAINING_UI_LABELS,
    TRAINING_UI_READY,
    TRAINING_UI_RUNNING,
    create_contact_sheet,
    get_roi_fractions,
    get_training_config,
    humanize_training_error,
    model_artifact_paths,
    quality_gate_staged_images,
    redact_secrets,
    relative_to_root,
    retrain_new_version_preconditions,
)
import ai_training  # noqa: E402


def try_db_connection():
    import mysql.connector

    try:
        conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=os.environ.get(
                "MYSQL_DATABASE",
                "textile_quality_db",
            ),
            connection_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return None


DB_AVAILABLE = try_db_connection() is not None


# Secuencia global para que cada frame sembrado sea único, incluso entre
# sesiones distintas del mismo modelo (el dataset acumula todas las
# imágenes ACEPTADAS).
_FRAME_SEQUENCE = itertools.count()


def make_training_frame(index: int = 0) -> bytes:
    """Frame sintético distinto por llamada: blusa crema sobre mesa blanca.

    Reproduce el caso real de la sesión 558 (máscara vacía → fallback de
    ROI) y garantiza una marca de posición/color única por imagen para que
    ninguna fuente colapse al mismo output.
    """
    import cv2
    import numpy as np

    sequence = next(_FRAME_SEQUENCE)
    frame = np.full((600, 800, 3), 250, dtype=np.uint8)
    frame[120:450, 200:560] = (215, 225, 235)

    # Marca interior al ROI: 4 posiciones horizontales x 52 tonos de gris
    # (paso 4 niveles) = 208 combinaciones únicas.
    marker_x = 150 + (sequence // 52) % 4 * 130
    marker = 250 - (sequence % 52) * 4
    frame[440:500, marker_x:marker_x + 100] = marker

    ok, buffer = cv2.imencode(".png", frame)

    if not ok:
        raise RuntimeError("No se pudo generar el frame de prueba.")

    return buffer.tobytes()


# ============================================================
# PRUEBAS PURAS (sin base de datos)
# ============================================================


class TrainingConfigTests(unittest.TestCase):
    def test_defaults_match_verified_history(self):
        config = get_training_config(env={})

        self.assertEqual(config["backbone"], "wide_resnet50_2")
        self.assertEqual(config["layers"], ["layer2", "layer3"])
        self.assertEqual(config["coreset_ratio"], 0.05)
        self.assertEqual(config["num_neighbors"], 9)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["batch_size"], 4)
        self.assertTrue(config["pre_trained"])
        self.assertEqual(config["accelerator"], "cpu")
        self.assertTrue(config["deterministic"])

    def test_default_input_size_matches_production_inference(self):
        config = get_training_config(env={})

        self.assertEqual(config["input_size"], 256)
        self.assertEqual(config["production_inference_input_size"], 256)
        self.assertTrue(config["input_size_consistent_with_production"])

    def test_explicit_input_size_override_is_reported(self):
        config = get_training_config(env={"PATCHCORE_INPUT_SIZE": "384"})

        self.assertEqual(config["input_size"], 384)
        self.assertFalse(config["input_size_consistent_with_production"])

    def test_layers_env_is_parsed_as_list(self):
        config = get_training_config(
            env={"PATCHCORE_LAYERS": "layer2; layer3, layer1"}
        )

        self.assertEqual(config["layers"], ["layer2", "layer3", "layer1"])

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(AIDomainError):
            get_training_config(env={"PATCHCORE_INPUT_SIZE": "32"})

        with self.assertRaises(AIDomainError):
            get_training_config(env={"PATCHCORE_CORESET_RATIO": "0"})

        with self.assertRaises(AIDomainError):
            get_training_config(env={"PATCHCORE_BATCH_SIZE": "0"})

        with self.assertRaises(AIDomainError):
            get_training_config(env={"PATCHCORE_LAYERS": ","})

        with self.assertRaises(AIDomainError):
            get_training_config(env={"PATCHCORE_INPUT_SIZE": "abc"})

    def test_blank_env_values_keep_defaults(self):
        config = get_training_config(
            env={"PATCHCORE_INPUT_SIZE": "   ", "PATCHCORE_LAYERS": ""}
        )

        self.assertEqual(config["input_size"], 256)
        self.assertEqual(config["layers"], ["layer2", "layer3"])

    def test_roi_comes_from_capture_environment(self):
        roi = get_roi_fractions(env={"ROI_X1": "0.16", "ROI_Y1": "0.05"})

        self.assertEqual(roi[0], 0.16)
        self.assertEqual(roi[1], 0.05)

    def test_defaults_dictionary_is_documented(self):
        for key in (
            "backbone",
            "layers",
            "input_size",
            "coreset_ratio",
            "num_neighbors",
            "seed",
            "batch_size",
            "keep_staging",
        ):
            self.assertIn(key, TRAINING_CONFIG_DEFAULTS)


class ArtifactPathTests(unittest.TestCase):
    def setUp(self):
        self._previous = os.environ.get("AI_ARTIFACTS_ROOT")
        self._tmp = tempfile.mkdtemp(prefix="astrid_paths_")
        os.environ["AI_ARTIFACTS_ROOT"] = self._tmp

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("AI_ARTIFACTS_ROOT", None)
        else:
            os.environ["AI_ARTIFACTS_ROOT"] = self._previous
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_training_layout_under_root(self):
        paths = model_artifact_paths(35, 155)
        root = Path(self._tmp)

        self.assertEqual(paths["root"], root)
        self.assertTrue(str(paths["training_dir"]).startswith(str(root)))
        self.assertEqual(paths["checkpoint_dir"].name, "checkpoint")
        self.assertEqual(paths["staging_dir"].name, "staging")
        self.assertEqual(paths["config_path"].name, "config.json")
        self.assertEqual(paths["metadata_path"].name, "metadata.json")
        self.assertEqual(paths["log_path"].name, "training.log")

    def test_relative_path_stays_inside_root(self):
        root = Path(self._tmp)
        inside = root / "garment_1" / "models" / "ai_model_2" / "a.ckpt"
        inside.parent.mkdir(parents=True, exist_ok=True)
        inside.write_bytes(b"x")

        self.assertEqual(
            relative_to_root(inside, root),
            "garment_1/models/ai_model_2/a.ckpt",
        )

    def test_relative_path_rejects_escapes(self):
        root = Path(self._tmp) / "inner"
        root.mkdir(parents=True, exist_ok=True)
        outside = Path(self._tmp) / "outside.ckpt"
        outside.write_bytes(b"x")

        with self.assertRaises(AIDomainError):
            relative_to_root(outside, root)


class TrainingLogAndMessageTests(unittest.TestCase):
    def test_secrets_are_redacted(self):
        previous = os.environ.get("MYSQL_PASSWORD")
        os.environ["MYSQL_PASSWORD"] = "super-secret-123"
        try:
            text = redact_secrets(
                "connect failed password=super-secret-123 "
                "url rtsp://admin:super-secret-123@10.0.0.1/x"
            )
        finally:
            if previous is None:
                os.environ.pop("MYSQL_PASSWORD", None)
            else:
                os.environ["MYSQL_PASSWORD"] = previous

        self.assertNotIn("super-secret-123", text)
        self.assertIn("***", text)

    def test_human_messages_hide_technical_details(self):
        cases = {
            RuntimeError("CUDA out of memory"): "recursos",
            RuntimeError("cannot download weights from url"): "conexión",
            ValueError("La silueta detectada es demasiado pequeña."): (
                "Capture de nuevo"
            ),
            AIDomainError("El dataset no está cerrado."): "cerrado",
            AIDomainError("Ya hay un entrenamiento en curso."): "en curso",
            AIDomainError("Ya existe una versión de IA en proceso."): (
                "en proceso"
            ),
            RuntimeError("boom at line 1"): "no pudo completarse",
        }

        for error, fragment in cases.items():
            with self.subTest(error=str(error)):
                message = humanize_training_error(error)
                self.assertTrue(message.strip())
                self.assertIn(fragment, message)

    def test_stage_protocol_is_monotonic_and_finishes(self):
        progresses = [step[0] for step in STAGE_PROTOCOL]

        self.assertEqual(progresses, sorted(progresses))
        self.assertEqual(progresses[-1], 100)
        self.assertEqual(STAGE_PROTOCOL[-1][1], "COMPLETADO")
        self.assertTrue(all(step[2].strip() for step in STAGE_PROTOCOL))

    def test_every_ui_status_has_a_human_label(self):
        for status in (
            TRAINING_UI_BLOCKED,
            TRAINING_UI_READY,
            TRAINING_UI_RUNNING,
            TRAINING_UI_COMPLETED,
            TRAINING_UI_FAILED,
            TRAINING_UI_HISTORICAL,
        ):
            label = TRAINING_UI_LABELS[status]
            self.assertTrue(label.strip())
            self.assertNotIn("_", label)

    def test_historical_label_states_it_is_not_validatable(self):
        label = TRAINING_UI_LABELS[TRAINING_UI_HISTORICAL]
        self.assertIn("HIST", label.upper())
        self.assertIn("NO APTA", label.upper())


class RetrainAvailabilityTests(unittest.TestCase):
    """§4: reglas puras de «Reentrenar como nueva versión»."""

    INVALIDATED = (
        f"{AI_MODEL_TECHNICAL_INVALIDATION_PREFIX}: dataset inválido"
    )

    @staticmethod
    def _versions(*items):
        return [
            {"version": version, "status": status, "notes": notes}
            for version, status, notes in items
        ]

    def test_requires_a_previous_version(self):
        available, reason = evaluate_retrain_availability([])

        self.assertFalse(available)
        self.assertIn("versión anterior", reason)

    def test_invalidated_version_does_not_block(self):
        available, reason = evaluate_retrain_availability(
            self._versions(("v1", "ENTRENADO", self.INVALIDATED))
        )

        self.assertTrue(available)
        self.assertIsNone(reason)

    def test_valid_version_pending_validation_blocks(self):
        for status in ("ENTRENADO", "VALIDACION", "VALIDADO"):
            available, reason = evaluate_retrain_availability(
                self._versions(("v1", status, None))
            )

            with self.subTest(status=status):
                self.assertFalse(available)
                self.assertIn("complete la validación", reason.lower())

    def test_invalid_version_does_not_hide_a_newer_valid_one(self):
        available, reason = evaluate_retrain_availability(
            self._versions(
                ("v1", "ENTRENADO", self.INVALIDATED),
                ("v2", "PREPARACION", None),
            )
        )

        self.assertFalse(available)
        self.assertIn("v2", reason)

    def test_failed_and_rejected_versions_do_not_block(self):
        for status in ("FALLIDO", "RECHAZADO"):
            available, _ = evaluate_retrain_availability(
                self._versions(("v1", status, None))
            )

            with self.subTest(status=status):
                self.assertTrue(available)

    def test_next_version_counts_every_version(self):
        self.assertEqual(next_version_label([]), "v1")
        self.assertEqual(
            next_version_label(
                self._versions(
                    ("v1", "ENTRENADO", self.INVALIDATED),
                    ("v2", "FALLIDO", None),
                )
            ),
            "v3",
        )


class ProductionSafetyTests(unittest.TestCase):
    """Guardas: el entrenamiento jamás toca la inferencia productiva."""

    def test_training_sources_never_read_production_checkpoint(self):
        reads = (
            'getenv("PATCHCORE_CKPT"',
            "environ.get(\"PATCHCORE_CKPT\"",
            "environ[\"PATCHCORE_CKPT\"]",
        )

        for name in ("ai_training.py", "ai_worker.py", "ai_domain.py"):
            with self.subTest(source=name):
                source = (ROOT / name).read_text(encoding="utf-8")

                for needle in reads:
                    self.assertNotIn(needle, source)

                # Mención documental permitida, lectura prohibida.
                self.assertNotIn("os.environ['PATCHCORE_CKPT']", source)

    def test_worker_service_does_not_receive_production_checkpoint(self):
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        worker_block = compose.split("ai-worker:", 1)

        self.assertEqual(len(worker_block), 2, "Falta el servicio ai-worker")
        self.assertNotIn("PATCHCORE_CKPT", worker_block[1])

    def test_template_renders_training_panel(self):
        template = (
            ROOT / "templates" / "garment_model_detail.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="aiTrainingPanel"', template)
        self.assertIn("ASTRID_AI_TRAINING", template)
        self.assertIn("ENTRENAR IA", template)


class WorkerHelperTests(unittest.TestCase):
    def test_worker_id_honours_environment(self):
        import ai_worker

        previous = os.environ.get("AI_WORKER_ID")
        os.environ["AI_WORKER_ID"] = "mi-worker"

        try:
            self.assertEqual(ai_worker.default_worker_id(), "mi-worker")
        finally:
            if previous is None:
                os.environ.pop("AI_WORKER_ID", None)
            else:
                os.environ["AI_WORKER_ID"] = previous

    def test_invalid_float_falls_back_to_default(self):
        import ai_worker

        self.assertEqual(ai_worker._env_float("NO_EXISTE_XYZ", 3.0), 3.0)

        previous = os.environ.get("AI_WORKER_POLL_SECONDS")
        os.environ["AI_WORKER_POLL_SECONDS"] = "no-numero"

        try:
            self.assertEqual(ai_worker._env_float("AI_WORKER_POLL_SECONDS", 3.0), 3.0)
        finally:
            if previous is None:
                os.environ.pop("AI_WORKER_POLL_SECONDS", None)
            else:
                os.environ["AI_WORKER_POLL_SECONDS"] = previous


# ============================================================
# PRUEBAS CON BASE DE DATOS (omitidas sin MySQL)
# ============================================================


class _TrainingDbCase(unittest.TestCase):
    """Base compartida: app importada, usuarios/modelos y raíz temporal."""

    MODELS: tuple = ()
    USERS: tuple = ()  # (username, role)

    @classmethod
    def setUpClass(cls):
        import importlib

        import mysql.connector

        cls.A = importlib.import_module("app")

        cls._orig_camera_worker = cls.A.ensure_camera_capture_worker
        cls._orig_ai_worker = cls.A._ensure_ai_capture_worker
        cls.A.ensure_camera_capture_worker = lambda *a, **k: False
        cls.A._ensure_ai_capture_worker = lambda *a, **k: False

        cls._previous_root = os.environ.get("AI_ARTIFACTS_ROOT")
        cls.tmp_root = Path(tempfile.mkdtemp(prefix="astrid_ph3a_"))
        os.environ["AI_ARTIFACTS_ROOT"] = str(cls.tmp_root)

        cls.database = os.environ.get(
            "MYSQL_DATABASE",
            "textile_quality_db",
        )
        cls.conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=cls.database,
        )
        cls.cur = cls.conn.cursor(dictionary=True)
        from ai_domain import ensure_ai_schema
        ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()
        cls.client = cls.A.app.test_client()
        cls.min_images = int(get_ai_capture_config()["min_images"])

        cls.user_ids = {}
        for username, role in cls.USERS:
            cls.user_ids[username] = cls._ensure_user(username, role)

        cls.model_ids = {}
        for code in cls.MODELS:
            cls.model_ids[code] = cls._ensure_model(code, "APROBADO")

    @classmethod
    def _ensure_user(cls, username, role):
        cls.cur.execute(
            "SELECT id FROM users WHERE username = %s",
            (username,),
        )
        row = cls.cur.fetchone()

        if row:
            cls.cur.execute(
                "UPDATE users SET role = %s, active = 1 WHERE id = %s",
                (role, row["id"]),
            )
        else:
            cls.cur.execute(
                """
                INSERT INTO users (username, password_hash, role, active)
                VALUES (%s, 'x', %s, 1)
                """,
                (username, role),
            )

        cls.conn.commit()
        cls.cur.execute(
            "SELECT id FROM users WHERE username = %s",
            (username,),
        )
        return int(cls.cur.fetchone()["id"])

    @classmethod
    def _ensure_model(cls, code, status):
        cls.cur.execute(
            "SELECT id FROM garment_models WHERE code = %s",
            (code,),
        )
        row = cls.cur.fetchone()

        if row:
            cls.cur.execute(
                """
                UPDATE garment_models
                SET status = %s, active = 1
                WHERE id = %s
                """,
                (status, row["id"]),
            )
            model_id = int(row["id"])
        else:
            cls.cur.execute(
                """
                INSERT INTO garment_models (code, name, status, active)
                VALUES (%s, %s, %s, 1)
                """,
                (code, f"Modelo fase 3a {code}", status),
            )
            model_id = int(cls.cur.lastrowid)

        cls.conn.commit()
        return model_id

    @classmethod
    def _seed_accepted(cls, model_id, actor_id, count=None):
        """Sesión COMPLETADA con N imágenes ACEPTADAS + archivos reales."""
        total = int(count if count is not None else cls.min_images)

        cls.cur.execute(
            """
            INSERT INTO ai_capture_sessions
                (garment_model_id, status, created_by, finished_at)
            VALUES (%s, 'COMPLETADA', %s, NOW())
            """,
            (model_id, actor_id),
        )
        session_id = int(cls.cur.lastrowid)
        cls.conn.commit()

        root = get_ai_artifacts_root()

        for index in range(total):
            frame = make_training_frame(index)
            filename = f"frame_{index:04d}.png"
            relative = capture_image_relative_path(
                model_id,
                session_id,
                filename,
            )
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(frame)

            cls.cur.execute(
                """
                INSERT INTO ai_training_images (
                    capture_session_id,
                    garment_model_id,
                    image_path,
                    sha256,
                    status,
                    frame_sequence,
                    coverage,
                    quality_score
                )
                VALUES (%s, %s, %s, %s, 'ACEPTADA', %s, 0.5, 90.0)
                """,
                (
                    session_id,
                    model_id,
                    relative,
                    hashlib.sha256(frame).hexdigest(),
                    index + 1,
                ),
            )

        cls.conn.commit()
        return session_id

    @classmethod
    def _connect(cls):
        import mysql.connector

        return mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=cls.database,
        )

    @classmethod
    def _delete_model(cls, model_id):
        from ai_domain import ensure_ai_schema

        cls.cur.execute(
            """
            SELECT COUNT(*) AS closed
            FROM ai_datasets
            WHERE garment_model_id = %s
              AND status <> 'ABIERTO'
            """,
            (model_id,),
        )
        closed = int(cls.cur.fetchone()["closed"] or 0) > 0

        if closed:
            # El trigger de inmutabilidad impide purgar datasets ya
            # cerrados; se retira solo para la limpieza de los tests
            # y se recrea de inmediato con ensure_ai_schema.
            cls.cur.execute(
                "DROP TRIGGER IF EXISTS trg_ai_dataset_images_before_delete"
            )
            cls.conn.commit()

        cls.cur.execute(
            """
            DELETE FROM ai_events
            WHERE ai_model_id IN (
                    SELECT id FROM garment_ai_models
                    WHERE garment_model_id = %s
                )
               OR dataset_id IN (
                    SELECT id FROM ai_datasets
                    WHERE garment_model_id = %s
                )
               OR capture_session_id IN (
                    SELECT id FROM ai_capture_sessions
                    WHERE garment_model_id = %s
                )
            """,
            (model_id, model_id, model_id),
        )
        cls.cur.execute(
            """
            DELETE FROM ai_jobs
            WHERE ai_model_id IN (
                SELECT id FROM garment_ai_models
                WHERE garment_model_id = %s
            )
            """,
            (model_id,),
        )
        cls.cur.execute(
            "DELETE FROM garment_ai_models WHERE garment_model_id = %s",
            (model_id,),
        )
        cls.cur.execute(
            """
            DELETE di FROM ai_dataset_images di
            JOIN ai_datasets d ON d.id = di.dataset_id
            WHERE d.garment_model_id = %s
            """,
            (model_id,),
        )
        cls.cur.execute(
            "DELETE FROM ai_datasets WHERE garment_model_id = %s",
            (model_id,),
        )
        cls.cur.execute(
            "DELETE FROM ai_training_images WHERE garment_model_id = %s",
            (model_id,),
        )
        cls.cur.execute(
            "DELETE FROM ai_capture_sessions WHERE garment_model_id = %s",
            (model_id,),
        )
        cls.conn.commit()

        if closed:
            ensure_ai_schema(cls.cur, cls.database)
            cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        from ai_domain import ensure_ai_schema

        cls.A.ensure_camera_capture_worker = cls._orig_camera_worker
        cls.A._ensure_ai_capture_worker = cls._orig_ai_worker

        for username in cls.user_ids:
            cls.cur.execute(
                "DELETE FROM ai_events WHERE actor_id = ("
                "SELECT id FROM users WHERE username = %s)",
                (username,),
            )
        cls.conn.commit()

        for code in cls.MODELS:
            model_id = cls.model_ids.get(code)

            if model_id:
                cls._delete_model(model_id)

            cls.cur.execute(
                "DELETE FROM garment_models WHERE code = %s",
                (code,),
            )
        cls.conn.commit()

        for username in cls.user_ids:
            cls.cur.execute(
                "DELETE FROM users WHERE username = %s",
                (username,),
            )
        cls.conn.commit()

        ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

        if cls._previous_root is None:
            os.environ.pop("AI_ARTIFACTS_ROOT", None)
        else:
            os.environ["AI_ARTIFACTS_ROOT"] = cls._previous_root

        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    # --------------------------------------------------------
    # utilidades
    # --------------------------------------------------------

    def _login(self, username):
        with self.client.session_transaction() as sess:
            sess.clear()
            sess["user_id"] = self.user_ids[username]

    def _freeze_threshold(self, model_id, ai_model_id, value=50.0,
                          username=None):
        self._login(username or self.ADMIN_USERNAME)
        return self.client.post(
            "/api/ai/validation/threshold/freeze",
            json={
                "garment_model_id": model_id,
                "ai_model_id": ai_model_id,
                "threshold_final": value,
                "provenance": "Confirmación técnica de prueba unitaria.",
            },
        )

    def _json(self, response):
        return json.loads(response.get_data(as_text=True))

    def _actor(self, username=None):
        return username or self.ADMIN_USERNAME

    def _start(self, model_id, username=None):
        self._login(self._actor(username))
        return self.client.post(
            "/api/ai/training/start",
            json={"garment_model_id": model_id},
        )

    def _status(self, model_id, username=None):
        self._login(self._actor(username))
        return self.client.get(
            f"/api/ai/training/status?garment_model_id={model_id}"
        )

    def _cancel(self, job_id, username=None):
        self._login(self._actor(username))
        return self.client.post(
            "/api/ai/training/cancel",
            json={"job_id": job_id},
        )

    def _latest_job(self, model_id):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT j.id, j.status, j.progress, j.stage, j.stage_label,
                   j.error_message, j.log_path, j.worker_id, j.config_json,
                   j.ai_model_id, j.dataset_id
            FROM ai_jobs j
            JOIN garment_ai_models m ON m.id = j.ai_model_id
            WHERE m.garment_model_id = %s AND j.kind = 'TRAINING'
            ORDER BY j.id DESC
            LIMIT 1
            """,
            (model_id,),
        )
        return self.cur.fetchone()

    def _model_row(self, model_id):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT id, version, status, checkpoint_path, checkpoint_hash,
                   input_size, dataset_id, normal_images_count, metrics_json
            FROM garment_ai_models
            WHERE garment_model_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_id,),
        )
        return self.cur.fetchone()

    def _job_row(self, job_id):
        self.conn.commit()
        self.cur.execute(
            "SELECT * FROM ai_jobs WHERE id = %s",
            (job_id,),
        )
        return self.cur.fetchone()


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de API de entrenamiento omitidos.",
)
class TrainingApiDbTests(_TrainingDbCase):
    ADMIN_USERNAME = "ph3a_admin"
    QM_USERNAME = "ph3a_qm"

    MODELS = (
        "TEST-PH3A-EMPTY",
        "TEST-PH3A-READY",
        "TEST-PH3A-FLOW",
    )
    USERS = (
        ("ph3a_admin", "ADMIN"),
        ("ph3a_qm", "QUALITY_MANAGER"),
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor = cls.user_ids[cls.ADMIN_USERNAME]

        cls._seed_accepted(cls.model_ids["TEST-PH3A-READY"], actor)
        cls._seed_accepted(cls.model_ids["TEST-PH3A-FLOW"], actor)

    def setUp(self):
        self._login(self.ADMIN_USERNAME)

    def test_status_without_dataset_is_blocked(self):
        model_id = self.model_ids["TEST-PH3A-EMPTY"]
        data = self._json(self._status(model_id))

        self.assertTrue(data["ok"])
        training = data["training"]
        self.assertEqual(training["ui_status"], TRAINING_UI_BLOCKED)
        self.assertFalse(training["can_train"])
        self.assertTrue(training["blocked_reason"])
        self.assertEqual(training["accepted_count"], 0)

    def test_status_ready_for_seeded_model(self):
        model_id = self.model_ids["TEST-PH3A-READY"]
        data = self._json(self._status(model_id))

        self.assertTrue(data["ok"], data)
        training = data["training"]
        self.assertEqual(
            training["ui_status"],
            TRAINING_UI_READY,
            training,
        )
        self.assertTrue(training["can_train"], training)
        self.assertIsNone(training["blocked_reason"], training)
        self.assertEqual(training["accepted_count"], self.min_images, training)

    def test_detail_page_renders_training_panel(self):
        model_id = self.model_ids["TEST-PH3A-READY"]
        self._login(self.ADMIN_USERNAME)
        response = self.client.get(f"/modelos-prenda/{model_id}")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("aiTrainingPanel", html)
        self.assertIn("ASTRID_AI_TRAINING", html)

    def test_start_status_and_cancel_flow(self):
        model_id = self.model_ids["TEST-PH3A-FLOW"]
        actor = self.user_ids[self.ADMIN_USERNAME]

        # 1) solicitud inicial
        data = self._json(self._start(model_id))
        self.assertEqual(data["ok"], True, data)
        job_id = data["job_id"]
        training = data["training"]
        self.assertEqual(training["ui_status"], TRAINING_UI_RUNNING)
        self.assertTrue(training["can_cancel"])

        # dataset materializado y cerrado por el backend
        job = self._job_row(job_id)
        self.assertEqual(job["status"], "PENDIENTE")
        self.assertEqual(job["kind"], "TRAINING")
        self.assertTrue(job["config_json"])

        model = self._model_row(model_id)
        self.assertEqual(model["status"], "PREPARACION")
        self.assertIsNotNone(model["dataset_id"])

        dataset_id = int(model["dataset_id"])
        self.conn.commit()
        self.cur.execute(
            "SELECT status, image_count FROM ai_datasets WHERE id = %s",
            (dataset_id,),
        )
        dataset = self.cur.fetchone()
        self.assertEqual(dataset["status"], "CERRADO")
        self.assertEqual(int(dataset["image_count"]), self.min_images)

        # 2) segunda solicitud: barrera activa
        second = self._start(model_id)
        self.assertEqual(second.status_code, 409)
        payload = self._json(second)
        self.assertFalse(payload["ok"])
        self.assertIn("entrenamiento", payload["error"].lower())

        # 3) estado por polling
        status = self._json(self._status(model_id))
        self.assertEqual(
            status["training"]["ui_status"],
            TRAINING_UI_RUNNING,
        )
        self.assertEqual(status["training"]["job_id"], job_id)

        # 4) cancelación del job aún pendiente
        cancelled = self._json(self._cancel(job_id))
        self.assertEqual(cancelled["ok"], True, cancelled)
        self.assertEqual(
            self._job_row(job_id)["status"],
            "CANCELADO",
        )
        self.assertEqual(
            cancelled["training"]["ui_status"],
            TRAINING_UI_READY,
        )

        # 5) reintento: crea un job nuevo y vuelve a quedar activo
        retry = self._json(self._start(model_id))
        self.assertEqual(retry["ok"], True, retry)
        retry_job_id = retry["job_id"]
        self.assertNotEqual(retry_job_id, job_id)
        self.assertEqual(
            self._job_row(retry_job_id)["status"],
            "PENDIENTE",
        )

        # 6) cancelar dos veces no está permitido
        again = self._cancel(retry_job_id)
        self.assertEqual(again.status_code, 200)

        blocked = self._cancel(retry_job_id)
        self.assertEqual(blocked.status_code, 409)

        # el modelo sigue en preparación: nunca se activa solo
        self.assertEqual(self._model_row(model_id)["status"], "PREPARACION")
        self._ = actor

    def test_start_requires_management_permission(self):
        model_id = self.model_ids["TEST-PH3A-READY"]
        response = self._start(model_id, username=self.QM_USERNAME)

        self.assertEqual(response.status_code, 403)
        self.assertFalse(self._json(response)["ok"])

    def test_cancel_requires_management_permission(self):
        # La barrera de rol responde antes de mirar el job.
        self._login(self.QM_USERNAME)
        response = self.client.post(
            "/api/ai/training/cancel",
            json={"job_id": 1},
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(self._json(response)["ok"])

    def test_status_for_unknown_model(self):
        response = self._status(99999999)
        self.assertEqual(response.status_code, 404)

    def test_status_rejects_invalid_parameters(self):
        self._login(self.ADMIN_USERNAME)
        response = self.client.get("/api/ai/training/status")
        self.assertEqual(response.status_code, 400)


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de ejecución del worker omitidos.",
)
class TrainingExecutionDbTests(_TrainingDbCase):
    ADMIN_USERNAME = "ph3w_admin"

    MODELS = (
        "TEST-PH3W-OK",
        "TEST-PH3W-FAIL",
        "TEST-PH3W-ORPH",
        "TEST-PH3W-STAT",
    )
    USERS = (("ph3w_admin", "ADMIN"),)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor = cls.user_ids[cls.ADMIN_USERNAME]

        for code in cls.MODELS:
            cls._seed_accepted(cls.model_ids[code], actor)

    class _FakeTrainer:
        """Trainer falso: escribe un checkpoint trivial (sin anomalib)."""

        def train(self, ctx):
            ctx["report"](35, "EMBEDDINGS", "Generando representaciones")
            ctx["report"](70, "MEMORY_BANK", "Construyendo modelo")

            checkpoint = Path(ctx["checkpoint_dir"]) / "model.ckpt"
            checkpoint.write_bytes(b"FAKE-CHECKPOINT-" + b"0" * 512)

            return {
                "checkpoint": str(checkpoint),
                "memory_bank": {
                    "shape": [13, 1536],
                    "dtype": "torch.float32",
                },
                "image_size": [256, 256],
                "accelerator": "cpu",
            }

    class _BrokenTrainer:
        def train(self, ctx):
            raise RuntimeError("falla deliberada del trainer")

    def setUp(self):
        self._login(self.ADMIN_USERNAME)
        # Ningún job ajeno debe interferir con el reclamo del worker.
        self.cur.execute(
            """
            UPDATE ai_jobs
            SET status = 'CANCELADO', finished_at = NOW()
            WHERE status = 'PENDIENTE'
            """
        )
        self.conn.commit()

    def _request_job(self, model_id):
        response = self._start(model_id)
        self.assertEqual(response.status_code, 201, self._json(response))
        return int(self._json(response)["job_id"])

    def test_worker_claims_and_completes_job_with_fake_trainer(self):
        import ai_worker
        from ai_training import execute_training_job

        model_id = self.model_ids["TEST-PH3W-OK"]
        job_id = self._request_job(model_id)

        claimed = ai_worker.claim_job(self._connect, "ph3w-worker-1")
        self.assertIsNotNone(claimed)
        self.assertEqual(int(claimed["id"]), job_id)
        self.assertEqual(claimed["status"], "EN_CURSO")
        self.assertEqual(claimed["worker_id"], "ph3w-worker-1")

        result = execute_training_job(
            job_id,
            connect=self._connect,
            trainer=self._FakeTrainer(),
        )

        self.assertTrue(result["ok"], result)

        job = self._job_row(job_id)
        self.assertEqual(job["status"], "COMPLETADO")
        self.assertEqual(float(job["progress"]), 100.0)
        self.assertEqual(job["stage"], "COMPLETADO")
        self.assertIsNotNone(job["finished_at"])
        self.assertTrue(job["artifacts_json"])

        model = self._model_row(model_id)
        self.assertEqual(model["status"], "ENTRENADO")
        self.assertTrue(model["checkpoint_path"])
        self.assertTrue(model["checkpoint_hash"])
        self.assertEqual(str(model["input_size"]), "256")
        self.assertEqual(int(model["normal_images_count"]), self.min_images)
        self.assertIsNotNone(model["metrics_json"])

        paths = model_artifact_paths(model_id, int(model["id"]))
        self.assertTrue(paths["config_path"].is_file())
        self.assertTrue(paths["metadata_path"].is_file())
        self.assertTrue(paths["log_path"].is_file())

        config = json.loads(paths["config_path"].read_text("utf-8"))
        self.assertEqual(config["config"]["input_size"], 256)
        self.assertEqual(config["dataset_id"], int(model["dataset_id"]))

        metadata = json.loads(paths["metadata_path"].read_text("utf-8"))
        self.assertEqual(metadata["result"], "COMPLETADO")
        self.assertEqual(len(metadata["artifacts"]), 1)
        self.assertTrue(metadata["stages"])

        log = paths["log_path"].read_text("utf-8")
        self.assertIn("INTENTO job_id", log)
        self.assertIn("ETAPA", log)
        self.assertIn("resultado=COMPLETADO", log)
        self.assertNotIn("MYSQL_PASSWORD", log)

        checkpoint = Path(paths["root"]) / model["checkpoint_path"]
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            model["checkpoint_hash"],
        )

        self.conn.commit()
        self.cur.execute(
            "SELECT event_type FROM ai_events WHERE ai_model_id = %s",
            (int(model["id"]),),
        )
        events = {row["event_type"] for row in self.cur.fetchall()}
        self.assertIn("TRAINING_STARTED", events)
        self.assertIn("TRAINING_COMPLETED", events)

        # estado visible para la UI
        status = self._json(self._status(model_id))
        self.assertEqual(
            status["training"]["ui_status"],
            TRAINING_UI_COMPLETED,
        )
        self.assertEqual(status["training"]["ui_label"], "PENDIENTE DE VALIDACIÓN")

    def test_status_stays_pending_validation_without_job_row(self):
        """La versión ENTRENADO manda aunque no exista job asociado.

        El historial de jobs puede purgarse (limpiezas o migraciones);
        la ficha no debe volver a ofrecer "ENTRENAR IA" mientras la
        versión siga pendiente de validación.
        """
        import ai_worker
        from ai_training import execute_training_job

        model_id = self.model_ids["TEST-PH3W-STAT"]
        job_id = self._request_job(model_id)

        claimed = ai_worker.claim_job(self._connect, "ph3w-worker-stat")
        self.assertIsNotNone(claimed)

        result = execute_training_job(
            job_id,
            connect=self._connect,
            trainer=self._FakeTrainer(),
        )
        self.assertTrue(result["ok"], result)

        self.conn.commit()
        self.cur.execute(
            """
            DELETE j FROM ai_jobs j
            JOIN garment_ai_models m ON m.id = j.ai_model_id
            WHERE m.garment_model_id = %s
            """,
            (model_id,),
        )
        self.conn.commit()

        training = self._json(self._status(model_id))["training"]
        self.assertEqual(training["ui_status"], TRAINING_UI_COMPLETED)
        self.assertEqual(training["ui_label"], "PENDIENTE DE VALIDACIÓN")
        self.assertFalse(training["can_train"])
        self.assertEqual(training["model_status"], "ENTRENADO")
        self.assertEqual(training["version"], "v1")
        self.assertIsNone(training["job_id"])

    def test_failed_training_marks_job_and_model_failed(self):
        from ai_training import execute_training_job

        model_id = self.model_ids["TEST-PH3W-FAIL"]
        job_id = self._request_job(model_id)

        self.cur.execute(
            "UPDATE ai_jobs SET status = 'EN_CURSO', started_at = NOW() "
            "WHERE id = %s",
            (job_id,),
        )
        self.conn.commit()

        result = execute_training_job(
            job_id,
            connect=self._connect,
            trainer=self._BrokenTrainer(),
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])

        job = self._job_row(job_id)
        self.assertEqual(job["status"], "FALLIDO")
        self.assertTrue(job["error_message"])
        self.assertIsNotNone(job["finished_at"])

        model = self._model_row(model_id)
        self.assertEqual(model["status"], "FALLIDO")
        self.assertIsNone(model["checkpoint_path"])

        status = self._json(self._status(model_id))
        training = status["training"]
        self.assertEqual(training["ui_status"], TRAINING_UI_FAILED)
        self.assertTrue(training["error_human"])
        self.assertIn("no pudo completarse", training["error_human"])

        # reintento: nueva versión vN + job nuevo
        retry = self._json(self._start(model_id))
        self.assertTrue(retry["ok"], retry)
        retry_job_id = int(retry["job_id"])

        self.conn.commit()
        self.cur.execute(
            "SELECT version FROM garment_ai_models WHERE id = %s",
            (int(retry["ai_model_id"]),),
        )
        self.assertEqual(self.cur.fetchone()["version"], "v2")
        self.assertEqual(self._job_row(retry_job_id)["status"], "PENDIENTE")

        # limpieza del job de reintento
        self._cancel(retry_job_id)

    def test_orphaned_job_is_reclaimed_then_retry_uses_new_version(self):
        import ai_worker

        model_id = self.model_ids["TEST-PH3W-ORPH"]
        job_id = self._request_job(model_id)

        claimed = ai_worker.claim_job(self._connect, "ph3w-dead-worker")
        self.assertIsNotNone(claimed)
        self.assertEqual(int(claimed["id"]), job_id)

        model = self._model_row(model_id)
        self.cur.execute(
            "UPDATE garment_ai_models SET status = 'ENTRENANDO' "
            "WHERE id = %s",
            (int(model["id"]),),
        )
        self.cur.execute(
            """
            UPDATE ai_jobs
            SET heartbeat_at = NOW() - INTERVAL 3600 SECOND
            WHERE id = %s
            """,
            (job_id,),
        )
        self.conn.commit()

        reclaimed = ai_worker.reclaim_stale_jobs(self._connect, 900)
        self.assertEqual(reclaimed, 1)

        self.assertEqual(self._job_row(job_id)["status"], "FALLIDO")
        self.assertEqual(self._model_row(model_id)["status"], "FALLIDO")

        # reintento sin reanudación ciegas: versión nueva + job nuevo
        retry = self._start(model_id)
        self.assertEqual(retry.status_code, 201, self._json(retry))
        retry_job_id = int(self._json(retry)["job_id"])

        self.cur.execute(
            """
            SELECT version FROM garment_ai_models
            WHERE id = %s
            """,
            (self._job_row(retry_job_id)["ai_model_id"],),
        )
        self.assertEqual(self.cur.fetchone()["version"], "v2")
        self.assertEqual(
            self._job_row(retry_job_id)["status"],
            "PENDIENTE",
        )

        self._cancel(retry_job_id)

    def test_run_once_without_jobs_does_nothing(self):
        import ai_worker

        worked = ai_worker.run_once(
            self._connect,
            "ph3w-idle-worker",
            orphan_seconds=900,
        )

        self.assertFalse(worked)


# ============================================================
# FASE 3A.2 — FALLBACK ROI, QUALITY GATE, HOJA DE CONTACTOS Y
# INVALIDACIÓN TÉCNICA DE LA VERSIÓN
# ============================================================

ROI_TEST = (0.16, 0.05, 0.84, 0.88)

STAGING_GATE_EXACT_MESSAGE = (
    "El conjunto preparado para entrenamiento no superó la validación "
    "de calidad. No se inició el entrenamiento."
)


def cream_blouse_frame():
    """Blusa crema sobre mesa blanca: el caso real que colapsaba el staging."""
    import numpy as np

    frame = np.full((600, 800, 3), 250, dtype=np.uint8)
    frame[120:450, 200:560] = (215, 225, 235)
    return frame


def rose_blouse_frame(background=255):
    """Prenda rosa con silueta válida dentro del ROI."""
    import cv2
    import numpy as np

    frame = np.full((600, 800, 3), background, dtype=np.uint8)
    cv2.rectangle(frame, (240, 100), (560, 470), (140, 90, 230), -1)
    return frame


class StagingFallbackTests(unittest.TestCase):
    """§5: una máscara inútil nunca produce una imagen en blanco."""

    def test_cream_blouse_mask_is_empty_and_roi_fallback_keeps_content(self):
        import numpy as np

        import patchcore_preprocess

        frame = cream_blouse_frame()
        x1, y1, x2, y2 = patchcore_preprocess.roi_bounds_from_fractions(
            frame,
            ROI_TEST,
        )
        mask = patchcore_preprocess.create_garment_mask(frame, (x1, y1, x2, y2))

        self.assertEqual(int(np.count_nonzero(mask)), 0)

        out = patchcore_preprocess.build_patchcore_input(
            frame,
            mask,
            (x1, y1, x2, y2),
        )
        roi = frame[y1:y2, x1:x2]

        self.assertTrue(np.array_equal(out, roi))
        self.assertGreater(float(out.std()), 2.0)
        self.assertGreater(int(len(np.unique(out.reshape(-1, 3), axis=0))), 1)

    def test_valid_mask_keeps_white_background(self):
        import numpy as np

        import patchcore_preprocess

        frame = rose_blouse_frame(background=200)
        x1, y1, x2, y2 = patchcore_preprocess.roi_bounds_from_fractions(
            frame,
            ROI_TEST,
        )
        roi_bounds = (x1, y1, x2, y2)
        mask = patchcore_preprocess.create_garment_mask(frame, roi_bounds)
        self.assertGreater(int(np.count_nonzero(mask)), 0)

        out = patchcore_preprocess.build_patchcore_input(
            frame,
            mask,
            roi_bounds,
        )
        roi = frame[y1:y2, x1:x2]

        self.assertTrue((out == 255).any(), "Debe conservar fondo blanco")
        self.assertTrue(
            (out != 255).any(),
            "Debe conservar los píxeles de la prenda",
        )
        self.assertFalse(np.array_equal(out, roi), "Sí pinta el fondo")

    def test_mask_below_coverage_threshold_falls_back_to_roi(self):
        import numpy as np

        import patchcore_preprocess

        frame = rose_blouse_frame()
        x1, y1, x2, y2 = patchcore_preprocess.roi_bounds_from_fractions(
            frame,
            ROI_TEST,
        )
        roi_bounds = (x1, y1, x2, y2)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # Cobertura ~10% del ROI: insuficiente para describir la prenda.
        mask[y1:y1 + 40, x1:x2] = 255

        coverage = patchcore_preprocess.mask_coverage(mask[y1:y2, x1:x2])
        self.assertLess(coverage, patchcore_preprocess.MIN_MASK_COVERAGE)

        out = patchcore_preprocess.build_patchcore_input(
            frame,
            mask,
            roi_bounds,
        )
        self.assertTrue(np.array_equal(out, frame[y1:y2, x1:x2]))

    def test_real_session_558_image_is_not_blank_after_fix(self):
        import cv2
        import numpy as np

        import patchcore_preprocess

        source = (
            ROOT
            / "ai_artifacts/garment_35/capture_sessions/session_558/"
            "accepted/frame_10081.jpg"
        )

        if not source.is_file():
            self.skipTest("No hay capturas reales de la sesión 558.")

        frame = cv2.imread(str(source))
        self.assertIsNotNone(frame)

        mask = patchcore_preprocess.create_garment_mask(
            frame,
            patchcore_preprocess.roi_bounds_from_fractions(frame, ROI_TEST),
        )
        out = patchcore_preprocess.build_patchcore_input(
            frame,
            mask,
            patchcore_preprocess.roi_bounds_from_fractions(frame, ROI_TEST),
        )

        self.assertGreater(float(out.std()), 2.0)
        self.assertGreater(
            int(len(np.unique(out.reshape(-1, 3)[::16], axis=0))),
            100,
        )


class AcceptedImageStagingContractTests(unittest.TestCase):
    """La aceptación de captura permite ROI fallback, nunca relaja integridad."""

    GARMENT_ID = 42
    DATASET_ID = 12
    CAPTURE_ID = 13392
    SESSION_ID = 81
    RELATIVE = "garment_42/datasets/dataset_12/images/13392.png"
    MANIFEST = "garment_42/datasets/dataset_12/manifest.json"

    class Cursor:
        def __init__(self, row, dataset, validation_hashes=()):
            self.row = dict(row)
            self.dataset = dict(dataset)
            self.validation_hashes = list(validation_hashes)
            self.result = None

        def execute(self, sql, params=()):
            if "FROM ai_dataset_images di" in sql:
                self.result = [self.row]
            elif "FROM ai_datasets WHERE id" in sql:
                self.result = self.dataset
            elif "FROM ai_validation_cases" in sql:
                self.result = [
                    {"image_sha256": value} for value in self.validation_hashes
                ]
            elif "FROM ai_final_test_cases" in sql:
                self.result = []
            else:
                raise AssertionError(f"Consulta inesperada de staging: {sql}")

        def fetchall(self):
            result, self.result = self.result, None
            return result

        def fetchone(self):
            result, self.result = self.result, None
            return result

    def setUp(self):
        import cv2

        self.tmp = Path(tempfile.mkdtemp(prefix="accepted_staging_"))
        self.root = self.tmp / "artifacts"
        self.staging = self.tmp / "staging"
        source = self.root.joinpath(*self.RELATIVE.split("/"))
        source.parent.mkdir(parents=True, exist_ok=True)
        self.frame = rose_blouse_frame()
        self.assertTrue(cv2.imwrite(str(source), self.frame))
        source_bytes = source.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        self.row = {
            "id": self.CAPTURE_ID,
            "image_path": "garment_42/capture_sessions/session_81/accepted/capture.png",
            "sha256": digest,
            "garment_model_id": self.GARMENT_ID,
            "status": "ACEPTADA",
            "capture_session_id": self.SESSION_ID,
            "session_status": "COMPLETADA",
        }
        self.item = {
            "image_id": self.CAPTURE_ID,
            "path": self.RELATIVE,
            "sha256": digest,
            "size": len(source_bytes),
            "category": "NORMAL",
            "capture_session_id": self.SESSION_ID,
        }
        self.dataset = {
            "id": self.DATASET_ID,
            "garment_model_id": self.GARMENT_ID,
            "status": "CERRADO",
            "image_count": 1,
            "dataset_path": "garment_42/datasets/dataset_12",
            "manifest_path": self.MANIFEST,
        }
        self._write_manifest()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_manifest(self, *, category="NORMAL", item=None):
        payload_item = dict(item or self.item, category=category)
        payload = {
            "dataset_id": self.DATASET_ID,
            "garment_model_id": self.GARMENT_ID,
            "version": "d1",
            "image_count": 1,
            "images": [payload_item],
            "content_hash": "regression-test",
        }
        target = self.root.joinpath(*self.MANIFEST.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        self.dataset["manifest_hash"] = hashlib.sha256(target.read_bytes()).hexdigest()

    def _stage(self, *, row=None, validation_hashes=()):
        from unittest import mock

        cursor = self.Cursor(row or self.row, self.dataset, validation_hashes)
        with mock.patch("ai_training.get_ai_artifacts_root", return_value=self.root):
            return ai_training.stage_training_images(
                cursor, self.DATASET_ID, self.staging, ROI_TEST
            )

    def test_accepted_image_with_good_segmentation_uses_normal_pipeline(self):
        staged = self._stage()
        self.assertEqual(len(staged), 1)
        self.assertFalse(staged[0]["segmentation_fallback"])
        output = self._cv2_read(staged[0]["path"])
        bounds = ai_training.patchcore_preprocess.roi_bounds_from_fractions(
            self.frame, ROI_TEST
        )
        self.assertEqual(output.shape, self.frame[bounds[1]:bounds[3], bounds[0]:bounds[2]].shape)

    def test_accepted_image_false_negative_uses_full_roi_fallback(self):
        from unittest import mock

        cursor = self.Cursor(self.row, self.dataset)
        with mock.patch("ai_training.get_ai_artifacts_root", return_value=self.root), \
             mock.patch.object(
                 ai_training.patchcore_preprocess,
                 "create_garment_mask",
                 side_effect=RuntimeError("La silueta detectada es demasiado pequeña."),
             ):
            staged = ai_training.stage_training_images(
                cursor, self.DATASET_ID, self.staging, ROI_TEST
            )
        self.assertTrue(staged[0]["segmentation_fallback"])
        output = self._cv2_read(staged[0]["path"])
        bounds = ai_training.patchcore_preprocess.roi_bounds_from_fractions(
            self.frame, ROI_TEST
        )
        self.assertTrue(
            self._np_equal(output, self.frame[bounds[1]:bounds[3], bounds[0]:bounds[2]])
        )

    def test_corrupt_image_is_blocked(self):
        source = self.root.joinpath(*self.RELATIVE.split("/"))
        source.write_bytes(b"not an image")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        row = dict(self.row, sha256=digest)
        item = dict(self.item, sha256=digest, size=source.stat().st_size)
        self._write_manifest(item=item)
        with self.assertRaisesRegex(AIDomainError, "No se pudo leer"):
            self._stage(row=row)

    def test_hash_incorrecto_is_blocked(self):
        with self.assertRaisesRegex(AIDomainError, "hash"):
            self._stage(row=dict(self.row, sha256="0" * 64))

    def test_nonaccepted_capture_is_blocked(self):
        with self.assertRaisesRegex(AIDomainError, "ACEPTADA"):
            self._stage(row=dict(self.row, status="RECHAZADA"))

    def test_validation_hash_is_blocked_from_training(self):
        with self.assertRaisesRegex(AIDomainError, "validation"):
            self._stage(validation_hashes=(self.row["sha256"],))

    def test_mixed_training_validation_content_is_blocked(self):
        with self.assertRaises(AIDomainError):
            self._stage(validation_hashes=(self.row["sha256"],))

    def test_manifest_defect_category_is_blocked(self):
        self._write_manifest(category="DEFECT")
        with self.assertRaisesRegex(AIDomainError, "NORMAL"):
            self._stage()

    @staticmethod
    def _cv2_read(path):
        import cv2
        return cv2.imread(str(path))

    @staticmethod
    def _np_equal(left, right):
        import numpy as np
        return np.array_equal(left, right)


class VersionedPreprocessingProfileTests(unittest.TestCase):
    def test_model_bundle_reads_its_versioned_preprocessing_profile(self):
        import ai_validation
        from unittest import mock

        temp = Path(tempfile.mkdtemp(prefix="preprocessing_bundle_"))
        model_id = 901
        root = temp
        relative_checkpoint = (
            f"garment_42/models/ai_model_{model_id}/checkpoint/model.ckpt"
        )
        checkpoint = root.joinpath(*relative_checkpoint.split("/"))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint-fixture")
        config_path = (
            root / "garment_42" / "models" / f"ai_model_{model_id}"
            / "training" / "config.json"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps({"config": {"input_size": 256,
                                    "validation_preprocessing_profile": "FULL_ROI"}}),
            encoding="utf-8",
        )

        class Cursor:
            def execute(self, _sql, _params):
                pass

            def fetchone(self):
                return {
                    "id": model_id, "garment_model_id": 42, "version": "v7",
                    "status": "ENTRENADO", "notes": None,
                    "checkpoint_path": relative_checkpoint,
                    "checkpoint_hash": "a" * 64, "input_size": "256",
                    "dataset_id": 8, "active": 0, "threshold_final": None,
                    "threshold_frozen_at": None, "threshold_frozen_by": None,
                    "threshold_provenance": None,
                }

        try:
            with mock.patch("ai_validation.get_ai_artifacts_root", return_value=root):
                bundle = ai_validation.load_model_bundle(Cursor(), model_id)
            self.assertEqual(bundle["preprocessing_profile"], "FULL_ROI")
            self.assertEqual(bundle["input_size"], 256)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_training_profile_is_summarized_from_actual_staging_modes(self):
        self.assertEqual(
            ai_training.summarize_preprocessing_profile([
                {"preprocessing_mode": "FULL_ROI"},
                {"preprocessing_mode": "FULL_ROI"},
            ]),
            "FULL_ROI",
        )
        self.assertEqual(
            ai_training.summarize_preprocessing_profile([
                {"preprocessing_mode": "SEGMENTED"},
                {"preprocessing_mode": "SEGMENTED"},
            ]),
            "SEGMENTED",
        )
        self.assertEqual(
            ai_training.summarize_preprocessing_profile([
                {"preprocessing_mode": "FULL_ROI"},
                {"preprocessing_mode": "SEGMENTED"},
            ]),
            "PER_IMAGE",
        )

    def test_full_roi_validation_profile_matches_training_roi_exactly(self):
        import cv2
        import numpy as np

        import ai_validation
        import patchcore_preprocess

        temp = Path(tempfile.mkdtemp(prefix="validation_full_roi_"))
        image_path = temp / "accepted.png"
        frame = rose_blouse_frame()
        self.assertTrue(cv2.imwrite(str(image_path), frame))
        output_path = None
        try:
            from unittest import mock
            with mock.patch.object(
                patchcore_preprocess,
                "create_garment_mask",
                side_effect=AssertionError("FULL_ROI profile must not segment"),
            ):
                output_path = ai_validation.build_validation_input(
                    image_path, preprocessing_profile="FULL_ROI"
                )
            bounds = ai_validation._roi_bounds(frame)
            expected = frame[bounds[1]:bounds[3], bounds[0]:bounds[2]]
            actual = cv2.imread(str(output_path))
            self.assertTrue(np.array_equal(actual, expected))
        finally:
            if output_path:
                Path(output_path).unlink(missing_ok=True)
            shutil.rmtree(temp, ignore_errors=True)


class QualityGateTests(unittest.TestCase):
    """§6/§7: gate de calidad y hoja de contactos."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="astrid_gate_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, image):
        import cv2

        path = self.tmp / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return {"image_id": len(list(self.tmp.glob("*.png"))), "path": path}

    def test_gate_rejects_uniform_images(self):
        import numpy as np

        staged = [
            self._write(f"blank_{index}.png", np.full((64, 64, 3), 255, np.uint8))
            for index in range(3)
        ]

        gate = quality_gate_staged_images(staged)

        self.assertFalse(gate["ok"])
        self.assertTrue(gate["errors"])
        self.assertIn("sin variación", gate["errors"][0])

    def test_gate_accepts_real_images(self):
        staged = [
            self._write(f"ok_{index}.png", rose_blouse_frame(background=value))
            for index, value in enumerate((255, 210, 165))
        ]

        gate = quality_gate_staged_images(staged, expected_count=3)

        self.assertTrue(gate["ok"], gate["errors"])
        self.assertEqual(gate["details"]["images"], 3)
        self.assertGreater(gate["details"]["std_min"], 2.0)

    def test_gate_rejects_fully_duplicated_dataset(self):
        staged = [
            self._write(f"dup_{index}.png", rose_blouse_frame())
            for index in range(5)
        ]

        gate = quality_gate_staged_images(staged, expected_count=5)

        self.assertFalse(gate["ok"])
        self.assertTrue(
            any("idénticas" in error for error in gate["errors"]),
            gate["errors"],
        )

    def test_fifty_two_distinct_sources_do_not_collapse(self):
        import hashlib

        import cv2
        import numpy as np

        import patchcore_preprocess

        def source(index):
            image = np.full((240, 320, 3), 250, np.uint8)
            image[40:150, 80:240] = (215, 225, 235)
            image[170:210, 40:100] = 250 - (index % 52) * 4
            return image

        first = source(0)
        bounds = patchcore_preprocess.roi_bounds_from_fractions(
            first,
            ROI_TEST,
        )

        outputs = set()
        staged = []

        for index in range(52):
            image = source(index)
            mask = patchcore_preprocess.create_garment_mask(image, bounds)
            out = patchcore_preprocess.build_patchcore_input(
                image,
                mask,
                bounds,
            )
            outputs.add(hashlib.sha256(out.tobytes()).hexdigest())
            staged.append(self._write(f"n_{index}.png", out))

        self.assertEqual(
            len(outputs),
            52,
            "52 fuentes distintas colapsaron al mismo output",
        )

        gate = quality_gate_staged_images(staged, expected_count=52)

        self.assertTrue(gate["ok"], gate["errors"])
        self.assertEqual(gate["details"]["distinct_visual"], 52)

    def test_gate_requires_every_dataset_image(self):
        staged = [self._write("only.png", rose_blouse_frame())]

        gate = quality_gate_staged_images(staged, expected_count=52)

        self.assertFalse(gate["ok"])
        self.assertTrue(
            any("declara 52" in error for error in gate["errors"]),
            gate["errors"],
        )

    def test_gate_message_is_the_required_one(self):
        self.assertEqual(STAGING_GATE_EXACT_MESSAGE, STAGING_QUALITY_GATE_MESSAGE)
        self.assertEqual(
            STAGING_QUALITY_GATE_MESSAGE,
            "El conjunto preparado para entrenamiento no superó la "
            "validación de calidad. No se inició el entrenamiento.",
        )

    def test_contact_sheet_builds_expected_grid(self):
        import numpy as np

        sources = [
            self._write(f"sheet_{index}.png", rose_blouse_frame())
            for index in range(5)
        ]
        target = self.tmp / "sheet.png"

        result = create_contact_sheet(
            [item["path"] for item in sources],
            target,
            columns=4,
        )

        import cv2

        sheet = cv2.imread(str(result))
        self.assertIsNotNone(sheet)
        self.assertEqual(sheet.shape[1], 4 * 320)
        self.assertEqual(sheet.shape[0], 2 * (240 + 22))
        self.assertGreater(float(np.asarray(sheet).std()), 2.0)


class TrainingServingPipelineTests(unittest.TestCase):
    """§4: entrenamiento e inferencia comparten el mismo preprocesamiento."""

    @classmethod
    def setUpClass(cls):
        import importlib

        cls.A = importlib.import_module("app")
        cls._orig_camera = cls.A.ensure_camera_capture_worker
        cls._orig_ai_worker = cls.A._ensure_ai_capture_worker
        cls.A.ensure_camera_capture_worker = lambda *a, **k: False
        cls.A._ensure_ai_capture_worker = lambda *a, **k: False

    @classmethod
    def tearDownClass(cls):
        cls.A.ensure_camera_capture_worker = cls._orig_camera
        cls.A._ensure_ai_capture_worker = cls._orig_ai_worker

    def test_same_roi_fractions_give_the_same_bounds(self):
        import patchcore_preprocess

        frame = rose_blouse_frame()

        self.assertEqual(
            patchcore_preprocess.roi_bounds_from_fractions(
                frame,
                get_roi_fractions(),
            ),
            self.A.get_roi_bounds(frame),
        )

    def test_both_paths_use_the_same_segmentation_code(self):
        import inspect

        import patchcore_preprocess

        training = inspect.getsource(ai_training.stage_training_images)
        self.assertIn("create_garment_mask(", training)
        self.assertIn("build_patchcore_input(", training)

        serving = inspect.getsource(self.A.detect_defect)
        self.assertIn("create_garment_mask(", serving)
        self.assertIn("build_patchcore_regions(", serving)

        # Un solo código: la entrada de entrenamiento delega en regions,
        # que es exactamente lo que ejecuta producción.
        self.assertIn(
            "build_patchcore_regions(",
            inspect.getsource(patchcore_preprocess.build_patchcore_input),
        )

    def test_same_frame_gives_the_same_patchcore_input(self):
        import cv2

        import numpy as np

        import patchcore_preprocess

        frame = rose_blouse_frame(background=180)
        train_bounds = patchcore_preprocess.roi_bounds_from_fractions(
            frame,
            get_roi_fractions(),
        )
        serve_bounds = self.A.get_roi_bounds(frame)
        self.assertEqual(train_bounds, serve_bounds)

        mask_training = patchcore_preprocess.create_garment_mask(
            frame,
            train_bounds,
        )
        mask_serving = self.A.create_garment_mask(frame)
        self.assertTrue(np.array_equal(mask_training, mask_serving))

        training_input = patchcore_preprocess.build_patchcore_input(
            frame,
            mask_training,
            train_bounds,
        )
        _, _, serving_input = patchcore_preprocess.build_patchcore_regions(
            frame,
            mask_serving,
            serve_bounds,
            cv2_mod=cv2,
        )
        self.assertTrue(np.array_equal(training_input, serving_input))


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; pruebas del gate de staging omitidas.",
)
class TrainingStagingGateDbTests(_TrainingDbCase):
    """§6: el gate impide crear el job TRAINING si el staging falla."""

    ADMIN_USERNAME = "ph3g_admin"

    MODELS = ("TEST-PH3A-GATE", "TEST-PH3A-GATE-LOW")
    USERS = (("ph3g_admin", "ADMIN"),)

    @staticmethod
    def _blank_stage(cur, dataset_id, staging_dir, roi_fractions):
        """Sustituto de stage_training_images: escribe PNGs colapsados."""
        import cv2
        import numpy as np

        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=True)

        cur.execute(
            """
            SELECT ti.id
            FROM ai_dataset_images di
            JOIN ai_training_images ti ON ti.id = di.image_id
            WHERE di.dataset_id = %s
            ORDER BY ti.id ASC
            """,
            (int(dataset_id),),
        )
        rows = cur.fetchall()
        staged = []

        for row in rows:
            target = staging / f"frame_{int(row['id'])}.png"
            cv2.imwrite(
                str(target),
                np.full((120, 160, 3), 255, dtype=np.uint8),
            )
            staged.append(
                {"image_id": int(row["id"]), "path": target, "source": target}
            )

        return staged

    def _request(self, model_id, patch_target=None):
        from unittest import mock

        actor = self.user_ids[self.ADMIN_USERNAME]
        cur = self.conn.cursor(dictionary=True)

        try:
            self.conn.commit()

            if patch_target is None:
                return ai_training.request_training_job(
                    cur,
                    garment_model_id=model_id,
                    actor_id=actor,
                )

            with mock.patch(
                "ai_training.stage_training_images",
                side_effect=patch_target,
            ):
                return ai_training.request_training_job(
                    cur,
                    garment_model_id=model_id,
                    actor_id=actor,
                )
        finally:
            cur.close()

    def test_blank_staging_blocks_job_creation(self):
        model_id = self.model_ids["TEST-PH3A-GATE"]
        self._seed_accepted(model_id, self.user_ids[self.ADMIN_USERNAME])

        with self.assertRaises(AIDomainError) as ctx:
            self._request(model_id, self._blank_stage)

        self.assertEqual(str(ctx.exception), STAGING_QUALITY_GATE_MESSAGE)

        self.conn.rollback()
        self.conn.commit()
        self.cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM ai_jobs j
            JOIN garment_ai_models m ON m.id = j.ai_model_id
            WHERE m.garment_model_id = %s
            """,
            (model_id,),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 0)

        # Tampoco queda ninguna versión entrenada por el intento fallido.
        self.cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM garment_ai_models
            WHERE garment_model_id = %s
              AND status IN (
                  'ENTRENANDO', 'ENTRENADO', 'VALIDACION',
                  'VALIDADO', 'ACTIVO'
              )
            """,
            (model_id,),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 0)

    def test_real_staging_creates_job_and_contact_sheet(self):
        model_id = self.model_ids["TEST-PH3A-GATE"]
        self._seed_accepted(model_id, self.user_ids[self.ADMIN_USERNAME])

        result = self._request(model_id, None)
        self.conn.commit()

        self.assertEqual(result["job"]["status"], "PENDIENTE")

        ai_model_id = int(result["ai_model"]["id"])
        paths = model_artifact_paths(model_id, ai_model_id)
        staged_pngs = sorted(Path(paths["staging_dir"]).glob("*.png"))

        self.assertTrue(staged_pngs)
        self.assertTrue(Path(paths["training_dir"], "contact_sheet.png").is_file())

        gate = quality_gate_staged_images(
            [{"image_id": 0, "path": path} for path in staged_pngs],
        )
        self.assertTrue(gate["ok"], gate["errors"])

        dataset = result["dataset"]
        manifest_path = Path(get_ai_artifacts_root()) / dataset["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            dataset["manifest_hash"],
        )
        self.assertEqual(manifest["image_count"], dataset["image_count"])
        self.assertEqual(len(manifest["images"]), dataset["image_count"])
        self.assertTrue(all(item["category"] == "NORMAL" for item in manifest["images"]))
        self.assertTrue(all(item["capture_session_id"] for item in manifest["images"]))
        for item in manifest["images"]:
            path = Path(get_ai_artifacts_root()) / item["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, item["size"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])

        # Reintentar la materialización reutiliza la misma versión/conjunto.
        cur = self.conn.cursor(dictionary=True)
        try:
            again = ai_training.materialize_training_dataset(
                cur, model_id, self.user_ids[self.ADMIN_USERNAME]
            )
            self.assertEqual(again["id"], dataset["id"])
            self.assertEqual(again["image_count"], dataset["image_count"])
            cur.execute(
                "SELECT COUNT(*) AS total FROM ai_dataset_images WHERE dataset_id = %s",
                (int(dataset["id"]),),
            )
            self.assertEqual(int(cur.fetchone()["total"]), dataset["image_count"])

            original = manifest_path.read_bytes()
            try:
                manifest_path.unlink()
                missing_manifest = ai_training.verify_training_dataset(
                    cur, dataset_id=dataset["id"], ai_model_id=result["ai_model"]["id"]
                )
                self.assertFalse(missing_manifest["ok"])
            finally:
                manifest_path.write_bytes(original)

            try:
                manifest_path.write_text("{corrupt", encoding="utf-8")
                corrupt_manifest = ai_training.verify_training_dataset(
                    cur, dataset_id=dataset["id"], ai_model_id=result["ai_model"]["id"]
                )
                self.assertFalse(corrupt_manifest["ok"])
            finally:
                manifest_path.write_bytes(original)

            image_path = Path(get_ai_artifacts_root()) / manifest["images"][0]["path"]
            image_bytes = image_path.read_bytes()
            try:
                image_path.unlink()
                missing_image = ai_training.verify_training_dataset(
                    cur, dataset_id=dataset["id"], ai_model_id=result["ai_model"]["id"]
                )
                self.assertFalse(missing_image["ok"])
            finally:
                image_path.write_bytes(image_bytes)
        finally:
            cur.close()

    def test_fewer_than_minimum_images_rejects_materialization(self):
        model_id = self.model_ids["TEST-PH3A-GATE-LOW"]
        self._seed_accepted(
            model_id, self.user_ids[self.ADMIN_USERNAME],
            count=max(0, self.min_images - 1),
        )
        cur = self.conn.cursor(dictionary=True)
        try:
            with self.assertRaises(AIDomainError):
                ai_training.materialize_training_dataset(
                    cur, model_id, self.user_ids[self.ADMIN_USERNAME]
                )
        finally:
            cur.close()


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; pruebas de invalidación técnica omitidas.",
)
class TechnicalInvalidationDbTests(_TrainingDbCase):
    """§8: la versión queda documentada como NO APTA PARA ACTIVACIÓN."""

    ADMIN_USERNAME = "ph3i_admin"

    MODELS = ("TEST-PH3A-INV",)
    USERS = (("ph3i_admin", "ADMIN"),)

    def _new_version(self, model_id, status="PREPARACION"):
        cls = type(self)
        cls._version_seq = getattr(cls, "_version_seq", 0) + 1
        version = f"v{cls._version_seq}"

        self.conn.commit()
        self.cur.execute(
            """
            INSERT INTO garment_ai_models
                (garment_model_id, version, model_type, status, active,
                 notes, created_by)
            VALUES (%s, %s, 'PatchCore', %s, 0, NULL, NULL)
            """,
            (model_id, version, status),
        )
        version_id = int(self.cur.lastrowid)
        self.conn.commit()
        return version_id

    def _walk_to(self, version_id, statuses):
        actor = self.user_ids[self.ADMIN_USERNAME]

        for status in statuses:
            self.conn.commit()
            transition_ai_model_status(self.cur, version_id, status, actor)
            self.conn.commit()

    def test_annotation_marks_notes_and_blocks_activation(self):
        from ai_domain import (
            AI_MODEL_STATUS_ACTIVO,
            AI_MODEL_STATUS_ENTRENADO,
            AI_MODEL_STATUS_ENTRENANDO,
            AI_MODEL_STATUS_VALIDACION,
            AI_MODEL_STATUS_VALIDADO,
        )

        model_id = self.model_ids["TEST-PH3A-INV"]
        version_id = self._new_version(model_id)
        self._walk_to(
            version_id,
            (
                AI_MODEL_STATUS_ENTRENANDO,
                AI_MODEL_STATUS_ENTRENADO,
                AI_MODEL_STATUS_VALIDACION,
                AI_MODEL_STATUS_VALIDADO,
            ),
        )

        result = annotate_technical_invalidation(
            self.cur,
            version_id,
            "dataset de prueba inválido",
            actor_id=self.user_ids[self.ADMIN_USERNAME],
        )
        self.conn.commit()

        self.assertTrue(is_technically_invalidated(result["notes"]))
        self.assertTrue(
            AI_MODEL_TECHNICAL_INVALIDATION_PREFIX in result["notes"]
        )
        self.assertEqual(result["status"], AI_MODEL_STATUS_VALIDADO)

        self.cur.execute(
            "SELECT COUNT(*) AS total FROM ai_events "
            "WHERE ai_model_id = %s AND event_type = 'TECHNICAL_INVALIDATION'",
            (version_id,),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 1)

        self.conn.commit()
        with self.assertRaises(AIDomainError) as ctx:
            transition_ai_model_status(
                self.cur,
                version_id,
                AI_MODEL_STATUS_ACTIVO,
                self.user_ids[self.ADMIN_USERNAME],
            )
        self.conn.rollback()

        self.assertIn("NO APTO PARA ACTIVACIÓN", str(ctx.exception))

    def test_annotation_refuses_active_version(self):
        from ai_domain import (
            AI_MODEL_STATUS_ACTIVO,
            AI_MODEL_STATUS_ENTRENADO,
            AI_MODEL_STATUS_ENTRENANDO,
            AI_MODEL_STATUS_VALIDACION,
            AI_MODEL_STATUS_VALIDADO,
        )

        model_id = self.model_ids["TEST-PH3A-INV"]
        version_id = self._new_version(model_id)
        self._walk_to(
            version_id,
            (
                AI_MODEL_STATUS_ENTRENANDO,
                AI_MODEL_STATUS_ENTRENADO,
                AI_MODEL_STATUS_VALIDACION,
                AI_MODEL_STATUS_VALIDADO,
                AI_MODEL_STATUS_ACTIVO,
            ),
        )

        with self.assertRaises(AIDomainError):
            annotate_technical_invalidation(
                self.cur,
                version_id,
                "no debe poder marcarse estando activa",
                actor_id=self.user_ids[self.ADMIN_USERNAME],
            )
        self.conn.rollback()


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; pruebas de reentrenamiento omitidas.",
)
class RetrainNewVersionDbTests(_TrainingDbCase):
    """FASE 3A.3: «Entrenar nueva versión» crea vN+1 sin tocar vN."""

    ADMIN_USERNAME = "ph3r_admin"
    QM_USERNAME = "ph3r_qm"

    MODELS = (
        "TEST-PH3R-EMPTY",
        "TEST-PH3R-GATE",
        "TEST-PH3R-HIST",
        "TEST-PH3R-PEND",
        "TEST-PH3R-RUN",
        "TEST-PH3R-STD",
        "TEST-PH3R-TRAN",
        "TEST-PH3R-V2N",
    )
    USERS = (
        ("ph3r_admin", "ADMIN"),
        ("ph3r_qm", "QUALITY_MANAGER"),
    )

    class _FakeTrainer:
        """Trainer falso: escribe un checkpoint trivial (sin anomalib)."""

        def train(self, ctx):
            ctx["report"](35, "EMBEDDINGS", "Generando representaciones")

            checkpoint = Path(ctx["checkpoint_dir"]) / "model.ckpt"
            checkpoint.write_bytes(b"FAKE-CHECKPOINT-" + b"0" * 512)

            return {
                "checkpoint": str(checkpoint),
                "memory_bank": {
                    "shape": [13, 1536],
                    "dtype": "torch.float32",
                },
                "image_size": [256, 256],
                "accelerator": "cpu",
            }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor = cls.user_ids[cls.ADMIN_USERNAME]

        for code in cls.MODELS:
            cls._seed_accepted(cls.model_ids[code], actor)

    def setUp(self):
        self._login(self.ADMIN_USERNAME)
        # Ningún job ajeno debe interferir con el reclamo del worker.
        self.cur.execute(
            """
            UPDATE ai_jobs
            SET status = 'CANCELADO', finished_at = NOW()
            WHERE status = 'PENDIENTE'
            """
        )
        self.conn.commit()

    # --------------------------------------------------------
    # utilidades
    # --------------------------------------------------------

    def _retrain(self, model_id, username=None):
        self._login(username or self.ADMIN_USERNAME)
        return self.client.post(
            "/api/ai/training/retrain",
            json={"garment_model_id": model_id},
        )

    def _versions(self, model_id):
        self.conn.commit()
        self.cur.execute(
            """
            SELECT id, version, status, notes, dataset_id,
                   checkpoint_path, checkpoint_hash
            FROM garment_ai_models
            WHERE garment_model_id = %s
            ORDER BY id ASC
            """,
            (model_id,),
        )
        return self.cur.fetchall()

    @staticmethod
    def _snapshot(directory):
        base = Path(directory)

        if not base.is_dir():
            return {}

        return {
            str(path.relative_to(base)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(base.rglob("*"))
            if path.is_file()
        }

    def _train_to_entrenado(self, model_id):
        """Encola y ejecuta un TRAINING real con el trainer falso."""
        import ai_worker
        from ai_training import execute_training_job

        response = self._start(model_id)
        self.assertEqual(response.status_code, 201, self._json(response))
        job_id = int(self._json(response)["job_id"])

        claimed = ai_worker.claim_job(self._connect, "ph3r-worker")
        self.assertIsNotNone(claimed)

        result = execute_training_job(
            job_id,
            connect=self._connect,
            trainer=self._FakeTrainer(),
        )
        self.assertTrue(result["ok"], result)
        self.conn.commit()

        versions = self._versions(model_id)
        self.assertEqual(versions[-1]["status"], "ENTRENADO")
        return job_id, int(versions[-1]["id"])

    def _invalidate_latest(self, model_id, detail="dataset de prueba inválido"):
        versions = self._versions(model_id)
        latest = versions[-1]
        annotate_technical_invalidation(
            self.cur,
            int(latest["id"]),
            detail,
            actor_id=self.user_ids[self.ADMIN_USERNAME],
        )
        self.conn.commit()
        return int(latest["id"])

    # --------------------------------------------------------
    # estado para la UI
    # --------------------------------------------------------

    def test_trained_version_without_invalidation_is_not_retrainable(self):
        model_id = self.model_ids["TEST-PH3R-PEND"]
        self._train_to_entrenado(model_id)

        training = self._json(self._status(model_id))["training"]

        self.assertEqual(training["ui_status"], TRAINING_UI_COMPLETED)
        self.assertFalse(training["retrain_available"], training)
        self.assertIn(
            "complete la validación",
            (training["retrain_blocked_reason"] or "").lower(),
        )
        self.assertFalse(training["can_train"])
        self.assertEqual(training["historical_version"], None)
        self.assertEqual(training["next_version"], "v2")

    def test_invalidated_version_is_reported_as_historical(self):
        model_id = self.model_ids["TEST-PH3R-HIST"]
        self._train_to_entrenado(model_id)
        self._invalidate_latest(model_id)

        training = self._json(self._status(model_id))["training"]

        self.assertEqual(training["ui_status"], TRAINING_UI_HISTORICAL)
        self.assertIn("HIST", training["ui_label"].upper())
        self.assertIn("NO APTA", training["ui_label"].upper())
        self.assertFalse(training["can_train"])
        self.assertTrue(training["retrain_available"], training)
        self.assertIsNone(training["retrain_blocked_reason"], training)
        self.assertEqual(training["version"], "v1")
        self.assertEqual(training["historical_version"], "v1")
        self.assertEqual(training["next_version"], "v2")
        self.assertEqual(training["model_status"], "ENTRENADO")
        self.assertIn("v1", training["invalidated_versions"])

        # La ficha lo pinta y enlaza la acción de nueva versión.
        self._login(self.ADMIN_USERNAME)
        html = self.client.get(
            f"/modelos-prenda/{model_id}"
        ).get_data(as_text=True)

        self.assertIn("aiTrainingPanel", html)
        self.assertIn("training/retrain", html)
        self.assertIn("no apta para validación", html)
        self.assertIn("NO VALIDADA / NO APTA PARA ACTIVACIÓN", html)

    # --------------------------------------------------------
    # la acción de reentrenamiento
    # --------------------------------------------------------

    def test_retrain_creates_new_version_and_keeps_history_intact(self):
        model_id = self.model_ids["TEST-PH3R-V2N"]
        old_job_id, _ = self._train_to_entrenado(model_id)
        self._invalidate_latest(model_id)

        before = self._versions(model_id)
        self.assertEqual(len(before), 1)
        v1 = before[0]
        v1_paths = model_artifact_paths(model_id, int(v1["id"]))
        v1_files = self._snapshot(v1_paths["training_dir"])

        response = self._retrain(model_id)
        data = self._json(response)
        self.assertEqual(response.status_code, 201, data)

        self.assertEqual(data["ok"], True)
        self.assertEqual(data["ai_model_version"], "v2")
        self.assertNotEqual(int(data["job_id"]), old_job_id)

        retrain = data["retrain"]
        self.assertEqual(retrain["version"], "v2")
        self.assertEqual(retrain["next_version"], "v2")
        self.assertEqual(
            [item["version"] for item in retrain["superseded_versions"]],
            ["v1"],
        )
        self.assertEqual(int(retrain["images_reused"]), self.min_images)

        # El dataset se reutiliza: no se clona ni se vuelve a cerrar.
        self.assertEqual(int(data["dataset_id"]), int(v1["dataset_id"]))

        after = self._versions(model_id)
        self.assertEqual(
            [item["version"] for item in after],
            ["v1", "v2"],
        )
        v2 = after[1]
        self.assertEqual(v2["status"], "PREPARACION")
        self.assertFalse(is_technically_invalidated(v2["notes"]))

        # La versión histórica queda exactamente igual, artefactos incluidos.
        self.assertEqual(after[0]["id"], v1["id"])
        self.assertEqual(after[0]["status"], v1["status"])
        self.assertEqual(after[0]["checkpoint_hash"], v1["checkpoint_hash"])
        self.assertTrue(is_technically_invalidated(after[0]["notes"]))
        self.assertEqual(
            self._snapshot(v1_paths["training_dir"]),
            v1_files,
        )

        v2_paths = model_artifact_paths(model_id, int(v2["id"]))
        self.assertNotEqual(
            str(v1_paths["training_dir"]),
            str(v2_paths["training_dir"]),
        )

        # Auditoría del reentrenamiento.
        self.cur.execute(
            """
            SELECT payload_json FROM ai_events
            WHERE ai_model_id = %s
              AND event_type = 'NEW_VERSION_TRAINING_REQUESTED'
            """,
            (int(v2["id"]),),
        )
        event = self.cur.fetchone()
        self.assertIsNotNone(event)

        payload = json.loads(event["payload_json"])
        self.assertEqual(payload["version"], "v2")
        self.assertEqual(payload["job_id"], int(data["job_id"]))
        self.assertEqual(payload["garment_model_id"], model_id)
        self.assertEqual(
            [item["version"] for item in payload["superseded_versions"]],
            ["v1"],
        )
        self.assertEqual(int(payload["images_reused"]), self.min_images)
        self.assertEqual(
            payload["artifacts_dir"],
            f"garment_{model_id}/models/ai_model_{int(v2['id'])}",
        )

        # La ficha ahora muestra el entrenamiento de v2 en curso.
        training = self._json(self._status(model_id))["training"]
        self.assertEqual(training["ui_status"], TRAINING_UI_RUNNING)
        self.assertFalse(training["retrain_available"])
        self.assertEqual(training["model_id"], int(v2["id"]))

    def test_historical_version_cannot_be_validated_nor_activated(self):
        from ai_domain import AI_MODEL_STATUS_ACTIVO, AI_MODEL_STATUS_VALIDACION

        model_id = self.model_ids["TEST-PH3R-TRAN"]
        self._train_to_entrenado(model_id)
        self._invalidate_latest(model_id)

        v1_id = int(self._versions(model_id)[0]["id"])
        actor = self.user_ids[self.ADMIN_USERNAME]

        # No vuelve a entrar al circuito de validación.
        with self.assertRaises(AIDomainError) as ctx:
            transition_ai_model_status(
                self.cur,
                v1_id,
                AI_MODEL_STATUS_VALIDACION,
                actor,
            )
        self.conn.rollback()
        self.assertIn("NO APTO PARA ACTIVACIÓN", str(ctx.exception))

        # Tampoco hay camino hasta ACTIVO.
        with self.assertRaises(AIDomainError):
            transition_ai_model_status(
                self.cur,
                v1_id,
                AI_MODEL_STATUS_ACTIVO,
                actor,
            )
        self.conn.rollback()

        versions = self._versions(model_id)
        self.assertEqual(versions[0]["status"], "ENTRENADO")
        self.assertTrue(is_technically_invalidated(versions[0]["notes"]))

        # La única salida es entrenar una versión nueva.
        training = self._json(self._status(model_id))["training"]
        self.assertTrue(training["retrain_available"], training)

    def test_retrain_blocked_while_another_training_runs(self):
        model_id = self.model_ids["TEST-PH3R-RUN"]
        self._train_to_entrenado(model_id)
        self._invalidate_latest(model_id)

        # El camino normal de /start también sigue funcionando.
        started = self._start(model_id)
        self.assertEqual(started.status_code, 201, self._json(started))
        self.assertEqual(self._json(started)["ai_model_version"], "v2")

        blocked = self._retrain(model_id)
        self.assertEqual(blocked.status_code, 409)
        payload = self._json(blocked)
        self.assertFalse(payload["ok"])
        self.assertIn("entrenamiento", payload["error"].lower())

        # Solo se creó una v2, nunca una v3.
        self.assertEqual(
            [item["version"] for item in self._versions(model_id)],
            ["v1", "v2"],
        )

        training = self._json(self._status(model_id))["training"]
        self.assertFalse(training["retrain_available"])
        self.assertEqual(training["ui_status"], TRAINING_UI_RUNNING)

    def test_retrain_requires_a_previous_version(self):
        model_id = self.model_ids["TEST-PH3R-EMPTY"]

        blocked = self._retrain(model_id)
        self.assertEqual(blocked.status_code, 409)

        payload = self._json(blocked)
        self.assertFalse(payload["ok"])
        self.assertIn("versión anterior", payload["error"].lower())

        training = payload["training"]
        self.assertFalse(training["retrain_available"])
        self.assertIn(
            "versión anterior",
            (training["retrain_blocked_reason"] or "").lower(),
        )
        self.assertIsNone(training["next_version"])

        self.assertEqual(self._versions(model_id), [])

    def test_retrain_requires_management_permission(self):
        model_id = self.model_ids["TEST-PH3R-EMPTY"]
        response = self._retrain(model_id, username=self.QM_USERNAME)

        self.assertEqual(response.status_code, 403)
        self.assertFalse(self._json(response)["ok"])

    def test_quality_gate_failure_rolls_back_the_new_version(self):
        from unittest import mock

        model_id = self.model_ids["TEST-PH3R-GATE"]
        self._train_to_entrenado(model_id)
        self._invalidate_latest(model_id)

        with mock.patch(
            "ai_training.stage_training_images",
            side_effect=TrainingStagingGateDbTests._blank_stage,
        ):
            response = self._retrain(model_id)

        self.assertEqual(response.status_code, 409, self._json(response))

        # Sin versión nueva, sin job y sin evento de reentrenamiento.
        self.assertEqual(
            [item["version"] for item in self._versions(model_id)],
            ["v1"],
        )
        self.assertEqual(
            self._json(response)["training"]["ui_status"],
            TRAINING_UI_HISTORICAL,
        )

        self.conn.commit()
        self.cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM ai_jobs j
            JOIN garment_ai_models m ON m.id = j.ai_model_id
            WHERE m.garment_model_id = %s
            """,
            (model_id,),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 1)

        self.cur.execute(
            """
            SELECT COUNT(*) AS total FROM ai_events
            WHERE event_type = 'NEW_VERSION_TRAINING_REQUESTED'
              AND ai_model_id IN (
                  SELECT id FROM garment_ai_models
                  WHERE garment_model_id = %s
              )
            """,
            (model_id,),
        )
        self.assertEqual(int(self.cur.fetchone()["total"]), 0)

    def test_normal_start_still_works_when_only_version_is_historical(self):
        model_id = self.model_ids["TEST-PH3R-STD"]
        self._train_to_entrenado(model_id)
        self._invalidate_latest(model_id)

        self._start(model_id)

        self.assertEqual(
            [item["version"] for item in self._versions(model_id)],
            ["v1", "v2"],
        )


if __name__ == "__main__":
    unittest.main()
