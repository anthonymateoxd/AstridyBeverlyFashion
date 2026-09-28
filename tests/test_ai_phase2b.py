"""Tests FASE 2B — interfaz simple y dinámica de preparación de IA.

Cubre:
- Generación automática y persistente de códigos de modelo de prenda.
- Estados de UI, etiquetas en español y mensajes humanos.
- API de captura desde la ficha (start/status/stop/cancel) y permisos.
- Error amigable de la inspección manual (sin traceback).

NO entrena PatchCore, NO toca PATCHCORE_CKPT ni la inferencia productiva.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from ai_domain import (
    AIDomainError,
    CAPTURE_REJECT_FALLBACK,
    CAPTURE_UI_STATE_CAPTURA_COMPLETADA,
    CAPTURE_UI_STATE_CAPTURANDO,
    CAPTURE_UI_STATE_LABELS,
    CAPTURE_UI_STATE_LISTO_PARA_ENTRENAR,
    CAPTURE_UI_STATE_PREPARANDO,
    CAPTURE_UI_STATE_SIN_PREPARAR,
    CAPTURE_UI_STATES,
    ensure_garment_code_state,
    format_garment_model_code,
    humanize_capture_error,
    humanize_reject_reason,
    next_garment_model_code,
    resolve_capture_ui_state,
)


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


# ============================================================
# PRUEBAS PURAS (sin base de datos)
# ============================================================


class GarmentCodeFormatTests(unittest.TestCase):
    def test_pads_to_three_digits(self):
        self.assertEqual(format_garment_model_code(1), "BLUSA-001")
        self.assertEqual(format_garment_model_code(5), "BLUSA-005")
        self.assertEqual(format_garment_model_code(42), "BLUSA-042")

    def test_no_artificial_cap_after_width(self):
        self.assertEqual(format_garment_model_code(1000), "BLUSA-1000")

    def test_invalid_values_rejected(self):
        with self.assertRaises(AIDomainError):
            format_garment_model_code(0)

        with self.assertRaises(AIDomainError):
            format_garment_model_code("texto")

        with self.assertRaises(AIDomainError):
            format_garment_model_code(None)

    def test_custom_prefix_and_width(self):
        self.assertEqual(
            format_garment_model_code(7, prefix="FALDA", width=4),
            "FALDA-0007",
        )


class CaptureUiStateTests(unittest.TestCase):
    def test_every_state_has_a_spanish_label(self):
        for state in CAPTURE_UI_STATES:
            self.assertIn(state, CAPTURE_UI_STATE_LABELS)
            label = CAPTURE_UI_STATE_LABELS[state]
            self.assertTrue(label.strip())
            self.assertNotIn("_", label)

    def test_open_session_is_capturando(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status="ABIERTA",
                accepted_count=3,
                has_preparation_version=True,
            ),
            CAPTURE_UI_STATE_CAPTURANDO,
        )

    def test_completed_at_training_minimum_is_ready(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status="COMPLETADA",
                accepted_count=20,
                has_preparation_version=True,
            ),
            CAPTURE_UI_STATE_LISTO_PARA_ENTRENAR,
        )

    def test_completed_below_minimum_is_not_ready(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status="COMPLETADA",
                accepted_count=19,
                has_preparation_version=True,
            ),
            CAPTURE_UI_STATE_CAPTURA_COMPLETADA,
        )

    def test_completed_without_images(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status="COMPLETADA",
                accepted_count=0,
                has_preparation_version=True,
            ),
            CAPTURE_UI_STATE_CAPTURA_COMPLETADA,
        )

    def test_preparation_without_session(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status=None,
                accepted_count=0,
                has_preparation_version=True,
            ),
            CAPTURE_UI_STATE_PREPARANDO,
        )

    def test_cancelled_session_returns_to_preparation(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status="CANCELADA",
                accepted_count=0,
                has_preparation_version=True,
            ),
            CAPTURE_UI_STATE_PREPARANDO,
        )

    def test_without_preparation_version(self):
        self.assertEqual(
            resolve_capture_ui_state(
                session_status=None,
                accepted_count=0,
                has_preparation_version=False,
            ),
            CAPTURE_UI_STATE_SIN_PREPARAR,
        )


class RejectReasonHumanizationTests(unittest.TestCase):
    def test_known_reasons_are_human(self):
        self.assertEqual(humanize_reject_reason("BLUR"), "Movimiento excesivo.")
        self.assertEqual(humanize_reject_reason("blur"), "Movimiento excesivo.")
        self.assertEqual(
            humanize_reject_reason("DUPLICATE_SHA256"),
            "Imagen repetida.",
        )

    def test_unknown_reason_uses_fallback(self):
        self.assertEqual(
            humanize_reject_reason("ERROR_CUALQUIERA"),
            CAPTURE_REJECT_FALLBACK,
        )

    def test_empty_reason_is_none(self):
        self.assertIsNone(humanize_reject_reason(None))
        self.assertIsNone(humanize_reject_reason(""))


class CaptureErrorHumanizationTests(unittest.TestCase):
    def test_lock_errors_do_not_leak_internals(self):
        message = humanize_capture_error(
            "No se pudo adquirir el lock astrid_ai_capture_station."
        )
        self.assertNotIn("lock", message.lower())
        self.assertIn("estación", message.lower())

    def test_active_session_error_hides_ids(self):
        message = humanize_capture_error(
            "Ya hay una sesión de captura activa en la estación "
            "para garment_model_id=35."
        )
        self.assertNotIn("garment_model_id", message)
        self.assertNotIn("id=", message)
        self.assertIn("Finalícela o cancélela", message)

    def test_open_session_error_hides_ids(self):
        message = humanize_capture_error(
            "Ya existe una sesión de captura ABIERTA en la estación "
            "(id=17, garment_model_id=35)."
        )
        self.assertNotIn("id=", message)
        self.assertNotIn("garment_model_id", message)

    def test_missing_model_error(self):
        self.assertEqual(
            humanize_capture_error(
                "El modelo de prenda no existe."
            ),
            "El modelo de prenda no existe.",
        )

    def test_empty_message(self):
        self.assertEqual(
            humanize_capture_error(""),
            "No se pudo completar la operación de captura.",
        )


# ============================================================
# GENERACIÓN DE CÓDIGOS (base de datos)
# ============================================================


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de códigos de modelo omitidos.",
)
class GarmentCodeGenerationDbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mysql.connector

        cls.conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=os.environ.get(
                "MYSQL_DATABASE",
                "textile_quality_db",
            ),
        )
        cls.cur = cls.conn.cursor(dictionary=True)
        ensure_garment_code_state(cls.cur)
        cls.conn.commit()

        cls.created_codes = []

    @classmethod
    def tearDownClass(cls):
        for code in cls.created_codes:
            cls.cur.execute(
                "DELETE FROM garment_models WHERE code = %s",
                (code,),
            )
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def _next_code(self, cur=None, conn=None):
        cursor = cur or self.cur
        connection = conn or self.conn
        if connection.in_transaction:
            connection.commit()
        connection.start_transaction()
        code = next_garment_model_code(cursor)
        connection.commit()
        return code

    def test_code_follows_the_series(self):
        first = self._next_code()
        second = self._next_code()
        self.created_codes.extend([first, second])

        self.assertRegex(first, r"^BLUSA-\d{3,}$")
        self.assertRegex(second, r"^BLUSA-\d{3,}$")
        self.assertNotEqual(first, second)
        self.assertGreater(
            int(second.split("-")[1]),
            int(first.split("-")[1]),
        )

    def test_historical_codes_are_preserved(self):
        self.cur.execute("SELECT code FROM garment_models")
        before = {row["code"] for row in self.cur.fetchall()}

        generated = self._next_code()
        self.created_codes.append(generated)
        self.cur.execute(
            """
            INSERT INTO garment_models (code, name, status, active)
            VALUES (%s, 'Historico fase 2b', 'BORRADOR', 1)
            """,
            (generated,),
        )
        self.conn.commit()

        self.cur.execute("SELECT code FROM garment_models")
        after = {row["code"] for row in self.cur.fetchall()}

        self.assertTrue(before.issubset(after))
        self.assertIn(generated, after)

    def test_codes_from_other_formats_are_not_reused(self):
        historical = "HISTORICO-9"
        self.cur.execute(
            """
            INSERT INTO garment_models (code, name, status, active)
            VALUES (%s, 'Codigo historico', 'APROBADO', 1)
            """,
            (historical,),
        )
        self.conn.commit()

        try:
            generated = self._next_code()
            self.created_codes.append(generated)

            self.assertNotEqual(generated, historical)
            self.assertTrue(generated.startswith("BLUSA-"))
            self.assertRegex(generated, r"^BLUSA-\d{3,}$")
        finally:
            self.cur.execute(
                "DELETE FROM garment_models WHERE code = %s",
                (historical,),
            )
            self.conn.commit()

    def test_deleted_code_is_not_reused(self):
        first = self._next_code()
        self.cur.execute(
            """
            INSERT INTO garment_models (code, name, status, active)
            VALUES (%s, 'Temporal fase 2b', 'BORRADOR', 1)
            """,
            (first,),
        )
        self.conn.commit()

        self.cur.execute(
            "DELETE FROM garment_models WHERE code = %s",
            (first,),
        )
        self.conn.commit()

        second = self._next_code()
        self.created_codes.append(second)

        self.assertNotEqual(first, second)
        self.assertGreater(
            int(second.split("-")[1]),
            int(first.split("-")[1]),
        )

    def test_concurrent_generation_is_unique(self):
        results = []
        errors = []
        lock = threading.Lock()

        def worker():
            import mysql.connector

            conn = None
            cur = None
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
                )
                cur = conn.cursor(dictionary=True)
                codes = []
                for _ in range(3):
                    conn.start_transaction()
                    codes.append(next_garment_model_code(cur))
                    conn.commit()
                with lock:
                    results.extend(codes)
            except Exception as error:  # noqa: BLE001
                with lock:
                    errors.append(error)
            finally:
                # Nunca dejar transacciones abiertas: bloquean los
                # tests siguientes.
                if conn is not None:
                    try:
                        if conn.in_transaction:
                            conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass
                if cur is not None:
                    try:
                        cur.close()
                    except Exception:  # noqa: BLE001
                        pass
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        threads = [
            threading.Thread(target=worker) for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertFalse(errors, f"Errores concurrentes: {errors}")
        self.assertEqual(len(results), 12)
        self.assertEqual(len(set(results)), 12)
        for code in results:
            self.assertRegex(code, r"^BLUSA-\d{3,}$")


# ============================================================
# INTERFAZ / API DE PREPARACIÓN (base de datos + Flask)
# ============================================================


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de interfaz de captura omitidos.",
)
class AiCaptureUiApiTests(unittest.TestCase):
    ADMIN_USERNAME = "ph2b_admin"
    QM_USERNAME = "ph2b_qm"
    MODEL_A = "TEST-PH2B-A"
    MODEL_B = "TEST-PH2B-B"
    MODEL_DRAFT = "TEST-PH2B-DRAFT"

    @classmethod
    def setUpClass(cls):
        import importlib

        import mysql.connector

        cls.A = importlib.import_module("app")

        # Los tests nunca deben arrancar hilos de cámara ni de captura.
        cls._orig_camera_worker = cls.A.ensure_camera_capture_worker
        cls._orig_ai_worker = cls.A._ensure_ai_capture_worker
        cls.A.ensure_camera_capture_worker = lambda *a, **k: False
        cls.A._ensure_ai_capture_worker = lambda *a, **k: False

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

        cls._ensure_user(cls.ADMIN_USERNAME, "ADMIN")
        cls._ensure_user(cls.QM_USERNAME, "QUALITY_MANAGER")

        cls.model_a = cls._ensure_model(cls.MODEL_A, "APROBADO")
        cls.model_b = cls._ensure_model(cls.MODEL_B, "APROBADO")
        cls.model_draft = cls._ensure_model(cls.MODEL_DRAFT, "BORRADOR")

        cls.client = cls.A.app.test_client()

    @classmethod
    def _ensure_user(cls, username, role):
        cls.cur.execute(
            "SELECT id FROM users WHERE username = %s",
            (username,),
        )
        row = cls.cur.fetchone()
        if row:
            cls.cur.execute(
                """
                UPDATE users
                SET role = %s, active = 1
                WHERE id = %s
                """,
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
                (code, f"Modelo fase 2b {code}", status),
            )
            model_id = int(cls.cur.lastrowid)
        cls.conn.commit()
        return model_id

    @classmethod
    def tearDownClass(cls):
        from ai_domain import ensure_ai_schema

        cls.A.ensure_camera_capture_worker = cls._orig_camera_worker
        cls.A._ensure_ai_capture_worker = cls._orig_ai_worker
        cls.A.set_ai_capture_mode(False)
        cls.A._stop_ai_capture_worker()

        trigger_names = (
            "trg_ai_dataset_images_before_insert",
            "trg_ai_dataset_images_before_update",
            "trg_ai_dataset_images_before_delete",
            "trg_ai_datasets_before_update",
            "trg_ai_training_images_before_update",
            "trg_ai_training_images_before_delete",
        )
        for name in trigger_names:
            cls.cur.execute(f"DROP TRIGGER IF EXISTS {name}")
        cls.conn.commit()

        cls.cur.execute(
            "DELETE FROM ai_events WHERE actor_id IN "
            "(SELECT id FROM users WHERE username IN (%s, %s))",
            (cls.ADMIN_USERNAME, cls.QM_USERNAME),
        )

        for code in (
            cls.MODEL_A,
            cls.MODEL_B,
            cls.MODEL_DRAFT,
        ):
            cls.cur.execute(
                "SELECT id FROM garment_models WHERE code = %s",
                (code,),
            )
            row = cls.cur.fetchone()
            if not row:
                continue
            mid = int(row["id"])

            cls.cur.execute(
                """
                DELETE di FROM ai_dataset_images di
                JOIN ai_datasets d ON d.id = di.dataset_id
                WHERE d.garment_model_id = %s
                """,
                (mid,),
            )
            cls.cur.execute(
                "DELETE FROM ai_datasets WHERE garment_model_id = %s",
                (mid,),
            )
            cls.cur.execute(
                "DELETE FROM ai_training_images "
                "WHERE garment_model_id = %s",
                (mid,),
            )
            cls.cur.execute(
                "DELETE FROM ai_capture_sessions "
                "WHERE garment_model_id = %s",
                (mid,),
            )
            cls.cur.execute(
                "DELETE FROM garment_ai_models "
                "WHERE garment_model_id = %s",
                (mid,),
            )
            cls.cur.execute(
                "DELETE FROM garment_models WHERE id = %s",
                (mid,),
            )

        cls.cur.execute(
            "DELETE FROM users WHERE username IN (%s, %s)",
            (cls.ADMIN_USERNAME, cls.QM_USERNAME),
        )
        cls.conn.commit()

        ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    # --------------------------------------------------------
    # utilidades
    # --------------------------------------------------------

    def _login(self, username):
        with self.client.session_transaction() as sess:
            sess.clear()
            sess["user_id"] = self._user_id(username)

    @classmethod
    def _user_id(cls, username):
        cls.cur.execute(
            "SELECT id FROM users WHERE username = %s",
            (username,),
        )
        return int(cls.cur.fetchone()["id"])

    def _close_all_open(self):
        self.cur.execute(
            """
            UPDATE ai_capture_sessions
            SET status = 'CANCELADA', finished_at = NOW()
            WHERE status = 'ABIERTA'
            """
        )
        self.conn.commit()
        self.A.set_ai_capture_mode(False)

    def setUp(self):
        self._close_all_open()
        self._login(self.ADMIN_USERNAME)

    def tearDown(self):
        self._close_all_open()

    def _json(self, response):
        return json.loads(response.get_data(as_text=True))

    def _prepare_version(self, model_id):
        return self.client.post(
            f"/modelos-prenda/{model_id}/ia/preparar",
            follow_redirects=False,
        )

    def _start(self, model_id):
        return self.client.post(
            "/api/ai/capture/start",
            json={"garment_model_id": model_id},
        )

    def _status(self, **params):
        query = "&".join(
            f"{key}={value}" for key, value in params.items()
        )
        return self.client.get(f"/api/ai/capture/status?{query}")

    def _insert_image(self, session_id, model_id, *, status, reason=None):
        digest = os.urandom(32).hex()
        self.cur.execute(
            """
            INSERT INTO ai_training_images (
                capture_session_id, garment_model_id, image_path,
                sha256, frame_sequence, coverage, quality_score,
                status, reject_reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                model_id,
                f"captures/phase2b/{digest}.jpg",
                digest,
                1,
                0.8000,
                90.000,
                status,
                reason,
            ),
        )
        self.conn.commit()

    # --------------------------------------------------------
    # estados y página
    # --------------------------------------------------------

    def test_detail_page_shows_capture_controls(self):
        response = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Preparación de IA", html)
        self.assertIn("startGuidedPreparation", html)

        # La captura se abre en una única estación guiada; el panel
        # inline heredado de Fase 2B ya no se renderiza.
        self._prepare_version(self.model_a)
        response = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        )
        html = response.get_data(as_text=True)

        self.assertIn("startGuidedPreparation", html)
        self.assertIn("/captura-ia", html)
        self.assertNotIn("aiCapturePanel", html)
        self.assertNotIn("ASTRID_AI_CAPTURE", html)
        self.assertNotIn("INICIAR CAPTURA", html)
        self.assertNotIn("window.alert(", html)
        self.assertNotIn("window.confirm(", html)

    def test_detail_page_reports_initial_state(self):
        self._prepare_version(self.model_a)

        html = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        ).get_data(as_text=True)

        config = self._json(self._status(garment_model_id=self.model_a))
        self.assertEqual(config["ui_state"], "PREPARANDO")
        self.assertTrue(config["has_preparation_version"])
        self.assertEqual(config["target_count"], 40)

    def _page_config(self, html):
        match = re.search(
            r"window\.ASTRID_AI_CAPTURE = (\{.*?\});",
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "No se encontró la configuración de captura IA en la página.",
        )
        return json.loads(match.group(1))

    def test_start_then_status_then_stop(self):
        self._prepare_version(self.model_a)

        started = self._json(self._start(self.model_a))
        self.assertTrue(started["ok"])
        self.assertEqual(started["session"]["ui_state"], "CAPTURANDO")
        self.assertEqual(
            started["session"]["ui_state_label"],
            "CAPTURANDO",
        )
        self.assertTrue(started["session"]["active"])

        status = self._json(
            self._status(garment_model_id=self.model_a)
        )
        self.assertTrue(status["ok"])
        self.assertEqual(status["ui_state"], "CAPTURANDO")
        self.assertEqual(status["accepted_count"], 0)

        html = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        ).get_data(as_text=True)
        self.assertNotIn("ASTRID_AI_CAPTURE", html)
        self.assertNotIn("aiCapturePanel", html)
        self.assertEqual(
            self._json(self._status(garment_model_id=self.model_a))["ui_state"],
            "CAPTURANDO",
        )

        stopped_response = self.client.post("/api/ai/capture/stop", json={})
        self.assertEqual(stopped_response.status_code, 409)
        self.assertFalse(self._json(stopped_response)["ok"])
        self.assertTrue(self._json(self._status(garment_model_id=self.model_a))["active"])

    def test_stop_with_accepted_images_is_ready(self):
        self._prepare_version(self.model_a)

        started = self._json(self._start(self.model_a))
        session_id = int(started["session"]["session_id"])

        for _ in range(20):
            self._insert_image(
                session_id,
                self.model_a,
                status="ACEPTADA",
            )
        self._insert_image(
            session_id,
            self.model_a,
            status="RECHAZADA",
            reason="BLUR",
        )

        status = self._json(
            self._status(garment_model_id=self.model_a)
        )
        self.assertEqual(status["ui_state"], "CAPTURANDO")
        self.assertEqual(status["accepted_count"], 20)
        self.assertEqual(status["rejected_count"], 1)
        self.assertIsNotNone(status["last_capture_human"])
        self.assertFalse(status["last_capture_human"]["accepted"])

        stopped = self._json(
            self.client.post("/api/ai/capture/stop", json={})
        )
        self.assertEqual(
            stopped["session"]["ui_state"],
            "LISTO_PARA_ENTRENAR",
        )
        self.assertEqual(
            stopped["session"]["ui_state_label"],
            "LISTO PARA ENTRENAR",
        )
        self.assertEqual(stopped["session"]["accepted_count"], 20)
        self.assertEqual(stopped["session"]["rejected_count"], 1)

    def test_rejected_last_capture_is_humanized(self):
        self._prepare_version(self.model_a)

        started = self._json(self._start(self.model_a))
        session_id = int(started["session"]["session_id"])
        self._insert_image(
            session_id,
            self.model_a,
            status="RECHAZADA",
            reason="BLUR",
        )

        status = self._json(
            self._status(garment_model_id=self.model_a)
        )
        last = status["last_capture_human"]
        self.assertFalse(last["accepted"])
        self.assertEqual(last["label"], "Descartada")
        self.assertEqual(last["reason"], "Movimiento excesivo.")

    def test_cancel_returns_to_preparation(self):
        self._prepare_version(self.model_a)
        self._start(self.model_a)

        cancelled = self._json(
            self.client.post("/api/ai/capture/cancel", json={})
        )
        self.assertTrue(cancelled["ok"])
        self.assertEqual(
            cancelled["session"]["ui_state"],
            "PREPARANDO",
        )

        status = self._json(
            self._status(garment_model_id=self.model_a)
        )
        self.assertEqual(status["ui_state"], "PREPARANDO")
        self.assertFalse(status["active"])

    def test_stop_is_idempotent_without_session(self):
        stopped = self._json(
            self.client.post("/api/ai/capture/stop", json={})
        )
        self.assertTrue(stopped["ok"])
        self.assertIsNone(stopped["session"])

    def test_active_session_blocks_another_model(self):
        self._prepare_version(self.model_a)
        self._prepare_version(self.model_b)

        self._json(self._start(self.model_a))

        blocked = self.client.post(
            "/api/ai/capture/start",
            json={"garment_model_id": self.model_b},
        )
        payload = self._json(blocked)

        self.assertEqual(blocked.status_code, 409)
        self.assertFalse(payload["ok"])
        self.assertNotIn("garment_model_id", payload["error"])
        self.assertNotIn("id=", payload["error"])
        self.assertIn("Finalícela o cancélela", payload["error"])

        other = self._json(
            self._status(garment_model_id=self.model_b)
        )
        self.assertTrue(other["station_busy"])
        self.assertEqual(
            other["station_busy_model_code"],
            self.MODEL_A,
        )

    def test_unknown_model_is_rejected(self):
        response = self.client.post(
            "/api/ai/capture/start",
            json={"garment_model_id": 99999999},
        )
        payload = self._json(response)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            payload["error"],
            "El modelo de prenda no existe.",
        )

    def test_not_approved_model_is_rejected(self):
        response = self._start(self.model_draft)
        payload = self._json(response)

        self.assertEqual(response.status_code, 409)
        self.assertIn("aprobado", payload["error"])

    def test_invalid_payload_is_rejected(self):
        response = self.client.post(
            "/api/ai/capture/start",
            json={"garment_model_id": "no-numerico"},
        )
        self.assertEqual(response.status_code, 400)

    def test_quality_manager_has_no_access(self):
        self._prepare_version(self.model_a)
        self._login(self.QM_USERNAME)

        response = self._start(self.model_a)
        self.assertEqual(response.status_code, 403)

        response = self._status(garment_model_id=self.model_a)
        self.assertEqual(response.status_code, 403)

        response = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(
            "aiCapturePanel",
            response.get_data(as_text=True),
        )

    def test_status_without_model_returns_idle_state(self):
        status = self._json(self.client.get("/api/ai/capture/status"))
        self.assertTrue(status["ok"])
        self.assertEqual(status["ui_state"], "SIN_PREPARAR")

    def test_status_for_unknown_garment_still_works(self):
        status = self._json(
            self._status(garment_model_id=99999999)
        )
        self.assertTrue(status["ok"])
        self.assertEqual(status["ui_state"], "SIN_PREPARAR")

    def test_status_rejects_invalid_garment_id(self):
        response = self._status(garment_model_id="texto")
        self.assertEqual(response.status_code, 400)


# ============================================================
# INSPECCIÓN MANUAL — MENSAJES AMIGABLES
# ============================================================


@unittest.skipUnless(
    DB_AVAILABLE,
    "MySQL no disponible; tests de inspección manual omitidos.",
)
class ManualInspectionErrorTests(unittest.TestCase):
    USERNAME = "ph2b_inspect"

    @classmethod
    def setUpClass(cls):
        import importlib

        import mysql.connector

        cls.A = importlib.import_module("app")
        cls._orig_camera_worker = cls.A.ensure_camera_capture_worker
        cls.A.ensure_camera_capture_worker = lambda *a, **k: False

        cls.conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=os.environ.get(
                "MYSQL_DATABASE",
                "textile_quality_db",
            ),
        )
        cls.cur = cls.conn.cursor(dictionary=True)
        cls.cur.execute(
            "SELECT id FROM users WHERE username = %s",
            (cls.USERNAME,),
        )
        row = cls.cur.fetchone()
        if row:
            cls.user_id = int(row["id"])
            cls.cur.execute(
                "UPDATE users SET role = 'ADMIN', active = 1 "
                "WHERE id = %s",
                (cls.user_id,),
            )
        else:
            cls.cur.execute(
                """
                INSERT INTO users (username, password_hash, role, active)
                VALUES (%s, 'x', 'ADMIN', 1)
                """,
                (cls.USERNAME,),
            )
            cls.user_id = int(cls.cur.lastrowid)
        cls.conn.commit()

        cls.client = cls.A.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.A.ensure_camera_capture_worker = cls._orig_camera_worker
        cls.cur.execute(
            "DELETE FROM inspections WHERE notes = %s",
            (
                "Registro generado automáticamente por el sistema "
                "de inspección.",
            ),
        )
        cls.cur.execute(
            "DELETE FROM users WHERE username = %s",
            (cls.USERNAME,),
        )
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def setUp(self):
        with self.client.session_transaction() as sess:
            sess.clear()
            sess["user_id"] = self.user_id

    def _post(self):
        return self.client.post("/inspeccion", data={})

    def test_garment_not_detected_error_is_recognized(self):
        self.assertTrue(
            self.A.is_garment_not_detected_error(
                "No se detectó una blusa completa en el área."
            )
        )
        self.assertTrue(
            self.A.is_garment_not_detected_error(
                RuntimeError("No se encontró la prenda completa.")
            )
        )
        self.assertFalse(
            self.A.is_garment_not_detected_error(
                ValueError("boom")
            )
        )

    def test_garment_not_detected_shows_friendly_message(self):
        with patch.object(
            self.A,
            "save_image",
            return_value=(Path("tmp") / "x.jpg", "captures/x.jpg"),
        ), patch.object(
            self.A,
            "detect_defect",
            side_effect=RuntimeError(
                "No se detectó una blusa completa en el área."
            ),
        ):
            response = self._post()

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "No se detectó una prenda completa en el área de "
            "inspección.",
            html,
        )
        self.assertNotIn("Traceback", html)
        self.assertNotIn("RuntimeError", html)

    def test_unexpected_error_shows_generic_message(self):
        with patch.object(
            self.A,
            "save_image",
            return_value=(Path("tmp") / "x.jpg", "captures/x.jpg"),
        ), patch.object(
            self.A,
            "detect_defect",
            side_effect=ValueError("detalle_interno_123"),
        ):
            response = self._post()

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "No se pudo completar la inspección.",
            html,
        )
        self.assertNotIn("detalle_interno_123", html)
        self.assertNotIn("Traceback", html)


if __name__ == "__main__":
    unittest.main()
