"""Tests FASE 2C — estación guiada de captura.

Cubre:
- Umbrales y configuración de posicionamiento (env con defaults).
- Clasificación de posición: prenda ausente, entrando, movida,
  desenfocada, desbordando el área y posición correcta.
- Máquina de estados: estabilidad, 1 paso = 1 captura, espera de
  salida y selección del mejor frame.
- Pantalla guiada: página, video con overlay, arranque del motor,
  progreso dinámico y cierre (finalizar/cancelar) sin crear
  inspecciones ni tocar la estación de calidad.

NO entrena PatchCore, NO cambia la producción, NO toca PATCHCORE_CKPT.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from ai_capture import (
    AI_GUIDED_DEFAULTS,
    GUIDED_MESSAGES,
    GUIDED_STATE_LABELS,
    GUIDED_STATES,
    GUIDED_VALID_STATES,
    GuidedTracker,
    classify_position,
    get_ai_guided_config,
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

ROI = (100, 50, 900, 750)
ROI_CENTER = (500, 400)
FAKE_FRAME = "FAKE_FRAME"


def box(cx, cy, width=360, height=360):
    """Bounding box centrado en (cx, cy)."""
    return (
        int(cx - width / 2),
        int(cy - height / 2),
        int(cx + width / 2),
        int(cy + height / 2),
    )


def classify(bbox, coverage, sharpness, config=None):
    return classify_position(
        roi=ROI,
        bbox=bbox,
        coverage=coverage,
        sharpness=sharpness,
        config=config or get_ai_guided_config(),
    )


# ------------------------------------------------------------
# Fotogramas reales de la estación (para validar el segmentador)
# ------------------------------------------------------------

BACKGROUND_FRAME_PATH = ROOT / "calibration" / "background_full.jpg"
EMPTY_FRAME_PATHS = tuple(
    sorted((ROOT / "calibration").glob("*_full.jpg"))
) + (ROOT / "calibration" / "rtsp_confirmacion.jpg",)

# Elipse "prenda" dibujada sobre el fondo real (coordenadas de frame).
SYNTH_ELLIPSE_CENTER = (1330, 800)
SYNTH_ELLIPSE_AXES = (450, 380)


def synthetic_garment_frame(color=(60, 150, 60)):
    """
    Frame real de la estación con una elipse sólida dentro del ROI.

    Devuelve None si no hay fotogramas de calibración en el repositorio.
    """
    if not BACKGROUND_FRAME_PATH.exists():
        return None

    import cv2

    frame = cv2.imread(str(BACKGROUND_FRAME_PATH))
    if frame is None:
        return None

    cv2.ellipse(
        frame,
        SYNTH_ELLIPSE_CENTER,
        SYNTH_ELLIPSE_AXES,
        0,
        0,
        360,
        color,
        -1,
    )
    return frame


# ============================================================
# PRUEBAS PURAS (sin base de datos)
# ============================================================


class GuidedConfigTests(unittest.TestCase):
    def test_defaults_are_documented(self):
        cfg = get_ai_guided_config()

        self.assertEqual(cfg["present_coverage"], 0.28)
        self.assertEqual(cfg["exit_coverage"], 0.12)
        self.assertEqual(cfg["max_coverage"], 0.98)
        self.assertEqual(cfg["center_tol_x"], 0.12)
        self.assertEqual(cfg["edge_margin"], 0.05)
        self.assertEqual(cfg["stability_frames"], 4)
        self.assertEqual(cfg["enter_frames"], 3)
        self.assertEqual(cfg["exit_frames"], 3)
        self.assertEqual(cfg["min_coverage"], 0.48)
        self.assertEqual(cfg["min_sharpness"], 40.0)

    def test_defaults_have_no_underscored_values(self):
        for key, value in AI_GUIDED_DEFAULTS.items():
            self.assertIsInstance(
                value,
                (int, float),
                f"{key} debe ser numérico.",
            )

    def test_auto_capture_is_off_by_default(self):
        cfg = get_ai_guided_config()

        self.assertIn("auto_capture", cfg)
        self.assertFalse(cfg["auto_capture"])
        self.assertFalse(bool(AI_GUIDED_DEFAULTS["auto_capture"]))

    def test_capture_config_exposes_minimum_for_training(self):
        from ai_domain import capture_progress_gate, get_ai_capture_config

        cfg = get_ai_capture_config()

        self.assertEqual(cfg["min_images"], 20)

        vacio = capture_progress_gate(0, min_images=cfg["min_images"])
        self.assertFalse(vacio["can_finalize"])
        self.assertFalse(vacio["can_train"])
        self.assertEqual(vacio["missing_to_train"], 20)

        una = capture_progress_gate(1, min_images=cfg["min_images"])
        self.assertFalse(una["can_finalize"])
        self.assertFalse(una["can_train"])
        self.assertEqual(una["missing_to_train"], 19)

        minimo = capture_progress_gate(20, min_images=cfg["min_images"])
        self.assertTrue(minimo["can_finalize"])
        self.assertTrue(minimo["can_train"])
        self.assertEqual(minimo["missing_to_train"], 0)

    def test_capture_minimum_can_be_configured_by_env(self):
        from ai_domain import get_ai_capture_config

        os.environ["AI_CAPTURE_MIN_IMAGES"] = "5"
        try:
            cfg = get_ai_capture_config()
            self.assertEqual(cfg["min_images"], 5)
        finally:
            os.environ.pop("AI_CAPTURE_MIN_IMAGES", None)

    def test_env_override_is_honoured(self):
        with self.subTest("estabilidad"):
            os.environ["AI_GUIDED_STABILITY_FRAMES"] = "7"
            try:
                self.assertEqual(
                    get_ai_guided_config()["stability_frames"],
                    7,
                )
            finally:
                os.environ.pop("AI_GUIDED_STABILITY_FRAMES", None)

        with self.subTest("tolerancia"):
            os.environ["AI_GUIDED_CENTER_TOL_X"] = "0.2"
            try:
                self.assertEqual(
                    get_ai_guided_config()["center_tol_x"],
                    0.2,
                )
            finally:
                os.environ.pop("AI_GUIDED_CENTER_TOL_X", None)

    def test_invalid_env_is_rejected_with_message(self):
        from ai_domain import AIDomainError

        os.environ["AI_GUIDED_STABILITY_FRAMES"] = "0"
        try:
            with self.assertRaises(AIDomainError) as ctx:
                get_ai_guided_config()
            self.assertIn("stability_frames", str(ctx.exception))
        finally:
            os.environ.pop("AI_GUIDED_STABILITY_FRAMES", None)

    def test_present_coverage_must_be_below_min_coverage(self):
        from ai_domain import AIDomainError

        os.environ["AI_GUIDED_PRESENT_COVERAGE"] = "0.9"
        try:
            with self.assertRaises(AIDomainError):
                get_ai_guided_config()
        finally:
            os.environ.pop("AI_GUIDED_PRESENT_COVERAGE", None)

    def test_exit_coverage_must_be_below_present(self):
        from ai_domain import AIDomainError

        os.environ["AI_GUIDED_EXIT_COVERAGE"] = "0.5"
        try:
            with self.assertRaises(AIDomainError):
                get_ai_guided_config()
        finally:
            os.environ.pop("AI_GUIDED_EXIT_COVERAGE", None)


class GuidedPositionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = get_ai_guided_config()

    def test_no_garment_returns_place_instruction(self):
        decision = classify(None, 0.0, 0.0, self.cfg)

        self.assertFalse(decision["present"])
        self.assertFalse(decision["aligned"])
        self.assertEqual(decision["reason"], "NO_GARMENT")
        self.assertEqual(
            decision["message"],
            "Coloque la prenda dentro del área",
        )

    def test_coverage_below_present_threshold_is_no_garment(self):
        decision = classify(
            box(*ROI_CENTER),
            self.cfg["present_coverage"] - 0.01,
            120.0,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "NO_GARMENT")

    def test_coverage_between_present_and_min_is_entering(self):
        decision = classify(
            box(*ROI_CENTER),
            0.35,
            120.0,
            self.cfg,
        )
        self.assertTrue(decision["present"])
        self.assertEqual(decision["reason"], "LOW_COVERAGE")
        self.assertEqual(
            decision["message"],
            "Espere, la prenda está entrando",
        )

    def test_coverage_above_max_is_high(self):
        decision = classify(
            box(*ROI_CENTER),
            self.cfg["max_coverage"] + 0.02,
            120.0,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "HIGH_COVERAGE")
        self.assertEqual(
            decision["message"],
            "Coloque la prenda dentro del área",
        )

    def test_blurry_frame_is_not_captured(self):
        decision = classify(
            box(*ROI_CENTER),
            0.55,
            self.cfg["min_sharpness"] - 1,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "BLUR")
        self.assertFalse(decision["aligned"])
        self.assertEqual(
            decision["message"],
            "Imagen posiblemente borrosa. Mantenga la prenda quieta",
        )

    def test_missing_sharpness_is_blur(self):
        decision = classify(box(*ROI_CENTER), 0.55, None, self.cfg)
        self.assertEqual(decision["reason"], "BLUR")

    def test_garment_too_far_left_asks_for_right(self):
        decision = classify(box(200, 400), 0.55, 120.0, self.cfg)

        self.assertEqual(decision["reason"], "MOVE_RIGHT")
        self.assertEqual(
            decision["message"],
            "Mueva un poco a la derecha",
        )
        self.assertLess(decision["dx"], 0)

    def test_garment_too_far_right_asks_for_left(self):
        decision = classify(box(800, 400), 0.55, 120.0, self.cfg)

        self.assertEqual(decision["reason"], "MOVE_LEFT")
        self.assertEqual(
            decision["message"],
            "Mueva un poco a la izquierda",
        )
        self.assertGreater(decision["dx"], 0)

    def test_garment_off_center_but_inside_margins(self):
        decision = classify(
            box(380, 400, 200, 200),
            0.55,
            120.0,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "MOVE_RIGHT")

        decision = classify(
            box(620, 400, 200, 200),
            0.55,
            120.0,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "MOVE_LEFT")

    def test_garment_too_high_asks_to_move_down(self):
        decision = classify(
            box(500, 250, 260, 260),
            0.55,
            120.0,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "MOVE_DOWN")
        self.assertEqual(
            decision["message"],
            "Mueva un poco hacia abajo",
        )

    def test_garment_too_low_asks_to_move_up(self):
        decision = classify(
            box(500, 560, 260, 260),
            0.55,
            120.0,
            self.cfg,
        )
        self.assertEqual(decision["reason"], "MOVE_UP")

    def test_garment_touching_the_edge_is_rejected(self):
        decision = classify(box(870, 400, 300, 300), 0.55, 120.0, self.cfg)

        self.assertFalse(decision["aligned"])
        self.assertIn(
            decision["reason"],
            ("MOVE_LEFT", "MOVE_RIGHT"),
        )
        self.assertEqual(
            decision["message"],
            "Mueva un poco a la izquierda",
        )

    def test_centered_garment_is_aligned(self):
        decision = classify(box(*ROI_CENTER), 0.55, 120.0, self.cfg)

        self.assertTrue(decision["present"])
        self.assertTrue(decision["aligned"])
        self.assertEqual(decision["reason"], "ALIGNED")
        self.assertEqual(decision["message"], "Posición correcta")
        self.assertAlmostEqual(decision["dx"], 0.0, places=3)
        self.assertAlmostEqual(decision["dy"], 0.0, places=3)

    def test_messages_are_human_and_never_leak_internals(self):
        reasons = (
            "NO_GARMENT",
            "LOW_COVERAGE",
            "HIGH_COVERAGE",
            "BLUR",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "MOVE_UP",
            "MOVE_DOWN",
            "ALIGNED",
        )

        for reason in reasons:
            with self.subTest(reason=reason):
                message = GUIDED_MESSAGES[reason]
                self.assertTrue(message.strip())
                self.assertNotIn("{", message)
                self.assertNotIn("coverage", message)
                self.assertNotIn("Error", message)
                self.assertNotIn("None", message)

    def test_every_state_has_a_readable_label(self):
        for state in GUIDED_STATES:
            with self.subTest(state=state):
                self.assertIn(state, GUIDED_STATE_LABELS)
                label = GUIDED_STATE_LABELS[state]
                self.assertTrue(label.strip())
                self.assertNotIn("_", label)

    def test_valid_states_are_grouped(self):
        self.assertEqual(
            GUIDED_VALID_STATES,
            {
                "POSICION_VALIDA",
                "CAPTURANDO",
                "CAPTURADA",
            },
        )


class GuidedTrackerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = get_ai_guided_config()
        self.tracker = GuidedTracker(dict(self.cfg))
        self.captures = 0

    def _feed(self, coverage=0.55, sharpness=120.0, bbox=None, seq=0,
              frame=FAKE_FRAME):
        return self.tracker.step(
            roi=ROI,
            bbox=box(*ROI_CENTER) if bbox is None and coverage > 0.4 else bbox,
            coverage=coverage,
            sharpness=sharpness,
            sequence=seq,
            frame=frame,
        )

    def _simulate(self, frames, **kwargs):
        """Alimenta N frames repitiendo el rol del worker."""
        for index in range(frames):
            guide = self._feed(seq=index, **kwargs)
            if guide["wants_capture"]:
                self.tracker.begin_capture()
                self.tracker.confirm_capture({"accepted": True})
                self.captures += 1
            yield guide

    def test_starts_without_garment(self):
        guide = self._feed(coverage=0.0, sharpness=0.0, bbox=None)

        self.assertEqual(guide["state"], "SIN_PRENDA")
        self.assertEqual(
            guide["message"],
            "Coloque la prenda dentro del área",
        )
        self.assertFalse(guide["wants_capture"])

    def test_enters_before_becoming_valid(self):
        guides = list(self._simulate(2))

        self.assertEqual(guides[0]["state"], "ENTRANDO")
        self.assertEqual(guides[1]["state"], "ENTRANDO")

    def test_becomes_valid_after_enter_frames(self):
        guides = list(self._simulate(4))

        self.assertEqual(guides[-1]["state"], "POSICION_VALIDA")
        self.assertEqual(guides[-1]["message"], "Posición correcta")
        self.assertTrue(guides[-1]["aligned"])

    def test_stability_frames_required_before_capture(self):
        cfg = dict(self.cfg)
        cfg["stability_frames"] = 5
        tracker = GuidedTracker(cfg)
        wanted = []

        for index in range(5):
            guide = tracker.step(
                roi=ROI,
                bbox=box(*ROI_CENTER),
                coverage=0.55,
                sharpness=120.0,
                sequence=index,
                frame=FAKE_FRAME,
            )
            wanted.append(guide["wants_capture"])

        self.assertEqual(wanted, [False, False, False, False, True])

    def test_one_capture_per_pass_even_if_garment_stays(self):
        list(self._simulate(12))

        self.assertEqual(self.captures, 1)

    def test_second_garment_produces_second_capture(self):
        # primer paso
        list(self._simulate(4))
        # la prenda sale
        for index in range(3):
            self._feed(coverage=0.0, sharpness=0.0, bbox=None, seq=50 + index)
        self.assertEqual(self.tracker.state, "SIN_PRENDA")

        # segundo paso
        for index in range(6):
            guide = self._feed(seq=100 + index)
            if guide["wants_capture"]:
                self.tracker.begin_capture()
                self.tracker.confirm_capture({"accepted": True})
                self.captures += 1

        self.assertEqual(self.captures, 2)

    def test_waits_for_full_exit_before_resetting(self):
        cfg = dict(self.cfg)
        cfg["exit_frames"] = 3
        tracker = GuidedTracker(cfg)

        tracker.step(
            roi=ROI,
            bbox=box(*ROI_CENTER),
            coverage=0.55,
            sharpness=120.0,
            sequence=0,
            frame=FAKE_FRAME,
        )
        tracker.begin_capture()
        tracker.confirm_capture({"accepted": True})

        states = []
        for index in range(3):
            guide = tracker.step(
                roi=ROI,
                bbox=None,
                coverage=0.0,
                sharpness=0.0,
                sequence=index + 1,
                frame=FAKE_FRAME,
            )
            states.append(guide["state"])

        self.assertEqual(states, ["CAPTURADA", "CAPTURADA", "SIN_PRENDA"])
        self.assertFalse(tracker.capture_reached)

    def test_no_capture_while_garment_is_moving(self):
        cfg = dict(self.cfg)
        cfg["stability_frames"] = 3
        tracker = GuidedTracker(cfg)
        wanted = False

        for index in range(8):
            guide = tracker.step(
                roi=ROI,
                bbox=box(200, 400),
                coverage=0.55,
                sharpness=120.0,
                sequence=index,
                frame=FAKE_FRAME,
            )
            wanted = wanted or guide["wants_capture"]
            self.assertEqual(guide["state"], "AJUSTAR")

        self.assertFalse(wanted)

    def test_best_frame_is_the_sharpest_centered_one(self):
        cfg = dict(self.cfg)
        cfg["stability_frames"] = 6
        tracker = GuidedTracker(cfg)

        tracker.step(
            roi=ROI,
            bbox=box(470, 400),
            coverage=0.55,
            sharpness=60.0,
            sequence=1,
            frame="SUAVE",
        )
        tracker.step(
            roi=ROI,
            bbox=box(500, 400),
            coverage=0.60,
            sharpness=200.0,
            sequence=2,
            frame="NITIDA",
        )
        tracker.step(
            roi=ROI,
            bbox=box(530, 400),
            coverage=0.55,
            sharpness=70.0,
            sequence=3,
            frame="SUAVE_2",
        )

        candidate = tracker.best_candidate(9)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.best_frame, "NITIDA")
        self.assertEqual(candidate.best_sequence, 2)
        self.assertEqual(candidate.garment_token, 9)

    def test_snapshot_exposes_human_state(self):
        guide = self._feed(coverage=0.55)

        self.assertIn(guide["state"], GUIDED_STATES)
        self.assertNotIn("_", guide["state_label"])
        self.assertTrue(guide["message"])
        self.assertIn("stability_frames", guide)
        self.assertEqual(guide["stability_frames"], 4)

    def test_capture_marks_step_as_done(self):
        tracker = GuidedTracker(dict(self.cfg))
        tracker.step(
            roi=ROI,
            bbox=box(*ROI_CENTER),
            coverage=0.55,
            sharpness=120.0,
            sequence=0,
            frame=FAKE_FRAME,
        )
        tracker.begin_capture()

        self.assertEqual(tracker.snapshot()["state"], "CAPTURANDO")
        self.assertTrue(tracker.capture_reached)

        tracker.confirm_capture({"accepted": True})

        guide = tracker.step(
            roi=ROI,
            bbox=box(*ROI_CENTER),
            coverage=0.55,
            sharpness=120.0,
            sequence=1,
            frame=FAKE_FRAME,
        )
        self.assertEqual(guide["state"], "CAPTURADA")
        self.assertFalse(guide["wants_capture"])


# ============================================================
# PRUEBAS HTTP (requieren MySQL; se omiten si no hay conexión)
# ============================================================


@unittest.skipUnless(DB_AVAILABLE, "MySQL no disponible")
class GuidedCaptureHttpTests(unittest.TestCase):
    ADMIN_USERNAME = "ph2c_admin"
    QM_USERNAME = "ph2c_qm"
    MM_USERNAME = "ph2c_mm"
    MODEL_A = "TEST-PH2C-A"
    MODEL_DRAFT = "TEST-PH2C-D"

    @classmethod
    def setUpClass(cls):
        import importlib

        import mysql.connector

        cls.A = importlib.import_module("app")

        # Los tests nunca deben arrancar hilos de cámara ni de captura.
        cls._orig_camera_worker = cls.A.ensure_camera_capture_worker
        cls._orig_ai_worker = cls.A._ensure_ai_capture_worker
        cls._orig_guided_worker = cls.A._ensure_ai_guided_worker
        cls.A.ensure_camera_capture_worker = lambda *a, **k: False
        cls.A._ensure_ai_capture_worker = lambda *a, **k: False
        cls.A._ensure_ai_guided_worker = lambda *a, **k: False

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
        cls._ensure_user(cls.MM_USERNAME, "MODEL_MANAGER")

        cls.model_a = cls._ensure_model(cls.MODEL_A, "APROBADO")
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
                (code, f"Modelo fase 2c {code}", status),
            )
            model_id = int(cls.cur.lastrowid)
        cls.conn.commit()
        return model_id

    @classmethod
    def tearDownClass(cls):
        from ai_domain import ensure_ai_schema

        cls.A.ensure_camera_capture_worker = cls._orig_camera_worker
        cls.A._ensure_ai_capture_worker = cls._orig_ai_worker
        cls.A._ensure_ai_guided_worker = cls._orig_guided_worker
        cls.A.set_ai_capture_mode(False)
        cls.A._stop_ai_capture_worker()
        cls.A._stop_ai_guided_worker()
        cls.A._ai_guided_reset_state()

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
            "(SELECT id FROM users WHERE username IN (%s, %s, %s))",
            (cls.ADMIN_USERNAME, cls.QM_USERNAME, cls.MM_USERNAME),
        )

        for code in (cls.MODEL_A, cls.MODEL_DRAFT):
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
            "DELETE FROM users WHERE username IN (%s, %s, %s)",
            (cls.ADMIN_USERNAME, cls.QM_USERNAME, cls.MM_USERNAME),
        )
        cls.conn.commit()

        ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    # --------------------------------------------------------
    # utilidades
    # --------------------------------------------------------

    def setUp(self):
        self._close_all_open()
        self._login(self.ADMIN_USERNAME)
        self.A._ai_guided_reset_state()
        # Anti-mismo-fotograma: los tests no esperan entre tomas.
        self._cooldown_env = os.environ.get("AI_GUIDED_MANUAL_COOLDOWN_SECONDS")
        os.environ["AI_GUIDED_MANUAL_COOLDOWN_SECONDS"] = "0"
        self.A.AI_GUIDED_MANUAL_LAST.update({"at": 0.0, "sequence": -1})

    def tearDown(self):
        self._close_all_open()
        self.A._ai_guided_reset_state()
        if self._cooldown_env is None:
            os.environ.pop("AI_GUIDED_MANUAL_COOLDOWN_SECONDS", None)
        else:
            os.environ["AI_GUIDED_MANUAL_COOLDOWN_SECONDS"] = self._cooldown_env
        self.A.AI_GUIDED_MANUAL_LAST.update({"at": 0.0, "sequence": -1})

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

    def _json(self, response):
        return json.loads(response.get_data(as_text=True))

    def _prepare_version(self, model_id):
        return self.client.post(
            f"/modelos-prenda/{model_id}/ia/preparar",
            follow_redirects=False,
        )

    def _start(self, model_id, guided=False):
        return self.client.post(
            "/api/ai/capture/start",
            json={"garment_model_id": model_id, "guided": guided},
        )

    def _guide_start(self, model_id):
        return self.client.post(
            "/api/ai/capture/guide/start",
            json={"garment_model_id": model_id},
        )

    def _status(self, model_id):
        return self.client.get(
            f"/api/ai/capture/status?garment_model_id={model_id}"
        )

    def _guided_url(self, model_id, suffix=""):
        return f"/modelos-prenda/{model_id}/captura-ia{suffix}"

    def _count_inspections(self):
        self.cur.execute("SELECT COUNT(*) AS c FROM inspections")
        return int(self.cur.fetchone()["c"])

    # --------------------------------------------------------
    # pantalla guiada
    # --------------------------------------------------------

    def test_detail_page_links_to_guided_screen(self):
        self._prepare_version(self.model_a)

        html = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        ).get_data(as_text=True)

        self.assertIn("Iniciar preparación", html)
        self.assertIn('id="startGuidedPreparation"', html)
        self.assertIn(f'href="{self._guided_url(self.model_a)}"', html)
        self.assertNotIn("ASTRID_AI_CAPTURE", html)
        self.assertNotIn("aiCapturePanel", html)

    def test_guided_page_renders_video_and_actions(self):
        html = self.client.get(
            self._guided_url(self.model_a)
        ).get_data(as_text=True)

        self.assertIn("Captura guiada", html)
        self.assertIn("frame_fresh", html)
        self.assertIn("Se activa cuando la cámara tiene una imagen reciente", html)
        self.assertIn(self._guided_url(self.model_a, "/video"), html)
        self.assertIn("FINALIZAR PREPARACIÓN", html)
        self.assertIn("CANCELAR SESIÓN", html)
        self.assertIn("Capturas válidas", html)
        self.assertIn("Objetivo recomendado alcanzado", html)
        self.assertIn("ASTRID_AI_GUIDED", html)
        self.assertNotIn("window.alert(", html)
        self.assertNotIn("window.confirm(", html)

    def test_open_guided_page_and_start_resume_same_open_session(self):
        self._prepare_version(self.model_a)
        first = self._json(self._start(self.model_a, guided=True))
        session_id = int(first["session"]["session_id"])

        page = self.client.get(self._guided_url(self.model_a))
        resumed = self._json(self._start(self.model_a, guided=True))

        self.assertEqual(page.status_code, 200)
        self.assertEqual(resumed["session"]["session_id"], session_id)
        self.conn.commit()
        self.cur.execute(
            """SELECT COUNT(*) AS n FROM ai_capture_sessions
               WHERE garment_model_id = %s AND status = 'ABIERTA'""",
            (self.model_a,),
        )
        self.assertEqual(int(self.cur.fetchone()["n"]), 1)

    def test_guided_page_is_forbidden_for_quality_manager(self):
        self._login(self.QM_USERNAME)

        response = self.client.get(
            self._guided_url(self.model_a),
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertNotIn(
            "/captura-ia",
            response.headers.get("Location", ""),
        )

    def test_guided_page_rejects_missing_model(self):
        response = self.client.get(
            self._guided_url(999999),
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)

    def test_guided_video_returns_mjpeg(self):
        response = self.client.get(
            self._guided_url(self.model_a, "/video"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "multipart/x-mixed-replace",
            response.headers.get("Content-Type", ""),
        )

    def test_guided_video_is_forbidden_for_quality_manager(self):
        self._login(self.QM_USERNAME)

        response = self.client.get(
            self._guided_url(self.model_a, "/video"),
        )

        self.assertNotEqual(response.status_code, 200)

    def test_guided_video_is_forbidden_for_foreign_manager(self):
        self._login(self.MM_USERNAME)

        response = self.client.get(
            self._guided_url(self.model_a, "/video"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(self._json(response)["ok"])

    def test_guided_page_redirects_foreign_manager(self):
        self._login(self.MM_USERNAME)

        response = self.client.get(
            self._guided_url(self.model_a),
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertNotIn(
            "/captura-ia",
            response.headers.get("Location", ""),
        )

    # --------------------------------------------------------
    # arranque del motor guiado
    # --------------------------------------------------------

    def test_guide_start_without_session_is_rejected(self):
        self._prepare_version(self.model_a)

        response = self._guide_start(self.model_a)

        self.assertEqual(response.status_code, 409)
        payload = self._json(response)
        self.assertFalse(payload["ok"])
        self.assertNotIn("{", payload["error"])
        self.assertNotIn("None", payload["error"])

    def test_guide_start_without_preparation_version_is_rejected(self):
        started = self._json(self._start(self.model_a, guided=True))
        self.assertTrue(started["ok"])

        self.cur.execute(
            "DELETE FROM garment_ai_models WHERE garment_model_id = %s",
            (self.model_a,),
        )
        self.conn.commit()

        response = self._guide_start(self.model_a)

        self.assertEqual(response.status_code, 409)
        self.assertIn("preparación", self._json(response)["error"])

    def test_guide_start_bootstraps_after_session(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        self.assertTrue(started["ok"])

        response = self._guide_start(self.model_a)
        payload = self._json(response)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["session"]["active"])
        self.assertIn("guide", payload)
        self.assertIn("state", payload["guide"])

    def test_guide_start_rejects_draft_model(self):
        response = self._guide_start(self.model_draft)

        self.assertEqual(response.status_code, 409)
        self.assertFalse(self._json(response)["ok"])

    def test_guide_start_rejects_invalid_model_id(self):
        response = self.client.post(
            "/api/ai/capture/guide/start",
            json={"garment_model_id": "abc"},
        )
        self.assertEqual(response.status_code, 400)

    def test_status_exposes_guide_and_does_not_create_inspections(self):
        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)
        self._guide_start(self.model_a)

        before = self._count_inspections()
        payload = self._json(self._status(self.model_a))
        after = self._count_inspections()

        self.assertTrue(payload["ok"])
        self.assertIn("guide", payload)
        self.assertIsInstance(payload["guide"], dict)
        self.assertEqual(before, after)

    def test_status_progress_is_dynamic_without_inspections(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])

        before = self._count_inspections()

        for index in range(3):
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
                    self.model_a,
                    f"captures/phase2c/{digest}.jpg",
                    digest,
                    index + 1,
                    0.8000,
                    90.000,
                    "ACEPTADA",
                    None,
                ),
            )
        self.conn.commit()

        payload = self._json(self._status(self.model_a))
        after = self._count_inspections()

        self.assertEqual(payload["accepted_count"], 3)
        self.assertEqual(before, after)
        self.assertTrue(payload["active"])

    def test_guide_reset_keeps_session_active(self):
        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)
        self._guide_start(self.model_a)

        response = self.client.post(
            "/api/ai/capture/guide/reset",
            json={},
        )
        payload = self._json(response)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(self._json(self._status(self.model_a))["active"])

    def test_guide_reset_without_session_is_rejected(self):
        response = self.client.post(
            "/api/ai/capture/guide/reset",
            json={},
        )
        self.assertEqual(response.status_code, 409)

    # --------------------------------------------------------
    # cierre de la sesión desde la pantalla guiada
    # --------------------------------------------------------

    def test_finish_redirects_to_detail_with_human_flash(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])

        for index in range(20):
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
                    self.model_a,
                    f"captures/phase2c/{digest}.jpg",
                    digest,
                    index + 1,
                    0.8000,
                    90.000,
                    "ACEPTADA",
                    None,
                ),
            )
        self.conn.commit()

        response = self.client.post(
            self._guided_url(self.model_a, "/finalizar"),
            data={},
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Captura completada", html)
        self.assertIn("imágenes normales", html)
        self.assertIn("Se alcanzó el mínimo", html)
        self.assertNotIn("Traceback", html)

        self.assertFalse(self._json(self._status(self.model_a))["active"])

    def test_cancel_redirects_to_detail_with_flash(self):
        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)

        response = self.client.post(
            self._guided_url(self.model_a, "/cancelar"),
            data={},
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("cancelada", html)
        self.assertNotIn("Traceback", html)
        self.assertFalse(self._json(self._status(self.model_a))["active"])

    def test_finish_without_open_session_shows_message(self):
        response = self.client.post(
            self._guided_url(self.model_a, "/finalizar"),
            data={},
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("No hay una captura activa", html)

    def test_finish_forbidden_for_quality_manager(self):
        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)
        self._login(self.QM_USERNAME)

        response = self.client.post(
            self._guided_url(self.model_a, "/finalizar"),
            data={},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertNotIn(
            "/captura-ia/finalizar",
            response.headers.get("Location", ""),
        )

        # La sesión sigue abierta: el rol no autorizado no pudo cerrarla.
        self._login(self.ADMIN_USERNAME)
        self.assertTrue(self._json(self._status(self.model_a))["active"])

    # --------------------------------------------------------
    # botón manual, revisión y compuertas de finalización
    # --------------------------------------------------------

    def _insert_image(self, session_id, status, reject_reason=None,
                      sequence=1):
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
                self.model_a,
                f"captures/phase2c/{digest}.jpg",
                digest,
                sequence,
                0.8000,
                90.000,
                status,
                reject_reason,
            ),
        )
        self.conn.commit()
        return int(self.cur.lastrowid)

    def test_capture_button_is_visible_and_disabled_until_ready(self):
        html = self.client.get(
            self._guided_url(self.model_a)
        ).get_data(as_text=True)

        self.assertIn("CAPTURAR IMAGEN", html)
        self.assertRegex(
            html,
            r'id="captureBtn"[^>]*disabled',
        )
        self.assertIn(
            "Se activa cuando la cámara tiene una imagen reciente",
            html,
        )
        self.assertIn("Lista para capturar", html)

    def test_guided_page_shows_finalize_block_message(self):
        html = self.client.get(
            self._guided_url(self.model_a)
        ).get_data(as_text=True)

        self.assertIn(
            "Debe capturar al menos 20 imágenes válidas antes de finalizar",
            html,
        )
        self.assertIn("Faltan 20 imágenes para alcanzar el mínimo", html)

    def test_finish_is_blocked_without_valid_images(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])

        # Una imagen descartada NO habilita finalizar.
        self._insert_image(session_id, "RECHAZADA", "MANUAL_DISCARD")

        payload = self._json(self._status(self.model_a))
        self.assertFalse(payload["can_finalize"])
        self.assertEqual(payload["accepted_count"], 0)

        response = self.client.post(
            self._guided_url(self.model_a, "/finalizar"),
            data={},
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "Faltan 20 imágenes para alcanzar el mínimo de 20 imágenes válidas antes de finalizar.",
            html,
        )
        self.assertNotIn("Traceback", html)

        # La sesión NO se cerró.
        self.assertTrue(self._json(self._status(self.model_a))["active"])

    def test_status_exposes_minimum_for_training(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])

        self._insert_image(session_id, "ACEPTADA")

        payload = self._json(self._status(self.model_a))

        self.assertEqual(payload["min_count"], 20)
        self.assertEqual(payload["target_count"], 40)
        self.assertFalse(payload["can_finalize"])
        self.assertFalse(payload["can_train"])
        self.assertEqual(payload["missing_to_train"], 19)

    def test_review_endpoint_updates_session_counters(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])
        image_id = self._insert_image(session_id, "ACEPTADA")

        self.assertEqual(
            self._json(self._status(self.model_a))["accepted_count"],
            1,
        )

        repetir = self._json(
            self.client.post(
                "/api/ai/capture/guide/review",
                json={
                    "garment_model_id": self.model_a,
                    "image_id": image_id,
                    "decision": "repeat",
                },
            )
        )
        self.assertTrue(repetir["ok"])
        self.assertEqual(repetir["session"]["accepted_count"], 0)
        self.assertEqual(repetir["session"]["rejected_count"], 1)
        self.assertFalse(repetir["session"]["can_finalize"])

        aceptar = self._json(
            self.client.post(
                "/api/ai/capture/guide/review",
                json={
                    "garment_model_id": self.model_a,
                    "image_id": image_id,
                    "decision": "accept",
                },
            )
        )
        self.assertTrue(aceptar["ok"])
        self.assertEqual(aceptar["session"]["accepted_count"], 1)
        self.assertEqual(aceptar["session"]["rejected_count"], 0)
        self.assertFalse(aceptar["session"]["can_finalize"])

        self.cur.execute(
            "SELECT status FROM ai_training_images WHERE id = %s",
            (image_id,),
        )
        self.assertEqual(self.cur.fetchone()["status"], "ACEPTADA")

    def test_review_rejects_unknown_decision_and_foreign_image(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))

        malo = self.client.post(
            "/api/ai/capture/guide/review",
            json={
                "garment_model_id": self.model_a,
                "image_id": 999999,
                "decision": "whatever",
            },
        )
        self.assertEqual(malo.status_code, 400)

        desconocida = self.client.post(
            "/api/ai/capture/guide/review",
            json={
                "garment_model_id": self.model_a,
                "image_id": 999999,
                "decision": "accept",
            },
        )
        self.assertEqual(desconocida.status_code, 404)
        self.assertFalse(self._json(desconocida)["ok"])
        self.assertTrue(self._json(self._status(self.model_a))["active"])
        self.assertTrue(started["ok"])

    def test_manual_capture_without_session_is_rejected(self):
        response = self.client.post(
            "/api/ai/capture/guide/manual",
            json={"garment_model_id": self.model_a},
        )

        self.assertEqual(response.status_code, 409)
        payload = self._json(response)
        self.assertFalse(payload["ok"])
        self.assertNotIn("{", payload["error"])

    def test_manual_capture_rejects_draft_model(self):
        response = self.client.post(
            "/api/ai/capture/guide/manual",
            json={"garment_model_id": self.model_draft},
        )

        self.assertEqual(response.status_code, 409)
        self.assertFalse(self._json(response)["ok"])

    def test_manual_capture_rejects_stale_camera_frame(self):
        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)
        original = self.A.get_latest_camera_frame
        try:
            import numpy as np

            self.A.get_latest_camera_frame = lambda: (
                np.zeros((480, 640, 3), dtype=np.uint8),
                time.perf_counter() - 5.0,
                123,
            )
            response = self.client.post(
                "/api/ai/capture/guide/manual",
                json={"garment_model_id": self.model_a},
            )
        finally:
            self.A.get_latest_camera_frame = original

        self.assertEqual(response.status_code, 409)
        self.assertIn("imagen reciente", self._json(response)["error"])

    def test_analysis_and_overlay_share_original_frame_roi(self):
        import numpy as np

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = self.A.get_roi_bounds(frame)
        from ai_capture import classify_position

        analysis = classify_position(
            roi=roi,
            bbox=None,
            coverage=0.0,
            sharpness=0.0,
        )
        self.assertEqual(analysis["roi"], roi)
        self.assertEqual(
            roi,
            (
                int(self.A.ROI_X1 * 1920),
                int(self.A.ROI_Y1 * 1080),
                int(self.A.ROI_X2 * 1920),
                int(self.A.ROI_Y2 * 1080),
            ),
        )
        overlaid = self.A.draw_guided_overlay(frame, {"state": None})
        self.assertEqual(overlaid.shape, frame.shape)
        self.assertEqual(self.A.get_roi_bounds(overlaid), analysis["roi"])

    def test_false_negative_still_allows_manual_preview_and_review(self):
        import numpy as np

        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        self.assertTrue(started["ok"])

        original_frame_fn = self.A.get_latest_camera_frame
        original_mask_fn = self.A.create_garment_mask
        sequence = [100]
        frame = np.full((1080, 1920, 3), 180, dtype=np.uint8)

        def fresh_frame():
            sequence[0] += 1
            return frame.copy(), time.perf_counter(), sequence[0]

        self.A.get_latest_camera_frame = fresh_frame
        self.A.create_garment_mask = lambda image: np.zeros(
            image.shape[:2], dtype=np.uint8
        )
        try:
            preview = self._json(
                self.client.post(
                    "/api/ai/capture/guide/manual",
                    json={"garment_model_id": self.model_a},
                )
            )
            self.assertTrue(preview["ok"])
            self.assertIn("data:image/png;base64,", preview["preview_data_url"])
            self.assertFalse(preview["metrics"]["detected"])
            self.assertEqual(preview["metrics"]["frame_size"], [1920, 1080])
            self.assertIn("no pudo confirmar", " ".join(preview["warnings"]).lower())
            self.assertEqual(
                self._json(self._status(self.model_a))["accepted_count"],
                0,
                "El preview no debe persistir ni contar la captura.",
            )

            repeated = self._json(
                self.client.post(
                    "/api/ai/capture/guide/review",
                    json={
                        "garment_model_id": self.model_a,
                        "pending_token": preview["pending_token"],
                        "decision": "repeat",
                    },
                )
            )
            self.assertEqual(repeated["session"]["accepted_count"], 0)
            self.assertEqual(repeated["session"]["rejected_count"], 0)

            preview = self._json(
                self.client.post(
                    "/api/ai/capture/guide/manual",
                    json={"garment_model_id": self.model_a},
                )
            )
            accepted = self._json(
                self.client.post(
                    "/api/ai/capture/guide/review",
                    json={
                        "garment_model_id": self.model_a,
                        "pending_token": preview["pending_token"],
                        "decision": "accept",
                    },
                )
            )
            self.assertEqual(accepted["session"]["accepted_count"], 1)
            self.assertEqual(accepted["session"]["rejected_count"], 0)

            preview = self._json(
                self.client.post(
                    "/api/ai/capture/guide/manual",
                    json={"garment_model_id": self.model_a},
                )
            )
            discarded = self._json(
                self.client.post(
                    "/api/ai/capture/guide/review",
                    json={
                        "garment_model_id": self.model_a,
                        "pending_token": preview["pending_token"],
                        "decision": "discard",
                    },
                )
            )
            self.assertEqual(discarded["session"]["accepted_count"], 1)
            self.assertEqual(discarded["session"]["rejected_count"], 1)
            self.conn.commit()
            self.cur.execute(
                """SELECT COUNT(*) AS n FROM ai_events
                   WHERE capture_session_id = %s
                     AND event_type = 'CAPTURE_IMAGE_REVIEWED'""",
                (started["session"]["session_id"],),
            )
            self.assertEqual(int(self.cur.fetchone()["n"]), 2)
        finally:
            self.A.get_latest_camera_frame = original_frame_fn
            self.A.create_garment_mask = original_mask_fn

    # --------------------------------------------------------
    # anti-mismo-fotograma, métricas y reconexión de cámara
    # --------------------------------------------------------

    def test_manual_capture_blocks_pending_same_frame_and_cooldown(self):
        import numpy as np

        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)

        state = {"sequence": 480}
        frame = np.full((1080, 1920, 3), 180, dtype=np.uint8)
        original = self.A.get_latest_camera_frame

        def frozen_frame():
            return frame.copy(), time.perf_counter(), state["sequence"]

        self.A.get_latest_camera_frame = frozen_frame
        try:
            first = self.client.post(
                "/api/ai/capture/guide/manual",
                json={"garment_model_id": self.model_a},
            )
            self.assertEqual(first.status_code, 200)
            token = self._json(first)["pending_token"]

            # Con vista previa pendiente no se admite otra captura.
            blocked = self.client.post(
                "/api/ai/capture/guide/manual",
                json={"garment_model_id": self.model_a},
            )
            self.assertEqual(blocked.status_code, 409)
            self.assertIn("pendiente", self._json(blocked)["error"])

            repeated = self._json(
                self.client.post(
                    "/api/ai/capture/guide/review",
                    json={
                        "garment_model_id": self.model_a,
                        "pending_token": token,
                        "decision": "repeat",
                    },
                )
            )
            self.assertTrue(repeated["ok"])
            self.assertEqual(repeated["session"]["accepted_count"], 0)
            self.assertEqual(repeated["session"]["rejected_count"], 0)

            # Se liberó la vista previa, pero sigue siendo el MISMO frame.
            os.environ["AI_GUIDED_MANUAL_COOLDOWN_SECONDS"] = "5"
            same = self.client.post(
                "/api/ai/capture/guide/manual",
                json={"garment_model_id": self.model_a},
            )
            self.assertEqual(same.status_code, 409)
            self.assertIn(
                "fotograma nuevo",
                self._json(same)["error"],
            )

            # Frame nuevo, pero dentro del cooldown entre tomas.
            state["sequence"] += 1
            too_fast = self.client.post(
                "/api/ai/capture/guide/manual",
                json={"garment_model_id": self.model_a},
            )
            self.assertEqual(too_fast.status_code, 429)
            self.assertIn("Espere", self._json(too_fast)["error"])
        finally:
            self.A.get_latest_camera_frame = original
            os.environ["AI_GUIDED_MANUAL_COOLDOWN_SECONDS"] = "0"

    def test_manual_preview_reports_segmentation_metrics(self):
        frame = synthetic_garment_frame()
        if frame is None:
            self.skipTest(
                "No hay fotogramas de calibración en el repositorio."
            )

        self._prepare_version(self.model_a)
        self._start(self.model_a, guided=True)

        original = self.A.get_latest_camera_frame
        self.A.get_latest_camera_frame = lambda: (
            frame.copy(),
            time.perf_counter(),
            9001,
        )
        try:
            preview = self._json(
                self.client.post(
                    "/api/ai/capture/guide/manual",
                    json={"garment_model_id": self.model_a},
                )
            )
        finally:
            self.A.get_latest_camera_frame = original

        self.assertTrue(preview["ok"])
        metrics = preview["metrics"]

        self.assertTrue(metrics["detected"])
        self.assertIn(metrics["mask_method"], ("semilla", "adaptativo"))
        self.assertGreater(metrics["mask_nonzero"], 0)
        self.assertGreater(metrics["coverage"], 0.15)
        self.assertLess(metrics["coverage"], 0.75)
        self.assertIsNotNone(metrics["bbox"])
        self.assertEqual(sorted(metrics["roi_stats"]), ["max", "mean", "min"])
        self.assertIsNone(metrics["segmentation_reason"])

        # La vista previa queda pendiente de revisión, sin persistir.
        status = self._json(self._status(self.model_a))
        self.assertEqual(status["accepted_count"], 0)
        self.assertEqual(status["rejected_count"], 0)

    def test_model_manager_can_reconnect_camera(self):
        self._login(self.MM_USERNAME)

        response = self.client.post(
            "/api/station/camera/reconnect",
            json={},
        )

        self.assertEqual(response.status_code, 202)
        payload = self._json(response)
        self.assertTrue(payload["ok"])
        self.assertIn("camera", payload)

    def test_guided_page_has_reconnect_button_and_review_flow(self):
        html = self.client.get(
            self._guided_url(self.model_a)
        ).get_data(as_text=True)

        self.assertIn("Reconectar cámara", html)
        self.assertIn('id="reconnectBtn"', html)
        self.assertIn("/api/station/camera/reconnect", html)
        self.assertIn("pending_token", html)
        # Guarda que rompía el flujo: exigía last_capture aunque la
        # revisión fuera sobre la vista previa pendiente (primer toma).
        self.assertNotIn("if (!last || !last.id) return;", html)

    def test_finish_with_19_images_is_rejected_by_backend(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])
        for index in range(19):
            self._insert_image(
                session_id, "ACEPTADA", sequence=index + 1
            )

        response = self.client.post("/api/ai/capture/stop", json={})

        self.assertEqual(response.status_code, 409)
        self.assertFalse(self._json(response)["ok"])
        self.assertTrue(self._json(self._status(self.model_a))["active"])

    def test_40_images_are_recommended_not_a_hard_limit(self):
        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])
        for index in range(40):
            self._insert_image(
                session_id, "ACEPTADA", sequence=index + 1
            )

        status = self._json(self._status(self.model_a))
        self.assertTrue(status["can_finalize"])
        self.assertTrue(status["can_train"])
        self.assertTrue(status["target_reached"])

        response = self.client.post(
            self._guided_url(self.model_a, "/finalizar"),
            data={},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "Objetivo recomendado alcanzado",
            response.get_data(as_text=True),
        )

    def test_detail_page_links_to_guided_screen_without_version(self):
        self.cur.execute(
            "DELETE FROM garment_ai_models WHERE garment_model_id = %s",
            (self.model_a,),
        )
        self.conn.commit()

        html = self.client.get(
            f"/modelos-prenda/{self.model_a}"
        ).get_data(as_text=True)

        self.assertIn("Iniciar preparación", html)
        self.assertIn(self._guided_url(self.model_a), html)

    def test_capture_image_endpoint_serves_existing_file(self):
        from ai_domain import (
            capture_image_relative_path,
            ensure_capture_session_dirs,
        )

        self._prepare_version(self.model_a)
        started = self._json(self._start(self.model_a, guided=True))
        session_id = int(started["session"]["session_id"])

        filename = "revision_manual.jpg"
        relative = capture_image_relative_path(
            self.model_a, session_id, filename
        )

        directory = ensure_capture_session_dirs(
            self.model_a, session_id
        )
        file_path = Path(directory) / "accepted" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"\xff\xd8\xff\xe0fake")

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
                self.model_a,
                relative,
                digest,
                1,
                0.8000,
                90.000,
                "ACEPTADA",
                None,
            ),
        )
        self.conn.commit()
        image_id = int(self.cur.lastrowid)

        response = self.client.get(
            self._guided_url(
                self.model_a,
                f"/imagen/{image_id}",
            )
        )

        body = response.data
        response.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body[:2], b"\xff\xd8")

        self.cur.execute(
            "DELETE FROM ai_training_images WHERE id = %s",
            (image_id,),
        )
        self.conn.commit()

        # El cliente de prueba puede retener el manejador en Windows.
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass


# ============================================================
# SEGMENTADOR GUIADO (sin base de datos)
# ============================================================


class GuidedSegmentationTests(unittest.TestCase):
    """
    create_guided_garment_mask / _ai_guided_segment.

    Regla dura: un frame sin prenda devuelve máscara VACÍA (nunca una
    silueta inventada) y registra el diagnóstico completo.
    """

    @classmethod
    def setUpClass(cls):
        import importlib

        cls.A = importlib.import_module("app")

    def _coverage(self, mask, frame):
        roi = self.A.get_roi_bounds(frame)
        roi_mask = mask[roi[1]:roi[3], roi[0]:roi[2]]
        return float(self.A.cv2.countNonZero(roi_mask)) / max(
            1,
            roi_mask.size,
        )

    def _loadable_frames(self):
        frames = []
        for path in EMPTY_FRAME_PATHS:
            if not path.exists():
                continue
            frame = self.A.cv2.imread(str(path))
            if frame is not None:
                frames.append((path.name, frame))
        return frames

    def test_empty_frames_report_no_garment(self):
        frames = self._loadable_frames()
        if not frames:
            self.skipTest(
                "No hay fotogramas de calibración en el repositorio."
            )

        for name, frame in frames:
            diagnostics = {}
            mask = self.A.create_guided_garment_mask(frame, diagnostics)

            self.assertEqual(diagnostics["method"], "ninguno", name)
            self.assertEqual(diagnostics["mask_nonzero"], 0, name)
            self.assertEqual(diagnostics["mask_bbox"], None, name)
            self.assertEqual(self._coverage(mask, frame), 0.0, name)
            self.assertEqual(
                sorted(diagnostics["roi_stats"]),
                ["max", "mean", "min"],
                name,
            )
            self.assertTrue(diagnostics.get("reason"), name)
            self.assertEqual(diagnostics["frame"][0], frame.shape[1], name)

    def test_synthetic_garment_is_detected(self):
        frame = synthetic_garment_frame()
        if frame is None:
            self.skipTest(
                "No hay fotogramas de calibración en el repositorio."
            )

        diagnostics = {}
        mask = self.A.create_guided_garment_mask(frame, diagnostics)

        import numpy as np

        roi = self.A.get_roi_bounds(frame)
        roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
        truth = (
            np.pi * SYNTH_ELLIPSE_AXES[0] * SYNTH_ELLIPSE_AXES[1] / roi_area
        )

        self.assertIn(diagnostics["method"], ("semilla", "adaptativo"))
        self.assertGreater(diagnostics["mask_nonzero"], 0)
        self.assertAlmostEqual(self._coverage(mask, frame), truth, delta=0.05)

        left, top, right, bottom = diagnostics["mask_bbox"]
        exp_left = SYNTH_ELLIPSE_CENTER[0] - SYNTH_ELLIPSE_AXES[0]
        exp_top = SYNTH_ELLIPSE_CENTER[1] - SYNTH_ELLIPSE_AXES[1]
        exp_right = SYNTH_ELLIPSE_CENTER[0] + SYNTH_ELLIPSE_AXES[0]
        exp_bottom = SYNTH_ELLIPSE_CENTER[1] + SYNTH_ELLIPSE_AXES[1]
        self.assertLessEqual(abs(left - exp_left), 40)
        self.assertLessEqual(abs(top - exp_top), 40)
        self.assertLessEqual(abs(right - exp_right), 40)
        self.assertLessEqual(abs(bottom - exp_bottom), 40)

    def test_guided_segment_uses_a_single_mask(self):
        frame = synthetic_garment_frame()
        if frame is None:
            self.skipTest(
                "No hay fotogramas de calibración en el repositorio."
            )

        roi = self.A.get_roi_bounds(frame)
        mask, bbox, coverage, diagnostics = self.A._ai_guided_segment(
            frame,
            roi,
        )

        self.assertIsNotNone(mask)
        self.assertIsNotNone(bbox)
        self.assertGreater(coverage, 0.15)
        self.assertIn(
            diagnostics.get("method"),
            ("semilla", "adaptativo"),
        )

        # bbox y cobertura provienen de la MISMA máscara devuelta.
        self.assertEqual(bbox, self.A.mask_bbox(mask, self.A.cv2))
        roi_mask = mask[roi[1]:roi[3], roi[0]:roi[2]]
        expected = float(
            self.A.cv2.countNonZero(roi_mask) / max(1, roi_mask.size)
        )
        self.assertAlmostEqual(coverage, expected, places=6)

    def test_worker_uses_guided_segmenter_not_production_coverage(self):
        import inspect

        source = inspect.getsource(self.A.ai_guided_worker)

        self.assertIn("_ai_guided_segment", source)
        self.assertNotIn("compute_roi_coverage", source)
        self.assertNotIn("create_garment_mask(", source)


if __name__ == "__main__":
    unittest.main()
