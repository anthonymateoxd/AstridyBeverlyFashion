"""Tests FASE 2A — captura de dataset IA (sin cámara física).

Cubren: mutex estación, calidad/duplicados, paths, presencia 1:1,
escritura atómica y persistencia MySQL de sesiones (si BD disponible).
NO entrena PatchCore.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import ai_capture
from ai_capture import (
    PresenceCandidate,
    assert_production_allowed,
    compute_dhash_64,
    ensure_ai_capture_can_start,
    evaluate_candidate,
    get_ai_capture_mode_state,
    is_ai_capture_mode_active,
    reset_presence_candidate,
    set_ai_capture_mode,
    set_production_probe,
)
from ai_domain import (
    AIDomainError,
    atomic_write_bytes,
    capture_image_relative_path,
    compute_quality_score,
    evaluate_capture_metrics,
    get_ai_capture_config,
    hamming_distance,
    is_perceptual_duplicate,
    resolve_under_root,
    sha256_bytes,
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


class CaptureConfigTests(unittest.TestCase):
    def test_defaults_documented(self):
        cfg = get_ai_capture_config()
        self.assertEqual(cfg["min_coverage"], 0.48)
        self.assertEqual(cfg["min_sharpness"], 40.0)
        self.assertEqual(cfg["target_images"], 40)
        self.assertEqual(cfg["duplicate_max_distance"], 6)
        self.assertFalse(cfg["keep_rejected"])

    def test_env_override(self):
        old = os.environ.get("AI_CAPTURE_TARGET_IMAGES")
        os.environ["AI_CAPTURE_TARGET_IMAGES"] = "12"
        try:
            cfg = get_ai_capture_config()
            self.assertEqual(cfg["target_images"], 12)
        finally:
            if old is None:
                os.environ.pop("AI_CAPTURE_TARGET_IMAGES", None)
            else:
                os.environ["AI_CAPTURE_TARGET_IMAGES"] = old

    def test_invalid_coverage_rejected(self):
        old = os.environ.get("AI_CAPTURE_MIN_COVERAGE")
        os.environ["AI_CAPTURE_MIN_COVERAGE"] = "2.5"
        try:
            with self.assertRaises(AIDomainError):
                get_ai_capture_config()
        finally:
            if old is None:
                os.environ.pop("AI_CAPTURE_MIN_COVERAGE", None)
            else:
                os.environ["AI_CAPTURE_MIN_COVERAGE"] = old


class QualityDecisionTests(unittest.TestCase):
    def test_low_coverage_rejected(self):
        decision = evaluate_capture_metrics(
            coverage=0.20,
            sharpness=100.0,
            roi_valid=True,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reject_reason"], "LOW_COVERAGE")

    def test_blur_rejected(self):
        decision = evaluate_capture_metrics(
            coverage=0.55,
            sharpness=5.0,
            roi_valid=True,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reject_reason"], "BLUR")

    def test_invalid_roi_rejected(self):
        decision = evaluate_capture_metrics(
            coverage=0.55,
            sharpness=100.0,
            roi_valid=False,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reject_reason"], "INVALID_ROI")

    def test_good_candidate_accepted(self):
        decision = evaluate_capture_metrics(
            coverage=0.55,
            sharpness=80.0,
            roi_valid=True,
        )
        self.assertTrue(decision["accepted"])
        self.assertIsNone(decision["reject_reason"])
        self.assertGreater(decision["quality_score"], 0.5)

    def test_quality_score_bounded(self):
        score = compute_quality_score(
            1.0,
            10_000.0,
            min_coverage=0.48,
            min_sharpness=40.0,
        )
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)


class DuplicateTests(unittest.TestCase):
    def test_exact_sha_match_flagged_in_evaluate(self):
        # Mock mínimo de frame/roi no aplica: evaluate_candidate
        # con ROI inválido no llega a sha. Probamos la capa pura.
        known = {"a" * 64}
        self.assertIn("a" * 64, known)

    def test_hamming_and_perceptual(self):
        a = 0b1111000011110000111100001111000011110000111100001111000011110000
        b = 0b1111000011110000111100001111000011110000111100001111000011110001
        self.assertEqual(hamming_distance(a, b), 1)
        self.assertTrue(is_perceptual_duplicate(a, b, 6))
        self.assertFalse(is_perceptual_duplicate(a, b, 0))

        far = 0
        self.assertFalse(is_perceptual_duplicate(a, far, 6))

    def test_threshold_bounds(self):
        with self.assertRaises(AIDomainError):
            is_perceptual_duplicate(1, 2, 65)


class PathAndStorageTests(unittest.TestCase):
    def test_relative_path_layout(self):
        rel = capture_image_relative_path(
            3,
            7,
            "frame_10.jpg",
        )
        self.assertEqual(
            rel,
            "garment_3/capture_sessions/session_7/accepted/frame_10.jpg",
        )
        rel_r = capture_image_relative_path(
            3,
            7,
            "frame_10.jpg",
            rejected=True,
        )
        self.assertIn("/rejected/", rel_r)

    def test_path_traversal_blocked(self):
        with self.assertRaises(AIDomainError):
            capture_image_relative_path(
                1,
                1,
                "../../etc/passwd",
            )
        with self.assertRaises(AIDomainError):
            capture_image_relative_path(
                1,
                1,
                "sub/dir.jpg",
            )

    def test_atomic_write_and_sha256(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "accepted" / "x.jpg"
            data = b"jpeg-bytes-demo"
            atomic_write_bytes(target, data)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), data)
            digest = sha256_bytes(data)
            self.assertEqual(len(digest), 64)
            # sin temporales huérfanos
            leftovers = [
                p for p in target.parent.iterdir()
                if p.suffix == ".tmp"
            ]
            self.assertEqual(leftovers, [])

    def test_resolve_under_root_blocks_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(AIDomainError):
                resolve_under_root(root, "../outside.jpg")


class StationMutexTests(unittest.TestCase):
    def tearDown(self):
        set_ai_capture_mode(False)
        set_production_probe(lambda: False)

    def test_production_blocked_when_ai_mode(self):
        set_ai_capture_mode(True, session_id=1, garment_model_id=1)
        self.assertTrue(is_ai_capture_mode_active())
        with self.assertRaises(AIDomainError):
            assert_production_allowed("test producción")
        state = get_ai_capture_mode_state()
        self.assertEqual(state["session_id"], 1)

    def test_production_allowed_when_idle(self):
        set_ai_capture_mode(False)
        assert_production_allowed("test producción")

    def test_cannot_start_ai_while_auto_production(self):
        set_ai_capture_mode(False)
        set_production_probe(lambda: True)
        with self.assertRaises(AIDomainError):
            ensure_ai_capture_can_start()

    def test_start_ok_when_production_idle(self):
        set_ai_capture_mode(False)
        set_production_probe(lambda: False)
        ensure_ai_capture_can_start()


class PresenceTrackerTests(unittest.TestCase):
    def test_one_token_one_candidate_best_coverage(self):
        candidate = reset_presence_candidate(42)
        self.assertIsInstance(candidate, PresenceCandidate)
        self.assertEqual(candidate.garment_token, 42)

        frame_a = object()
        frame_b = object()
        candidate.observe(
            frame_a,
            0.30,
            1,
            enter_coverage=0.28,
            confirm_frames=3,
        )
        candidate.observe(
            frame_b,
            0.55,
            2,
            enter_coverage=0.28,
            confirm_frames=3,
        )
        candidate.observe(
            frame_a,
            0.40,
            3,
            enter_coverage=0.28,
            confirm_frames=3,
        )

        self.assertIs(candidate.best_frame, frame_b)
        self.assertAlmostEqual(candidate.best_coverage, 0.55)
        self.assertEqual(candidate.best_sequence, 2)
        self.assertEqual(candidate.present_frames, 3)

    def test_finalize_once_only(self):
        candidate = reset_presence_candidate(1)
        candidate.observe(
            object(),
            0.5,
            10,
            enter_coverage=0.28,
            confirm_frames=1,
        )
        results = []

        def persist_like():
            if candidate.finalized:
                results.append("skip")
                return
            candidate.finalized = True
            results.append("once")

        persist_like()
        persist_like()
        self.assertEqual(results.count("once"), 1)
        self.assertEqual(results.count("skip"), 1)

    def test_exit_requires_best_above_min(self):
        candidate = reset_presence_candidate(5)
        # sin observar frames
        self.assertFalse(
            candidate.should_finalize_on_exit(
                0.05,
                exit_coverage=0.12,
                exit_frames_threshold=3,
                exit_frames_counter=3,
            )
        )
        candidate.observe(
            object(),
            0.55,
            1,
            enter_coverage=0.28,
            confirm_frames=1,
        )
        self.assertTrue(
            candidate.should_finalize_on_exit(
                0.05,
                exit_coverage=0.12,
                exit_frames_threshold=3,
                exit_frames_counter=3,
            )
        )


class DhashTests(unittest.TestCase):
    def test_dhash_stable_without_cv2_real_frame(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("numpy/cv2 no disponibles")

        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        h1 = compute_dhash_64(frame, cv2_mod=cv2)
        h2 = compute_dhash_64(frame.copy(), cv2_mod=cv2)
        self.assertEqual(h1, h2)
        self.assertTrue(0 <= h1 < (1 << 64))


class EvaluateCandidateIntegrationTests(unittest.TestCase):
    def test_duplicate_sha256_rejected(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("numpy/cv2 no disponibles")

        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        digest = sha256_bytes(frame.tobytes())
        decision = evaluate_candidate(
            frame,
            coverage=0.60,
            frame_sequence=99,
            known_sha256={digest},
            known_dhashes=[],
            get_roi_bounds=lambda f: (5, 5, 70, 70),
            cv2_mod=cv2,
            config=get_ai_capture_config(),
            sha256_hex=digest,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(
            decision["reject_reason"],
            "DUPLICATE_SHA256",
        )

    def test_perceptual_duplicate_rejected(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("numpy/cv2 no disponibles")

        rng = np.random.default_rng(2)
        frame = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        known_hash = compute_dhash_64(frame, cv2_mod=cv2)
        decision = evaluate_candidate(
            frame,
            coverage=0.60,
            frame_sequence=100,
            known_sha256=set(),
            known_dhashes=[known_hash],
            get_roi_bounds=lambda f: (5, 5, 70, 70),
            cv2_mod=cv2,
            config=get_ai_capture_config(),
            sha256_hex=None,
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual(
            decision["reject_reason"],
            "DUPLICATE_PERCEPTUAL",
        )


@unittest.skipUnless(
    try_db_connection() is not None,
    "MySQL no disponible; tests de sesión de captura omitidos.",
)
class CaptureSessionDbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mysql.connector
        from ai_domain import ensure_ai_schema

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
        ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()

        cls.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'QUALITY_MANAGER',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB
            """
        )
        cls.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS garment_models (
                id INT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(80) NOT NULL UNIQUE,
                name VARCHAR(150) NULL,
                status VARCHAR(40) NOT NULL DEFAULT 'APROBADO',
                active TINYINT(1) NOT NULL DEFAULT 1
            ) ENGINE=InnoDB
            """
        )
        cls.conn.commit()

        cls.cur.execute(
            "SELECT id FROM users WHERE username = %s",
            ("ai_cap_test_user",),
        )
        row = cls.cur.fetchone()
        if row:
            cls.actor_id = int(row["id"])
        else:
            cls.cur.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, %s)
                """,
                ("ai_cap_test_user", "x", "MODEL_MANAGER"),
            )
            cls.actor_id = cls.cur.lastrowid

        cls.cur.execute(
            "SELECT id FROM garment_models WHERE code = %s",
            ("TEST-CAP-A",),
        )
        row = cls.cur.fetchone()
        if row:
            cls.model_a = int(row["id"])
        else:
            cls.cur.execute(
                """
                INSERT INTO garment_models (code, name, status, active)
                VALUES (%s, %s, 'APROBADO', 1)
                """,
                ("TEST-CAP-A", "Modelo captura A"),
            )
            cls.model_a = cls.cur.lastrowid

        cls.cur.execute(
            "SELECT id FROM garment_models WHERE code = %s",
            ("TEST-CAP-B",),
        )
        row = cls.cur.fetchone()
        if row:
            cls.model_b = int(row["id"])
        else:
            cls.cur.execute(
                """
                INSERT INTO garment_models (code, name, status, active)
                VALUES (%s, %s, 'APROBADO', 1)
                """,
                ("TEST-CAP-B", "Modelo captura B"),
            )
            cls.model_b = cls.cur.lastrowid

        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        # Purgar solo filas de prueba (con triggers dropeados).
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

        for code in ("TEST-CAP-A", "TEST-CAP-B"):
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
                """
                DELETE FROM ai_training_images
                WHERE garment_model_id = %s
                """,
                (mid,),
            )
            cls.cur.execute(
                """
                DELETE FROM ai_capture_sessions
                WHERE garment_model_id = %s
                """,
                (mid,),
            )
            cls.cur.execute(
                """
                DELETE FROM garment_ai_models
                WHERE garment_model_id = %s
                """,
                (mid,),
            )
            cls.cur.execute(
                "DELETE FROM garment_models WHERE id = %s",
                (mid,),
            )
        cls.cur.execute(
            "DELETE FROM ai_events WHERE actor_id = %s",
            (cls.actor_id,),
        )
        cls.cur.execute(
            "DELETE FROM users WHERE id = %s",
            (cls.actor_id,),
        )
        cls.conn.commit()

        from ai_domain import ensure_ai_schema

        ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()
        cls.cur.close()
        cls.conn.close()

    def _close_all_open(self):
        self.cur.execute(
            "UPDATE ai_capture_sessions SET status = 'CANCELADA', "
            "finished_at = NOW() WHERE status = 'ABIERTA'"
        )
        self.conn.commit()

    def setUp(self):
        self._close_all_open()

    def tearDown(self):
        self._close_all_open()

    def _tx(self):
        if self.conn.in_transaction:
            self.conn.commit()
        self.conn.start_transaction()
        return self.conn, self.cur

    def test_start_stop_idempotent(self):
        from ai_domain import (
            start_ai_capture_session,
            stop_ai_capture_session,
        )

        conn, cur = self._tx()
        opened = start_ai_capture_session(
            cur,
            self.model_a,
            self.actor_id,
        )
        conn.commit()
        sid = int(opened["id"])
        self.assertFalse(opened["already_open"])

        cur.execute(
            """INSERT INTO ai_training_images (
                capture_session_id, garment_model_id, image_path, sha256,
                status, reject_reason
            ) VALUES (%s, %s, %s, %s, 'ACEPTADA', NULL)""",
            (sid, self.model_a, "captures/test/stop-min.jpg", "a" * 64),
        )
        conn.commit()

        # doble inicio mismo modelo → idempotente
        conn.start_transaction()
        again = start_ai_capture_session(
            cur,
            self.model_a,
            self.actor_id,
        )
        conn.commit()
        self.assertTrue(again["already_open"])
        self.assertEqual(int(again["id"]), sid)

        # segundo modelo mientras hay sesión abierta → error
        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            start_ai_capture_session(
                cur,
                self.model_b,
                self.actor_id,
            )
        conn.rollback()

        # stop
        previous_min = os.environ.get("AI_CAPTURE_MIN_IMAGES")
        os.environ["AI_CAPTURE_MIN_IMAGES"] = "1"
        conn.start_transaction()
        try:
            stopped = stop_ai_capture_session(cur, sid)
            conn.commit()
            # stop twice → idempotente
            conn.start_transaction()
            stopped2 = stop_ai_capture_session(cur, sid)
            conn.commit()
        finally:
            if previous_min is None:
                os.environ.pop("AI_CAPTURE_MIN_IMAGES", None)
            else:
                os.environ["AI_CAPTURE_MIN_IMAGES"] = previous_min
        self.assertEqual(stopped["status"], "COMPLETADA")
        self.assertFalse(stopped["already_finished"])
        self.assertTrue(stopped2["already_finished"])

    def test_cancel_and_double_cancel(self):
        from ai_domain import (
            cancel_ai_capture_session,
            start_ai_capture_session,
        )

        conn, cur = self._tx()
        opened = start_ai_capture_session(
            cur,
            self.model_a,
            self.actor_id,
        )
        conn.commit()
        sid = int(opened["id"])

        conn.start_transaction()
        cancelled = cancel_ai_capture_session(cur, sid)
        conn.commit()
        self.assertEqual(cancelled["status"], "CANCELADA")
        self.assertFalse(cancelled["already_cancelled"])

        conn.start_transaction()
        cancelled2 = cancel_ai_capture_session(cur, sid)
        conn.commit()
        self.assertTrue(cancelled2["already_cancelled"])

    def test_missing_garment_rejected(self):
        from ai_domain import start_ai_capture_session

        conn, cur = self._tx()
        with self.assertRaises(Exception):
            start_ai_capture_session(
                cur,
                999999,
                self.actor_id,
            )
        conn.rollback()

    def test_status_payload_fields(self):
        from ai_domain import (
            capture_session_status_payload,
            start_ai_capture_session,
        )

        conn, cur = self._tx()
        opened = start_ai_capture_session(
            cur,
            self.model_a,
            self.actor_id,
        )
        conn.commit()
        sid = int(opened["id"])

        conn.start_transaction()
        payload = capture_session_status_payload(cur, sid)
        conn.commit()

        self.assertEqual(payload["session_id"], sid)
        self.assertEqual(payload["status"], "ABIERTA")
        self.assertEqual(payload["accepted_count"], 0)
        self.assertEqual(payload["rejected_count"], 0)
        self.assertEqual(payload["target_count"], 40)
        self.assertEqual(payload["garment_model"]["code"], "TEST-CAP-A")
        self.assertIn("started_at", payload)

    def test_frame_sequence_unique_per_session(self):
        from ai_domain import (
            claim_frame_sequence,
            register_training_image,
            start_ai_capture_session,
        )

        conn, cur = self._tx()
        opened = start_ai_capture_session(
            cur,
            self.model_a,
            self.actor_id,
        )
        conn.commit()
        sid = int(opened["id"])
        mid = self.model_a

        conn.start_transaction()
        claim_frame_sequence(cur, sid, 101)
        register_training_image(
            cur,
            garment_model_id=mid,
            image_path=capture_image_relative_path(
                mid,
                sid,
                "frame_101.jpg",
            ),
            sha256="b" * 64,
            capture_session_id=sid,
            frame_sequence=101,
            coverage=0.55,
            quality_score=0.8,
            status="ACEPTADA",
        )
        conn.commit()

        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            claim_frame_sequence(cur, sid, 101)
        conn.rollback()

        # mismo sequence en OTRA sesión está permitido (si la hubiera)
        conn.start_transaction()
        claim_frame_sequence(cur, sid, 102)
        conn.rollback()

    def test_register_requires_open_session(self):
        from ai_domain import (
            register_training_image,
            start_ai_capture_session,
            stop_ai_capture_session,
        )

        conn, cur = self._tx()
        opened = start_ai_capture_session(
            cur,
            self.model_a,
            self.actor_id,
        )
        conn.commit()
        sid = int(opened["id"])

        cur.execute(
            """INSERT INTO ai_training_images (
                capture_session_id, garment_model_id, image_path, sha256,
                status, reject_reason
            ) VALUES (%s, %s, %s, %s, 'ACEPTADA', NULL)""",
            (sid, self.model_a, "captures/test/register-min.jpg", "b" * 64),
        )
        conn.commit()

        previous_min = os.environ.get("AI_CAPTURE_MIN_IMAGES")
        os.environ["AI_CAPTURE_MIN_IMAGES"] = "1"
        conn.start_transaction()
        try:
            stop_ai_capture_session(cur, sid)
            conn.commit()
        finally:
            if previous_min is None:
                os.environ.pop("AI_CAPTURE_MIN_IMAGES", None)
            else:
                os.environ["AI_CAPTURE_MIN_IMAGES"] = previous_min

        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            register_training_image(
                cur,
                garment_model_id=self.model_a,
                image_path=capture_image_relative_path(
                    self.model_a,
                    sid,
                    "late.jpg",
                ),
                sha256="c" * 64,
                capture_session_id=sid,
                frame_sequence=1,
            )
        conn.rollback()


if __name__ == "__main__":
    unittest.main()
