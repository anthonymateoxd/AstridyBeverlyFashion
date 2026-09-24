"""Tests de dominio y persistencia — Fases 1 / 1.1 (sin PatchCore).

Los tests puros siempre se ejecutan. Los tests MySQL se omiten si la
base del proyecto no está disponible.
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

import ai_domain
from ai_domain import (
    AI_MODEL_STATUS_ACTIVO,
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_PREPARACION,
    AI_MODEL_STATUS_RETIRADO,
    AI_MODEL_STATUS_VALIDADO,
    AI_MODEL_STATUSES,
    AIDomainError,
    JOB_KINDS,
    JOB_STATUSES,
    add_images_to_dataset,
    build_garment_artifact_dir,
    close_ai_dataset,
    compute_manifest_hash,
    compute_manifest_path,
    create_ai_dataset,
    create_ai_job,
    create_next_ai_model_version,
    ensure_safe_relative_path,
    is_valid_sha256,
    next_dataset_version_label,
    next_version_label,
    normalize_sha256,
    open_ai_capture_session,
    record_ai_event,
    register_training_image,
    resolve_under_root,
    sanitize_event_payload,
    transition_ai_model_status,
    transition_capture_session_status,
    validate_ai_status_transition,
    validate_dataset_status_transition,
    validate_job_kind,
)

AI_TRIGGER_NAMES = (
    "trg_ai_dataset_images_before_insert",
    "trg_ai_dataset_images_before_update",
    "trg_ai_dataset_images_before_delete",
    "trg_ai_datasets_before_update",
    "trg_ai_training_images_before_update",
    "trg_ai_training_images_before_delete",
)


def try_db_connection():
    import mysql.connector

    try:
        conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=os.environ.get(
                "MYSQL_DATABASE",
                "textile_quality_db",
            ),
            connection_timeout=3,
        )
        return conn
    except Exception:
        return None


class PureDomainTests(unittest.TestCase):
    def test_next_version_starts_at_v1(self):
        self.assertEqual(next_version_label([]), "v1")
        self.assertEqual(next_version_label(None), "v1")

    def test_next_version_increments(self):
        rows = [{"version": "v1"}, {"version": "v3"}, {"version": "x"}]
        self.assertEqual(next_version_label(rows), "v4")

    def test_next_dataset_version(self):
        self.assertEqual(next_dataset_version_label([]), "d1")
        self.assertEqual(
            next_dataset_version_label([{"version": "d2"}]),
            "d3",
        )

    def test_valid_transition_chain(self):
        validate_ai_status_transition(
            AI_MODEL_STATUS_PREPARACION,
            "ENTRENANDO",
        )
        validate_ai_status_transition("ENTRENANDO", "ENTRENADO")
        validate_ai_status_transition("ENTRENADO", "VALIDACION")
        validate_ai_status_transition("VALIDACION", "VALIDADO")
        validate_ai_status_transition("VALIDADO", "ACTIVO")
        validate_ai_status_transition("ACTIVO", "RETIRADO")
        validate_ai_status_transition("RETIRADO", "ACTIVO")

    def test_invalid_transition_rejected(self):
        with self.assertRaises(AIDomainError):
            validate_ai_status_transition(
                AI_MODEL_STATUS_PREPARACION,
                AI_MODEL_STATUS_ACTIVO,
            )

    def test_active_without_validated_rejected(self):
        for status in (
            AI_MODEL_STATUS_PREPARACION,
            AI_MODEL_STATUS_ENTRENADO,
            "ENTRENANDO",
            "VALIDACION",
        ):
            with self.assertRaises(AIDomainError):
                validate_ai_status_transition(
                    status,
                    AI_MODEL_STATUS_ACTIVO,
                )

    def test_same_status_rejected(self):
        with self.assertRaises(AIDomainError):
            validate_ai_status_transition("VALIDADO", "VALIDADO")

    def test_terminal_statuses_have_no_exits(self):
        for status in ("RECHAZADO", "FALLIDO"):
            self.assertEqual(
                ai_domain.AI_MODEL_TRANSITIONS[status],
                set(),
            )

    def test_status_sets_complete(self):
        expected = {
            "PREPARACION",
            "ENTRENANDO",
            "ENTRENADO",
            "VALIDACION",
            "VALIDADO",
            "ACTIVO",
            "RETIRADO",
            "RECHAZADO",
            "FALLIDO",
        }
        self.assertEqual(AI_MODEL_STATUSES, expected)
        self.assertEqual(set(ai_domain.AI_MODEL_TRANSITIONS), expected)

    def test_job_kinds_and_statuses(self):
        self.assertEqual(JOB_KINDS, {"TRAINING", "VALIDATION"})
        self.assertEqual(
            JOB_STATUSES,
            {
                "PENDIENTE",
                "EN_CURSO",
                "COMPLETADO",
                "FALLIDO",
                "CANCELADO",
            },
        )
        self.assertEqual(validate_job_kind("training"), "TRAINING")
        with self.assertRaises(AIDomainError):
            validate_job_kind("BACKFILL")

    def test_dataset_transition_rules(self):
        validate_dataset_status_transition("ABIERTO", "CERRADO")
        validate_dataset_status_transition("CERRADO", "ARCHIVADO")
        with self.assertRaises(AIDomainError):
            validate_dataset_status_transition("CERRADO", "ABIERTO")
        with self.assertRaises(AIDomainError):
            validate_dataset_status_transition("ARCHIVADO", "CERRADO")

    def test_sha256_validation(self):
        digest = "a" * 64
        self.assertTrue(is_valid_sha256(digest))
        self.assertEqual(normalize_sha256(digest.upper()), digest)
        self.assertFalse(is_valid_sha256("nope"))
        self.assertFalse(is_valid_sha256("a" * 63))
        with self.assertRaises(AIDomainError):
            normalize_sha256("short")

    def test_relative_path_blocks_traversal(self):
        self.assertEqual(
            ensure_safe_relative_path("garment_1/datasets/a.jpg"),
            "garment_1/datasets/a.jpg",
        )
        blocked = (
            "",
            "/etc/passwd",
            "../secret",
            "..\\secret",
            "a/../../b",
            "a\\..\\..\\b",
            "C:/windows/system32",
            "C:\\windows",
            "\\\\server\\share\\x",
            "/var/tmp/x",
            "a\x00b",
            "a\nb",
        )
        for bad in blocked:
            with self.assertRaises(
                AIDomainError,
                msg=f"debió rechazar: {bad!r}",
            ):
                ensure_safe_relative_path(bad)

    def test_resolve_under_root_blocks_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            inside = resolve_under_root(
                root,
                "garment_1/checkpoints/x.ckpt",
            )
            self.assertTrue(str(inside).startswith(str(root.resolve())))
            for bad in (
                "../outside.txt",
                "..\\outside.txt",
                "/etc/passwd",
                "C:/windows/win.ini",
            ):
                with self.assertRaises(AIDomainError):
                    resolve_under_root(root, bad)

    def test_resolve_under_root_blocks_symlink_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            target = outside / "secret.txt"
            target.write_text("x", encoding="utf-8")

            link = root / "escape"
            try:
                link.symlink_to(str(target))
            except OSError:
                self.skipTest("symlinks no disponibles sin privilegios")

            with self.assertRaises(AIDomainError):
                resolve_under_root(root, "escape")

    def test_build_garment_artifact_dir(self):
        with TemporaryDirectory() as tmp:
            path = build_garment_artifact_dir(7, root=Path(tmp))
            self.assertEqual(path.name, "garment_7")
        with self.assertRaises(AIDomainError):
            build_garment_artifact_dir(0)
        with self.assertRaises(AIDomainError):
            build_garment_artifact_dir("abc")

    def test_manifest_hash_deterministic(self):
        rows = [(2, "B" * 64), (1, "A" * 64)]
        again = [(1, "A" * 64), (2, "B" * 64)]
        self.assertEqual(
            compute_manifest_hash(rows),
            compute_manifest_hash(again),
        )
        self.assertEqual(len(compute_manifest_hash(rows)), 64)

    def test_manifest_path_safe(self):
        path = compute_manifest_path(3, 9)
        self.assertEqual(
            path,
            "garment_3/datasets/dataset_9/manifest.json",
        )
        with self.assertRaises(AIDomainError):
            compute_manifest_path(0, 9)

    def test_payload_blocks_secrets(self):
        self.assertIsNone(sanitize_event_payload(None))
        cleaned = sanitize_event_payload({"note": "ok", "count": 2})
        self.assertEqual(cleaned["note"], "ok")
        for bad in (
            {"password": "x"},
            {"nested": {"api_key": "x"}},
            {"token": "t"},
        ):
            with self.assertRaises(AIDomainError):
                sanitize_event_payload(bad)
        with self.assertRaises(AIDomainError):
            sanitize_event_payload(["not", "a", "dict"])


@unittest.skipUnless(
    try_db_connection() is not None,
    "MySQL no disponible; tests de persistencia omitidos.",
)
class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mysql.connector

        cls.database = os.environ.get(
            "MYSQL_DATABASE",
            "textile_quality_db",
        )
        cls.conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=cls.database,
        )
        cls.cur = cls.conn.cursor(dictionary=True)
        cls.plain_cur = cls.conn.cursor()

        base = cls.plain_cur
        base.execute(
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
        base.execute(
            """
            CREATE TABLE IF NOT EXISTS garment_models (
                id INT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(80) NOT NULL,
                name VARCHAR(150) NOT NULL,
                garment_type VARCHAR(100) NOT NULL DEFAULT 'Blusa',
                color VARCHAR(80) NULL,
                size VARCHAR(20) NOT NULL DEFAULT 'S',
                inspection_side VARCHAR(30) NOT NULL DEFAULT 'Frente',
                description TEXT NULL,
                status VARCHAR(30) NOT NULL DEFAULT 'BORRADOR',
                created_by INT NULL,
                approved_by INT NULL,
                rejection_reason TEXT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
                approved_at DATETIME NULL,
                active TINYINT(1) NOT NULL DEFAULT 1,
                UNIQUE KEY uq_garment_model_code (code)
            ) ENGINE=InnoDB
            """
        )
        base.execute(
            """
            CREATE TABLE IF NOT EXISTS garment_ai_models (
                id INT AUTO_INCREMENT PRIMARY KEY,
                garment_model_id INT NOT NULL,
                version VARCHAR(40) NOT NULL,
                model_type VARCHAR(50) NOT NULL DEFAULT 'PatchCore',
                dataset_name VARCHAR(180) NULL,
                dataset_path VARCHAR(500) NULL,
                checkpoint_path VARCHAR(500) NULL,
                status VARCHAR(30) NOT NULL DEFAULT 'PREPARACION',
                normal_images_count INT NULL,
                metrics_json JSON NULL,
                notes TEXT NULL,
                created_by INT NULL,
                trained_by INT NULL,
                validated_by INT NULL,
                activated_by INT NULL,
                trained_at DATETIME NULL,
                validated_at DATETIME NULL,
                activated_at DATETIME NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                active TINYINT(1) NOT NULL DEFAULT 0,
                UNIQUE KEY uq_garment_ai_version (
                    garment_model_id,
                    version
                )
            ) ENGINE=InnoDB
            """
        )
        cls.conn.commit()
        base.close()

        ai_domain.ensure_ai_schema(cls.cur, cls.database)
        cls.conn.commit()

        suffix = f"{os.getpid()}_{threading.get_ident()}"
        cls.cur.execute(
            """
            INSERT INTO garment_models (code, name, status, active)
            VALUES (%s, %s, 'APROBADO', 1)
            """,
            (f"TEST-AI-{suffix}", "Modelo Fase 1"),
        )
        cls.model_id_a = cls.cur.lastrowid

        cls.cur.execute(
            """
            INSERT INTO garment_models (code, name, status, active)
            VALUES (%s, %s, 'APROBADO', 1)
            """,
            (f"TEST-AI-B-{suffix}", "Modelo Fase 1 B"),
        )
        cls.model_id_b = cls.cur.lastrowid
        cls.conn.commit()

        # Compat con tests que aún usan model_id.
        cls.model_id = cls.model_id_a

        cls.cur.execute(
            "SELECT id FROM users ORDER BY id ASC LIMIT 1"
        )
        user_row = cls.cur.fetchone()

        if user_row is None:
            cls.cur.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, 'ADMIN')
                """,
                (f"test_ai_{suffix}", "x"),
            )
            cls.conn.commit()
            cls.actor_id = cls.cur.lastrowid
        else:
            cls.actor_id = user_row["id"]

        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.conn.in_transaction:
                cls.conn.commit()

            # Los triggers de inmutabilidad bloquean el purge de datos
            # de prueba; se dropean solo aquí y se recrean al final.
            for name in AI_TRIGGER_NAMES:
                cls.cur.execute(f"DROP TRIGGER IF EXISTS {name}")
            cls.conn.commit()

            model_ids = [
                mid
                for mid in (
                    getattr(cls, "model_id_a", None),
                    getattr(cls, "model_id_b", None),
                )
                if mid
            ]

            for mid in model_ids:
                cls.cur.execute(
                    "DELETE FROM ai_events "
                    "WHERE ai_model_id IN ("
                    "  SELECT id FROM garment_ai_models "
                    "  WHERE garment_model_id = %s"
                    ") OR dataset_id IN ("
                    "  SELECT id FROM ai_datasets "
                    "  WHERE garment_model_id = %s"
                    ") OR capture_session_id IN ("
                    "  SELECT id FROM ai_capture_sessions "
                    "  WHERE garment_model_id = %s"
                    ")",
                    (mid, mid, mid),
                )
                cls.cur.execute(
                    """
                    DELETE di FROM ai_dataset_images di
                    JOIN ai_datasets d ON d.id = di.dataset_id
                    WHERE d.garment_model_id = %s
                    """,
                    (mid,),
                )
                cls.cur.execute(
                    "DELETE FROM ai_jobs "
                    "WHERE ai_model_id IN ("
                    "  SELECT id FROM garment_ai_models "
                    "  WHERE garment_model_id = %s"
                    ") OR dataset_id IN ("
                    "  SELECT id FROM ai_datasets "
                    "  WHERE garment_model_id = %s"
                    ")",
                    (mid, mid),
                )
                cls.cur.execute(
                    "DELETE FROM ai_datasets "
                    "WHERE garment_model_id = %s",
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

            cls.conn.commit()

            # Restaurar triggers de inmutabilidad.
            ai_domain.ensure_ai_schema(cls.cur, cls.database)
            cls.conn.commit()
        finally:
            cls.cur.close()
            cls.plain_cur.close()
            cls.conn.close()

    def _tx(self):
        if getattr(self.conn, "in_transaction", False):
            self.conn.commit()
        self.conn.start_transaction()
        return self.conn, self.cur

    def _promote_to_validado(self, version_id):
        conn, cur = self._tx()
        for status in (
            "ENTRENANDO",
            "ENTRENADO",
            "VALIDACION",
            "VALIDADO",
        ):
            transition_ai_model_status(
                cur,
                version_id,
                status,
                self.actor_id,
            )
            conn.commit()
            if status != "VALIDADO":
                conn.start_transaction()
        if conn.in_transaction:
            conn.commit()

    def _cleanup_ai_rows(self, model_id):
        conn, cur = self._tx()
        # Los triggers de inmutabilidad impiden purgar datasets cerrados.
        for name in AI_TRIGGER_NAMES:
            cur.execute(f"DROP TRIGGER IF EXISTS {name}")
        cur.execute(
            "DELETE FROM ai_events WHERE 1=1"
        )
        cur.execute(
            """
            DELETE di FROM ai_dataset_images di
            JOIN ai_datasets d ON d.id = di.dataset_id
            WHERE d.garment_model_id = %s
            """,
            (model_id,),
        )
        cur.execute(
            "DELETE FROM ai_jobs WHERE 1=1"
        )
        cur.execute(
            "DELETE FROM ai_datasets WHERE garment_model_id = %s",
            (model_id,),
        )
        cur.execute(
            "DELETE FROM ai_training_images "
            "WHERE garment_model_id = %s",
            (model_id,),
        )
        cur.execute(
            "DELETE FROM ai_capture_sessions "
            "WHERE garment_model_id = %s",
            (model_id,),
        )
        cur.execute(
            "DELETE FROM garment_ai_models "
            "WHERE garment_model_id = %s",
            (model_id,),
        )
        conn.commit()
        ai_domain.ensure_ai_schema(cur, os.environ.get("MYSQL_DATABASE"))
        conn.commit()

    # --------------------------------------------------------
    # A/B: create_next_ai_model_version
    # --------------------------------------------------------
    def test_create_next_version_and_duplicate_rejected(self):
        conn, cur = self._tx()
        first = create_next_ai_model_version(
            cur,
            self.model_id,
            self.actor_id,
            dataset_name="TEST_DS",
            dataset_path="ai_artifacts/test/TEST_DS",
        )
        conn.commit()
        self.assertEqual(first["version"], "v1")
        self.assertEqual(first["status"], "PREPARACION")

        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            create_next_ai_model_version(
                cur,
                self.model_id,
                self.actor_id,
            )
        conn.rollback()

        # B: segunda versión cuando no hay pendiente.
        cur.execute(
            "DELETE FROM garment_ai_models WHERE id = %s",
            (first["id"],),
        )
        conn.commit()

        conn.start_transaction()
        second = create_next_ai_model_version(
            cur,
            self.model_id,
            self.actor_id,
        )
        conn.commit()
        self.assertEqual(second["version"], "v1")

        # Violación de UNIQUE si se inserta versión duplicada a mano.
        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                """
                INSERT INTO garment_ai_models (
                    garment_model_id, version, status, active
                ) VALUES (%s, %s, 'PREPARACION', 0)
                """,
                (self.model_id, second["version"]),
            )
        conn.rollback()

        cur.execute(
            "DELETE FROM garment_ai_models WHERE id = %s",
            (second["id"],),
        )
        conn.commit()

    # --------------------------------------------------------
    # C/D: transiciones inválidas y VALIDATED -> ACTIVE
    # --------------------------------------------------------
    def test_transition_invalid_and_active_requires_validated(self):
        conn, cur = self._tx()
        version = create_next_ai_model_version(
            cur,
            self.model_id,
            self.actor_id,
        )
        conn.commit()

        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            transition_ai_model_status(
                cur,
                version["id"],
                "ACTIVO",
                self.actor_id,
            )
        conn.rollback()

        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            transition_ai_model_status(
                cur,
                version["id"],
                "VALIDADO",
                self.actor_id,
            )
        conn.rollback()

        self._promote_to_validado(version["id"])

        conn.start_transaction()
        row = transition_ai_model_status(
            cur,
            version["id"],
            "ACTIVO",
            self.actor_id,
        )
        conn.commit()
        self.assertEqual(row["status"], "ACTIVO")

        cur.execute(
            "SELECT status, active FROM garment_ai_models WHERE id = %s",
            (version["id"],),
        )
        stored = cur.fetchone()
        self.assertEqual(stored["status"], "ACTIVO")
        self.assertEqual(int(stored["active"]), 1)

        cur.execute(
            "DELETE FROM garment_ai_models WHERE id = %s",
            (version["id"],),
        )
        conn.commit()

    # --------------------------------------------------------
    # E: segundo ACTIVE rechazado / un solo ACTIVE
    # --------------------------------------------------------
    def test_single_active_constraint(self):
        conn, cur = self._tx()
        first = create_next_ai_model_version(
            cur,
            self.model_id,
            self.actor_id,
        )
        cur.execute(
            """
            INSERT INTO garment_ai_models (
                garment_model_id, version, status, active
            ) VALUES (%s, 'v99', 'VALIDADO', 0)
            """,
            (self.model_id,),
        )
        second_id = cur.lastrowid
        conn.commit()

        self._promote_to_validado(first["id"])

        conn.start_transaction()
        transition_ai_model_status(
            cur,
            first["id"],
            "ACTIVO",
            self.actor_id,
        )
        conn.commit()

        conn.start_transaction()
        transition_ai_model_status(
            cur,
            second_id,
            "ACTIVO",
            self.actor_id,
        )
        conn.commit()

        cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM garment_ai_models
            WHERE garment_model_id = %s AND status = 'ACTIVO'
            """,
            (self.model_id,),
        )
        self.assertEqual(int(cur.fetchone()["total"]), 1)

        cur.execute(
            "SELECT status, active FROM garment_ai_models WHERE id = %s",
            (first["id"],),
        )
        retired = cur.fetchone()
        self.assertEqual(retired["status"], "RETIRADO")
        self.assertEqual(int(retired["active"]), 0)

        # Segundo ACTIVE bruto (sin helper) debe chocar con el índice.
        if getattr(conn, "in_transaction", False):
            conn.commit()
        conn.start_transaction()
        try:
            cur.execute(
                """
                UPDATE garment_ai_models
                SET status = 'ACTIVO', active = 1
                WHERE id = %s
                """,
                (first["id"],),
            )
            conn.commit()
            self.fail("El índice único debió rechazar el segundo ACTIVE")
        except Exception:
            if getattr(conn, "in_transaction", False):
                conn.rollback()

        self._cleanup_ai_rows(self.model_id)

    def test_active_allowed_for_different_garment_models(self):
        conn, cur = self._tx()
        ver_a = create_next_ai_model_version(
            cur,
            self.model_id_a,
            self.actor_id,
        )
        ver_b = create_next_ai_model_version(
            cur,
            self.model_id_b,
            self.actor_id,
        )
        conn.commit()

        self._promote_to_validado(ver_a["id"])
        self._promote_to_validado(ver_b["id"])

        conn.start_transaction()
        transition_ai_model_status(
            cur,
            ver_a["id"],
            "ACTIVO",
            self.actor_id,
        )
        conn.commit()

        conn.start_transaction()
        transition_ai_model_status(
            cur,
            ver_b["id"],
            "ACTIVO",
            self.actor_id,
        )
        conn.commit()

        cur.execute(
            """
            SELECT garment_model_id, COUNT(*) AS c
            FROM garment_ai_models
            WHERE garment_model_id IN (%s, %s)
              AND status = 'ACTIVO'
            GROUP BY garment_model_id
            """,
            (self.model_id_a, self.model_id_b),
        )
        counts = {
            row["garment_model_id"]: int(row["c"])
            for row in cur.fetchall()
        }
        self.assertEqual(counts.get(self.model_id_a), 1)
        self.assertEqual(counts.get(self.model_id_b), 1)

        self._cleanup_ai_rows(self.model_id_a)
        self._cleanup_ai_rows(self.model_id_b)

    def test_conflict_blocks_activation_and_schema(self):
        """Duplicado histórico ACTIVE: schema falla y activación bloquea."""
        conn, cur = self._tx()
        ver = create_next_ai_model_version(
            cur,
            self.model_id_a,
            self.actor_id,
        )
        cur.execute(
            """
            INSERT INTO garment_ai_models (
                garment_model_id, version, status, active
            ) VALUES (%s, 'v88', 'VALIDADO', 0)
            """,
            (self.model_id_a,),
        )
        conflict_b = cur.lastrowid
        conn.commit()
        self._promote_to_validado(ver["id"])

        # Simular histórico corrupto: sin índice único.
        conn.start_transaction()
        cur.execute(
            "DROP INDEX uq_garment_ai_one_active "
            "ON garment_ai_models"
        )
        conn.commit()

        try:
            conn.start_transaction()
            cur.execute(
                """
                UPDATE garment_ai_models
                SET status = 'ACTIVO', active = 1
                WHERE id IN (%s, %s)
                """,
                (ver["id"], conflict_b),
            )
            conn.commit()

            # ensure_ai_schema debe fallar de forma explícita.
            with self.assertRaises(AIDomainError) as ctx:
                ai_domain.ensure_ai_schema(cur, self.database)
            self.assertIn("CONFLICTO ACTIVE", str(ctx.exception))
            if conn.in_transaction:
                conn.rollback()

            # Nueva activación de otra versión del mismo modelo: bloqueada.
            cur.execute(
                """
                INSERT INTO garment_ai_models (
                    garment_model_id, version, status, active
                ) VALUES (%s, 'v77', 'VALIDADO', 0)
                """,
                (self.model_id_a,),
            )
            third = cur.lastrowid
            conn.commit()

            conn.start_transaction()
            with self.assertRaises(AIDomainError):
                transition_ai_model_status(
                    cur,
                    third,
                    "ACTIVO",
                    self.actor_id,
                )
            conn.rollback()

            # Modelo distinto sigue poder activar (sin conflicto).
            ver_other = create_next_ai_model_version(
                cur,
                self.model_id_b,
                self.actor_id,
            )
            conn.commit()
            self._promote_to_validado(ver_other["id"])
            conn.start_transaction()
            transition_ai_model_status(
                cur,
                ver_other["id"],
                "ACTIVO",
                self.actor_id,
            )
            conn.commit()
        finally:
            # Restaurar unicidad para el resto de la suite.
            conn.start_transaction()
            cur.execute(
                """
                UPDATE garment_ai_models
                SET status = 'RETIRADO', active = 0
                WHERE garment_model_id = %s
                  AND status = 'ACTIVO'
                  AND id <> (
                      SELECT * FROM (
                          SELECT id FROM garment_ai_models
                          WHERE garment_model_id = %s
                            AND status = 'ACTIVO'
                          ORDER BY id ASC LIMIT 1
                      ) AS keep_one
                  )
                """,
                (self.model_id_a, self.model_id_a),
            )
            conn.commit()

            # Si aún hay 2 ACTIVO, retirar todos los de prueba.
            cur.execute(
                """
                SELECT id FROM garment_ai_models
                WHERE garment_model_id IN (%s, %s)
                """,
                (self.model_id_a, self.model_id_b),
            )
            ids = [r["id"] for r in cur.fetchall()]
            if ids:
                placeholders = ",".join(["%s"] * len(ids))
                cur.execute(
                    f"""
                    UPDATE garment_ai_models
                    SET status = 'RETIRADO', active = 0
                    WHERE id IN ({placeholders})
                    """,
                    tuple(ids),
                )
                conn.commit()

            # Si quedan 0 o 1 ACTIVO por modelo, recrear índice.
            ai_domain.ensure_ai_schema(cur, self.database)
            conn.commit()

            # Un solo ACTIVO puede quedar en cada modelo de prueba.
            cur.execute(
                """
                UPDATE garment_ai_models
                SET status = 'RETIRADO', active = 0
                WHERE garment_model_id IN (%s, %s)
                """,
                (self.model_id_a, self.model_id_b),
            )
            conn.commit()

            # Asegurar índice aunque haya sobrantes retirados.
            ai_domain.ensure_ai_schema(cur, self.database)
            conn.commit()

            self._cleanup_ai_rows(self.model_id_a)
            self._cleanup_ai_rows(self.model_id_b)

    def test_concurrent_create_version_single_winner(self):
        """Dos hilos creando versión: solo uno obtiene la nueva."""
        errors = []
        results = []
        barrier = threading.Barrier(2)

        def worker():
            import mysql.connector

            try:
                local = mysql.connector.connect(
                    host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
                    port=int(os.environ.get("MYSQL_PORT", "3306")),
                    user=os.environ.get("MYSQL_USER", "root"),
                    password=os.environ.get("MYSQL_PASSWORD", ""),
                    database=self.database,
                )
                cur = local.cursor(dictionary=True)
                barrier.wait(timeout=10)
                local.start_transaction()
                try:
                    row = create_next_ai_model_version(
                        cur,
                        self.model_id_a,
                        self.actor_id,
                    )
                    local.commit()
                    results.append(row)
                except Exception as error:
                    local.rollback()
                    errors.append(error)
                finally:
                    cur.close()
                    local.close()
            except Exception as error:  # pragma: no cover
                errors.append(error)

        threads = [
            threading.Thread(target=worker),
            threading.Thread(target=worker),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(len(results), 1, f"results={results}")
        self.assertEqual(len(errors), 1, f"errors={errors}")
        self._cleanup_ai_rows(self.model_id_a)

    # --------------------------------------------------------
    # F/G/H/I: datasets OPEN/CLOSED e immutabilidad
    # --------------------------------------------------------
    def test_dataset_open_modifiable_closed_immutable(self):
        conn, cur = self._tx()
        session = open_ai_capture_session(
            cur,
            self.model_id,
            self.actor_id,
        )
        image = register_training_image(
            cur,
            garment_model_id=self.model_id,
            capture_session_id=session["id"],
            image_path="garment_test/sessions/1/open.jpg",
            sha256="D" * 64,
        )
        dataset = create_ai_dataset(
            cur,
            self.model_id,
            self.actor_id,
        )
        add_images_to_dataset(cur, dataset["id"], [image["id"]])
        conn.commit()

        # F: OPEN modificable — se aceptan más imágenes.
        image2 = register_training_image(
            cur,
            garment_model_id=self.model_id,
            capture_session_id=session["id"],
            image_path="garment_test/sessions/1/open2.jpg",
            sha256="E" * 64,
        )
        add_images_to_dataset(cur, dataset["id"], [image2["id"]])
        conn.commit()

        with TemporaryDirectory() as tmp:
            closed = close_ai_dataset(
                cur,
                dataset["id"],
                self.actor_id,
                artifacts_root=Path(tmp),
            )
        conn.commit()
        self.assertEqual(closed["status"], "CERRADO")

        # G: CLOSED no acepta nuevas imágenes (app + trigger).
        image3 = register_training_image(
            cur,
            garment_model_id=self.model_id,
            capture_session_id=session["id"],
            image_path="garment_test/sessions/1/late.jpg",
            sha256="F" * 64,
        )
        conn.commit()

        conn.start_transaction()
        with self.assertRaises(Exception):
            add_images_to_dataset(
                cur,
                dataset["id"],
                [image3["id"]],
            )
        conn.rollback()

        # H: membership no modificable (trigger en ai_dataset_images).
        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                "DELETE FROM ai_dataset_images "
                "WHERE dataset_id = %s AND image_id = %s",
                (dataset["id"], image["id"]),
            )
        conn.rollback()

        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                "UPDATE ai_dataset_images SET image_id = %s "
                "WHERE dataset_id = %s AND image_id = %s",
                (image2["id"], dataset["id"], image["id"]),
            )
        conn.rollback()

        # I: imagen en CLOSED no se altera ni se elimina.
        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                "UPDATE ai_training_images SET sha256 = %s "
                "WHERE id = %s",
                ("A" * 64, image["id"]),
            )
        conn.rollback()

        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                "UPDATE ai_training_images SET image_path = %s "
                "WHERE id = %s",
                ("garment_test/hacked.jpg", image["id"]),
            )
        conn.rollback()

        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                "DELETE FROM ai_training_images WHERE id = %s",
                (image["id"],),
            )
        conn.rollback()

        # Imagen fuera de CLOSED sí es editable (status libre).
        conn.start_transaction()
        cur.execute(
            "UPDATE ai_training_images SET quality_score = 55.5 "
            "WHERE id = %s",
            (image3["id"],),
        )
        conn.commit()

        # Dataset CLOSED no se reabre.
        conn.start_transaction()
        with self.assertRaises(Exception):
            cur.execute(
                "UPDATE ai_datasets SET status = 'ABIERTO' "
                "WHERE id = %s",
                (dataset["id"],),
            )
        conn.rollback()

        # Snapshot: manifest_hash estable tras intentos de mutación.
        cur.execute(
            "SELECT manifest_hash, status FROM ai_datasets WHERE id = %s",
            (dataset["id"],),
        )
        snap = cur.fetchone()
        self.assertEqual(snap["status"], "CERRADO")
        self.assertEqual(snap["manifest_hash"], closed["manifest_hash"])

        transition_capture_session_status(
            cur,
            session["id"],
            "COMPLETADA",
        )
        conn.commit()

        counts = ai_domain.count_session_images(cur, session["id"])
        self.assertEqual(counts["accepted_images"], 3)

        # Limpieza al final de la clase vía drop de triggers.

    # --------------------------------------------------------
    # J: ai_job
    # --------------------------------------------------------
    def test_create_ai_job_and_statuses(self):
        conn, cur = self._tx()

        with self.assertRaises(AIDomainError):
            create_ai_job(cur, "TRAINING")

        version = create_next_ai_model_version(
            cur,
            self.model_id,
            self.actor_id,
        )
        job = create_ai_job(
            cur,
            "VALIDATION",
            ai_model_id=version["id"],
        )
        self.assertEqual(job["status"], "PENDIENTE")
        self.assertEqual(job["kind"], "VALIDATION")
        conn.commit()

        conn.start_transaction()
        ai_domain.transition_ai_job_status(
            cur,
            job["id"],
            "EN_CURSO",
            progress=10,
        )
        conn.commit()

        conn.start_transaction()
        with self.assertRaises(AIDomainError):
            ai_domain.transition_ai_job_status(
                cur,
                job["id"],
                "PENDIENTE",
            )
        conn.rollback()

        cur.execute("DELETE FROM ai_jobs WHERE id = %s", (job["id"],))
        cur.execute(
            "DELETE FROM garment_ai_models WHERE id = %s",
            (version["id"],),
        )
        conn.commit()

    def test_job_training_requires_dataset(self):
        conn, cur = self._tx()
        version = create_next_ai_model_version(
            cur,
            self.model_id,
            self.actor_id,
        )
        with self.assertRaises(AIDomainError):
            create_ai_job(
                cur,
                "TRAINING",
                ai_model_id=version["id"],
            )
        conn.rollback()
        cur.execute(
            "DELETE FROM garment_ai_models WHERE id = %s",
            (version["id"],),
        )
        conn.commit()

    def test_event_payload_rejects_secrets(self):
        conn, cur = self._tx()
        with self.assertRaises(AIDomainError):
            record_ai_event(
                cur,
                "VERSION_PREPARED",
                actor_id=self.actor_id,
                payload={"password": "leak"},
            )
        conn.rollback()

        conn.start_transaction()
        event = record_ai_event(
            cur,
            "VERSION_PREPARED",
            actor_id=self.actor_id,
            payload={"note": "ok"},
        )
        conn.commit()
        self.assertGreater(event["id"], 0)

        cur.execute(
            "DELETE FROM ai_events WHERE id = %s",
            (event["id"],),
        )
        conn.commit()

    # --------------------------------------------------------
    # K/L: init_db idempotente (triggers/indexes no se duplican)
    # --------------------------------------------------------
    def test_ensure_schema_idempotent_triggers_and_index(self):
        conn, cur = self._tx()
        if conn.in_transaction:
            conn.commit()

        ai_domain.ensure_ai_schema(cur, self.database)
        conn.commit()

        cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM information_schema.TRIGGERS
            WHERE TRIGGER_SCHEMA = %s
              AND TRIGGER_NAME IN (
                'trg_ai_dataset_images_before_insert',
                'trg_ai_dataset_images_before_update',
                'trg_ai_dataset_images_before_delete',
                'trg_ai_datasets_before_update',
                'trg_ai_training_images_before_update',
                'trg_ai_training_images_before_delete'
              )
            """,
            (self.database,),
        )
        self.assertEqual(int(cur.fetchone()["total"]), 6)

        cur.execute(
            """
            SELECT COUNT(DISTINCT INDEX_NAME) AS total
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'garment_ai_models'
              AND INDEX_NAME = 'uq_garment_ai_one_active'
            """,
            (self.database,),
        )
        self.assertEqual(int(cur.fetchone()["total"]), 1)

        # Segunda corrida no debe fallar ni duplicar.
        ai_domain.ensure_ai_schema(cur, self.database)
        conn.commit()


if __name__ == "__main__":
    unittest.main()
