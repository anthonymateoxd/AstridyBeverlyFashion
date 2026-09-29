from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
    Response,
    send_file,
)
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import quote
from patchcore_inference import PatchCoreInspector
import os
import base64
import time
import uuid
import json
import threading
from functools import wraps
import io
from zoneinfo import ZoneInfo

import mysql.connector

from ai_domain import (
    AIDomainError,
    CAPTURE_UI_STATE_LABELS,
    atomic_write_bytes,
    capture_image_relative_path,
    capture_progress_gate,
    capture_session_status_payload,
    cancel_ai_capture_session,
    claim_frame_sequence,
    count_session_images,
    ensure_ai_schema,
    ensure_capture_session_dirs,
    find_accepted_sha256,
    get_ai_capture_config,
    get_capture_session,
    get_open_capture_session,
    humanize_capture_error,
    humanize_reject_reason,
    is_technically_invalidated,
    load_session_accepted_hashes,
    next_garment_model_code,
    record_ai_event,
    register_training_image,
    resolve_capture_ui_state,
    resolve_under_root,
    sha256_bytes,
    start_ai_capture_session,
    stop_ai_capture_session,
    VALIDATION_CATEGORIES,
    VALIDATION_MODEL_STATUSES,
    get_ai_artifacts_root,
)
import ai_validation
import ai_capture
import patchcore_preprocess
from ai_capture import (
    GUIDED_MESSAGES,
    GUIDED_STATE_COLORS,
    GUIDED_VALID_STATES,
    GuidedTracker,
    assert_production_allowed,
    compute_sharpness,
    ensure_ai_capture_can_start,
    evaluate_candidate,
    get_ai_guided_config,
    is_ai_capture_mode_active,
    mask_bbox,
    reset_presence_candidate,
    set_ai_capture_mode,
    set_production_probe,
)
from ai_training import (
    cancel_pending_training_job,
    humanize_training_error,
    request_training_job,
    training_status_payload,
)


load_dotenv()
# RTSP mediante TCP: más estable que UDP para OpenCV
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
)

PATCHCORE_CKPT = os.getenv("PATCHCORE_CKPT", "").strip()

PATCHCORE_IMAGE_SIZE = int(
    os.getenv("PATCHCORE_IMAGE_SIZE", "384")
)

PATCHCORE_SCORE_THRESHOLD = float(
    os.getenv("PATCHCORE_SCORE_THRESHOLD", "55.20")
)

if PATCHCORE_IMAGE_SIZE <= 0:
    raise ValueError(
        "PATCHCORE_IMAGE_SIZE debe ser mayor que cero."
    )

if not 0.0 <= PATCHCORE_SCORE_THRESHOLD <= 100.0:
    raise ValueError(
        "PATCHCORE_SCORE_THRESHOLD debe estar entre 0 y 100."
    )

PATCHCORE_PIXEL_THRESHOLD = float(
    os.getenv("PATCHCORE_PIXEL_THRESHOLD", "0.72")
)

PATCHCORE_MAX_BOX_RATIO = float(
    os.getenv("PATCHCORE_MAX_BOX_RATIO", "0.15")
)

patchcore_inspector = None

if PATCHCORE_CKPT:
    try:
        patchcore_inspector = PatchCoreInspector(
            PATCHCORE_CKPT,
            image_size=PATCHCORE_IMAGE_SIZE,
        )
        print("[PATCHCORE] Modelo preparado para inspección.")
    except Exception as error:
        print(f"[PATCHCORE] No se pudo preparar el modelo: {error}")
else:
    print("[PATCHCORE] PATCHCORE_CKPT no está configurado.")

try:
    import cv2
    import numpy as np
except Exception as e:
    print(f"[ERROR] No se pudo importar OpenCV o NumPy: {e}")
    cv2 = None
    np = None

try:
    from ultralytics import YOLO
except Exception as e:
    print(f"[IA] Ultralytics no disponible: {e}")
    YOLO = None

ROOT = Path(__file__).resolve().parent


CAPTURE_DIR = ROOT / "static" / "captures"
RESULT_DIR = ROOT / "static" / "results"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best.pt"

# Experimento de latencia: candidatos 40/45/50 solo benchmark.
# Default false => no se copian frames extra ni se escribe a disco.
FRAME_SELECTION_BENCHMARK = str(
    os.getenv("FRAME_SELECTION_BENCHMARK", "false")
).strip().lower() in ("1", "true", "yes", "on")

BENCHMARK_FRAMES_DIR = ROOT / "benchmark_frames"
BENCHMARK_COVERAGE_LEVELS = (0.40, 0.45, 0.50)

DB_NAME = os.environ.get("MYSQL_DATABASE", "textile_quality_db")
DB_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

DEFECT_TYPES = [
    "Mancha",
    "Rotura",
    "Agujero",
    "Variación de color",
    "Sin defecto",
]

GARMENTS = ["Blusa", "Top corto", "Camisa cropped"]
SIZES = ["S"]

# protege el manejador VideoCapture (open/read/release del worker)
camera_lock = threading.Lock()

# protege latest_frame / timestamp / sequence (consumidores)
latest_frame_lock = threading.Lock()
latest_camera_frame = None
latest_frame_timestamp_monotonic = None
latest_frame_sequence = 0

CAMERA_CAPTURE_WORKER = None
CAMERA_CAPTURE_WORKER_LOCK = threading.Lock()
CAMERA_CAPTURE_WORKER_STARTED = False
CAMERA_FORCE_REOPEN = False
yolo_model = None

app = Flask(__name__)
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    raise RuntimeError("SECRET_KEY environment variable is required")
app.secret_key = _secret_key
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ============================================================
# BASE DE DATOS MYSQL
# ============================================================

def db_config(include_database=True):
    config = {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "charset": "utf8mb4",
        "use_unicode": True,
    }
    if include_database:
        config["database"] = DB_NAME
    return config


def db(include_database=True):
    return mysql.connector.connect(**db_config(include_database=include_database))


def init_db():
    """
    Inicializa MySQL de forma segura:
    1. Crea la base configurada si todavía no existe.
    2. Crea las tablas requeridas.
    3. Crea el usuario inicial solo cuando aún no existe.
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Primero se conecta sin seleccionar una base de datos. Esto evita el
    # error "Unknown database" en la primera ejecución del proyecto.
    conn = db(include_database=False)
    cur = conn.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()
    cur.close()
    conn.close()

    conn = db(include_database=True)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'QUALITY_MANAGER',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # ============================================================
    # ESQUEMA EMPRESARIAL V2
    # Usuarios, modelos de prenda y modelos de IA
    # ============================================================

    def ensure_column(table_name, column_name, definition):
        cur.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
              AND COLUMN_NAME = %s
            """,
            (DB_NAME, table_name, column_name),
        )

        if cur.fetchone()[0] == 0:
            cur.execute(
                f"ALTER TABLE `{table_name}` ADD COLUMN {definition}"
            )

    ensure_column(
        "users",
        "full_name",
        "full_name VARCHAR(150) NULL AFTER username",
    )
    ensure_column(
        "users",
        "active",
        "active TINYINT(1) NOT NULL DEFAULT 1 AFTER role",
    )
    ensure_column(
        "users",
        "updated_at",
        "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP "
        "ON UPDATE CURRENT_TIMESTAMP AFTER created_at",
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_production_goals (
            id INT AUTO_INCREMENT PRIMARY KEY,
            goal_date DATE NOT NULL,
            target_batches INT NOT NULL,
            target_garments INT NOT NULL,
            shift_start TIME NOT NULL DEFAULT '08:00:00',
            shift_end TIME NOT NULL DEFAULT '17:00:00',
            created_by INT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,

            UNIQUE KEY uq_daily_goal_date (goal_date),
            KEY idx_daily_goal_date (goal_date),

            CONSTRAINT fk_daily_goal_created_by
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
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
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            approved_at DATETIME NULL,
            active TINYINT(1) NOT NULL DEFAULT 1,

            UNIQUE KEY uq_garment_model_code (code),
            KEY idx_garment_model_status (status),
            KEY idx_garment_model_created_by (created_by),
            KEY idx_garment_model_approved_by (approved_by),

            CONSTRAINT fk_garment_model_created_by
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL,

            CONSTRAINT fk_garment_model_approved_by
                FOREIGN KEY (approved_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS garment_model_images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            garment_model_id INT NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            image_type VARCHAR(50) NOT NULL DEFAULT 'REFERENCIA',
            sort_order INT NOT NULL DEFAULT 0,
            created_by INT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            KEY idx_garment_image_model (garment_model_id),

            CONSTRAINT fk_garment_image_model
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE CASCADE,

            CONSTRAINT fk_garment_image_created_by
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
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
            ),
            KEY idx_garment_ai_model (garment_model_id),
            KEY idx_garment_ai_active (active),
            KEY idx_garment_ai_status (status),
            KEY idx_garment_ai_created_by (created_by),
            KEY idx_garment_ai_activated_by (activated_by),

            CONSTRAINT fk_garment_ai_garment
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE RESTRICT
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
    """)

    ensure_column(
        "garment_ai_models",
        "dataset_name",
        "dataset_name VARCHAR(180) NULL AFTER model_type",
    )
    ensure_column(
        "garment_ai_models",
        "dataset_path",
        "dataset_path VARCHAR(500) NULL AFTER dataset_name",
    )
    ensure_column(
        "garment_ai_models",
        "metrics_json",
        "metrics_json JSON NULL AFTER normal_images_count",
    )
    ensure_column(
        "garment_ai_models",
        "created_by",
        "created_by INT NULL AFTER notes",
    )
    ensure_column(
        "garment_ai_models",
        "trained_by",
        "trained_by INT NULL AFTER created_by",
    )
    ensure_column(
        "garment_ai_models",
        "validated_by",
        "validated_by INT NULL AFTER trained_by",
    )
    ensure_column(
        "garment_ai_models",
        "activated_by",
        "activated_by INT NULL AFTER validated_by",
    )
    ensure_column(
        "garment_ai_models",
        "validated_at",
        "validated_at DATETIME NULL AFTER trained_at",
    )
    ensure_column(
        "garment_ai_models",
        "activated_at",
        "activated_at DATETIME NULL AFTER validated_at",
    )

    # ========================================================
    # FASE 1 — PREPARACIÓN Y VERSIONADO DE IA
    # Tablas de dominio, extensiones de garment_ai_models,
    # constraint de un solo ACTIVO e inmutabilidad de datasets.
    # ========================================================
    ensure_ai_schema(cur, DB_NAME, ensure_column=ensure_column)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            code VARCHAR(80) NOT NULL UNIQUE,
            created_at DATETIME NOT NULL,
            garment_type VARCHAR(100) NOT NULL DEFAULT 'Prenda inspeccionada',
            size VARCHAR(20) NOT NULL DEFAULT 'S',
            status VARCHAR(50) NOT NULL,
            defect_type VARCHAR(150),
            confidence DECIMAL(5,2),
            zone VARCHAR(120),
            image_original VARCHAR(255),
            image_result VARCHAR(255),
            human_validation VARCHAR(50) DEFAULT 'Pendiente',
            notes TEXT,
            created_record_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_created_at (created_at),
            INDEX idx_status (status),
            INDEX idx_defect_type (defect_type),
            INDEX idx_validation (human_validation)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)


    # ========================================================
    # LOTES DE PRODUCCIÓN
    # ========================================================

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inspection_lines (
            id INT AUTO_INCREMENT PRIMARY KEY,
            code VARCHAR(40) NOT NULL,
            name VARCHAR(120) NOT NULL,
            camera_ip VARCHAR(255) NULL,
            camera_rtsp_port INT NOT NULL DEFAULT 554,
            camera_channel INT NOT NULL DEFAULT 1,
            camera_subtype INT NOT NULL DEFAULT 0,
            conveyor_speed_cm_s DECIMAL(8,3) NULL,
            withdrawal_distance_cm DECIMAL(8,2) NULL,
            active TINYINT(1) NOT NULL DEFAULT 1,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL
                DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_inspection_line_code (code),
            KEY idx_inspection_line_active (active)
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id INT AUTO_INCREMENT PRIMARY KEY,
            code VARCHAR(80) NOT NULL UNIQUE,
            planned_quantity INT NOT NULL,
            status VARCHAR(40) NOT NULL DEFAULT 'PREPARACION',
            started_at DATETIME NULL,
            inspection_completed_at DATETIME NULL,
            closed_at DATETIME NULL,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_batch_status (status),
            INDEX idx_batch_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    ensure_column(
        "batches",
        "garment_model_id",
        "garment_model_id INT NULL AFTER code",
    )
    ensure_column(
        "batches",
        "ai_model_id",
        "ai_model_id INT NULL AFTER garment_model_id",
    )
    ensure_column(
        "batches",
        "inspection_line_id",
        "inspection_line_id INT NULL AFTER ai_model_id",
    )
    ensure_column(
        "batches",
        "production_date",
        "production_date DATE NULL AFTER inspection_line_id",
    )
    ensure_column(
        "batches",
        "created_by",
        "created_by INT NULL AFTER notes",
    )

    cur.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = 'batches'
          AND INDEX_NAME = 'idx_batches_inspection_line'
        """,
        (DB_NAME,),
    )

    if cur.fetchone()[0] == 0:
        cur.execute(
            """
            ALTER TABLE batches
            ADD INDEX idx_batches_inspection_line (
                inspection_line_id
            )
            """
        )

    def ensure_inspection_column(column_name, definition):
        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'inspections'
              AND COLUMN_NAME = %s
            """,
            (DB_NAME, column_name),
        )

        if cur.fetchone()[0] == 0:
            cur.execute(
                f"ALTER TABLE inspections ADD COLUMN {definition}"
            )

    ensure_inspection_column(
        "batch_id",
        "batch_id INT NULL"
    )
    ensure_inspection_column(
        "inspection_line_id",
        "inspection_line_id INT NULL AFTER batch_id"
    )
    ensure_inspection_column(
        "batch_position",
        "batch_position INT NULL"
    )
    ensure_inspection_column(
        "ai_decision",
        "ai_decision VARCHAR(30) NULL"
    )
    ensure_inspection_column(
        "ai_model_id",
        "ai_model_id INT NULL"
    )
    ensure_inspection_column(
        "review_status",
        "review_status VARCHAR(50) NULL"
    )
    ensure_inspection_column(
        "audit_selected",
        "audit_selected TINYINT(1) NOT NULL DEFAULT 0"
    )

    def ensure_inspection_index(index_name, definition):
        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'inspections'
              AND INDEX_NAME = %s
            """,
            (DB_NAME, index_name),
        )

        if cur.fetchone()[0] == 0:
            cur.execute(
                f"ALTER TABLE inspections ADD {definition}"
            )

    ensure_inspection_index(
        "idx_batch_id",
        "INDEX idx_batch_id (batch_id)"
    )
    ensure_inspection_index(
        "idx_inspections_inspection_line",
        "INDEX idx_inspections_inspection_line (inspection_line_id)"
    )
    ensure_inspection_index(
        "idx_batch_position",
        "INDEX idx_batch_position (batch_id, batch_position)"
    )

    cur.execute(
        """
        INSERT IGNORE INTO inspection_lines (
            code,
            name,
            camera_ip,
            camera_rtsp_port,
            camera_channel,
            camera_subtype,
            conveyor_speed_cm_s,
            withdrawal_distance_cm,
            active
        )
        VALUES (
            'LINEA-1',
            %s,
            %s,
            554,
            1,
            0,
            6.000,
            120.00,
            1
        )
        """,
        (
            "L\u00ednea 1",
            os.getenv("CAMERA_IP", "").strip() or None,
        ),
    )

    cur.execute(
        """
        INSERT IGNORE INTO inspection_lines (
            code,
            name,
            camera_ip,
            camera_rtsp_port,
            camera_channel,
            camera_subtype,
            active
        )
        VALUES (
            'LINEA-2',
            %s,
            NULL,
            554,
            1,
            0,
            0
        )
        """,
        ("L\u00ednea 2",),
    )

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

    cur.execute("SELECT id FROM users WHERE username = %s", (admin_username,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
            (admin_username, generate_password_hash(admin_password), "ADMIN"),
        )

    # Compatibilidad con usuarios creados antes de incorporar roles formales.
    cur.execute("""
        UPDATE users
        SET role = 'ADMIN'
        WHERE LOWER(role) IN ('administrador', 'admin')
    """)

    cur.execute("""
        UPDATE users
        SET role = 'QUALITY_MANAGER'
        WHERE LOWER(role) IN ('operario', 'gestor_calidad', 'gestor de calidad')
    """)

    conn.commit()
    cur.close()
    conn.close()


def fetch_one(sql, params=None):
    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, params or ())
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def fetch_all(sql, params=None):
    conn = db()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def execute(sql, params=None):
    conn = db()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    conn.commit()
    last_id = cur.lastrowid
    cur.close()
    conn.close()
    return last_id


def get_inspection_lines(
    active_only=False,
    configured_only=False,
):
    conditions = []
    params = []

    if active_only:
        conditions.append("l.active = 1")

    if configured_only:
        conditions.append(
            "NULLIF(TRIM(l.camera_ip), '') IS NOT NULL"
        )

    where = ""

    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    return fetch_all(
        f"""
        SELECT
            l.*,

            CASE
                WHEN l.conveyor_speed_cm_s > 0
                 AND l.withdrawal_distance_cm IS NOT NULL
                THEN ROUND(
                    l.withdrawal_distance_cm /
                    l.conveyor_speed_cm_s,
                    2
                )
                ELSE NULL
            END AS travel_seconds,

            (
                SELECT b.code
                FROM batches b
                WHERE b.inspection_line_id = l.id
                  AND b.status = 'EN_INSPECCION'
                ORDER BY b.id DESC
                LIMIT 1
            ) AS active_batch_code

        FROM inspection_lines l
        {where}
        ORDER BY l.id
        """,
        tuple(params),
    )


def get_active_batch(line_id=None):
    conditions = [
        "b.status = 'EN_INSPECCION'"
    ]
    params = []

    if line_id is not None:
        conditions.append(
            "b.inspection_line_id = %s"
        )
        params.append(int(line_id))

    where = " AND ".join(conditions)

    return fetch_one(
        f"""
        SELECT
            b.*,

            (
                SELECT gm.code
                FROM garment_models gm
                WHERE gm.id = b.garment_model_id
                LIMIT 1
            ) AS garment_model_code,

            (
                SELECT gm.name
                FROM garment_models gm
                WHERE gm.id = b.garment_model_id
                LIMIT 1
            ) AS garment_model_name,

            (
                SELECT ai.version
                FROM garment_ai_models ai
                WHERE ai.id = b.ai_model_id
                LIMIT 1
            ) AS ai_version,

            (
                SELECT ai.model_type
                FROM garment_ai_models ai
                WHERE ai.id = b.ai_model_id
                LIMIT 1
            ) AS ai_model_type,

            (
                SELECT l.code
                FROM inspection_lines l
                WHERE l.id = b.inspection_line_id
                LIMIT 1
            ) AS inspection_line_code,

            (
                SELECT l.name
                FROM inspection_lines l
                WHERE l.id = b.inspection_line_id
                LIMIT 1
            ) AS inspection_line_name,

            (
                SELECT ROUND(
                    l.withdrawal_distance_cm /
                    NULLIF(l.conveyor_speed_cm_s, 0),
                    2
                )
                FROM inspection_lines l
                WHERE l.id = b.inspection_line_id
                LIMIT 1
            ) AS withdrawal_seconds,

            COUNT(i.id) AS processed_quantity,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'NORMAL'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS auto_approved,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'DEFECTO_CONFIRMADO'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'ALERTA_DESCARTADA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded_alerts

        FROM batches b
        LEFT JOIN inspections i
          ON i.batch_id = b.id

        WHERE {where}

        GROUP BY b.id
        ORDER BY b.id DESC
        LIMIT 1
        """,
        tuple(params),
    )


def get_batch(batch_id):
    return fetch_one(
        """
        SELECT
            b.*,
            COUNT(i.id) AS processed_quantity,
            COALESCE(
                SUM(CASE WHEN i.ai_decision = 'NORMAL' THEN 1 ELSE 0 END),
                0
            ) AS auto_approved,
            COALESCE(
                SUM(CASE WHEN i.ai_decision = 'ANOMALIA' THEN 1 ELSE 0 END),
                0
            ) AS alerts,
            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'DEFECTO_CONFIRMADO'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects,
            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'ALERTA_DESCARTADA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded_alerts
        FROM batches b
        LEFT JOIN inspections i ON i.batch_id = b.id
        WHERE b.id = %s
        GROUP BY b.id
        """,
        (batch_id,),
    )


def get_batches():
    return fetch_all(
        """
        SELECT
            b.*,

            (
                SELECT gm.code
                FROM garment_models gm
                WHERE gm.id = b.garment_model_id
                LIMIT 1
            ) AS garment_model_code,

            (
                SELECT gm.name
                FROM garment_models gm
                WHERE gm.id = b.garment_model_id
                LIMIT 1
            ) AS garment_model_name,

            (
                SELECT ai.version
                FROM garment_ai_models ai
                WHERE ai.id = b.ai_model_id
                LIMIT 1
            ) AS ai_version,

            (
                SELECT ai.model_type
                FROM garment_ai_models ai
                WHERE ai.id = b.ai_model_id
                LIMIT 1
            ) AS ai_model_type,

            (
                SELECT l.code
                FROM inspection_lines l
                WHERE l.id = b.inspection_line_id
                LIMIT 1
            ) AS inspection_line_code,

            (
                SELECT l.name
                FROM inspection_lines l
                WHERE l.id = b.inspection_line_id
                LIMIT 1
            ) AS inspection_line_name,

            (
                SELECT ROUND(
                    l.withdrawal_distance_cm /
                    NULLIF(l.conveyor_speed_cm_s, 0),
                    2
                )
                FROM inspection_lines l
                WHERE l.id = b.inspection_line_id
                LIMIT 1
            ) AS withdrawal_seconds,

            (
                SELECT COALESCE(
                    NULLIF(u.full_name, ''),
                    u.username
                )
                FROM users u
                WHERE u.id = b.created_by
                LIMIT 1
            ) AS creator_name,

            COUNT(i.id) AS processed_quantity,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'NORMAL'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS auto_approved,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'DEFECTO_CONFIRMADO'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'ALERTA_DESCARTADA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded_alerts

        FROM batches b

        LEFT JOIN inspections i
          ON i.batch_id = b.id

        GROUP BY b.id
        ORDER BY b.id DESC
        """
    )


ROLE_ADMIN = "ADMIN"
ROLE_MODEL_MANAGER = "MODEL_MANAGER"
ROLE_QUALITY_MANAGER = "QUALITY_MANAGER"

VALID_ROLES = {
    ROLE_ADMIN,
    ROLE_MODEL_MANAGER,
    ROLE_QUALITY_MANAGER,
}

ROLE_LABELS = {
    ROLE_ADMIN: "Administrador",
    ROLE_MODEL_MANAGER: "Encargado de modelos",
    ROLE_QUALITY_MANAGER: "Gestor de calidad",
}

ROLE_ALIASES = {
    "administrador": ROLE_ADMIN,
    "admin": ROLE_ADMIN,
    "encargado_modelos": ROLE_MODEL_MANAGER,
    "encargado de modelos": ROLE_MODEL_MANAGER,
    "model_manager": ROLE_MODEL_MANAGER,
    "gestor_calidad": ROLE_QUALITY_MANAGER,
    "gestor de calidad": ROLE_QUALITY_MANAGER,
    "quality_manager": ROLE_QUALITY_MANAGER,
    "operario": ROLE_QUALITY_MANAGER,
}

def normalize_role(value):
    raw = str(value or "").strip()

    if not raw:
        return ""

    return ROLE_ALIASES.get(raw.lower(), raw.upper())


def has_role(*roles):
    current_role = normalize_role(session.get("role"))
    allowed = {normalize_role(role) for role in roles}
    return current_role in allowed


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user_id = session.get("user_id")

        if not user_id:
            if request.path.startswith("/api/"):
                return jsonify({
                    "ok": False,
                    "error": "Autenticacion requerida."
                }), 401

            return redirect(url_for("login"))

        user = fetch_one(
            """
            SELECT id, username, role, active
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        )

        if not user or int(user.get("active") or 0) != 1:
            session.clear()

            if request.path.startswith("/api/"):
                return jsonify({
                    "ok": False,
                    "error": "La sesion ya no esta autorizada."
                }), 401

            flash(
                "La cuenta esta desactivada o ya no se encuentra disponible.",
                "error",
            )
            return redirect(url_for("login"))

        role = normalize_role(user.get("role"))

        session["username"] = user["username"]
        session["role"] = role

        return fn(*args, **kwargs)

    return wrapper


def role_required(*roles):
    allowed_roles = {normalize_role(role) for role in roles}

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            current_role = normalize_role(session.get("role"))

            if current_role not in allowed_roles:
                if request.path.startswith("/api/"):
                    return jsonify({
                        "ok": False,
                        "error": "No tiene permisos para realizar esta accion."
                    }), 403

                flash(
                    "No tiene permisos para acceder a esta opcion.",
                    "error",
                )
                return redirect(url_for("dashboard"))

            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# CÁMARA / ESTACIÓN AUTOMÁTICA
# ============================================================

CAMERA_IP = os.getenv("CAMERA_IP", "10.145.208.30").strip()
# CAMERA_IP = os.getenv("CAMERA_IP", "192.168.0.11").strip()
CAMERA_USER = os.getenv("CAMERA_USER", "admin").strip()
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")

CAMERA_RTSP_PORT = int(os.getenv("CAMERA_RTSP_PORT", "554"))
CAMERA_CHANNEL = int(os.getenv("CAMERA_CHANNEL", "1"))
CAMERA_SUBTYPE = int(os.getenv("CAMERA_SUBTYPE", "0"))

# Codifica automáticamente los caracteres especiales de las credenciales.
CAMERA_USER_ENCODED = quote(CAMERA_USER, safe="")
CAMERA_PASSWORD_ENCODED = quote(CAMERA_PASSWORD, safe="")

CAMERA_SOURCE = (
    f"rtsp://{CAMERA_USER_ENCODED}:{CAMERA_PASSWORD_ENCODED}"
    f"@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={CAMERA_CHANNEL}&subtype={CAMERA_SUBTYPE}"
)

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "15"))

print(
    f"[CAMARA] RTSP configurado: "
    f"{CAMERA_USER}@{CAMERA_IP}:{CAMERA_RTSP_PORT} "
    f"canal={CAMERA_CHANNEL}, flujo={CAMERA_SUBTYPE}"
)
print(
    "[CAMARA] IP .env="
    f"{CAMERA_IP} | URL="
    f"rtsp://{CAMERA_USER_ENCODED}:***"
    f"@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={CAMERA_CHANNEL}"
    f"&subtype={CAMERA_SUBTYPE}"
)

AUTO_INSPECTION_ENABLED = False
AUTO_THREAD = None
AUTO_LAST_RESULT = None
AUTO_LAST_ERROR = None
AUTO_LAST_CAPTURE_TIME = 0

# ------------------------------------------------------------
# FASE 2A — modo preparación IA (mutex con producción)
# ------------------------------------------------------------
AI_CAPTURE_THREAD = None
AI_CAPTURE_THREAD_LOCK = threading.Lock()
AI_CAPTURE_WORKER_ENABLED = False
AI_CAPTURE_RUNTIME = {
    "session_id": None,
    "garment_model_id": None,
    "config": None,
    "known_sha256": set(),
    "known_dhashes": [],
    "presence": None,
    "waiting_for_exit": False,
    "exit_frames": 0,
    "last_sequence": -1,
    "last_persisted_token": None,
    "last_error": None,
    "last_result": None,
}

# ------------------------------------------------------------
# FASE 2C — estación guiada de captura (mismo candado que el
# worker simple: nunca corren los dos a la vez).
# ------------------------------------------------------------
AI_GUIDED_ACTIVE = False
AI_GUIDED_ENABLED = False
AI_GUIDED_THREAD = None
AI_GUIDED_RUNTIME = {
    "garment_model_id": None,
    "session_id": None,
    "guide": None,
    "tracker": None,
    "tracker_model_id": None,
    "last_error": None,
    "pending_capture": None,
}

set_production_probe(lambda: bool(AUTO_INSPECTION_ENABLED))

AUTO_COOLDOWN_SECONDS = float(os.getenv("AUTO_COOLDOWN_SECONDS", "4"))
AUTO_MOTION_THRESHOLD = int(os.getenv("AUTO_MOTION_THRESHOLD", "18000"))
AUTO_STABILIZATION_SECONDS = float(os.getenv("AUTO_STABILIZATION_SECONDS", "0.6"))

# ------------------------------------------------------------
# Detección automática de presencia de la blusa.
#
# La blusa completa suele ocupar aproximadamente 50-52 % del ROI.
# El automático espera que entre suficientemente antes de capturar.
# ------------------------------------------------------------
AUTO_GARMENT_ENTER_COVERAGE = float(
    os.getenv(
        "AUTO_GARMENT_ENTER_COVERAGE",
        "0.28",
    )
)

AUTO_GARMENT_CAPTURE_COVERAGE = float(
    os.getenv(
        "AUTO_GARMENT_CAPTURE_COVERAGE",
        "0.48",
    )
)

AUTO_GARMENT_EXIT_COVERAGE = float(
    os.getenv(
        "AUTO_GARMENT_EXIT_COVERAGE",
        "0.12",
    )
)

AUTO_GARMENT_CONFIRM_FRAMES = int(
    os.getenv(
        "AUTO_GARMENT_CONFIRM_FRAMES",
        "3",
    )
)

AUTO_GARMENT_EXIT_FRAMES = int(
    os.getenv(
        "AUTO_GARMENT_EXIT_FRAMES",
        "3",
    )
)

AUTO_GARMENT_MAX_TRACK_FRAMES = int(
    os.getenv(
        "AUTO_GARMENT_MAX_TRACK_FRAMES",
        "16",
    )
)

# ------------------------------------------------------------
# Identidad temporal de prenda (Fase 3) + latencia (Fase 2).
# Solo en memoria y logging: no se persiste en BD todavía.
# ------------------------------------------------------------
GARMENT_TOKEN_LOCK = threading.Lock()
GARMENT_TOKEN_SEQ = 0


def next_garment_token():
    """Token secuencial único por ciclo de presencia de prenda."""
    global GARMENT_TOKEN_SEQ
    with GARMENT_TOKEN_LOCK:
        GARMENT_TOKEN_SEQ += 1
        return GARMENT_TOKEN_SEQ


# ------------------------------------------------------------
# Timeline por garment_token (experimento de latencia).
# perf_counter para duraciones; time.time() para correlación.
# ------------------------------------------------------------
TIMELINE_LOCK = threading.Lock()
TIMELINE_EVENTS = {}


def timeline_mark(token, event):
    """Registra el PRIMER cruce de un evento para el token."""
    if token is None or not event:
        return

    snapshot = {
        "perf": time.perf_counter(),
        "wall_ms": int(time.time() * 1000),
    }

    with TIMELINE_LOCK:
        bucket = TIMELINE_EVENTS.setdefault(token, {})
        if event in bucket:
            return
        bucket[event] = snapshot


def _timeline_delta_ms(token, from_event, to_event):
    with TIMELINE_LOCK:
        bucket = TIMELINE_EVENTS.get(token) or {}
        start = bucket.get(from_event)
        end = bucket.get(to_event)

    if start is None or end is None:
        return None

    return (end["perf"] - start["perf"]) * 1000.0


def log_timeline(token):
    """Emite el bloque [TIMELINE] de una prenda."""
    if token is None:
        return

    with TIMELINE_LOCK:
        bucket = dict(TIMELINE_EVENTS.get(token) or {})

    def fmt(value):
        if value is None:
            return "n/a"
        return f"{float(value):.1f}"

    detect = "garment_detected"

    lines = [
        f"[TIMELINE] token={token}",
        (
            "detect_to_40_ms="
            + fmt(_timeline_delta_ms(token, detect, "coverage_40_reached"))
        ),
        (
            "detect_to_45_ms="
            + fmt(_timeline_delta_ms(token, detect, "coverage_45_reached"))
        ),
        (
            "detect_to_50_ms="
            + fmt(_timeline_delta_ms(token, detect, "coverage_50_reached"))
        ),
        (
            "frame_selection_ms="
            + fmt(_timeline_delta_ms(token, detect, "frame_selected"))
        ),
        (
            "preprocess_ms="
            + fmt(_timeline_delta_ms(token, "preprocess_start", "preprocess_end"))
        ),
        (
            "inference_ms="
            + fmt(_timeline_delta_ms(token, "inference_start", "inference_end"))
        ),
        (
            "detect_to_decision_ms="
            + fmt(_timeline_delta_ms(token, detect, "decision_ready"))
        ),
        (
            "decision_to_alert_ms="
            + fmt(_timeline_delta_ms(token, "decision_ready", "alert_emitted"))
        ),
        (
            "detect_to_alert_ms="
            + fmt(_timeline_delta_ms(token, detect, "alert_emitted"))
        ),
        (
            "total_ms="
            + fmt(_timeline_delta_ms(token, detect, "inspection_complete"))
        ),
    ]

    print("\n".join(lines), flush=True)


def reset_timeline(token):
    if token is None:
        return
    with TIMELINE_LOCK:
        TIMELINE_EVENTS.pop(token, None)


def persist_benchmark_frames(
    garment_token,
    frames_by_level,
    production_result=None,
    production_score=None,
    batch_id=None,
    batch_position=None,
):
    """
    Escribe candidatos 40/45/50 SOLO después de la inspección
    productiva. No participa en la decisión.
    """
    if not FRAME_SELECTION_BENCHMARK:
        return
    if not frames_by_level or garment_token is None:
        return

    try:
        out_dir = BENCHMARK_FRAMES_DIR / f"token_{garment_token}"
        out_dir.mkdir(parents=True, exist_ok=True)

        coverages = {}
        for level, payload in frames_by_level.items():
            pct = int(round(float(level) * 100))
            frame = payload.get("frame")
            if frame is None:
                continue
            path = out_dir / f"coverage_{pct}.jpg"
            if not cv2.imwrite(str(path), frame):
                print(
                    "[BENCHMARK] No se pudo escribir "
                    f"{path}",
                    flush=True,
                )
                continue
            coverages[str(pct)] = float(payload.get("coverage") or 0.0)

        metadata = {
            "token": garment_token,
            "batch_id": batch_id,
            "batch_position": batch_position,
            "coverages": coverages,
            "production_result": production_result,
            "production_score": production_score,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "server_emitted_at_ms": int(time.time() * 1000),
        }

        meta_path = out_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            "[BENCHMARK] Candidatos persistidos "
            f"token={garment_token} dir={out_dir}",
            flush=True,
        )
    except Exception as error:
        print(
            "[BENCHMARK] Error al persistir "
            f"token={garment_token}: {error}",
            flush=True,
        )


# ------------------------------------------------------------
# Eventos de alarma de calidad en memoria (para SSE/polling).
# No se persisten en BD. Se publican en el instante de la
# decisión, antes de imagen/BD.
# ------------------------------------------------------------
QUALITY_ALERT_LOCK = threading.Lock()
QUALITY_ALERT_SEQ = 0
QUALITY_ALERT_EVENTS = []


def publish_quality_alert_event(
    garment_token,
    confidence=None,
    zone=None,
    defect_type=None,
    source="auto",
):
    """Publica un evento único de anomalía para el navegador."""
    global QUALITY_ALERT_SEQ

    with QUALITY_ALERT_LOCK:
        QUALITY_ALERT_SEQ += 1
        event = {
            "alert_id": f"ae{QUALITY_ALERT_SEQ}",
            "event_id": f"ae{QUALITY_ALERT_SEQ}",
            "seq": QUALITY_ALERT_SEQ,
            "token": garment_token,
            "garment_token": garment_token,
            "result": "ANOMALIA",
            "confidence": confidence,
            "zone": zone,
            "defect_type": defect_type,
            "source": source,
            "batch_id": None,
            "batch_position": None,
            "code": None,
            "ts": time.time(),
            "server_emitted_at_ms": int(time.time() * 1000),
        }
        QUALITY_ALERT_EVENTS.append(event)
        if len(QUALITY_ALERT_EVENTS) > 80:
            del QUALITY_ALERT_EVENTS[: len(QUALITY_ALERT_EVENTS) - 80]
        return dict(event)


def enrich_quality_alert_event(
    garment_token,
    batch_id=None,
    batch_position=None,
    code=None,
):
    """
    Completa posición/código tras el INSERT sin crear un nuevo
    alert_id (el navegador no vuelve a sonar; solo puede refrescar
    la alerta visual).
    """
    if garment_token is None:
        return

    global QUALITY_ALERT_SEQ

    with QUALITY_ALERT_LOCK:
        for event in reversed(QUALITY_ALERT_EVENTS):
            if event.get("token") == garment_token:
                if batch_id is not None:
                    event["batch_id"] = batch_id
                if batch_position is not None:
                    event["batch_position"] = batch_position
                if code is not None:
                    event["code"] = code

                update = dict(event)
                QUALITY_ALERT_SEQ += 1
                update["seq"] = QUALITY_ALERT_SEQ
                update["enriched"] = True
                QUALITY_ALERT_EVENTS.append(update)
                if len(QUALITY_ALERT_EVENTS) > 80:
                    del QUALITY_ALERT_EVENTS[
                        : len(QUALITY_ALERT_EVENTS) - 80
                    ]
                break


def get_quality_alerts_after(after_seq):
    """Devuelve eventos con seq > after_seq (orden ascendente)."""
    with QUALITY_ALERT_LOCK:
        return [
            dict(event)
            for event in QUALITY_ALERT_EVENTS
            if int(event.get("seq") or 0) > int(after_seq or 0)
        ]


def current_quality_alert_seq():
    with QUALITY_ALERT_LOCK:
        return QUALITY_ALERT_SEQ


def trigger_quality_alert(
    garment_token,
    confidence=None,
    zone=None,
    defect_type=None,
    source="auto",
):
    """
    Punto único de alerta de calidad.

    Se invoca INMEDIATAMENTE después de conocer ANOMALIA y ANTES de
    persistencia secundaria (imagen de resultado, MySQL, contadores).
    Publica el evento en memoria para que el navegador pueda sonar
    sin esperar BD.
    """
    publish_quality_alert_event(
        garment_token=garment_token,
        confidence=confidence,
        zone=zone,
        defect_type=defect_type,
        source=source,
    )
    timeline_mark(garment_token, "alert_emitted")

    confidence_text = (
        f"{confidence}"
        if confidence is not None
        else "n/a"
    )
    print(
        "[ALERT] "
        f"token={garment_token} "
        "result=ANOMALIA "
        f"confidence={confidence_text} "
        f"zone={zone or 'n/a'} "
        f"defect_type={defect_type or 'n/a'} "
        f"source={source}",
        flush=True,
    )


def log_latency_metrics(
    garment_token,
    metrics,
    ai_decision,
    batch_id=None,
    batch_position=None,
    source="auto",
):
    """Emite la línea [LATENCY] de una inspección completa."""
    if not metrics:
        return

    def _ms(key):
        value = metrics.get(key)
        if value is None:
            return "n/a"
        return f"{float(value):.1f}"

    print(
        "[LATENCY] "
        f"token={garment_token} "
        f"decision_ms={_ms('decision_ms')} "
        f"total_ms={_ms('total_complete_ms')} "
        f"inference_ms={_ms('inference_ms')} "
        f"capture_ms={_ms('capture_ms')} "
        f"frame_selection_ms={_ms('frame_selection_ms')} "
        f"preprocess_ms={_ms('preprocess_ms')} "
        f"postprocess_ms={_ms('postprocess_ms')} "
        f"image_storage_ms={_ms('image_storage_ms')} "
        f"database_ms={_ms('database_ms')} "
        f"batch_id={batch_id if batch_id is not None else 'n/a'} "
        f"batch_position={batch_position if batch_position is not None else 'n/a'} "
        f"source={source} "
        f"result={ai_decision}",
        flush=True,
    )


ROI_X1 = float(os.getenv("ROI_X1", "0.10"))
ROI_Y1 = float(os.getenv("ROI_Y1", "0.10"))
ROI_X2 = float(os.getenv("ROI_X2", "0.90"))
ROI_Y2 = float(os.getenv("ROI_Y2", "0.90"))

YOLO_INFERENCE_CONF = float(os.getenv("YOLO_INFERENCE_CONF", "0.40"))
YOLO_DEFECT_THRESHOLD = float(os.getenv("YOLO_DEFECT_THRESHOLD", "0.70"))

camera_capture = None

CAMERA_CONNECTED = False
CAMERA_LAST_OK_AT = None
CAMERA_LAST_ERROR = "Comprobando conexión con la cámara."
CAMERA_LAST_CONNECT_ATTEMPT = 0.0
CAMERA_FAILURE_COUNT = 0

CAMERA_RETRY_SECONDS = float(
    os.getenv("CAMERA_RETRY_SECONDS", "3")
)

CAMERA_FAILURE_LIMIT = int(
    os.getenv("CAMERA_FAILURE_LIMIT", "3")
)

CAMERA_OPEN_TIMEOUT_MS = int(
    os.getenv("CAMERA_OPEN_TIMEOUT_MS", "4000")
)

CAMERA_READ_TIMEOUT_MS = int(
    os.getenv("CAMERA_READ_TIMEOUT_MS", "4000")
)

CAMERA_RECONNECTING = False
CAMERA_RECONNECT_LOCK = threading.Lock()
CAMERA_CURRENT_IP = CAMERA_IP

CAMERA_LATENCY_LOG_LOCK = threading.Lock()
CAMERA_LATENCY_LAST_LOG = {}
CAMERA_LATENCY_LOG_INTERVAL_S = float(
    os.getenv("CAMERA_LATENCY_LOG_INTERVAL_S", "2.0")
)
CAMERA_READ_FAIL_RELEASE_LIMIT = int(
    os.getenv("CAMERA_READ_FAIL_RELEASE_LIMIT", "3")
)

CAMERA_DISCOVERY_ENABLED = (
    os.getenv(
        "CAMERA_DISCOVERY_ENABLED",
        "1",
    ).strip().lower()
    not in (
        "0",
        "false",
        "no",
        "off",
    )
)

CAMERA_DISCOVERY_CIDR = os.getenv(
    "CAMERA_DISCOVERY_CIDR",
    "",
).strip()

CAMERA_DISCOVERY_TIMEOUT = float(
    os.getenv(
        "CAMERA_DISCOVERY_TIMEOUT",
        "0.30",
    )
)

CAMERA_DISCOVERY_WORKERS = int(
    os.getenv(
        "CAMERA_DISCOVERY_WORKERS",
        "40",
    )
)



def build_runtime_camera_source(
    ip_address=None,
):
    """
    Construye el RTSP utilizando la IP que el sistema
    considera actualmente valida para la camara.
    """
    selected_ip = (
        ip_address
        or CAMERA_CURRENT_IP
        or CAMERA_IP
    )

    return (
        f"rtsp://"
        f"{CAMERA_USER_ENCODED}:"
        f"{CAMERA_PASSWORD_ENCODED}"
        f"@{selected_ip}:"
        f"{CAMERA_RTSP_PORT}"
        f"/cam/realmonitor"
        f"?channel={CAMERA_CHANNEL}"
        f"&subtype={CAMERA_SUBTYPE}"
    )


def get_camera_discovery_network():
    """
    Devuelve la red que se utilizara para localizar
    automaticamente la camara.
    """
    import ipaddress

    if CAMERA_DISCOVERY_CIDR:
        try:
            return ipaddress.ip_network(
                CAMERA_DISCOVERY_CIDR,
                strict=False,
            )
        except ValueError:
            pass

    try:
        return ipaddress.ip_network(
            f"{CAMERA_IP}/24",
            strict=False,
        )
    except ValueError:
        return None


def camera_port_is_open(ip_address):
    """
    Comprobacion TCP rapida antes de intentar RTSP.
    """
    import socket

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    sock.settimeout(
        CAMERA_DISCOVERY_TIMEOUT
    )

    try:
        return (
            sock.connect_ex(
                (
                    str(ip_address),
                    CAMERA_RTSP_PORT,
                )
            )
            == 0
        )

    except OSError:
        return False

    finally:
        try:
            sock.close()
        except Exception:
            pass


def validate_camera_ip(ip_address):
    """
    Valida que una IP no solamente tenga RTSP abierto,
    sino que entregue un frame real usando las
    credenciales de esta instalacion.
    """
    if cv2 is None:
        return False

    source = build_runtime_camera_source(
        str(ip_address)
    )

    capture_params = []

    validation_timeout = min(
        CAMERA_OPEN_TIMEOUT_MS,
        2500,
    )

    if hasattr(
        cv2,
        "CAP_PROP_OPEN_TIMEOUT_MSEC",
    ):
        capture_params.extend([
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            validation_timeout,
        ])

    if hasattr(
        cv2,
        "CAP_PROP_READ_TIMEOUT_MSEC",
    ):
        capture_params.extend([
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            validation_timeout,
        ])

    cap = None

    try:
        if capture_params:
            cap = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                capture_params,
            )
        else:
            cap = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
            )

        if not cap.isOpened():
            return False

        ok, frame = cap.read()

        return bool(
            ok
            and frame is not None
            and getattr(
                frame,
                "size",
                0,
            ) > 0
        )

    except Exception:
        return False

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def discover_camera_ip():
    """
    Busca automaticamente la camara en la LAN.

    Primero detecta hosts con el puerto RTSP abierto
    y despues valida cual entrega realmente video con
    las credenciales configuradas.
    """
    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )

    network = (
        get_camera_discovery_network()
    )

    if (
        not CAMERA_DISCOVERY_ENABLED
        or network is None
    ):
        return None

    hosts = [
        str(host)
        for host in network.hosts()
    ]

    # Priorizar las direcciones conocidas.
    priority = []

    for ip_address in (
        CAMERA_CURRENT_IP,
        CAMERA_IP,
    ):
        if (
            ip_address
            and ip_address in hosts
            and ip_address
            not in priority
        ):
            priority.append(
                ip_address
            )

    remaining = [
        ip_address
        for ip_address in hosts
        if ip_address not in priority
    ]

    candidates = []

    with ThreadPoolExecutor(
        max_workers=max(
            4,
            CAMERA_DISCOVERY_WORKERS,
        )
    ) as executor:

        future_map = {
            executor.submit(
                camera_port_is_open,
                ip_address,
            ): ip_address
            for ip_address
            in priority + remaining
        }

        for future in as_completed(
            future_map
        ):
            ip_address = (
                future_map[future]
            )

            try:
                if future.result():
                    candidates.append(
                        ip_address
                    )
            except Exception:
                pass

    # Volver a priorizar IP actual/configurada
    # si tienen RTSP abierto.
    candidates.sort(
        key=lambda ip_address: (
            0
            if ip_address
            == CAMERA_CURRENT_IP
            else (
                1
                if ip_address
                == CAMERA_IP
                else 2
            ),
            tuple(
                int(part)
                for part
                in ip_address.split(".")
            ),
        )
    )

    for ip_address in candidates:

        app.logger.info(
            "[CAMARA] Probando dispositivo "
            f"en {ip_address}:"
            f"{CAMERA_RTSP_PORT}"
        )

        if validate_camera_ip(
            ip_address
        ):
            app.logger.info(
                "[CAMARA] Camara localizada "
                f"automaticamente en "
                f"{ip_address}."
            )

            return ip_address

    return None


def parse_camera_source(source):
    """
    Permite usar cámara local con 0, 1, 2...
    o cámara IP/RTSP con una URL.
    """
    if str(source).isdigit():
        return int(source)

    return source


def log_camera_latency(source, sequence, timestamp_monotonic):
    """
    Log limitado de edad del frame. No escribe en BD.
    """
    if timestamp_monotonic is None:
        return

    now = time.perf_counter()

    with CAMERA_LATENCY_LOG_LOCK:
        last = CAMERA_LATENCY_LAST_LOG.get(source, 0.0)
        if (now - last) < CAMERA_LATENCY_LOG_INTERVAL_S:
            return
        CAMERA_LATENCY_LAST_LOG[source] = now

    frame_age_ms = (now - timestamp_monotonic) * 1000.0
    print(
        "[CAMERA_LATENCY]\n"
        f"sequence={sequence}\n"
        f"frame_age_ms={frame_age_ms:.1f}\n"
        f"source={source}",
        flush=True,
    )


def get_latest_camera_frame():
    """
    Copia thread-safe del frame más reciente.
    NO llama cap.read(). No mantiene lock durante procesamiento.
    Devuelve: (frame|None, timestamp_monotonic|None, sequence:int)
    """
    global latest_camera_frame
    global latest_frame_timestamp_monotonic
    global latest_frame_sequence

    with latest_frame_lock:
        frame = (
            latest_camera_frame.copy()
            if latest_camera_frame is not None
            else None
        )
        timestamp = latest_frame_timestamp_monotonic
        sequence = latest_frame_sequence

    return frame, timestamp, sequence


def _publish_latest_frame(frame):
    global latest_camera_frame
    global latest_frame_timestamp_monotonic
    global latest_frame_sequence

    timestamp = time.perf_counter()

    with latest_frame_lock:
        latest_camera_frame = frame
        latest_frame_timestamp_monotonic = timestamp
        latest_frame_sequence += 1
        sequence = latest_frame_sequence

    return timestamp, sequence


def _clear_latest_frame():
    global latest_camera_frame
    global latest_frame_timestamp_monotonic

    with latest_frame_lock:
        latest_camera_frame = None
        latest_frame_timestamp_monotonic = None


def _note_camera_signal_lost(error_message):
    global CAMERA_CONNECTED
    global CAMERA_FAILURE_COUNT
    global CAMERA_LAST_ERROR
    global AUTO_INSPECTION_ENABLED
    global AUTO_LAST_ERROR

    CAMERA_CONNECTED = False
    CAMERA_FAILURE_COUNT += 1

    if error_message:
        CAMERA_LAST_ERROR = error_message

    if (
        AUTO_INSPECTION_ENABLED
        and CAMERA_FAILURE_COUNT >= CAMERA_FAILURE_LIMIT
    ):
        AUTO_INSPECTION_ENABLED = False
        AUTO_LAST_ERROR = (
            "Inspección automática detenida: "
            "la cámara perdió la señal."
        )


def get_camera():
    """
    Abre/reutiliza la ÚNICA sesión VideoCapture del pipeline.

    No hace cap.read(). Los consumidores NO deben llamar esto
    para obtener frames: usan get_latest_camera_frame().
    Debe invocarse bajo camera_lock desde camera_capture_worker
    o desde reset/reconexión controlada.
    """
    global camera_capture
    global CAMERA_LAST_CONNECT_ATTEMPT
    global CAMERA_LAST_ERROR

    if cv2 is None:
        CAMERA_LAST_ERROR = (
            "OpenCV no esta disponible."
        )
        return None

    if (
        camera_capture is not None
        and camera_capture.isOpened()
    ):
        return camera_capture

    now = time.time()

    if (
        CAMERA_LAST_CONNECT_ATTEMPT > 0
        and (
            now
            - CAMERA_LAST_CONNECT_ATTEMPT
        )
        < CAMERA_RETRY_SECONDS
    ):
        return None

    CAMERA_LAST_CONNECT_ATTEMPT = now

    source = parse_camera_source(
        build_runtime_camera_source()
    )

    try:
        capture_params = []

        if hasattr(
            cv2,
            "CAP_PROP_OPEN_TIMEOUT_MSEC",
        ):
            capture_params.extend([
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                CAMERA_OPEN_TIMEOUT_MS,
            ])

        if hasattr(
            cv2,
            "CAP_PROP_READ_TIMEOUT_MSEC",
        ):
            capture_params.extend([
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                CAMERA_READ_TIMEOUT_MS,
            ])

        if capture_params:
            camera_capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                capture_params,
            )
        else:
            camera_capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
            )

        # Best-effort: FFmpeg/RTSP puede ignorarlo.
        try:
            camera_capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )
        except Exception:
            pass

        if not camera_capture.isOpened():
            try:
                camera_capture.release()
            except Exception:
                pass

            camera_capture = None

            CAMERA_LAST_ERROR = (
                "No se pudo establecer "
                "conexion con la camara."
            )

            return None

        return camera_capture

    except Exception as error:
        try:
            if camera_capture is not None:
                camera_capture.release()
        except Exception:
            pass

        camera_capture = None

        CAMERA_LAST_ERROR = (
            "Error al conectar con "
            f"la camara: {error}"
        )

        return None


def read_camera_frame(source="compat"):
    """
    Compatibilidad: NO ejecuta cap.read().

    Devuelve el latest_frame publicado por camera_capture_worker.
    """
    frame, timestamp, sequence = get_latest_camera_frame()

    if frame is None:
        return False, None

    log_camera_latency(
        source=source,
        sequence=sequence,
        timestamp_monotonic=timestamp,
    )

    return True, frame


def reset_camera_connection():
    """
    Libera la conexion RTSP actual y limpia latest_frame.
    No modifica lotes ni registra inspecciones.
    Solo debe usarse desde el capture worker / reconexión.
    """
    global camera_capture
    global CAMERA_CONNECTED
    global CAMERA_LAST_ERROR
    global CAMERA_LAST_CONNECT_ATTEMPT
    global CAMERA_FAILURE_COUNT

    with camera_lock:
        try:
            if camera_capture is not None:
                camera_capture.release()
        except Exception:
            pass

        camera_capture = None

    _clear_latest_frame()

    CAMERA_CONNECTED = False
    CAMERA_LAST_ERROR = (
        "Reconectando con la c\u00e1mara de inspecci\u00f3n."
    )
    CAMERA_LAST_CONNECT_ATTEMPT = 0.0
    CAMERA_FAILURE_COUNT = 0


def _try_discover_camera_after_failures(direct_failures):
    """
    Solo el capture worker llama esto (evita 2 reconexiones).
    """
    global CAMERA_CURRENT_IP
    global CAMERA_LAST_CONNECT_ATTEMPT

    if not CAMERA_DISCOVERY_ENABLED:
        return False

    if not (
        direct_failures == 2
        or direct_failures % 5 == 0
    ):
        return False

    app.logger.warning(
        "[CAMARA] IP actual sin respuesta. "
        "Buscando camara en la red local."
    )

    discovered_ip = discover_camera_ip()

    if (
        discovered_ip
        and discovered_ip != CAMERA_CURRENT_IP
    ):
        old_ip = CAMERA_CURRENT_IP
        CAMERA_CURRENT_IP = discovered_ip
        CAMERA_LAST_CONNECT_ATTEMPT = 0.0
        reset_camera_connection()

        app.logger.warning(
            "[CAMARA] Cambio automatico "
            f"de IP: {old_ip} -> "
            f"{CAMERA_CURRENT_IP}"
        )
        return True

    return False


def camera_capture_worker():
    """
    ÚNICO lector continuo de RTSP.

    while running:
        ok, frame = cap.read()
        if ok:
            latest_frame = frame  (reemplaza; sin cola histórica)

    Reconexión/discovery también viven aquí: no hay un segundo
    worker que haga cap.read() concurrente.
    """
    global CAMERA_CONNECTED
    global CAMERA_LAST_OK_AT
    global CAMERA_LAST_ERROR
    global CAMERA_FAILURE_COUNT
    global CAMERA_RECONNECTING
    global CAMERA_FORCE_REOPEN

    with CAMERA_RECONNECT_LOCK:
        CAMERA_RECONNECTING = True

    app.logger.info(
        "[CAMARA] camera_capture_worker iniciado "
        f"(ip={CAMERA_CURRENT_IP}, "
        f"channel={CAMERA_CHANNEL}, "
        f"subtype={CAMERA_SUBTYPE})."
    )

    read_failures = 0

    try:
        while True:
            if cv2 is None:
                CAMERA_CONNECTED = False
                CAMERA_LAST_ERROR = (
                    "OpenCV no esta disponible."
                )
                time.sleep(1.0)
                continue

            # Reapertura solicitada (botón reconectar / API).
            # Solo el worker libera el VideoCapture.
            force_reopen = False
            with camera_lock:
                force_reopen = CAMERA_FORCE_REOPEN
                if force_reopen:
                    CAMERA_FORCE_REOPEN = False
                    try:
                        if camera_capture is not None:
                            camera_capture.release()
                    except Exception:
                        pass
                    camera_capture = None
                    CAMERA_LAST_CONNECT_ATTEMPT = 0.0

            if force_reopen:
                _clear_latest_frame()
                CAMERA_CONNECTED = False

            # --- abrir sesión si hace falta ---
            with camera_lock:
                cap = get_camera()

            if cap is None:
                _note_camera_signal_lost(
                    CAMERA_LAST_ERROR
                    or "Cámara sin señal."
                )
                with CAMERA_RECONNECT_LOCK:
                    CAMERA_RECONNECTING = True

                # Política de discovery de la reconexión clásica.
                # direct_failures se deriva de CAMERA_FAILURE_COUNT.
                _try_discover_camera_after_failures(
                    CAMERA_FAILURE_COUNT
                )

                time.sleep(
                    max(float(CAMERA_RETRY_SECONDS), 1.0)
                )
                continue

            # --- ÚNICO cap.read() continuo del pipeline ---
            try:
                ok, frame = cap.read()
            except Exception as error:
                ok = False
                frame = None
                CAMERA_LAST_ERROR = (
                    f"Error de lectura RTSP: {error}"
                )

            if ok and frame is not None:
                timestamp, sequence = _publish_latest_frame(
                    frame
                )

                CAMERA_CONNECTED = True
                CAMERA_FAILURE_COUNT = 0
                CAMERA_LAST_ERROR = None
                CAMERA_LAST_OK_AT = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                read_failures = 0

                with CAMERA_RECONNECT_LOCK:
                    CAMERA_RECONNECTING = False

                # Diagnóstico opcional del lector (limitado).
                log_camera_latency(
                    source="camera_capture_worker",
                    sequence=sequence,
                    timestamp_monotonic=timestamp,
                )
                continue

            # --- fallo de lectura ---
            read_failures += 1
            _note_camera_signal_lost(
                CAMERA_LAST_ERROR
                or "No se está recibiendo video de la cámara."
            )
            _clear_latest_frame()

            if (
                read_failures
                >= max(1, CAMERA_READ_FAIL_RELEASE_LIMIT)
            ):
                with camera_lock:
                    try:
                        if camera_capture is not None:
                            camera_capture.release()
                    except Exception:
                        pass
                    camera_capture = None

                CAMERA_LAST_CONNECT_ATTEMPT = 0.0
                read_failures = 0

                with CAMERA_RECONNECT_LOCK:
                    CAMERA_RECONNECTING = True

                _try_discover_camera_after_failures(
                    CAMERA_FAILURE_COUNT
                )

            time.sleep(0.05)

    finally:
        with CAMERA_RECONNECT_LOCK:
            CAMERA_RECONNECTING = False


def ensure_camera_capture_worker():
    """
    Garantiza EXACTAMENTE UN camera_capture_worker.
    No crea un thread por request HTTP si ya corre.
    """
    global CAMERA_CAPTURE_WORKER
    global CAMERA_CAPTURE_WORKER_STARTED

    if cv2 is None:
        return False

    with CAMERA_CAPTURE_WORKER_LOCK:
        if (
            CAMERA_CAPTURE_WORKER_STARTED
            and CAMERA_CAPTURE_WORKER is not None
            and CAMERA_CAPTURE_WORKER.is_alive()
        ):
            return True

        CAMERA_CAPTURE_WORKER = threading.Thread(
            target=camera_capture_worker,
            daemon=True,
            name="camera-capture-worker",
        )
        CAMERA_CAPTURE_WORKER.start()
        CAMERA_CAPTURE_WORKER_STARTED = True

    return True


def ensure_camera_reconnect_worker():
    """
    Compatibilidad con rutas previas.
    Solo asegura el único capture worker (ya incluye reconexión).
    """
    if CAMERA_CONNECTED:
        return False

    with CAMERA_RECONNECT_LOCK:
        CAMERA_RECONNECTING = True

    return ensure_camera_capture_worker()


def get_camera_status():
    """
    Estado operativo. Si no hay señal, asegura el capture worker
    (que también reconecta). No abre RTSP por su cuenta.
    """
    ensure_camera_capture_worker()

    if CAMERA_CONNECTED:
        return {
            "connected": True,
            "reconnecting": False,
            "state": "CONNECTED",
            "message": (
                "Camara conectada y transmitiendo."
            ),
            "last_ok_at": CAMERA_LAST_OK_AT,
        }

    with CAMERA_RECONNECT_LOCK:
        reconnecting = CAMERA_RECONNECTING

    if reconnecting:
        return {
            "connected": False,
            "reconnecting": True,
            "state": "CHECKING",
            "message": (
                "Reconectando automaticamente "
                "con la camara."
            ),
            "last_ok_at": CAMERA_LAST_OK_AT,
        }

    return {
        "connected": False,
        "reconnecting": False,
        "state": "DISCONNECTED",
        "message": (
            CAMERA_LAST_ERROR
            or "No se recibe senal de la camara."
        ),
        "last_ok_at": CAMERA_LAST_OK_AT,
    }



def make_camera_error_frame(message="CAMARA NO DISPONIBLE"):
    """
    Genera una imagen de error para que la interfaz no se rompa
    cuando la cámara esté desconectada o apagada.
    """
    if cv2 is not None and np is not None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (22, 22, 22)

        cv2.putText(
            frame,
            message,
            (330, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "Verifique conexion, energia o CAMERA_SOURCE",
            (315, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )

        return frame

    return None


def generate_placeholder_frame(message="CAMARA NO DISPONIBLE"):
    """
    Devuelve bytes JPEG de respaldo. Se usa si OpenCV no puede codificar.
    """
    frame = make_camera_error_frame(message)

    if frame is not None:
        jpg = encode_jpeg(frame)

        if jpg:
            return jpg

    from PIL import Image, ImageDraw
    import io

    img = Image.new("RGB", (1280, 720), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    draw.text((430, 350), message, fill=(255, 255, 255))
    bio = io.BytesIO()
    img.save(bio, format="JPEG")
    return bio.getvalue()


def encode_jpeg(frame):
    if cv2 is None or frame is None:
        return None

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not ok:
        return None

    return buffer.tobytes()


def get_roi_bounds(frame):
    """Devuelve límites ROI válidos en píxeles para el frame recibido."""
    h, w = frame.shape[:2]
    x1 = int(max(0.0, min(ROI_X1, 0.99)) * w)
    y1 = int(max(0.0, min(ROI_Y1, 0.99)) * h)
    x2 = int(max(0.01, min(ROI_X2, 1.0)) * w)
    y2 = int(max(0.01, min(ROI_Y2, 1.0)) * h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("La región ROI configurada no es válida.")

    return x1, y1, x2, y2


def crop_inspection_roi(frame):
    x1, y1, x2, y2 = get_roi_bounds(frame)
    return frame[y1:y2, x1:x2]


def compute_roi_coverage(frame):
    """
    Cobertura de prenda dentro del ROI — MISMA fórmula que usaba
    auto_inspection_worker en línea (ahora compartida con captura IA).
    Lanza si la segmentación falla; el caller decide cómo tratarlo.
    """
    garment_mask = create_garment_mask(frame)
    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_bounds(frame)
    roi_mask = garment_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_area = float(
        max(
            1,
            roi_mask.shape[0] * roi_mask.shape[1],
        )
    )
    return float(cv2.countNonZero(roi_mask) / roi_area)


def draw_inspection_overlay(frame):
    """
    Dibuja el área de inspección sobre el video en vivo.
    No afecta la imagen guardada para análisis.
    """
    if cv2 is None or frame is None:
        return frame

    output = frame.copy()
    x1, y1, x2, y2 = get_roi_bounds(output)

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 180, 255),
        2,
    )

    cv2.putText(
        output,
        "AREA DE INSPECCION",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 180, 255),
        2,
    )

    return output


def generate_video_feed():
    """
    Stream MJPEG desde latest_frame.

    NO ejecuta cap.read(). El capture worker sigue drenando RTSP.
    """
    while True:
        ensure_camera_capture_worker()

        frame, timestamp, sequence = get_latest_camera_frame()

        if frame is None:
            frame_to_send = make_camera_error_frame(
                "RECONECTANDO CAMARA"
            )
        else:
            log_camera_latency(
                source="video_feed",
                sequence=sequence,
                timestamp_monotonic=timestamp,
            )
            frame_to_send = draw_inspection_overlay(frame)

        jpg = encode_jpeg(frame_to_send)

        if jpg is None:
            jpg = generate_placeholder_frame(
                "ERROR DE VIDEO"
            )

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg
            + b"\r\n"
        )

        time.sleep(0.08)

def save_frame_to_static(frame):
    """
    Guarda una captura tomada desde la cámara.
    Devuelve: ruta absoluta, ruta relativa dentro de /static.
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6].upper()}.jpg"
    abs_path = CAPTURE_DIR / filename
    rel_path = f"captures/{filename}"

    saved = cv2.imwrite(str(abs_path), frame)
    if not saved:
        raise IOError(f"No se pudo guardar la captura en {abs_path}")

    return abs_path, rel_path


def save_image(file_storage=None):
    """
    Compatibilidad con la pantalla anterior de inspección.
    Si viene archivo subido, lo guarda.
    Si no viene archivo, intenta capturar desde la cámara.
    Si la cámara no está disponible, devuelve None.
    """
    if file_storage and file_storage.filename:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

        name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        path = CAPTURE_DIR / name
        file_storage.save(path)

        return path, f"captures/{name}"

    ok, frame = read_camera_frame(source="save_image")

    if not ok:
        return None, None

    return save_frame_to_static(frame)


def register_inspection_from_frame(
    frame,
    notes="Registro generado por estación de inspección.",
    garment_token=None,
    decision_start=None,
    frame_selection_ms=None,
    source="auto",
):
    """
    Ejecuta la detección, guarda la evidencia y registra la blusa
    dentro del lote activo.

    Los contadores del lote se sincronizan con las inspecciones
    realmente almacenadas para evitar inconsistencias.

    garment_token: identidad temporal de la prenda (Fase 3).
    decision_start: time.perf_counter() cuando el frame válido
        de esa prenda quedó disponible (para decision_ms).
    frame_selection_ms: duración de la selección de best_frame.
    source: "auto" | "manual" (solo logging).
    """
    global AUTO_INSPECTION_ENABLED

    # Mutex estación: preparación IA no genera inspections productivas.
    assert_production_allowed("registro de inspección productiva")

    if frame is None:
        raise ValueError(
            "No se recibió imagen de cámara para registrar la inspección."
        )

    if garment_token is None:
        garment_token = next_garment_token()

    total_start = (
        decision_start
        if decision_start is not None
        else time.perf_counter()
    )
    metrics = {}

    if frame_selection_ms is not None:
        metrics["frame_selection_ms"] = float(frame_selection_ms)

    t_capture_0 = time.perf_counter()
    img_path, img_rel = save_frame_to_static(frame)
    metrics["capture_ms"] = (
        time.perf_counter() - t_capture_0
    ) * 1000.0

    detect_metrics = {}
    status, defect, conf, zone, result_rel = detect_defect(
        img_path,
        metrics=detect_metrics,
        garment_token=garment_token,
        decision_start=total_start,
        source=source,
    )
    metrics.update(detect_metrics)

    if "decision_ms" not in metrics:
        metrics["decision_ms"] = (
            time.perf_counter() - total_start
        ) * 1000.0

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    code = (
        f"INS-{datetime.now().strftime('%H%M%S')}-"
        f"{uuid.uuid4().hex[:4].upper()}"
    )

    ai_decision = (
        "NORMAL"
        if status == "Aprobado"
        else "ANOMALIA"
    )

    review_status = (
        "NO_REQUIERE_REVISION"
        if ai_decision == "NORMAL"
        else "PENDIENTE"
    )

    t_database_0 = time.perf_counter()
    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()

        cur.execute(
            """
            SELECT
                id,
                code,
                garment_model_id,
                ai_model_id,
                planned_quantity,
                status
            FROM batches
            WHERE status = 'EN_INSPECCION'
            ORDER BY id DESC
            LIMIT 1
            FOR UPDATE
            """
        )

        batch = cur.fetchone()

        if batch is None:
            raise RuntimeError(
                "No hay un lote activo. "
                "Crea o inicia un lote antes de inspeccionar."
            )

        cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM inspections
            WHERE batch_id = %s
            """,
            (batch["id"],),
        )

        processed_before = int(
            cur.fetchone()["total"] or 0
        )

        if processed_before >= int(
            batch["planned_quantity"]
        ):
            cur.execute(
                """
                UPDATE batches
                SET status = 'REVISION_PENDIENTE',
                    inspection_completed_at = %s
                WHERE id = %s
                """,
                (
                    created_at,
                    batch["id"],
                ),
            )

            conn.commit()
            AUTO_INSPECTION_ENABLED = False

            raise RuntimeError(
                "El lote ya alcanz? la cantidad planificada."
            )

        batch_position = processed_before + 1

        cur.execute(
            """
            INSERT INTO inspections (
                code,
                created_at,
                garment_type,
                size,
                status,
                defect_type,
                confidence,
                zone,
                image_original,
                image_result,
                human_validation,
                notes,
                batch_id,
                ai_model_id,
                batch_position,
                ai_decision,
                review_status,
                audit_selected
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                code,
                created_at,
                "Blusa",
                "S",
                status,
                defect,
                conf,
                zone,
                img_rel,
                result_rel,
                (
                    "No requerida"
                    if ai_decision == "NORMAL"
                    else "Pendiente"
                ),
                notes,
                batch["id"],
                batch["ai_model_id"],
                batch_position,
                ai_decision,
                review_status,
                0,
            ),
        )

        # La propia tabla de inspecciones es la fuente de verdad.
        cur.execute(
            """
            SELECT
                COUNT(*) AS processed_quantity,
                COALESCE(
                    SUM(
                        CASE
                            WHEN ai_decision = 'NORMAL'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS auto_approved,
                COALESCE(
                    SUM(
                        CASE
                            WHEN ai_decision = 'ANOMALIA'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS alerts
            FROM inspections
            WHERE batch_id = %s
            """,
            (batch["id"],),
        )

        stats = cur.fetchone()

        processed_quantity = int(
            stats["processed_quantity"] or 0
        )

        auto_approved = int(
            stats["auto_approved"] or 0
        )

        alerts = int(
            stats["alerts"] or 0
        )

        batch_complete = (
            processed_quantity
            >= int(batch["planned_quantity"])
        )

        if batch_complete:
            cur.execute(
                """
                UPDATE batches
                SET status = 'REVISION_PENDIENTE',
                    inspection_completed_at = %s
                WHERE id = %s
                """,
                (
                    created_at,
                    batch["id"],
                ),
            )

            AUTO_INSPECTION_ENABLED = False

        conn.commit()
        metrics["database_ms"] = (
            time.perf_counter() - t_database_0
        ) * 1000.0

    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

    metrics["total_complete_ms"] = (
        time.perf_counter() - total_start
    ) * 1000.0

    if ai_decision == "ANOMALIA":
        enrich_quality_alert_event(
            garment_token=garment_token,
            batch_id=batch["id"],
            batch_position=processed_quantity,
            code=code,
        )

    log_latency_metrics(
        garment_token=garment_token,
        metrics=metrics,
        ai_decision=ai_decision,
        batch_id=batch["id"],
        batch_position=processed_quantity,
        source=source,
    )

    return {
        "code": code,
        "status": status,
        "defect_type": defect,
        "confidence": (
            float(conf)
            if conf is not None
            else None
        ),
        "zone": zone,
        "image_original": img_rel,
        "image_result": result_rel,
        "created_at": created_at,
        "batch_id": batch["id"],
        "batch_code": batch["code"],
        "batch_position": processed_quantity,
        "processed_quantity": processed_quantity,
        "planned_quantity": int(
            batch["planned_quantity"]
        ),
        "auto_approved": auto_approved,
        "alerts": alerts,
        "ai_decision": ai_decision,
        "review_status": review_status,
        "batch_complete": batch_complete,
        "garment_token": garment_token,
    }



def auto_inspection_worker():
    """
    Inspección automática orientada a cinta transportadora.

    Flujo:
    1. Espera el área vacía.
    2. Detecta entrada de una blusa mediante su máscara.
    3. Sigue varios frames mientras entra.
    4. Conserva el frame con mayor cobertura.
    5. Ejecuta exactamente el mismo pipeline de IA que el manual.
    6. No vuelve a registrar hasta que esa blusa haya salido.
    """
    global AUTO_INSPECTION_ENABLED
    global AUTO_LAST_RESULT
    global AUTO_LAST_ERROR
    global AUTO_LAST_CAPTURE_TIME

    # Estado de la prenda que actualmente atraviesa la estación.
    tracking = False
    waiting_for_exit = False

    present_frames = 0
    exit_frames = 0
    tracked_frames = 0

    best_frame = None
    best_coverage = 0.0

    # Identidad temporal de la prenda del ciclo actual (Fase 3).
    garment_token = None
    tracking_started_perf = None

    # Experimento: primer cruce 40/45/50 + frames candidato en RAM.
    seen_coverage_levels = set()
    benchmark_frames = {}

    # Secuencia last para no reprocesar el mismo frame.
    last_camera_sequence = -1

    print(
        "[AUTO] Modo automático iniciado. "
        "Esperando entrada de una blusa."
    )

    while AUTO_INSPECTION_ENABLED:
        try:
            # Mutex: si hay sesión de captura IA, la producción no consume frames.
            if is_ai_capture_mode_active():
                AUTO_LAST_ERROR = (
                    "Modo preparación IA activo: producción en pausa."
                )
                time.sleep(0.5)
                continue

            # Consumidor de latest_frame: NO cap.read().
            # Mientras PatchCore corre, camera_capture_worker
            # sigue drenando RTSP.
            frame, frame_timestamp, sequence = (
                get_latest_camera_frame()
            )

            if frame is None:
                AUTO_LAST_ERROR = (
                    "Cámara no disponible."
                )
                time.sleep(0.5)
                continue

            if sequence == last_camera_sequence:
                # Misma secuencia: no reintentar el mismo frame.
                time.sleep(0.05)
                continue

            last_camera_sequence = sequence
            log_camera_latency(
                source="auto_inspection",
                sequence=sequence,
                timestamp_monotonic=frame_timestamp,
            )

            # -------------------------------------------------
            # Medir presencia de prenda.
            #
            # Se reutiliza EXACTAMENTE la segmentación de la
            # blusa que usa PatchCore (compute_roi_coverage).
            # -------------------------------------------------
            try:
                coverage = compute_roi_coverage(frame)

            except Exception:
                # Una silueta demasiado pequeña normalmente
                # significa que la prenda apenas está entrando,
                # saliendo, o que el área está vacía.
                coverage = 0.0

            # -------------------------------------------------
            # ESTADO 1: ya inspeccionamos esta blusa.
            # No rearmar hasta verla salir completamente.
            # -------------------------------------------------
            if waiting_for_exit:

                if (
                    coverage
                    <= AUTO_GARMENT_EXIT_COVERAGE
                ):
                    exit_frames += 1
                else:
                    exit_frames = 0

                if (
                    exit_frames
                    >= AUTO_GARMENT_EXIT_FRAMES
                ):
                    waiting_for_exit = False
                    tracking = False

                    present_frames = 0
                    exit_frames = 0
                    tracked_frames = 0

                    best_frame = None
                    best_coverage = 0.0
                    garment_token = None
                    tracking_started_perf = None
                    seen_coverage_levels = set()
                    benchmark_frames = {}

                    print(
                        "[AUTO] Blusa anterior sali?. "
                        "Sistema rearmado."
                    )

                time.sleep(0.20)
                continue

            # -------------------------------------------------
            # ESTADO 2: detectar comienzo de entrada.
            # -------------------------------------------------
            if (
                coverage
                >= AUTO_GARMENT_ENTER_COVERAGE
            ):
                present_frames += 1

                if not tracking:
                    tracking = True
                    tracked_frames = 0
                    best_frame = None
                    best_coverage = 0.0
                    garment_token = next_garment_token()
                    tracking_started_perf = time.perf_counter()
                    seen_coverage_levels = set()
                    benchmark_frames = {}
                    timeline_mark(
                        garment_token,
                        "garment_detected",
                    )

                    print(
                        "[AUTO] Entrada de prenda detectada. "
                        f"token={garment_token}"
                    )

                tracked_frames += 1

                # Conservar siempre el frame donde la prenda
                # ocupa mayor superficie dentro del ROI.
                # BEST_FRAME DE PRODUCCIÓN: misma regla que antes.
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_frame = frame.copy()

                # -------------------------------------------------
                # Experimento (no altera best_frame productivo):
                # primer cruce de 40/45/50 => marca timeline y,
                # solo si FRAME_SELECTION_BENCHMARK, copia en RAM.
                # -------------------------------------------------
                if tracking and garment_token is not None:
                    for level in BENCHMARK_COVERAGE_LEVELS:
                        if level in seen_coverage_levels:
                            continue
                        if coverage < level:
                            continue

                        seen_coverage_levels.add(level)
                        pct = int(round(level * 100))
                        timeline_mark(
                            garment_token,
                            f"coverage_{pct}_reached",
                        )

                        if FRAME_SELECTION_BENCHMARK:
                            benchmark_frames[level] = {
                                "frame": frame.copy(),
                                "coverage": float(coverage),
                            }

                print(
                    "[AUTO] "
                    f"cobertura={coverage * 100:.2f}%, "
                    f"mejor={best_coverage * 100:.2f}%, "
                    f"frames={present_frames}"
                )

            else:
                # Si apenas comenzó una detección pero no llegó
                # a ser una prenda válida, descartarla.
                if (
                    tracking
                    and best_coverage
                    < AUTO_GARMENT_CAPTURE_COVERAGE
                ):
                    tracking = False
                    present_frames = 0
                    tracked_frames = 0
                    best_frame = None
                    best_coverage = 0.0
                    if garment_token is not None:
                        reset_timeline(garment_token)
                    garment_token = None
                    tracking_started_perf = None
                    seen_coverage_levels = set()
                    benchmark_frames = {}

            # -------------------------------------------------
            # ESTADO 3: determinar el momento de captura.
            #
            # Esperamos varios frames y seleccionamos el mejor.
            # Si empieza a salir, usamos el máximo ya observado.
            # -------------------------------------------------
            if (
                tracking
                and present_frames
                >= AUTO_GARMENT_CONFIRM_FRAMES
                and best_coverage
                >= AUTO_GARMENT_CAPTURE_COVERAGE
            ):
                coverage_dropped = (
                    coverage
                    < best_coverage - 0.015
                )

                maximum_tracking = (
                    tracked_frames
                    >= AUTO_GARMENT_MAX_TRACK_FRAMES
                )

                # No capturar ?nicamente porque hayan pasado
                # algunos frames: eso hacía que la blusa se
                # inspeccionara cuando todavía estaba entrando.
                #
                # Se captura cuando:
                # 1. ya alcanz? cobertura de prenda completa y
                #    comienza a salir, o
                # 2. lleva suficientes frames prácticamente
                #    centrada en el área.
                if (
                    coverage_dropped
                    or maximum_tracking
                ):
                    now = time.time()

                    if (
                        now - AUTO_LAST_CAPTURE_TIME
                        < AUTO_COOLDOWN_SECONDS
                    ):
                        time.sleep(0.20)
                        continue

                    if best_frame is None:
                        time.sleep(0.20)
                        continue

                    print(
                        "[AUTO] Prenda completa confirmada. "
                        f"Capturando mejor frame "
                        f"({best_coverage * 100:.2f}%). "
                        f"token={garment_token}"
                    )

                    timeline_mark(
                        garment_token,
                        "frame_selected",
                    )

                    frame_selection_ms = None
                    if tracking_started_perf is not None:
                        frame_selection_ms = (
                            time.perf_counter()
                            - tracking_started_perf
                        ) * 1000.0

                    # La captura válida de ESTA prenda está
                    # disponible a partir de este instante.
                    decision_start = time.perf_counter()

                    result = (
                        register_inspection_from_frame(
                            best_frame,
                            notes=(
                                "Registro automático generado "
                                "por presencia de prenda en "
                                "cinta transportadora."
                            ),
                            garment_token=garment_token,
                            decision_start=decision_start,
                            frame_selection_ms=frame_selection_ms,
                            source="auto",
                        )
                    )

                    timeline_mark(
                        garment_token,
                        "inspection_complete",
                    )
                    log_timeline(garment_token)

                    # Persistir candidatos SOLO después de la
                    # inspección productiva terminar.
                    if FRAME_SELECTION_BENCHMARK and benchmark_frames:
                        persist_benchmark_frames(
                            garment_token=garment_token,
                            frames_by_level=dict(benchmark_frames),
                            production_result=result.get("ai_decision"),
                            production_score=result.get("confidence"),
                            batch_id=result.get("batch_id"),
                            batch_position=result.get("batch_position"),
                        )

                    AUTO_LAST_RESULT = result
                    AUTO_LAST_ERROR = None
                    AUTO_LAST_CAPTURE_TIME = (
                        time.time()
                    )

                    print(
                        "[AUTO] Inspección registrada: "
                        f"{result.get('code')} | "
                        f"{result.get('status')} | "
                        f"{result.get('defect_type')} | "
                        f"token={result.get('garment_token')}"
                    )

                    # Esta misma blusa no puede generar otro
                    # registro hasta abandonar completamente ROI.
                    waiting_for_exit = True
                    tracking = False

                    present_frames = 0
                    exit_frames = 0
                    tracked_frames = 0

                    best_frame = None
                    best_coverage = 0.0
                    garment_token = None
                    tracking_started_perf = None
                    seen_coverage_levels = set()
                    benchmark_frames = {}

                    continue

        except Exception as error:
            AUTO_LAST_ERROR = str(error)

            print(
                "[AUTO] Error controlado: "
                f"{error}"
            )

        time.sleep(0.20)



def normalize_for_json(value):
    """
    Convierte objetos de MySQL como Decimal o datetime a valores serializables.
    """
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    try:
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
    except Exception:
        pass

    return value


def row_to_json(row):
    if row is None:
        return None

    return {key: normalize_for_json(value) for key, value in row.items()}

# ============================================================
# VISIÓN ARTIFICIAL / IA
# ============================================================

def load_yolo_model():
    global yolo_model

    if YOLO is None:
        return None

    if not MODEL_PATH.exists():
        return None

    if MODEL_PATH.stat().st_size < 1024:
        print("[IA] models/best.pt existe, pero está vacío o incompleto. Se usará OpenCV.")
        return None

    if yolo_model is None:
        try:
            yolo_model = YOLO(str(MODEL_PATH))
            print("[IA] Modelo YOLO cargado correctamente.")
        except Exception as e:
            print(f"[IA] No se pudo cargar YOLO. Se usará OpenCV. Error: {e}")
            yolo_model = None
            return None

    return yolo_model

def create_garment_mask(image):
    """
    Obtiene la silueta completa de la prenda (delegado compartido).

    La implementacion vive en patchcore_preprocess para que el
    entrenador de FASE 3A use exactamente el mismo preprocesamiento
    que la inferencia productiva.
    """
    return patchcore_preprocess.create_garment_mask(
        image,
        get_roi_bounds(image),
        cv2_mod=cv2,
        np_mod=np,
    )






GUIDED_SEGMENTATION_SCALE = 0.5
GUIDED_MIN_CANDIDATE_FRACTION = 0.04
GUIDED_COVERAGE_MIN = 0.20
GUIDED_COVERAGE_MAX = 0.75


def _guided_fill_holes(mask):
    """Rellena los huecos encerrados por una silueta ya formada."""
    inverted = cv2.bitwise_not(mask)
    flood = inverted.copy()
    seed_mask = np.zeros(
        (inverted.shape[0] + 2, inverted.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flood, seed_mask, (0, 0), 0)
    return cv2.bitwise_or(mask, flood)


def _guided_panel_bounds(gray, diagnostics, scale):
    """
    Límites del panel iluminado dentro del ROI.

    El ROI de la estación a veces incluye la zona oscura de fuera del
    panel. Si no se recorta, esa zona se confunde con una prenda.
    """
    percentile = float(np.percentile(gray, 35))
    bright = np.where(gray >= percentile, 255, 0).astype(np.uint8)
    kernel_size = max(5, int(25 * scale) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        bright,
        8,
    )

    if count <= 1:
        diagnostics["panel"] = None
        return None

    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left, top, width, height, area = (
        int(stats[index, key]) for key in range(5)
    )
    fraction = float(area) / float(gray.size)
    diagnostics["panel"] = {
        "percentile": round(percentile, 1),
        "fraction": round(fraction, 3),
        "box": (left, top, width, height),
    }

    if fraction < 0.50:
        return None

    return left, top, width, height


def _guided_adaptive_mask(sub_roi, roi_area, diagnostics, scale):
    """
    Segmentación sin color fijo: bordes en espacio LAB + islas.

    Reglas, todas derivadas del propio fotograma (sin umbrales fijos
    de color):

    1. Umbral Otsu sobre la magnitud del gradiente en LAB, con
       histéresis (fuerte = Otsu, débil = 40% de Otsu) para no romper
       el contorno de la prenda.
    2. Se consideran islas a las regiones NO unidas al borde del panel:
       lo que está pegado al borde es el fondo, no la prenda.
    3. Fracción de candidata entre 4% y 75% del ROI.
    4. Cobertura final dentro de los mismos límites que producción
       (20%–75%): fuera de ese rango no se devuelve silueta.
    """
    height, width = sub_roi.shape[:2]
    lab = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2LAB)
    blur = cv2.GaussianBlur(lab, (5, 5), 0).astype(np.float32)

    magnitude = np.zeros(blur.shape[:2], dtype=np.float32)
    for channel in range(3):
        grad_x = cv2.Scharr(blur[:, :, channel], cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(blur[:, :, channel], cv2.CV_32F, 0, 1)
        magnitude += cv2.magnitude(grad_x, grad_y) ** 2
    magnitude = np.sqrt(magnitude)

    magnitude8 = np.clip(magnitude, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(
        magnitude8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu = float(max(otsu, 8.0))

    strong = np.where(magnitude8 >= otsu, 255, 0).astype(np.uint8)
    weak = np.where(magnitude8 >= 0.4 * otsu, 255, 0).astype(np.uint8)
    linked = strong.copy()
    hysteresis = max(3, int(9 * scale) | 1)
    for _ in range(4):
        grown = cv2.dilate(
            linked,
            np.ones((hysteresis, hysteresis), dtype=np.uint8),
        )
        linked = np.where((grown > 0) & (weak > 0), 255, 0).astype(np.uint8)

    close_size = max(9, int(41 * scale) | 1)
    barriers = cv2.morphologyEx(
        linked,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), dtype=np.uint8),
    )
    barriers = cv2.dilate(
        barriers,
        np.ones((max(3, int(5 * scale)),) * 2, dtype=np.uint8),
    )

    free = cv2.bitwise_not(barriers)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        free,
        4,
    )

    border = max(3, int(6 * scale))
    candidates = []
    rejected = []
    selected = np.zeros((height, width), dtype=np.uint8)

    for index in range(1, count):
        left, top, box_width, box_height, area = (
            int(stats[index, key]) for key in range(5)
        )
        fraction = float(area) / float(roi_area)
        touches_border = (
            left <= border
            or top <= border
            or (left + box_width) >= width - border
            or (top + box_height) >= height - border
        )
        candidates.append({
            "fraction": round(fraction, 3),
            "box": (left, top, box_width, box_height),
            "touch_border": bool(touches_border),
        })

        if touches_border:
            rejected.append({"fraction": round(fraction, 3), "why": "borde"})
            continue
        if fraction < GUIDED_MIN_CANDIDATE_FRACTION:
            rejected.append({"fraction": round(fraction, 3), "why": "minima"})
            continue
        if fraction > GUIDED_COVERAGE_MAX:
            rejected.append({"fraction": round(fraction, 3), "why": "maxima"})
            continue

        selected[labels == index] = 255

    diagnostics["gradient_otsu"] = round(otsu, 1)
    diagnostics["candidates"] = candidates[:5]
    diagnostics["rejected"] = rejected[:5]

    if cv2.countNonZero(selected) == 0:
        diagnostics["reason"] = "sin_candidata"
        return None

    selected = _guided_fill_holes(selected)
    contours, _ = cv2.findContours(
        selected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        diagnostics["reason"] = "sin_contorno"
        return None

    garment_contour = max(contours, key=cv2.contourArea)
    garment = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(
        garment,
        [garment_contour],
        -1,
        255,
        cv2.FILLED,
    )

    coverage = float(cv2.countNonZero(garment)) / float(roi_area)
    diagnostics["adaptive_coverage"] = round(coverage, 4)

    if coverage < GUIDED_COVERAGE_MIN or coverage > GUIDED_COVERAGE_MAX:
        diagnostics["reason"] = "cobertura_adaptativa"
        return None

    return garment


def create_guided_garment_mask(image, diagnostics=None):
    """
    Silueta de la prenda para la captura guiada (fallback incluido).

    Cascada:
      1. Semilla cromática de producción (create_garment_mask): misma
         regla que la estación automática, para la blusa rosa.
      2. Segmentación adaptativa sin color fijo, para prendas que la
         semilla no reconoce (azul, verde, gris, negra, blanca...).
      3. Máscara vacía con el motivo registrado.

    Rellena `diagnostics` con nonzero, %, dimensiones, ROI min/max/mean
    y el método usado, para poder diagnosticar SIN_PRENDA desde el log.
    """
    if image is None:
        raise ValueError("No se recibió una imagen válida.")

    if diagnostics is None:
        diagnostics = {}

    height, width = image.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_bounds(image)
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    if roi is None or roi.size == 0:
        raise ValueError("El ROI de inspección está vacío.")

    diagnostics["roi"] = (int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2))
    diagnostics["frame"] = (int(width), int(height))

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    diagnostics["roi_stats"] = {
        "min": int(gray_roi.min()),
        "max": int(gray_roi.max()),
        "mean": round(float(gray_roi.mean()), 1),
    }

    def report(method, mask, reason=None):
        diagnostics["method"] = method
        nonzero = int(cv2.countNonZero(mask))
        roi_mask = mask[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_nonzero = int(cv2.countNonZero(roi_mask))
        roi_area = float(max(1, roi_mask.size))
        diagnostics["mask_nonzero"] = nonzero
        # % sobre el ROI (la misma base que coverage) y sobre el frame.
        diagnostics["mask_pct"] = round(roi_nonzero / roi_area, 4)
        diagnostics["mask_frame_pct"] = round(
            nonzero / float(height * width),
            4,
        )
        diagnostics["mask_bbox"] = mask_bbox(mask, cv2)
        if reason is not None:
            diagnostics["reason"] = reason
        return mask

    zeros = np.zeros((height, width), dtype=np.uint8)

    seed_diagnostics = {}
    try:
        seed_mask = create_garment_mask(image)
    except Exception as error:  # noqa: BLE001 - el motivo queda en el log
        seed_mask = None
        seed_diagnostics["error"] = str(error)

    if seed_mask is not None:
        seed_roi = seed_mask[roi_y1:roi_y2, roi_x1:roi_x2]
        seed_coverage = float(cv2.countNonZero(seed_roi)) / float(
            max(1, seed_roi.size)
        )
        diagnostics["seed"] = {
            "nonzero": int(cv2.countNonZero(seed_roi)),
            "coverage": round(seed_coverage, 4),
            **seed_diagnostics,
        }
        if GUIDED_COVERAGE_MIN <= seed_coverage <= GUIDED_COVERAGE_MAX:
            return report("semilla", seed_mask)
    else:
        diagnostics["seed"] = dict(seed_diagnostics)

    scale = float(GUIDED_SEGMENTATION_SCALE)
    working = cv2.resize(
        roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    working_area = float(working.shape[0] * working.shape[1])
    adaptive_diagnostics = diagnostics.setdefault("adaptive", {})

    panel = _guided_panel_bounds(
        cv2.cvtColor(working, cv2.COLOR_BGR2GRAY),
        adaptive_diagnostics,
        scale,
    )

    if panel is None:
        adaptive_diagnostics["reason"] = "sin_panel"
        return report("ninguno", zeros, "sin_prenda")

    left, top, box_width, box_height = panel
    sub_roi = working[top:top + box_height, left:left + box_width]
    garment_small = _guided_adaptive_mask(
        sub_roi,
        working_area,
        adaptive_diagnostics,
        scale,
    )

    if garment_small is None:
        return report(
            "ninguno",
            zeros,
            adaptive_diagnostics.get("reason") or "sin_prenda",
        )

    # garment_small vive en el recorte del panel: hay que devolverlo a
    # coordenadas del ROI antes de escalar al tamaño del frame.
    garment_working = np.zeros(working.shape[:2], dtype=np.uint8)
    garment_working[top:top + box_height, left:left + box_width] = garment_small

    garment_roi = cv2.resize(
        garment_working,
        (roi_x2 - roi_x1, roi_y2 - roi_y1),
        interpolation=cv2.INTER_NEAREST,
    )
    garment_mask = np.zeros((height, width), dtype=np.uint8)
    garment_mask[roi_y1:roi_y2, roi_x1:roi_x2] = garment_roi

    return report("adaptativo", garment_mask)


def localize_patchcore_anomaly(
    image,
    anomaly_map,
    confidence,
    is_anomaly,
    garment_mask=None,
):
    """
    Localiza la anomalía principal dentro de la blusa.

    No depende de una posición fija. Combina el mapa PatchCore con
    información de contraste visual y aplica solamente una
    penalización suave a los bordes.
    """
    if image is None:
        raise ValueError(
            "No se recibió una imagen válida."
        )

    output = image.copy()
    height, width = image.shape[:2]

    if garment_mask is None:
        garment_mask = (
            create_garment_mask(image)
        )

    garment_mask = np.asarray(
        garment_mask,
        dtype=np.uint8,
    )

    if garment_mask.shape[:2] != (
        height,
        width,
    ):
        garment_mask = cv2.resize(
            garment_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    if cv2.countNonZero(
        garment_mask
    ) == 0:
        raise RuntimeError(
            "No se pudo separar la blusa del fondo."
        )

    if not is_anomaly:
        cv2.putText(
            output,
            "APROBADO - SIN ANOMALIA",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 170, 0),
            2,
            cv2.LINE_AA,
        )

        return output, "No aplica", 0

    raw_map = np.asarray(
        anomaly_map,
        dtype=np.float32,
    ).squeeze()

    if raw_map.ndim != 2:
        raise ValueError(
            "Mapa PatchCore inválido: "
            f"{raw_map.shape}"
        )

    raw_map = np.nan_to_num(
        raw_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    raw_map = cv2.resize(
        raw_map,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    valid_pixels = raw_map[
        garment_mask > 0
    ]

    if valid_pixels.size == 0:
        raise RuntimeError(
            "No existen píxeles válidos en la prenda."
        )

    low = float(
        np.quantile(
            valid_pixels,
            0.05,
        )
    )

    high = float(
        np.quantile(
            valid_pixels,
            0.995,
        )
    )

    if high > low:
        patch_signal = np.clip(
            (raw_map - low)
            / (high - low),
            0.0,
            1.0,
        )
    else:
        patch_signal = np.zeros_like(
            raw_map,
            dtype=np.float32,
        )

    patch_signal[
        garment_mask == 0
    ] = 0.0

    # ---------------------------------------------------------
    # SEGUNDA SEÑAL:
    # cambio visual respecto del entorno local.
    #
    # No clasifica por color amarillo ni por una posición fija.
    # Ayuda a reforzar una región cuya apariencia difiere
    # localmente de la tela circundante.
    # ---------------------------------------------------------
    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    local_reference = cv2.GaussianBlur(
        lab_image,
        (0, 0),
        sigmaX=15.0,
        sigmaY=15.0,
    )

    delta = (
        lab_image
        - local_reference
    )

    visual_difference = np.sqrt(
        (delta[:, :, 0] * 0.35) ** 2
        + delta[:, :, 1] ** 2
        + delta[:, :, 2] ** 2
    )

    visual_values = visual_difference[
        garment_mask > 0
    ]

    visual_low = float(
        np.quantile(
            visual_values,
            0.40,
        )
    )

    visual_high = float(
        np.quantile(
            visual_values,
            0.995,
        )
    )

    if visual_high > visual_low:
        visual_signal = np.clip(
            (
                visual_difference
                - visual_low
            )
            / (
                visual_high
                - visual_low
            ),
            0.0,
            1.0,
        )
    else:
        visual_signal = np.zeros_like(
            visual_difference,
            dtype=np.float32,
        )

    visual_signal[
        garment_mask == 0
    ] = 0.0

    # PatchCore continúa siendo la señal dominante.
    combined = (
        patch_signal * 0.82
        + visual_signal * 0.18
    ).astype(np.float32)

    combined[
        garment_mask == 0
    ] = 0.0

    combined = cv2.GaussianBlur(
        combined,
        (0, 0),
        sigmaX=1.2,
        sigmaY=1.2,
    )

    inside = combined[
        garment_mask > 0
    ]

    threshold = float(
        np.quantile(
            inside,
            0.960,
        )
    )

    # Nunca utilizar un umbral excesivamente bajo.
    threshold = max(
        0.64,
        threshold,
    )

    binary_mask = np.where(
        combined >= threshold,
        255,
        0,
    ).astype(np.uint8)

    binary_mask = cv2.bitwise_and(
        binary_mask,
        garment_mask,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        ),
        iterations=1,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_area = max(
        100,
        int(
            width
            * height
            * 0.00005
        ),
    )

    # Distancia al borde.
    # IMPORTANTE: se usa como penalización, no como exclusión.
    # Una mancha próxima al borde sigue siendo candidata.
    distance_map = cv2.distanceTransform(
        (
            garment_mask > 0
        ).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    candidates = []

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )

        if area < minimum_area:
            continue

        x, y, bw, bh = (
            cv2.boundingRect(contour)
        )

        if bw <= 2 or bh <= 2:
            continue

        box_ratio = (
            bw * bh
        ) / float(
            width * height
        )

        if box_ratio > (
            PATCHCORE_MAX_BOX_RATIO
        ):
            continue

        region_binary = binary_mask[
            y:y + bh,
            x:x + bw,
        ]

        active = (
            region_binary > 0
        )

        if not np.any(active):
            continue

        patch_region = patch_signal[
            y:y + bh,
            x:x + bw,
        ][active]

        visual_region = visual_signal[
            y:y + bh,
            x:x + bw,
        ][active]

        combined_region = combined[
            y:y + bh,
            x:x + bw,
        ][active]

        distance_region = distance_map[
            y:y + bh,
            x:x + bw,
        ][active]

        patch_p95 = float(
            np.quantile(
                patch_region,
                0.95,
            )
        )

        patch_mean = float(
            patch_region.mean()
        )

        visual_mean = float(
            visual_region.mean()
        )

        combined_mean = float(
            combined_region.mean()
        )

        average_distance = float(
            distance_region.mean()
        )

        # Suave penalización de borde: 0.72 .. 1.00.
        # No elimina anomalías que están en un extremo.
        edge_factor = (
            0.72
            + 0.28
            * min(
                average_distance / 20.0,
                1.0,
            )
        )

        area_bonus = min(
            area
            / max(
                1.0,
                width
                * height
                * 0.004,
            ),
            1.0,
        )

        score = (
            patch_p95 * 0.52
            + patch_mean * 0.18
            + combined_mean * 0.12
            + visual_mean * 0.13
            + area_bonus * 0.05
        ) * edge_factor

        candidates.append({
            "x": x,
            "y": y,
            "width": bw,
            "height": bh,
            "score": score,
            "patch_p95": patch_p95,
            "patch_mean": patch_mean,
            "visual_mean": visual_mean,
            "combined_mean": combined_mean,
            "edge_factor": edge_factor,
            "area": area,
        })

    if not candidates:
        cv2.putText(
            output,
            "MANCHA DETECTADA - REGION NO LOCALIZABLE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return (
            output,
            "Sin zona localizada",
            0,
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # =========================================================
    # AGRUPACIÓN MULTIMANCHA
    #
    # Una misma mancha puede generar varios hotspots PatchCore.
    # Primero agrupamos fragmentos cercanos y DESPUÉS contamos
    # instancias independientes.
    # =========================================================

    if not candidates:
        cv2.putText(
            output,
            "MANCHA DETECTADA - REGION NO LOCALIZABLE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return (
            output,
            "Sin zona localizada",
            0,
        )

    horizontal_join = max(
        18,
        int(width * 0.025),
    )

    vertical_join = max(
        18,
        int(height * 0.035),
    )

    def should_merge(a, b):
        ax1 = a["x"]
        ay1 = a["y"]
        ax2 = ax1 + a["width"]
        ay2 = ay1 + a["height"]

        bx1 = b["x"]
        by1 = b["y"]
        bx2 = bx1 + b["width"]
        by2 = by1 + b["height"]

        overlap_x = min(
            ax2,
            bx2,
        ) - max(
            ax1,
            bx1,
        )

        overlap_y = min(
            ay2,
            by2,
        ) - max(
            ay1,
            by1,
        )

        gap_x = max(
            0,
            max(
                bx1 - ax2,
                ax1 - bx2,
            ),
        )

        gap_y = max(
            0,
            max(
                by1 - ay2,
                ay1 - by2,
            ),
        )

        # Fragmentos del mismo objeto suelen compartir
        # uno de los ejes y estar próximos en el otro.
        if (
            overlap_x > 0
            and gap_y <= vertical_join
        ):
            return True

        if (
            overlap_y > 0
            and gap_x <= horizontal_join
        ):
            return True

        # Fragmentos muy próximos en ambos ejes.
        if (
            gap_x <= horizontal_join * 0.55
            and gap_y <= vertical_join * 0.55
        ):
            return True

        return False

    # Union-Find: garantiza agrupación transitiva.
    parent = list(
        range(
            len(candidates)
        )
    )

    def find(index):
        while parent[index] != index:
            parent[index] = parent[
                parent[index]
            ]
            index = parent[index]

        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(
        len(candidates)
    ):
        for j in range(
            i + 1,
            len(candidates),
        ):
            if should_merge(
                candidates[i],
                candidates[j],
            ):
                union(i, j)

    grouped = {}

    for index, candidate in enumerate(
        candidates
    ):
        root = find(index)

        grouped.setdefault(
            root,
            [],
        ).append(candidate)

    clusters = []

    for members in grouped.values():
        x1 = min(
            item["x"]
            for item in members
        )

        y1 = min(
            item["y"]
            for item in members
        )

        x2 = max(
            item["x"]
            + item["width"]
            for item in members
        )

        y2 = max(
            item["y"]
            + item["height"]
            for item in members
        )

        best_member = max(
            members,
            key=lambda item: item["score"],
        )

        cluster_score = max(
            item["score"]
            for item in members
        )

        cluster_patch = max(
            item["patch_p95"]
            for item in members
        )

        cluster_visual = max(
            item["visual_mean"]
            for item in members
        )

        cluster_area = sum(
            item["area"]
            for item in members
        )

        # Pequeño refuerzo cuando varios hotspots coherentes
        # pertenecen al mismo objeto.
        cluster_score = min(
            1.0,
            cluster_score
            + min(
                0.045,
                0.012
                * (
                    len(members) - 1
                ),
            ),
        )

        clusters.append({
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "score": cluster_score,
            "patch_p95": cluster_patch,
            "visual_mean": cluster_visual,
            "area": cluster_area,
            "members": len(members),
        })

    clusters.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print(
        "[CLUSTER MANCHA] "
        f"hotspots={len(candidates)}, "
        f"grupos={len(clusters)}"
    )

    for index, cluster in enumerate(
        clusters,
        start=1,
    ):
        print(
            "[CLUSTER MANCHA] "
            f"grupo={index}, "
            f"miembros={cluster['members']}, "
            f"score={cluster['score']:.4f}, "
            f"patch={cluster['patch_p95']:.4f}, "
            f"visual={cluster['visual_mean']:.4f}, "
            f"x={cluster['x']}, "
            f"y={cluster['y']}, "
            f"w={cluster['width']}, "
            f"h={cluster['height']}"
        )

    # =========================================================
    # FILTRO DE INSTANCIAS
    #
    # Se filtra DESPUÉS de agrupar, para no contar cada hotspot
    # individual como si fuera una mancha diferente.
    # =========================================================

    best_score = clusters[0]["score"]
    best_patch = clusters[0]["patch_p95"]

    score_floor = max(
        0.62,
        best_score - 0.12,
    )

    patch_floor = max(
        0.82,
        best_patch - 0.15,
    )

    selected = []

    for index, cluster in enumerate(
        clusters,
        start=1,
    ):
        strong_patch = (
            cluster["score"]
            >= score_floor
            and cluster["patch_p95"]
            >= patch_floor
        )

        strong_visual = (
            cluster["score"]
            >= best_score - 0.08
            and cluster["patch_p95"]
            >= 0.75
            and cluster["visual_mean"]
            >= 0.72
        )

        accepted = (
            strong_patch
            or strong_visual
        )

        print(
            "[FILTRO CLUSTER] "
            f"grupo={index}, "
            f"aceptado={accepted}, "
            f"score={cluster['score']:.4f}, "
            f"patch={cluster['patch_p95']:.4f}, "
            f"visual={cluster['visual_mean']:.4f}, "
            f"score_floor={score_floor:.4f}, "
            f"patch_floor={patch_floor:.4f}"
        )

        if accepted:
            selected.append(
                cluster
            )

    if not selected:
        selected = [
            clusters[0]
        ]

    # No existe un máximo funcional de 2.
    # 10 es solamente una protección defensiva ante un mapa
    # completamente degradado.
    selected = selected[:10]

    # Bounding box global de la propia blusa.
    garment_contours, _ = (
        cv2.findContours(
            garment_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    )

    if garment_contours:
        largest = max(
            garment_contours,
            key=cv2.contourArea,
        )

        gx, gy, gw, gh = (
            cv2.boundingRect(
                largest
            )
        )
    else:
        gx, gy, gw, gh = (
            0,
            0,
            width,
            height,
        )

    padding = max(
        6,
        int(
            min(
                width,
                height,
            ) * 0.01
        ),
    )

    zones = []

    for index, candidate in enumerate(
        selected,
        start=1,
    ):
        x1 = max(
            0,
            candidate["x"] - padding,
        )

        y1 = max(
            0,
            candidate["y"] - padding,
        )

        x2 = min(
            width - 1,
            candidate["x"]
            + candidate["width"]
            + padding,
        )

        y2 = min(
            height - 1,
            candidate["y"]
            + candidate["height"]
            + padding,
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        label = (
            "MANCHA"
            if len(selected) == 1
            else f"MANCHA {index}"
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    30,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        relative_x = (
            center_x - gx
        ) / max(
            1.0,
            float(gw),
        )

        relative_y = (
            center_y - gy
        ) / max(
            1.0,
            float(gh),
        )

        if relative_x < 0.33:
            horizontal = "Izquierda"
        elif relative_x > 0.66:
            horizontal = "Derecha"
        else:
            horizontal = "Centro"

        if relative_y < 0.33:
            vertical = "Superior"
        elif relative_y > 0.66:
            vertical = "Inferior"
        else:
            vertical = "Media"

        candidate_zone = (
            f"{vertical} - {horizontal}"
        )

        zones.append(candidate_zone)

        print(
            "[LOCALIZACION MANCHA] "
            f"indice={index}, "
            f"zona={candidate_zone}, "
            f"score={candidate['score']:.4f}, "
            f"patch_p95={candidate['patch_p95']:.4f}, "
            f"visual_mean={candidate['visual_mean']:.4f}"
        )

    if len(zones) == 1:
        zone = zones[0]
    elif len(zones) == 2:
        zone = (
            f"{zones[0]}; {zones[1]}"
        )
    else:
        zone = (
            f"Varias zonas ({len(zones)})"
        )

    print(
        "[MULTIMANCHA] "
        f"regiones_detectadas={len(selected)}"
    )

    return (
        output,
        zone,
        len(selected),
    )





# PatchCore/Lightning mantiene estado interno durante predict().
# Flask puede atender varias solicitudes simultáneamente, por lo que
# las inferencias sobre el mismo inspector deben serializarse.
PATCHCORE_INFERENCE_LOCK = __import__("threading").Lock()


def detect_defect(
    image_path,
    metrics=None,
    garment_token=None,
    decision_start=None,
    source="auto",
):
    """
    Ejecuta la inferencia y, al conocer NORMAL/ANOMALIA, dispara
    trigger_quality_alert antes de la persistencia secundaria
    (imagen de resultado).

    metrics: dict opcional que se rellena con preprocess_ms,
    inference_ms, postprocess_ms, image_storage_ms y decision_ms.
    """
    m = metrics if metrics is not None else {}

    # ========================================================
    # 1. PATCHCORE: detector principal
    # ========================================================
    if patchcore_inspector is not None and cv2 is not None:
        try:
            timeline_mark(garment_token, "preprocess_start")
            t_preprocess_0 = time.perf_counter()
            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(
                    f"No se pudo abrir la imagen: {image_path}"
                )

            garment_mask_full = create_garment_mask(
                image
            )

            if cv2.countNonZero(
                garment_mask_full
            ) == 0:
                # La mascara de prenda no describio la prenda. El
                # fallback al ROI completo (compartido con el
                # entrenador) vive en build_patchcore_regions: asi
                # entrenamiento e inferencia reciben la misma imagen.
                print(
                    "[SEGMENTACION] Mascara de prenda vacia: "
                    "se usara el ROI completo."
                )

            roi_x1, roi_y1, roi_x2, roi_y2 = (
                get_roi_bounds(image)
            )

            # Fondo blanco uniforme sobre el ROI.
            # PatchCore recibe únicamente la prenda.
            # Misma implementación que usa el entrenador de FASE 3A
            # (patchcore_preprocess) para evitar training-serving skew.
            roi_image, roi_garment_mask, patchcore_input = (
                patchcore_preprocess.build_patchcore_regions(
                    image,
                    garment_mask_full,
                    (roi_x1, roi_y1, roi_x2, roi_y2),
                    cv2_mod=cv2,
                )
            )

            roi_name = (
                f"_patchcore_roi_"
                f"{uuid.uuid4().hex[:10]}.png"
            )

            roi_path = CAPTURE_DIR / roi_name

            if not cv2.imwrite(
                str(roi_path),
                patchcore_input,
            ):
                raise IOError(
                    "No se pudo preparar el ROI para PatchCore."
                )

            m["preprocess_ms"] = (
                time.perf_counter() - t_preprocess_0
            ) * 1000.0
            timeline_mark(garment_token, "preprocess_end")

            try:
                print(
                    "[PATCHCORE] Esperando acceso exclusivo "
                    "al motor de inferencia."
                )

                timeline_mark(garment_token, "inference_start")
                t_inference_0 = time.perf_counter()
                with PATCHCORE_INFERENCE_LOCK:
                    print(
                        "[PATCHCORE] Inferencia iniciada."
                    )

                    prediction = patchcore_inspector.inspect(
                        str(roi_path)
                    )

                    print(
                        "[PATCHCORE] Inferencia finalizada."
                    )
                m["inference_ms"] = (
                    time.perf_counter() - t_inference_0
                ) * 1000.0
                timeline_mark(garment_token, "inference_end")
            finally:
                try:
                    roi_path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            model_label = bool(
                prediction["is_anomaly"]
            )

            raw_score = float(
                prediction["score"]
            )

            # Se normaliza el score a la escala porcentual utilizada
            # durante la calibración de V2.1.
            confidence = (
                raw_score * 100
                if raw_score <= 1
                else raw_score
            )

            confidence = round(
                max(0.0, min(confidence, 100.0)),
                2,
            )

            # La decisión de producción utiliza el threshold calibrado.
            # El pred_label interno se conserva únicamente para diagnóstico.
            is_anomaly = (
                confidence >= PATCHCORE_SCORE_THRESHOLD
            )

            t_postprocess_0 = time.perf_counter()
            annotated_roi, zone, anomaly_count = localize_patchcore_anomaly(
                image=patchcore_input,
                anomaly_map=prediction["anomaly_map"],
                confidence=confidence,
                is_anomaly=is_anomaly,
                garment_mask=roi_garment_mask,
            )

            # La imagen procesada muestra visualmente
            # la eliminación del fondo.
            annotated = np.full_like(
                image,
                255,
            )

            annotated[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ] = annotated_roi

            if is_anomaly:
                status = "Defecto"
                if anomaly_count > 1:
                    defect_type = f"Manchas ({anomaly_count})"
                else:
                    defect_type = "Mancha"
            else:
                status = "Aprobado"
                defect_type = "Sin defecto"
                zone = "Centro"

            m["postprocess_ms"] = (
                time.perf_counter() - t_postprocess_0
            ) * 1000.0
            timeline_mark(garment_token, "decision_ready")

            # ====================================================
            # DECISIÓN NORMAL/ANOMALIA conocida.
            # Punto de alerta: ANTES de persistencia secundaria
            # (imagen de resultado, MySQL, contadores).
            # ====================================================
            if decision_start is not None:
                m["decision_ms"] = (
                    time.perf_counter() - decision_start
                ) * 1000.0

            if is_anomaly:
                trigger_quality_alert(
                    garment_token=garment_token,
                    confidence=confidence,
                    zone=zone,
                    defect_type=defect_type,
                    source=source,
                )

            # -------------------------------------------------
            # Persistencia secundaria: solo después de la
            # decisión y de haber disparado la alerta.
            # -------------------------------------------------
            t_image_storage_0 = time.perf_counter()
            result_name = (
                f"result_patchcore_"
                f"{uuid.uuid4().hex[:10]}.jpg"
            )

            result_path = RESULT_DIR / result_name

            if not cv2.imwrite(
                str(result_path),
                annotated,
            ):
                raise IOError(
                    "No se pudo guardar el resultado PatchCore."
                )

            m["image_storage_ms"] = (
                time.perf_counter() - t_image_storage_0
            ) * 1000.0

            print(
                f"[PATCHCORE] Estado={status}, "
                f"decision={int(is_anomaly)}, "
                f"model_label={int(model_label)}, "
                f"threshold={PATCHCORE_SCORE_THRESHOLD:.2f}, "
                f"raw_score={raw_score:.6f}, "
                f"confianza_mostrada={confidence}%, "
                f"zona={zone}"
            )

            return (
                status,
                defect_type,
                confidence,
                zone,
                f"results/{result_name}",
            )

        except Exception as error:
            error_text = str(error)
            error_lower = error_text.lower()

            # -------------------------------------------------
            # SIN PRENDA != ERROR DEL MODELO
            #
            # Si no existe una blusa válida, nunca ejecutar el
            # detector provisional ni registrar el panel vacío.
            # -------------------------------------------------
            garment_absent = any(
                marker in error_lower
                for marker in (
                    "separar la blusa",
                    "silueta detectada",
                    "roi de inspecci",
                )
            )

            if garment_absent:
                print(
                    "[PATCHCORE] Inspección cancelada: "
                    "no hay una blusa completa en el área."
                )

                raise RuntimeError(
                    "No se detect? una blusa completa "
                    "en el área de inspección."
                ) from error

            # Solamente un fallo técnico real de PatchCore
            # puede utilizar el método de respaldo.
            print(
                "[PATCHCORE] Error técnico durante "
                f"la inferencia: {error}. "
                "Se usará el método alternativo."
            )

    # ========================================================
    # 2. YOLO O DETECTOR PROVISIONAL COMO RESPALDO
    # ========================================================
    model = load_yolo_model()

    if model is not None and cv2 is not None:
        results = model(str(image_path), conf=YOLO_INFERENCE_CONF, verbose=False)
        result = results[0]

        result_name = f"result_{uuid.uuid4().hex[:10]}.jpg"
        result_path = RESULT_DIR / result_name

        annotated = result.plot()

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            best_index = int(boxes.conf.argmax().item())
            confidence = float(boxes.conf[best_index].item()) * 100
            class_id = int(boxes.cls[best_index].item())
            if isinstance(model.names, dict):
                defect_type = model.names.get(class_id, "Defecto visible")
            else:
                defect_type = model.names[class_id] if class_id < len(model.names) else "Defecto visible"

            x1, y1, x2, y2 = boxes.xyxy[best_index].tolist()
            cx = (x1 + x2) / 2

            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]

            if cx < w * 0.33:
                zone = "Lateral izquierdo"
            elif cx > w * 0.66:
                zone = "Lateral derecho"
            else:
                zone = "Zona frontal"

            status = "Defecto" if confidence >= (YOLO_DEFECT_THRESHOLD * 100) else "Revisar"

            if decision_start is not None:
                m["decision_ms"] = (
                    time.perf_counter() - decision_start
                ) * 1000.0

            if status != "Aprobado":
                trigger_quality_alert(
                    garment_token=garment_token,
                    confidence=round(confidence, 2),
                    zone=zone,
                    defect_type=defect_type,
                    source=source,
                )

            t_image_storage_0 = time.perf_counter()
            cv2.imwrite(str(result_path), annotated)
            m["image_storage_ms"] = (
                time.perf_counter() - t_image_storage_0
            ) * 1000.0
            return status, defect_type, round(confidence, 2), zone, f"results/{result_name}"

        if decision_start is not None:
            m["decision_ms"] = (
                time.perf_counter() - decision_start
            ) * 1000.0
        t_image_storage_0 = time.perf_counter()
        cv2.imwrite(str(result_path), annotated)
        m["image_storage_ms"] = (
            time.perf_counter() - t_image_storage_0
        ) * 1000.0
        return "Aprobado", "Sin defecto", 90.0, "Centro", f"results/{result_name}"

    if cv2 is None or np is None:
        raise RuntimeError(
            "OpenCV/NumPy no están disponibles y todavía no existe un modelo YOLO utilizable."
        )

    img = cv2.imread(str(image_path))

    if img is None:
        return "Revisar", "Imagen no válida", 0.0, "No definida", None

    original = img.copy()
    h, w = img.shape[:2]

    x1, y1, x2, y2 = get_roi_bounds(img)
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    dark_mask = cv2.inRange(gray, 0, 65)
    sat_mask = cv2.inRange(hsv[:, :, 1], 120, 255)

    mask = cv2.bitwise_or(dark_mask, sat_mask)
    mask = cv2.medianBlur(mask, 5)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    defect_found = False
    best_area = 0
    best_box = None
    min_area = max(250, (w * h) * 0.001)

    for c in contours[:8]:
        area = cv2.contourArea(c)

        if area > min_area:
            x, y, bw, bh = cv2.boundingRect(c)
            best_area = area
            best_box = (x + x1, y + y1, bw, bh)
            defect_found = True
            break

    result_name = f"result_{uuid.uuid4().hex[:10]}.jpg"
    result_path = RESULT_DIR / result_name

    if defect_found:
        bx, by, bw, bh = best_box
        confidence = min(95, 70 + (best_area / (w * h)) * 900)
        confidence = round(confidence, 2)

        defect_type = "Anomalía visual (modo prototipo, sin modelo entrenado)"
        status = "Revisar"

        cv2.rectangle(original, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
        cv2.putText(
            original,
            f"{defect_type} {confidence}%",
            (bx, max(30, by - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cx = bx + bw / 2
        if cx < w * 0.33:
            zone = "Lateral izquierdo"
        elif cx > w * 0.66:
            zone = "Lateral derecho"
        else:
            zone = "Zona frontal"
    else:
        confidence = 0.0
        defect_type = "Sin anomalía evidente (modo prototipo)"
        status = "Revisar"
        zone = "Centro"

        cv2.putText(
            original,
            "SIN MODELO ENTRENADO - REVISION HUMANA",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 170, 0),
            2,
        )

    if decision_start is not None:
        m["decision_ms"] = (
            time.perf_counter() - decision_start
        ) * 1000.0

    if status != "Aprobado":
        trigger_quality_alert(
            garment_token=garment_token,
            confidence=confidence,
            zone=zone,
            defect_type=defect_type,
            source=source,
        )

    t_image_storage_0 = time.perf_counter()
    cv2.imwrite(str(result_path), original)
    m["image_storage_ms"] = (
        time.perf_counter() - t_image_storage_0
    ) * 1000.0

    return status, defect_type, confidence, zone, f"results/{result_name}"



def get_batch_review_summary(batch_id):
    return fetch_one(
        """
        SELECT
            b.*,
            COUNT(i.id) AS processed_quantity,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'NORMAL'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS auto_approved,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'DEFECTO_CONFIRMADO'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'ALERTA_DESCARTADA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded_alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                         AND COALESCE(
                             i.review_status,
                             'PENDIENTE'
                         ) = 'PENDIENTE'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending_alerts

        FROM batches b
        LEFT JOIN inspections i
          ON i.batch_id = b.id

        WHERE b.id = %s

        GROUP BY b.id
        """,
        (batch_id,),
    )



def get_batch_alerts(batch_id):
    return fetch_all(
        """
        SELECT *
        FROM inspections
        WHERE batch_id = %s
          AND ai_decision = 'ANOMALIA'
          AND COALESCE(
              review_status,
              'PENDIENTE'
          ) = 'PENDIENTE'
        ORDER BY batch_position ASC, id ASC
        """,
        (batch_id,),
    )


def get_batch_garments(batch_id, page=1, per_page=50):
    """
    Todas las inspecciones del lote (sin filtrar por
    ai_decision ni review_status), con paginación.
    """
    per_page = max(1, int(per_page))
    page = max(1, int(page or 1))

    total_row = fetch_one(
        """
        SELECT COUNT(*) AS total
        FROM inspections
        WHERE batch_id = %s
        """,
        (batch_id,),
    )
    total = int((total_row or {}).get("total") or 0)

    max_page = max(1, (total + per_page - 1) // per_page)
    if page > max_page:
        page = max_page

    offset = (page - 1) * per_page

    rows = fetch_all(
        """
        SELECT *
        FROM inspections
        WHERE batch_id = %s
        ORDER BY
            batch_position IS NULL,
            batch_position ASC,
            id ASC
        LIMIT %s OFFSET %s
        """,
        (batch_id, per_page, offset),
    ) or []

    return {
        "items": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "max_page": max_page,
        "has_prev": page > 1,
        "has_next": page < max_page,
        "start_index": (offset + 1) if total else 0,
        "end_index": min(offset + per_page, total),
    }





# ============================================================
# RUTAS
# ============================================================

@app.route("/video_feed")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def video_feed():
    return Response(
        generate_video_feed(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = fetch_one("SELECT * FROM users WHERE username = %s", (username,))

        if user and check_password_hash(user["password_hash"], password):
            if int(user.get("active") or 0) != 1:
                flash(
                    "Esta cuenta se encuentra desactivada. "
                    "Contacte al administrador.",
                    "error",
                )
                return render_template("login.html")

            role = normalize_role(user.get("role"))

            if role not in VALID_ROLES:
                flash(
                    "La cuenta no tiene un rol valido asignado.",
                    "error",
                )
                return render_template("login.html")

            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = role

            return redirect(url_for("dashboard"))

        flash("Usuario o contraseña incorrectos.", "error")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    from datetime import date, time, timedelta
    from zoneinfo import ZoneInfo

    timezone_gt = ZoneInfo("America/Guatemala")
    timezone_utc = ZoneInfo("UTC")

    now_gt = datetime.now(timezone_gt)
    today = now_gt.date()

    # La base almacena los tiempos operativos usando UTC.
    # Convertimos el inicio y fin del día de Guatemala a UTC
    # antes de consultar los registros.
    day_start_gt = datetime.combine(
        today,
        time.min,
        tzinfo=timezone_gt,
    )
    day_end_gt = day_start_gt + timedelta(days=1)

    day_start = (
        day_start_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )

    day_end = (
        day_end_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )

    goal = fetch_one(
        """
        SELECT *
        FROM daily_production_goals
        WHERE goal_date = %s
        """,
        (today,),
    )

    today_stats = fetch_one(
        """
        SELECT
            COUNT(*) AS inspected,
            COALESCE(
                SUM(
                    CASE
                        WHEN ai_decision = 'NORMAL'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS auto_approved,
            COALESCE(
                SUM(
                    CASE
                        WHEN ai_decision = 'ANOMALIA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,
            COALESCE(
                SUM(
                    CASE
                        WHEN review_status = 'DEFECTO_CONFIRMADO'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects,
            COALESCE(
                SUM(
                    CASE
                        WHEN ai_decision = 'ANOMALIA'
                         AND COALESCE(
                             review_status,
                             'PENDIENTE'
                         ) = 'PENDIENTE'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending_alerts
        FROM inspections
        WHERE created_at >= %s
          AND created_at < %s
        """,
        (day_start, day_end),
    )

    completed_batches = fetch_one(
        """
        SELECT COUNT(*) AS c
        FROM batches b
        WHERE b.inspection_completed_at >= %s
          AND b.inspection_completed_at < %s
          AND EXISTS (
              SELECT 1
              FROM inspections i
              WHERE i.batch_id = b.id
                AND i.created_at >= %s
                AND i.created_at < %s
          )
        """,
        (
            day_start,
            day_end,
            day_start,
            day_end,
        ),
    )["c"]

    active_batch = get_active_batch()

    recent_batches = fetch_all(
        """
        SELECT
            b.*,
            gm.code AS garment_model_code,
            gm.name AS garment_model_name,
            COUNT(i.id) AS processed_quantity
        FROM batches b
        LEFT JOIN garment_models gm
          ON gm.id = b.garment_model_id
        LEFT JOIN inspections i
          ON i.batch_id = b.id
        GROUP BY b.id
        ORDER BY b.id DESC
        LIMIT 5
        """
    )

    operational = {
        "status": "SIN_META",
        "status_label": "Sin objetivo configurado",
        "progress": 0,
        "remaining_garments": 0,
        "remaining_batches": 0,
        "current_rate": 0.0,
        "required_rate": 0.0,
        "remaining_minutes": 0,
        "estimated_finish": None,
    }

    if goal:
        target_garments = int(goal["target_garments"])
        target_batches = int(goal["target_batches"])
        inspected = int(today_stats["inspected"] or 0)
        completed = int(completed_batches or 0)

        remaining_garments = max(
            target_garments - inspected,
            0,
        )
        remaining_batches = (
            (remaining_garments + 99) // 100
            if remaining_garments > 0
            else 0
        )

        progress = (
            min(inspected / target_garments * 100, 100)
            if target_garments
            else 0
        )

        shift_start_value = goal["shift_start"]
        shift_end_value = goal["shift_end"]

        if isinstance(shift_start_value, timedelta):
            shift_start_seconds = int(
                shift_start_value.total_seconds()
            )
            shift_start_time = time(
                shift_start_seconds // 3600,
                (shift_start_seconds % 3600) // 60,
                shift_start_seconds % 60,
            )
        else:
            shift_start_time = shift_start_value

        if isinstance(shift_end_value, timedelta):
            shift_end_seconds = int(
                shift_end_value.total_seconds()
            )
            shift_end_time = time(
                shift_end_seconds // 3600,
                (shift_end_seconds % 3600) // 60,
                shift_end_seconds % 60,
            )
        else:
            shift_end_time = shift_end_value

        shift_start = datetime.combine(
            today,
            shift_start_time,
            tzinfo=ZoneInfo("America/Guatemala"),
        )

        shift_end = datetime.combine(
            today,
            shift_end_time,
            tzinfo=ZoneInfo("America/Guatemala"),
        )

        elapsed_hours = max(
            (now_gt - shift_start).total_seconds() / 3600,
            0,
        )

        remaining_hours = max(
            (shift_end - now_gt).total_seconds() / 3600,
            0,
        )

        current_rate = (
            inspected / elapsed_hours
            if elapsed_hours > 0
            else 0
        )

        required_rate = (
            remaining_garments / remaining_hours
            if remaining_hours > 0
            else 0
        )

        estimated_finish = None

        if current_rate > 0 and remaining_garments > 0:
            estimated_finish = (
                now_gt
                + timedelta(
                    hours=remaining_garments / current_rate
                )
            )

        garments_met = (
            inspected >= target_garments
        )

        if garments_met:
            status = "META_CUMPLIDA"
            status_label = "Objetivo cumplido"

        elif now_gt >= shift_end:
            status = "ATRASADO"
            status_label = "Objetivo no cumplido"

        elif now_gt < shift_start:
            status = "PENDIENTE"
            status_label = "Turno pendiente"

        elif (
            current_rate > 0
            and current_rate >= required_rate
        ):
            status = "EN_TIEMPO"
            status_label = "En tiempo"

        else:
            status = "EN_RIESGO"
            status_label = "En riesgo"

        operational = {
            "status": status,
            "status_label": status_label,
            "progress": round(progress, 1),
            "remaining_garments": remaining_garments,
            "remaining_batches": remaining_batches,
            "current_rate": round(current_rate, 1),
            "required_rate": round(required_rate, 1),
            "remaining_minutes": max(
                int((shift_end - now_gt).total_seconds() / 60),
                0,
            ),
            "estimated_finish": estimated_finish,
            "shift_start": shift_start,
            "shift_end": shift_end,
        }

    # Informacion operativa en caliente:
    # que prendas y modelos fueron inspeccionados hoy.
    production_rows = fetch_all(
        """
        SELECT
            gm.id AS model_id,
            gm.code AS model_code,
            gm.name AS model_name,

            COUNT(i.id) AS inspected,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'NORMAL'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS without_alert,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status =
                             'DEFECTO_CONFIRMADO'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects,

            COALESCE(
                SUM(
                    CASE
                        WHEN
                            i.ai_decision = 'ANOMALIA'
                            AND COALESCE(
                                i.review_status,
                                'PENDIENTE'
                            ) = 'PENDIENTE'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending_alerts

        FROM inspections i

        LEFT JOIN batches b
            ON b.id = i.batch_id

        LEFT JOIN garment_models gm
            ON gm.id = b.garment_model_id

        WHERE
            i.created_at >= %s
            AND i.created_at < %s

        GROUP BY
            gm.id,
            gm.code,
            gm.name

        ORDER BY
            inspected DESC,
            gm.code ASC
        """,
        (
            day_start,
            day_end,
        ),
    )

    live_total_inspected = sum(
        int(
            row["inspected"]
            or 0
        )
        for row in production_rows
    )

    production_by_model = []

    for row in production_rows:
        item = dict(row)

        item["model_code"] = (
            item["model_code"]
            or "SIN-MODELO"
        )

        item["model_name"] = (
            item["model_name"]
            or "Modelo no identificado"
        )

        item["inspected"] = int(
            item["inspected"]
            or 0
        )

        item["without_alert"] = int(
            item["without_alert"]
            or 0
        )

        item["alerts"] = int(
            item["alerts"]
            or 0
        )

        item["confirmed_defects"] = int(
            item["confirmed_defects"]
            or 0
        )

        item["pending_alerts"] = int(
            item["pending_alerts"]
            or 0
        )

        item["share"] = (
            round(
                item["inspected"]
                / live_total_inspected
                * 100,
                1,
            )
            if live_total_inspected > 0
            else 0.0
        )

        production_by_model.append(
            item
        )


    latest_inspection = fetch_one(
        """
        SELECT
            i.id,
            i.code,
            i.created_at,
            i.batch_position,
            i.ai_decision,
            i.review_status,

            b.id AS batch_id,
            b.code AS batch_code,

            gm.code AS model_code,
            gm.name AS model_name

        FROM inspections i

        LEFT JOIN batches b
            ON b.id = i.batch_id

        LEFT JOIN garment_models gm
            ON gm.id = b.garment_model_id

        WHERE
            i.created_at >= %s
            AND i.created_at < %s

        ORDER BY
            i.created_at DESC,
            i.id DESC

        LIMIT 1
        """,
        (
            day_start,
            day_end,
        ),
    )


    if latest_inspection:
        latest_inspection = dict(
            latest_inspection
        )

        latest_inspection[
            "model_code"
        ] = (
            latest_inspection[
                "model_code"
            ]
            or "SIN-MODELO"
        )

        latest_inspection[
            "model_name"
        ] = (
            latest_inspection[
                "model_name"
            ]
            or "Modelo no identificado"
        )

        created_at = (
            latest_inspection[
                "created_at"
            ]
        )

        if created_at:
            latest_gt = (
                created_at
                .replace(
                    tzinfo=timezone_utc
                )
                .astimezone(
                    timezone_gt
                )
            )

            latest_inspection[
                "time_label"
            ] = latest_gt.strftime(
                "%H:%M"
            )

        else:
            latest_inspection[
                "time_label"
            ] = "--:--"


        ai_decision = (
            latest_inspection[
                "ai_decision"
            ]
            or ""
        )

        review_status = (
            latest_inspection[
                "review_status"
            ]
            or ""
        )

        if ai_decision == "NORMAL":
            result_label = (
                "Sin alerta"
            )

        elif (
            review_status
            == "DEFECTO_CONFIRMADO"
        ):
            result_label = (
                "Defecto confirmado"
            )

        elif (
            ai_decision == "ANOMALIA"
            and review_status
            in (
                "",
                "PENDIENTE",
            )
        ):
            result_label = (
                "Alerta pendiente"
            )

        elif ai_decision == "ANOMALIA":
            result_label = (
                "Alerta revisada"
            )

        else:
            result_label = (
                "Resultado pendiente"
            )

        latest_inspection[
            "result_label"
        ] = result_label


    return render_template(
        "dashboard.html",
        today=today,
        now_gt=now_gt,
        goal=goal,
        stats=today_stats,
        completed_batches=int(
            completed_batches or 0
        ),
        active_batch=active_batch,
        recent_batches=recent_batches,
        operational=operational,
        role=normalize_role(
            session.get("role")
        ),
        production_by_model=
            production_by_model,
        latest_inspection=
            latest_inspection,
        live_total_inspected=
            live_total_inspected,
    )




# ============================================================
# DASHBOARD_LIVE_API_V1
# Estado operativo en tiempo real del dia.
# ============================================================

@app.route("/api/dashboard/live")
@login_required
def dashboard_live_api():
    from datetime import time, timedelta
    from zoneinfo import ZoneInfo

    timezone_gt = ZoneInfo(
        "America/Guatemala"
    )

    timezone_utc = ZoneInfo(
        "UTC"
    )

    now_gt = datetime.now(
        timezone_gt
    )

    today = now_gt.date()

    day_start_gt = datetime.combine(
        today,
        time.min,
        tzinfo=timezone_gt,
    )

    day_end_gt = (
        day_start_gt
        + timedelta(days=1)
    )

    day_start = (
        day_start_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )

    day_end = (
        day_end_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )


    goal = fetch_one(
        """
        SELECT
            target_garments
        FROM daily_production_goals
        WHERE goal_date = %s
        LIMIT 1
        """,
        (today,),
    )


    stats = fetch_one(
        """
        SELECT
            COUNT(*) AS inspected,

            COALESCE(
                SUM(
                    CASE
                        WHEN ai_decision = 'ANOMALIA'
                         AND COALESCE(
                             review_status,
                             'PENDIENTE'
                         ) = 'PENDIENTE'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS pending_alerts

        FROM inspections
        WHERE created_at >= %s
          AND created_at < %s
        """,
        (
            day_start,
            day_end,
        ),
    )


    batch_states = fetch_one(
        """
        SELECT
            COALESCE(
                SUM(
                    CASE
                        WHEN status = 'PREPARACION'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS preparing,

            COALESCE(
                SUM(
                    CASE
                        WHEN status = 'EN_INSPECCION'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS inspecting,

            COALESCE(
                SUM(
                    CASE
                        WHEN status = 'REVISION_PENDIENTE'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS review_pending

        FROM batches
        WHERE production_date = %s
        """,
        (today,),
    )


    def get_batch_by_status(
        status,
        order_direction,
    ):
        sql = f"""
            SELECT
                b.id,
                b.code,
                b.status,
                b.planned_quantity,

                gm.code
                    AS garment_model_code,

                gm.name
                    AS garment_model_name,

                il.name
                    AS inspection_line_name,

                (
                    SELECT COUNT(*)
                    FROM inspections i
                    WHERE i.batch_id = b.id
                ) AS processed_quantity

            FROM batches b

            LEFT JOIN garment_models gm
              ON gm.id = b.garment_model_id

            LEFT JOIN inspection_lines il
              ON il.id = b.inspection_line_id

            WHERE b.production_date = %s
              AND b.status = %s

            ORDER BY b.id {order_direction}
            LIMIT 1
        """

        return fetch_one(
            sql,
            (
                today,
                status,
            ),
        )


    active_batch = get_batch_by_status(
        "EN_INSPECCION",
        "DESC",
    )

    next_batch = get_batch_by_status(
        "PREPARACION",
        "ASC",
    )

    review_batch = get_batch_by_status(
        "REVISION_PENDIENTE",
        "DESC",
    )


    production_by_model = fetch_all(
        """
        SELECT
            gm.id AS model_id,
            gm.code AS model_code,
            gm.name AS model_name,

            COUNT(i.id) AS inspected,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'NORMAL'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS without_alert,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                         AND COALESCE(
                             i.review_status,
                             'PENDIENTE'
                         ) = 'PENDIENTE'
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending_alerts

        FROM inspections i

        LEFT JOIN batches b
          ON b.id = i.batch_id

        LEFT JOIN garment_models gm
          ON gm.id = b.garment_model_id

        WHERE i.created_at >= %s
          AND i.created_at < %s

        GROUP BY
            gm.id,
            gm.code,
            gm.name

        ORDER BY inspected DESC
        """,
        (
            day_start,
            day_end,
        ),
    )


    inspected = int(
        stats["inspected"]
        or 0
    )

    target = int(
        goal["target_garments"]
        or 0
    ) if goal else 0

    remaining = max(
        target - inspected,
        0,
    )


    def serialize_batch(row):
        if not row:
            return None

        planned = int(
            row[
                "planned_quantity"
            ]
            or 0
        )

        processed = int(
            row[
                "processed_quantity"
            ]
            or 0
        )

        return {
            "id":
                int(row["id"]),

            "code":
                row["code"],

            "status":
                row["status"],

            "model_code":
                (
                    row[
                        "garment_model_code"
                    ]
                    or "SIN-MODELO"
                ),

            "model_name":
                (
                    row[
                        "garment_model_name"
                    ]
                    or "Modelo no identificado"
                ),

            "size":
                "S",

            "line_name":
                (
                    row[
                        "inspection_line_name"
                    ]
                    or "Sin linea"
                ),

            "planned":
                planned,

            "processed":
                processed,

            "remaining":
                max(
                    planned - processed,
                    0,
                ),
        }


    models = []

    for row in production_by_model:
        models.append({
            "model_id":
                row["model_id"],

            "model_code":
                (
                    row["model_code"]
                    or "SIN-MODELO"
                ),

            "model_name":
                (
                    row["model_name"]
                    or "Modelo no identificado"
                ),

            "size":
                "S",

            "inspected":
                int(
                    row["inspected"]
                    or 0
                ),

            "without_alert":
                int(
                    row["without_alert"]
                    or 0
                ),

            "alerts":
                int(
                    row["alerts"]
                    or 0
                ),

            "pending_alerts":
                int(
                    row["pending_alerts"]
                    or 0
                ),
        })


    return jsonify({
        "ok":
            True,

        "target_garments":
            target,

        "inspected":
            inspected,

        "remaining_garments":
            remaining,

        "pending_alerts":
            int(
                stats[
                    "pending_alerts"
                ]
                or 0
            ),

        "progress":
            (
                round(
                    min(
                        inspected
                        / target
                        * 100,
                        100,
                    ),
                    1,
                )
                if target
                else 0.0
            ),

        "objective_met":
            bool(
                target > 0
                and inspected >= target
            ),

        "batch_states": {
            "preparing":
                int(
                    batch_states[
                        "preparing"
                    ]
                    or 0
                ),

            "inspecting":
                int(
                    batch_states[
                        "inspecting"
                    ]
                    or 0
                ),

            "review_pending":
                int(
                    batch_states[
                        "review_pending"
                    ]
                    or 0
                ),
        },

        "active_batch":
            serialize_batch(
                active_batch
            ),

        "next_batch":
            serialize_batch(
                next_batch
            ),

        "review_batch":
            serialize_batch(
                review_batch
            ),

        "models":
            models,
    })


@app.route("/dashboard/meta", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN)
def dashboard_daily_goal():
    from datetime import time, timedelta
    from zoneinfo import ZoneInfo

    try:
        target_batches = int(
            request.form.get(
                "target_batches",
                "0",
            )
        )

        target_garments = int(
            request.form.get(
                "target_garments",
                "0",
            )
        )

    except (TypeError, ValueError):
        target_batches = 0
        target_garments = 0


    if (
        target_batches < 1
        or target_batches > 100
        or target_garments < 1
        or target_garments > 50000
    ):
        flash(
            "Ingrese un objetivo diario válido.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    if target_garments < target_batches:
        flash(
            "El objetivo de prendas no puede ser "
            "menor que el objetivo de lotes.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    shift_start = request.form.get(
        "shift_start",
        "08:00",
    ).strip()

    shift_end = request.form.get(
        "shift_end",
        "17:00",
    ).strip()


    try:
        start_time = datetime.strptime(
            shift_start,
            "%H:%M",
        ).time()

        end_time = datetime.strptime(
            shift_end,
            "%H:%M",
        ).time()

    except ValueError:
        flash(
            "El horario del turno no es válido.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    if end_time <= start_time:
        flash(
            "La hora de finalización debe ser "
            "posterior al inicio del turno.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    timezone_gt = ZoneInfo(
        "America/Guatemala"
    )

    timezone_utc = ZoneInfo(
        "UTC"
    )

    now_gt = datetime.now(
        timezone_gt
    )

    today = now_gt.date()


    # Produccion real ya realizada hoy.
    day_start_gt = datetime.combine(
        today,
        time.min,
        tzinfo=timezone_gt,
    )

    day_end_gt = (
        day_start_gt
        + timedelta(days=1)
    )

    day_start = (
        day_start_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )

    day_end = (
        day_end_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )


    actual = fetch_one(
        """
        SELECT COUNT(*) AS inspected
        FROM inspections
        WHERE created_at >= %s
          AND created_at < %s
        """,
        (
            day_start,
            day_end,
        ),
    )

    inspected_today = int(
        actual["inspected"]
        or 0
    )


    completed_today = int(
        fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM batches b
            WHERE
                b.inspection_completed_at >= %s
                AND b.inspection_completed_at < %s
                AND EXISTS (
                    SELECT 1
                    FROM inspections i
                    WHERE i.batch_id = b.id
                      AND i.created_at >= %s
                      AND i.created_at < %s
                )
            """,
            (
                day_start,
                day_end,
                day_start,
                day_end,
            ),
        )["total"]
        or 0
    )


    # Nunca permitir que un ajuste haga desaparecer
    # produccion que ya ocurrio.
    if target_garments < inspected_today:
        flash(
            "El objetivo no puede quedar por debajo "
            f"de las {inspected_today} prendas que "
            "ya fueron inspeccionadas hoy.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    if target_batches < completed_today:
        flash(
            "El objetivo no puede quedar por debajo "
            f"de los {completed_today} lotes que "
            "ya fueron completados hoy.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    existing = fetch_one(
        """
        SELECT *
        FROM daily_production_goals
        WHERE goal_date = %s
        LIMIT 1
        """,
        (today,),
    )


    goal_action = (
        request.form.get(
            "goal_action",
            "create",
        )
        .strip()
        .lower()
    )


    # Primera definicion del dia.
    if existing is None:

        execute(
            """
            INSERT INTO daily_production_goals (
                goal_date,
                target_batches,
                target_garments,
                shift_start,
                shift_end,
                created_by
            )
            VALUES (
                %s, %s, %s, %s, %s, %s
            )
            """,
            (
                today,
                target_batches,
                target_garments,
                start_time,
                end_time,
                session.get(
                    "user_id"
                ),
            ),
        )

        flash(
            "Objetivo de producción definido "
            "para hoy.",
            "success",
        )

        return redirect(
            url_for("dashboard")
        )


    # Ya existe: un POST normal NO puede
    # sobrescribirlo silenciosamente.
    if goal_action != "adjust":

        flash(
            "El objetivo de producción de hoy "
            "ya est? definido. Utilice "
            "\"Ajustar objetivo\" si necesita "
            "modificarlo.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    reason = (
        request.form.get(
            "adjustment_reason",
            "",
        )
        .strip()
    )


    if len(reason) < 10:
        flash(
            "Explique el motivo del ajuste "
            "con al menos 10 caracteres.",
            "error",
        )

        return redirect(
            url_for("dashboard")
        )


    # Registrar primero la trazabilidad.
    execute(
        """
        INSERT INTO production_goal_adjustments (
            goal_id,
            goal_date,

            previous_target_batches,
            previous_target_garments,

            new_target_batches,
            new_target_garments,

            previous_shift_start,
            previous_shift_end,

            new_shift_start,
            new_shift_end,

            reason,
            changed_by
        )
        VALUES (
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s
        )
        """,
        (
            existing["id"],
            today,

            existing[
                "target_batches"
            ],
            existing[
                "target_garments"
            ],

            target_batches,
            target_garments,

            existing[
                "shift_start"
            ],
            existing[
                "shift_end"
            ],

            start_time,
            end_time,

            reason,

            session.get(
                "user_id"
            ),
        ),
    )


    execute(
        """
        UPDATE daily_production_goals
        SET
            target_batches = %s,
            target_garments = %s,
            shift_start = %s,
            shift_end = %s,
            created_by = %s
        WHERE id = %s
        """,
        (
            target_batches,
            target_garments,
            start_time,
            end_time,
            session.get(
                "user_id"
            ),
            existing["id"],
        ),
    )


    flash(
        "Objetivo de producción ajustado. "
        "La producción registrada se conserva.",
        "success",
    )

    return redirect(
        url_for("dashboard")
    )


def get_today_production_plan():
    from zoneinfo import ZoneInfo

    today = datetime.now(
        ZoneInfo("America/Guatemala")
    ).date()

    goal = fetch_one(
        """
        SELECT
            goal_date,
            target_batches,
            target_garments,
            shift_start,
            shift_end
        FROM daily_production_goals
        WHERE goal_date = %s
        LIMIT 1
        """,
        (today,),
    )

    planned = fetch_one(
        """
        SELECT
            COUNT(*) AS planned_batches,
            COALESCE(
                SUM(planned_quantity),
                0
            ) AS planned_garments
        FROM batches
        WHERE production_date = %s
        """,
        (today,),
    )

    planned_batches = int(
        planned["planned_batches"]
        or 0
    )

    planned_garments = int(
        planned["planned_garments"]
        or 0
    )


    # SMART_BATCH_PLANNING_V1
    # El objetivo principal es la cantidad de prendas.
    # Los lotes se estiman automaticamente con una
    # referencia operativa de hasta 100 prendas por lote.

    if goal is None:

        return {
            "date":
                today,

            "has_goal":
                False,

            "target_batches":
                0,

            "target_garments":
                0,

            "planned_batches":
                planned_batches,

            "planned_garments":
                planned_garments,

            "remaining_batches":
                0,

            "remaining_garments":
                0,

            "estimated_remaining_batches":
                0,

            "suggested_quantity":
                100,

            "batches_excess":
                0,

            "garments_excess":
                0,

            "goal_fully_planned":
                False,
        }


    target_garments = int(
        goal["target_garments"]
        or 0
    )


    # El numero de lotes deja de ser una cuenta que
    # el encargado tenga que realizar manualmente.
    estimated_target_batches = (
        (target_garments + 99) // 100
        if target_garments > 0
        else 0
    )


    remaining_garments = max(
        target_garments
        - planned_garments,
        0,
    )


    garments_excess = max(
        planned_garments
        - target_garments,
        0,
    )


    estimated_remaining_batches = (
        (remaining_garments + 99) // 100
        if remaining_garments > 0
        else 0
    )


    # Cada nuevo lote recibe automaticamente hasta
    # 100 prendas o exactamente lo que falte.
    suggested_quantity = (
        min(
            100,
            remaining_garments,
        )
        if remaining_garments > 0
        else 0
    )


    goal_fully_planned = (
        remaining_garments == 0
    )


    return {
        "date":
            today,

        "has_goal":
            True,

        "target_batches":
            estimated_target_batches,

        "configured_target_batches":
            int(
                goal["target_batches"]
                or 0
            ),

        "target_garments":
            target_garments,

        "planned_batches":
            planned_batches,

        "planned_garments":
            planned_garments,

        "remaining_batches":
            estimated_remaining_batches,

        "remaining_garments":
            remaining_garments,

        "estimated_remaining_batches":
            estimated_remaining_batches,

        "suggested_quantity":
            suggested_quantity,

        "batches_excess":
            max(
                planned_batches
                - estimated_target_batches,
                0,
            ),

        "garments_excess":
            garments_excess,

        "goal_fully_planned":
            goal_fully_planned,
    }


@app.route("/lotes", methods=["GET", "POST"])
@login_required
def batches():
    from zoneinfo import ZoneInfo

    can_create = has_role(
        ROLE_ADMIN,
        ROLE_QUALITY_MANAGER,
    )

    production_plan = (
        get_today_production_plan()
    )

    available_models = fetch_all(
        """
        SELECT
            gm.id,
            gm.code,
            gm.name,
            gm.color,

            ai.id AS ai_model_id,
            ai.version AS ai_version,
            ai.model_type AS ai_model_type

        FROM garment_models gm

        JOIN garment_ai_models ai
          ON ai.garment_model_id = gm.id

        WHERE gm.status = 'APROBADO'
          AND gm.active = 1
          AND ai.status = 'ACTIVO'
          AND ai.active = 1

          AND ai.id = (
              SELECT ai2.id
              FROM garment_ai_models ai2
              WHERE ai2.garment_model_id = gm.id
                AND ai2.status = 'ACTIVO'
                AND ai2.active = 1
              ORDER BY ai2.id DESC
              LIMIT 1
          )

        ORDER BY gm.code
        """
    )

    available_lines = get_inspection_lines(
        active_only=True,
        configured_only=True,
    )

    if request.method == "POST":
        if not can_create:
            flash(
                "Su rol permite consultar lotes, "
                "pero no crear nuevos lotes.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        model_raw = request.form.get(
            "garment_model_id",
            "",
        ).strip()

        line_raw = request.form.get(
            "inspection_line_id",
            "",
        ).strip()

        quantity_raw = request.form.get(
            "planned_quantity",
            "",
        ).strip()

        try:
            garment_model_id = int(
                model_raw
            )
        except (TypeError, ValueError):
            garment_model_id = 0

        try:
            inspection_line_id = int(
                line_raw
            )
        except (TypeError, ValueError):
            inspection_line_id = 0

        try:
            planned_quantity = int(
                quantity_raw
            )
        except (TypeError, ValueError):
            planned_quantity = 0

        if (
            planned_quantity < 1
            or planned_quantity > 5000
        ):
            flash(
                "La cantidad del lote debe estar "
                "entre 1 y 5000 prendas.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        selected_line = fetch_one(
            """
            SELECT
                id,
                code,
                name,
                camera_ip,
                active

            FROM inspection_lines

            WHERE id = %s
              AND active = 1
              AND NULLIF(
                  TRIM(camera_ip),
                  ''
              ) IS NOT NULL

            LIMIT 1
            """,
            (inspection_line_id,),
        )

        if selected_line is None:
            flash(
                "Seleccione una l\u00ednea de inspecci\u00f3n "
                "activa y configurada.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        selected_model = fetch_one(
            """
            SELECT
                gm.id,
                gm.code,
                gm.name,

                ai.id AS ai_model_id,
                ai.version AS ai_version,
                ai.model_type AS ai_model_type

            FROM garment_models gm

            JOIN garment_ai_models ai
              ON ai.garment_model_id = gm.id

            WHERE gm.id = %s
              AND gm.status = 'APROBADO'
              AND gm.active = 1
              AND ai.status = 'ACTIVO'
              AND ai.active = 1

            ORDER BY ai.id DESC
            LIMIT 1
            """,
            (garment_model_id,),
        )

        if selected_model is None:
            flash(
                "Seleccione un modelo aprobado "
                "que tenga una IA activa.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        today = datetime.now(
            ZoneInfo("America/Guatemala")
        ).date()

        conn = db()
        cur = conn.cursor(
            dictionary=True
        )

        try:
            conn.start_transaction()

            temporary_code = (
                "PENDING-"
                + uuid.uuid4().hex.upper()
            )

            cur.execute(
                """
                INSERT INTO batches (
                    code,
                    garment_model_id,
                    ai_model_id,
                    inspection_line_id,
                    production_date,
                    planned_quantity,
                    status,
                    created_by
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    'PREPARACION',
                    %s
                )
                """,
                (
                    temporary_code,
                    selected_model["id"],
                    selected_model[
                        "ai_model_id"
                    ],
                    selected_line["id"],
                    today,
                    planned_quantity,
                    session.get("user_id"),
                ),
            )

            batch_id = int(
                cur.lastrowid
            )

            date_code = today.strftime(
                "%Y%m%d"
            )

            code = (
                f"LOT-{date_code}-"
                f"{batch_id:05d}"
            )

            cur.execute(
                """
                UPDATE batches
                SET code = %s
                WHERE id = %s
                """,
                (
                    code,
                    batch_id,
                ),
            )

            conn.commit()

        except Exception:
            if conn.in_transaction:
                conn.rollback()

            app.logger.exception(
                "No fue posible crear el lote."
            )

            flash(
                "No fue posible crear el lote.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        finally:
            cur.close()
            conn.close()

        flash(
            f"Lote {code} creado correctamente.",
            "success",
        )

        return redirect(
            url_for(
                "batches",
                created=batch_id,
            )
        )
    # SMART_BATCH_QUANTITY_GUARD_V1
    smart_plan = get_today_production_plan()

    quantity_override = (
        request.form.get(
            "quantity_override",
            "0",
        )
        == "1"
    )

    if request.method == "POST" and smart_plan["has_goal"]:

        remaining = int(
            smart_plan[
                "remaining_garments"
            ]
            or 0
        )

        suggested = int(
            smart_plan[
                "suggested_quantity"
            ]
            or 0
        )

        if (
            remaining <= 0
            and not quantity_override
        ):
            flash(
                "El objetivo de producci\u00f3n ya est\u00e1 "
                "completamente planificado. "
                "No es necesario crear otro lote.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        if (
            remaining > 0
            and not quantity_override
            and planned_quantity != suggested
        ):
            flash(
                "La planificaci\u00f3n cambi\u00f3. "
                f"El sistema recomienda "
                f"{suggested} prendas para el "
                "siguiente lote.",
                "error",
            )

            return redirect(
                url_for("batches")
            )


    all_batches = get_batches()

    today_key = (
        production_plan["date"].isoformat()
    )

    today_batches = [
        batch
        for batch in all_batches
        if str(
            batch.get("production_date")
            or ""
        ) == today_key
    ]

    today_batches_total = len(
        today_batches
    )

    visible_batches = (
        today_batches[:8]
    )

    return render_template(
        "batches.html",
        batches=visible_batches,
        today_batches_total=today_batches_total,
        active_batch=get_active_batch(),
        available_models=available_models,
        available_lines=available_lines,
        production_plan=production_plan,
        can_create=can_create,
    )



# DELETE_ACCIDENTAL_BATCH_V2
@app.route(
    "/lotes/<int:batch_id>/eliminar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN)
def delete_accidental_batch(batch_id):
    conn = db()
    cur = conn.cursor(
        dictionary=True
    )

    try:
        conn.start_transaction()

        cur.execute(
            """
            SELECT
                b.id,
                b.status,
                b.started_at,

                (
                    SELECT COUNT(*)
                    FROM inspections i
                    WHERE i.batch_id = b.id
                ) AS inspection_count

            FROM batches b
            WHERE b.id = %s
            FOR UPDATE
            """,
            (batch_id,),
        )

        batch = cur.fetchone()

        if batch is None:
            conn.rollback()
            return redirect(
                url_for("batches")
            )

        if (
            batch["status"] != "PREPARACION"
            or batch["started_at"] is not None
            or int(
                batch["inspection_count"]
                or 0
            ) != 0
        ):
            conn.rollback()
            return redirect(
                url_for("batches")
            )

        cur.execute(
            """
            DELETE FROM batches
            WHERE id = %s
              AND status = 'PREPARACION'
              AND started_at IS NULL
            """,
            (batch_id,),
        )

        if cur.rowcount != 1:
            conn.rollback()
            return redirect(
                url_for("batches")
            )

        conn.commit()

        return redirect(
            url_for("batches")
        )

    except Exception:
        if conn.in_transaction:
            conn.rollback()

        app.logger.exception(
            "No fue posible eliminar "
            "el lote accidental %s.",
            batch_id,
        )

        return redirect(
            url_for("batches")
        )

    finally:
        cur.close()
        conn.close()


@app.route("/lotes/<int:batch_id>/iniciar", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def start_batch(batch_id):
    global AUTO_INSPECTION_ENABLED

    conn = db()
    cur = conn.cursor(dictionary=True)
    lock_acquired = False

    try:
        # Serializa el inicio de lotes para impedir que dos
        # usuarios activen lotes diferentes simultaneamente.
        cur.execute(
            "SELECT GET_LOCK(%s, 10) AS acquired",
            ("astrid_quality_start_batch",),
        )

        lock_row = cur.fetchone()

        lock_acquired = bool(
            lock_row
            and int(lock_row["acquired"] or 0) == 1
        )

        if not lock_acquired:
            flash(
                "No fue posible asegurar el inicio del lote. "
                "Intente nuevamente.",
                "error",
            )
            return redirect(
                url_for("batches")
            )

        conn.start_transaction()

        cur.execute(
            """
            SELECT
                b.id,
                b.code,
                b.status,
                b.garment_model_id,
                b.ai_model_id,

                gm.code AS garment_model_code,
                gm.name AS garment_model_name,
                gm.status AS garment_model_status,
                gm.active AS garment_model_active,

                ai.id AS linked_ai_id,
                ai.garment_model_id AS ai_garment_model_id,
                ai.version AS ai_version,
                ai.model_type AS ai_model_type

            FROM batches b

            LEFT JOIN garment_models gm
              ON gm.id = b.garment_model_id

            LEFT JOIN garment_ai_models ai
              ON ai.id = b.ai_model_id

            WHERE b.id = %s

            FOR UPDATE
            """,
            (batch_id,),
        )

        batch = cur.fetchone()

        if batch is None:
            conn.rollback()

            flash(
                "El lote no existe.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        if batch["status"] != "PREPARACION":
            conn.rollback()

            flash(
                "Solo pueden iniciarse lotes en "
                "preparaci\u00f3n.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        if (
            batch["garment_model_id"] is None
            or batch["ai_model_id"] is None
        ):
            conn.rollback()

            flash(
                "Este lote no tiene un modelo y una "
                "versi\u00f3n de IA asociados. "
                "Los lotes hist\u00f3ricos sin trazabilidad "
                "no pueden iniciarse.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        if (
            batch["garment_model_status"] != "APROBADO"
            or int(
                batch["garment_model_active"]
                or 0
            ) != 1
        ):
            conn.rollback()

            flash(
                "El modelo asociado al lote ya no est\u00e1 "
                "aprobado y activo.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        if (
            batch["linked_ai_id"] is None
            or int(
                batch["ai_garment_model_id"]
                or 0
            )
            != int(
                batch["garment_model_id"]
            )
        ):
            conn.rollback()

            flash(
                "La versi\u00f3n de IA asociada al lote "
                "no es v\u00e1lida para este modelo.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        cur.execute(
            """
            SELECT
                id,
                code
            FROM batches
            WHERE status = 'EN_INSPECCION'
              AND id <> %s
            ORDER BY id DESC
            LIMIT 1
            FOR UPDATE
            """,
            (batch_id,),
        )

        active = cur.fetchone()

        if active is not None:
            conn.rollback()

            flash(
                f"Ya existe un lote activo: "
                f"{active['code']}.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        AUTO_INSPECTION_ENABLED = False

        cur.execute(
            """
            UPDATE batches
            SET status = 'EN_INSPECCION',
                started_at = %s
            WHERE id = %s
              AND status = 'PREPARACION'
            """,
            (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                batch_id,
            ),
        )

        if cur.rowcount != 1:
            conn.rollback()

            flash(
                "El lote cambi\u00f3 de estado antes de "
                "poder iniciarse.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        conn.commit()

        flash(
            f"Lote {batch['code']} iniciado con "
            f"{batch['garment_model_code']} "
            f"y {batch['ai_model_type']} "
            f"{batch['ai_version']}.",
            "success",
        )

        return redirect(
            url_for("station")
        )

    except Exception:
        if conn.in_transaction:
            conn.rollback()

        app.logger.exception(
            "No fue posible iniciar el lote %s.",
            batch_id,
        )

        flash(
            "No fue posible iniciar el lote.",
            "error",
        )

        return redirect(
            url_for("batches")
        )

    finally:
        if lock_acquired:
            try:
                cur.execute(
                    "SELECT RELEASE_LOCK(%s)",
                    ("astrid_quality_start_batch",),
                )
                cur.fetchone()
            except Exception:
                pass

        cur.close()
        conn.close()


@app.route(
    "/lotes/<int:batch_id>/finalizar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def finish_batch_manual(batch_id):
    global AUTO_INSPECTION_ENABLED

    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()

        cur.execute(
            """
            SELECT
                id,
                code,
                status,
                planned_quantity,
                inspection_line_id
            FROM batches
            WHERE id = %s
            FOR UPDATE
            """,
            (batch_id,),
        )

        batch = cur.fetchone()

        if batch is None:
            conn.rollback()

            flash(
                "El lote solicitado no existe.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        if batch["status"] != "EN_INSPECCION":
            conn.rollback()

            flash(
                "Solo puede finalizarse un lote "
                "que se encuentre en inspecci\u00f3n.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        cur.execute(
            """
            SELECT COUNT(*) AS total
            FROM inspections
            WHERE batch_id = %s
            """,
            (batch_id,),
        )

        processed_quantity = int(
            cur.fetchone()["total"] or 0
        )

        finished_at = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cur.execute(
            """
            UPDATE batches
            SET status = 'REVISION_PENDIENTE',
                inspection_completed_at = %s
            WHERE id = %s
              AND status = 'EN_INSPECCION'
            """,
            (
                finished_at,
                batch_id,
            ),
        )

        if cur.rowcount != 1:
            conn.rollback()

            flash(
                "El lote cambi\u00f3 de estado antes "
                "de poder finalizarlo.",
                "error",
            )

            return redirect(
                url_for("batches")
            )

        conn.commit()

        # Actualmente el modo automatico sigue siendo global.
        # Cuando la estacion sea multi-linea, este estado sera
        # independiente para cada linea.
        AUTO_INSPECTION_ENABLED = False

        flash(
            (
                f"Lote {batch['code']} finalizado. "
                f"Se procesaron {processed_quantity} de "
                f"{batch['planned_quantity']} prendas."
            ),
            "success",
        )

        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    except Exception:
        if conn.in_transaction:
            conn.rollback()

        app.logger.exception(
            "No fue posible finalizar manualmente "
            "el lote %s.",
            batch_id,
        )

        flash(
            "No fue posible finalizar el lote.",
            "error",
        )

        return redirect(
            url_for("station")
        )

    finally:
        cur.close()
        conn.close()


@app.route("/lotes/<int:batch_id>/revision")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def batch_review(batch_id):
    batch = get_batch_review_summary(batch_id)

    if batch is None:
        flash(
            "El lote solicitado no existe.",
            "error",
        )
        return redirect(url_for("batches"))

    alerts = get_batch_alerts(batch_id)

    garment_page = request.args.get(
        "garment_page",
        1,
        type=int,
    )
    garments = get_batch_garments(
        batch_id,
        page=garment_page,
        per_page=50,
    )

    return render_template(
        "batch_review.html",
        batch=batch,
        alerts=alerts,
        garments=garments,
        garment_page=garments["page"],
    )




# EXCEPTION_ONLY_REVIEW_V1
@app.route(
    "/lotes/<int:batch_id>/revision/finalizar",
    methods=["POST"],
)
@login_required
@role_required(
    ROLE_ADMIN,
    ROLE_QUALITY_MANAGER,
)
def complete_batch_review(batch_id):
    conn = db()

    cur = conn.cursor(
        dictionary=True
    )

    try:
        conn.start_transaction()

        cur.execute(
            """
            SELECT
                id,
                status
            FROM batches
            WHERE id = %s
            FOR UPDATE
            """,
            (batch_id,),
        )

        batch = cur.fetchone()

        if (
            batch is None
            or batch["status"]
            != "REVISION_PENDIENTE"
        ):
            conn.rollback()

            return redirect(
                url_for(
                    "batch_review",
                    batch_id=batch_id,
                )
            )

        cur.execute(
            """
            SELECT
                COUNT(*) AS processed,

                COALESCE(
                    SUM(
                        CASE
                            WHEN ai_decision
                                 = 'ANOMALIA'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS alerts,

                COALESCE(
                    SUM(
                        CASE
                            WHEN ai_decision
                                 = 'ANOMALIA'
                             AND COALESCE(
                                 review_status,
                                 'PENDIENTE'
                             ) = 'PENDIENTE'
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS pending_alerts

            FROM inspections
            WHERE batch_id = %s
            """,
            (batch_id,),
        )

        stats = cur.fetchone()

        processed = int(
            stats["processed"] or 0
        )

        alerts = int(
            stats["alerts"] or 0
        )

        pending = int(
            stats["pending_alerts"]
            or 0
        )

        mode = (
            request.form.get(
                "review_mode",
                "",
            )
            .strip()
        )

        if processed <= 0:
            conn.rollback()

            return redirect(
                url_for(
                    "batch_review",
                    batch_id=batch_id,
                )
            )

        # Nunca se puede cerrar un lote
        # mientras exista una alerta
        # pendiente de decision humana.
        if pending > 0:
            conn.rollback()

            return redirect(
                url_for(
                    "batch_review",
                    batch_id=batch_id,
                )
            )

        # "Aceptar todas" solo existe
        # cuando realmente no hubo alertas.
        if (
            mode == "sin_alertas"
            and alerts != 0
        ):
            conn.rollback()

            return redirect(
                url_for(
                    "batch_review",
                    batch_id=batch_id,
                )
            )

        if mode not in {
            "sin_alertas",
            "revisado",
        }:
            conn.rollback()

            return redirect(
                url_for(
                    "batch_review",
                    batch_id=batch_id,
                )
            )

        closed_at = (
            datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        cur.execute(
            """
            UPDATE batches
            SET status = 'COMPLETADO',
                closed_at = %s
            WHERE id = %s
              AND status
                  = 'REVISION_PENDIENTE'
            """,
            (
                closed_at,
                batch_id,
            ),
        )

        if cur.rowcount != 1:
            conn.rollback()

            return redirect(
                url_for(
                    "batch_review",
                    batch_id=batch_id,
                )
            )

        conn.commit()

        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    except Exception:
        if conn.in_transaction:
            conn.rollback()

        app.logger.exception(
            "No fue posible finalizar "
            "la revision del lote %s.",
            batch_id,
        )

        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    finally:
        cur.close()
        conn.close()



@app.route(
    "/lotes/<int:batch_id>/revision/"
    "<int:inspection_id>/<decision>",
    methods=["POST"],
)
@login_required
@role_required(
    ROLE_ADMIN,
    ROLE_QUALITY_MANAGER,
)
def review_batch_alert(
    batch_id,
    inspection_id,
    decision,
):
    batch = get_batch_review_summary(
        batch_id
    )

    if (
        batch is None
        or batch["status"]
        != "REVISION_PENDIENTE"
    ):
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    inspection = fetch_one(
        """
        SELECT
            id,
            batch_id,
            batch_position,
            ai_decision,
            review_status
        FROM inspections
        WHERE id = %s
          AND batch_id = %s
        """,
        (
            inspection_id,
            batch_id,
        ),
    )

    if inspection is None:
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    current_status = (
        inspection["review_status"]
        or "PENDIENTE"
    )

    if (
        inspection["ai_decision"]
        != "ANOMALIA"
        or current_status
        != "PENDIENTE"
    ):
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    if decision == "confirmar":
        review_status = (
            "DEFECTO_CONFIRMADO"
        )
        human_validation = "Correcto"

    elif decision == "descartar":
        review_status = (
            "ALERTA_DESCARTADA"
        )
        human_validation = "Incorrecto"

    else:
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    review_notes = (
        request.form.get(
            "review_notes",
            "",
        )
        .strip()
    )

    if len(review_notes) < 5:
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
            + f"#alert-{inspection_id}"
        )

    if len(review_notes) > 500:
        review_notes = (
            review_notes[:500]
        )

    reviewed_by = session.get(
        "user_id"
    )

    reviewed_at = (
        datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    execute(
        """
        UPDATE inspections
        SET review_status = %s,
            human_validation = %s,
            reviewed_by = %s,
            reviewed_at = %s,
            review_notes = %s
        WHERE id = %s
          AND batch_id = %s
          AND ai_decision = 'ANOMALIA'
          AND COALESCE(
              review_status,
              'PENDIENTE'
          ) = 'PENDIENTE'
        """,
        (
            review_status,
            human_validation,
            reviewed_by,
            reviewed_at,
            review_notes,
            inspection_id,
            batch_id,
        ),
    )

    return redirect(
        url_for(
            "batch_review",
            batch_id=batch_id,
        )
    )




@app.route("/estacion")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station():
    return render_template(
        "station.html",
        active_batch=get_active_batch(),
    )


@app.route("/api/station/manual", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_manual_inspect():
    global AUTO_LAST_RESULT
    global AUTO_LAST_ERROR

    try:
        if get_active_batch() is None:
            return jsonify({
                "ok": False,
                "message": (
                    "No hay un lote activo. "
                    "Inicia un lote antes de inspeccionar."
                ),
            }), 409

        # latest_frame: NO cap.read(); no abre otra sesión RTSP.
        ok, frame = read_camera_frame(source="manual")

        if not ok:
            return jsonify({
                "ok": False,
                "message": (
                    "No se pudo leer la cámara. "
                    "No se guardó ningún registro."
                ),
            }), 503

        frame_available = time.perf_counter()
        try:
            result = register_inspection_from_frame(
                frame,
                notes=(
                    "Registro manual generado desde "
                    "estación de inspección."
                ),
                decision_start=frame_available,
                source="manual",
            )
        except AIDomainError as domain_error:
            return jsonify({
                "ok": False,
                "message": str(domain_error),
            }), 409
        except RuntimeError as runtime_error:
            if not is_garment_not_detected_error(runtime_error):
                app.logger.exception(
                    "Error inesperado en la inspección manual."
                )
                return jsonify({
                    "ok": False,
                    "message": (
                        "No se pudo completar la inspección. "
                        "Intente de nuevo."
                    ),
                }), 500

            return jsonify({
                "ok": False,
                "message": (
                    "No se detectó una prenda completa en el área de "
                    "inspección. Colóquela correctamente e intente de nuevo."
                ),
            }), 409

        AUTO_LAST_RESULT = result
        AUTO_LAST_ERROR = None

        return jsonify({
            "ok": True,
            "message": (
                "Inspección manual registrada correctamente."
            ),
            "result": result,
        })

    except Exception:
        app.logger.exception("Error al ejecutar la inspección manual.")
        AUTO_LAST_ERROR = "Error al ejecutar la inspección manual."

        return jsonify({
            "ok": False,
            "message": (
                "Error al ejecutar la inspección manual."
            ),
        }), 500




@app.route("/api/station/camera/reconnect", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER, ROLE_MODEL_MANAGER)
def station_camera_reconnect():
    """
    Pide reabrir RTSP. No crea otro camera_capture_worker.
    La liberación real la hace el worker (evita release concurrente).
    """
    global CAMERA_FORCE_REOPEN

    with camera_lock:
        CAMERA_FORCE_REOPEN = True

    ensure_camera_capture_worker()

    return jsonify({
        "ok": True,
        "message": (
            "Reconexion de camara solicitada."
        ),
        "camera": get_camera_status(),
    }), 202


@app.route("/api/station/auto/start", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_auto_start():
    global AUTO_INSPECTION_ENABLED
    global AUTO_THREAD

    if is_ai_capture_mode_active():
        return jsonify({
            "ok": False,
            "message": (
                "Hay una sesión de captura IA activa. "
                "Finalízala o cancélala antes de iniciar producción."
            ),
        }), 409

    active_batch = get_active_batch()

    if active_batch is None:
        return jsonify({
            "ok": False,
            "message": (
                "No hay un lote activo. "
                "Inicia un lote antes de activar la inspección automática."
            ),
        }), 409

    if not CAMERA_CONNECTED:
        return jsonify({
            "ok": False,
            "message": (
                "No se puede iniciar la inspección automática: "
                "la cámara no tiene señal."
            ),
        }), 503

    if int(active_batch["processed_quantity"] or 0) >= int(
        active_batch["planned_quantity"]
    ):
        return jsonify({
            "ok": False,
            "message": "El lote ya completó todas sus prendas.",
        }), 409

    if AUTO_INSPECTION_ENABLED:
        return jsonify({
            "ok": True,
            "message": "El modo automático ya está activo.",
        })

    AUTO_INSPECTION_ENABLED = True
    AUTO_THREAD = threading.Thread(target=auto_inspection_worker, daemon=True)
    AUTO_THREAD.start()

    return jsonify({
        "ok": True,
        "message": "Modo automático iniciado.",
    })


@app.route("/api/station/auto/stop", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_auto_stop():
    global AUTO_INSPECTION_ENABLED

    AUTO_INSPECTION_ENABLED = False

    return jsonify({
        "ok": True,
        "message": "Modo automático detenido.",
    })


@app.route("/api/station/auto/status")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_auto_status():
    active_batch = get_active_batch()

    last_result = AUTO_LAST_RESULT

    # Evita mostrar el resultado de un lote anterior mientras
    # un lote nuevo todavía no tiene inspecciones.
    if (
        last_result is not None
        and active_batch is not None
        and last_result.get("batch_id") != active_batch["id"]
    ):
        last_result = None

    # Persistencia: si Flask se reinició o la página se recarg?,
    # recuperar el ?ltimo resultado directamente de MySQL.
    if last_result is None:
        if active_batch is not None:
            row = fetch_one(
                """
                SELECT *
                FROM inspections
                WHERE batch_id = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (active_batch["id"],),
            )
        else:
            row = fetch_one(
                """
                SELECT *
                FROM inspections
                ORDER BY id DESC
                LIMIT 1
                """
            )

        last_result = row_to_json(row)

    return jsonify({
        "ok": True,
        "automatic": AUTO_INSPECTION_ENABLED,
        "last_result": last_result,
        "last_error": AUTO_LAST_ERROR,
        "active_batch": row_to_json(active_batch),
        "camera": get_camera_status(),
        "alert_seq": current_quality_alert_seq(),
    })


@app.route("/api/station/alerts/stream")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_alerts_stream():
    """
    Server-Sent Events de anomalías.

    Por defecto (sin query after) solo emite eventos NUEVOS
    desde la conexión: un reload no reanima alarmas viejas.
    """
    after_raw = request.args.get("after")
    if after_raw is None:
        start_after = current_quality_alert_seq()
    else:
        try:
            start_after = int(after_raw)
        except (TypeError, ValueError):
            start_after = current_quality_alert_seq()

    def event_stream():
        last_sent = start_after
        last_beat = time.time()

        while True:
            events = get_quality_alerts_after(last_sent)
            for event in events:
                last_sent = int(event["seq"])
                payload = json.dumps(event, ensure_ascii=False)
                yield (
                    f"id: {event['seq']}\n"
                    f"event: quality-alert\n"
                    f"data: {payload}\n\n"
                )

            now = time.time()
            if now - last_beat >= 15.0:
                last_beat = now
                yield f": heartbeat {int(now)}\n\n"

            time.sleep(0.12)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/station/alerts/last")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_alerts_last():
    """Polling corto de respaldo si SSE no está disponible."""
    after_raw = request.args.get("after")

    if after_raw is None:
        # Suscripción desde ahora: no reenviar alarmas antiguas.
        events = []
        after = current_quality_alert_seq()
    else:
        try:
            after = int(after_raw)
        except (TypeError, ValueError):
            after = current_quality_alert_seq()
        events = get_quality_alerts_after(after)

    return jsonify({
        "ok": True,
        "seq": current_quality_alert_seq(),
        "after": after,
        "events": events,
    })


@app.route("/api/station/alerts/ack", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_alerts_ack():
    """
    Diagnóstico de entrega de alarma. Solo logging; no toca BD
    y no bloquea el sonido del navegador.
    """
    data = request.get_json(silent=True) or {}

    try:
        server_emitted_at_ms = int(
            data.get("server_emitted_at_ms") or 0
        )
    except (TypeError, ValueError):
        server_emitted_at_ms = 0

    try:
        browser_received_at_ms = int(
            data.get("browser_received_at_ms") or 0
        )
    except (TypeError, ValueError):
        browser_received_at_ms = 0

    try:
        audio_started_at_ms = int(
            data.get("audio_started_at_ms") or 0
        )
    except (TypeError, ValueError):
        audio_started_at_ms = 0

    event_id = data.get("event_id") or data.get("alert_id") or "n/a"
    garment_token = (
        data.get("garment_token")
        if data.get("garment_token") is not None
        else data.get("token")
    )

    transport_ms = "n/a"
    browser_audio_delay_ms = "n/a"
    server_to_audio_ms = "n/a"

    if server_emitted_at_ms and browser_received_at_ms:
        transport_ms = str(
            browser_received_at_ms - server_emitted_at_ms
        )

    if browser_received_at_ms and audio_started_at_ms:
        browser_audio_delay_ms = str(
            audio_started_at_ms - browser_received_at_ms
        )

    if server_emitted_at_ms and audio_started_at_ms:
        server_to_audio_ms = str(
            audio_started_at_ms - server_emitted_at_ms
        )

    print(
        "[ALERT_DELIVERY]\n"
        f"token={garment_token}\n"
        f"event_id={event_id}\n"
        f"transport_ms={transport_ms}\n"
        f"browser_audio_delay_ms={browser_audio_delay_ms}\n"
        f"server_to_audio_ms={server_to_audio_ms}",
        flush=True,
    )

    return jsonify({"ok": True})



@app.route("/api/station/latest")
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_latest():
    row = fetch_one(
        """
        SELECT *
        FROM inspections
        ORDER BY id DESC
        LIMIT 1
        """
    )

    return jsonify({
        "ok": True,
        "inspection": row_to_json(row),
    })


GARMENT_NOT_DETECTED_MARKERS = (
    "blusa completa",
    "prenda completa",
)


def is_garment_not_detected_error(error) -> bool:
    """
    Error de negocio: la prenda no quedó completa en el área.

    No es un fallo técnico: se muestra un mensaje amigable y jamás
    un traceback.
    """
    text = str(error or "").strip().lower()

    if not text:
        return False

    return any(marker in text for marker in GARMENT_NOT_DETECTED_MARKERS)


@app.route("/inspeccion", methods=["GET", "POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def inspection():
    result = None

    if request.method == "POST":
        garment = "Prenda inspeccionada"
        size = "S"
        notes = "Registro generado automáticamente por el sistema de inspección."

        img_path, img_rel = save_image(request.files.get("image"))

        if img_path is None:
            flash("No se pudo leer la cámara. No se guardó ningún registro.", "error")
            return render_template("inspection.html", result=None)

        decision_start = time.perf_counter()

        try:
            status, defect, conf, zone, result_rel = detect_defect(
                img_path,
                garment_token=next_garment_token(),
                decision_start=decision_start,
                source="legacy",
            )
        except Exception as error:
            if is_garment_not_detected_error(error):
                flash(
                    "No se detectó una prenda completa en el área de "
                    "inspección. Colóquela correctamente e intente de nuevo.",
                    "error",
                )
            else:
                app.logger.exception(
                    "Error inesperado durante la inspección manual."
                )
                flash(
                    "No se pudo completar la inspección. "
                    "Intente de nuevo.",
                    "error",
                )

            return render_template("inspection.html", result=None)

        code = f"INS-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

        execute(
            """
            INSERT INTO inspections (
                code, created_at, garment_type, size, status, defect_type,
                confidence, zone, image_original, image_result, notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                code,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                garment,
                size,
                status,
                defect,
                conf,
                zone,
                img_rel,
                result_rel,
                notes,
            ),
        )

        result = dict(
            code=code,
            status=status,
            defect_type=defect,
            confidence=conf,
            zone=zone,
            image_original=img_rel,
            image_result=result_rel,
        )

    return render_template("inspection.html", result=result)


@app.route("/registros")
@login_required
def records():
    from datetime import datetime as dt_datetime
    from datetime import time as dt_time
    from datetime import timedelta as dt_timedelta
    from zoneinfo import ZoneInfo

    timezone_gt = ZoneInfo("America/Guatemala")
    timezone_utc = ZoneInfo("UTC")

    today_gt = dt_datetime.now(timezone_gt).date()
    today_text = today_gt.strftime("%Y-%m-%d")

    q = request.args.get("q", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    batch_id = request.args.get("batch_id", "").strip()
    model_id = request.args.get("model_id", "").strip()
    reviewer_id = request.args.get("reviewer_id", "").strip()
    result_filter = request.args.get("result", "").strip()
    show_mode = request.args.get("show", "all").strip().lower() or "all"

    has_explicit_filters = any([
        q,
        date_from,
        date_to,
        batch_id,
        model_id,
        reviewer_id,
        result_filter,
    ])

    # Sin filtros, el resumen corresponde solamente al dia actual.
    if not has_explicit_filters:
        date_from = today_text
        date_to = today_text

    alert_condition = """
        (
            i.ai_decision = 'ANOMALIA'
            OR (
                i.ai_decision IS NULL
                AND i.status = 'Defecto'
            )
        )
    """

    normal_condition = """
        (
            i.ai_decision = 'NORMAL'
            OR (
                i.ai_decision IS NULL
                AND i.status = 'Aprobado'
            )
        )
    """

    confirmed_condition = f"""
        (
            {alert_condition}
            AND (
                i.review_status = 'DEFECTO_CONFIRMADO'
                OR (
                    COALESCE(
                        i.review_status,
                        'PENDIENTE'
                    ) = 'PENDIENTE'
                    AND i.human_validation = 'Correcto'
                )
            )
        )
    """

    discarded_condition = f"""
        (
            {alert_condition}
            AND (
                i.review_status = 'ALERTA_DESCARTADA'
                OR (
                    COALESCE(
                        i.review_status,
                        'PENDIENTE'
                    ) = 'PENDIENTE'
                    AND i.human_validation = 'Incorrecto'
                )
            )
        )
    """

    pending_condition = f"""
        (
            {alert_condition}
            AND COALESCE(
                i.review_status,
                'PENDIENTE'
            ) = 'PENDIENTE'
            AND COALESCE(
                i.human_validation,
                'Pendiente'
            ) = 'Pendiente'
        )
    """

    passed_condition = f"""
        (
            {normal_condition}
            OR {discarded_condition}
        )
    """

    conditions = []
    params = []

    if q:
        wildcard = f"%{q}%"

        conditions.append(
            """
            (
                i.code LIKE %s
                OR i.defect_type LIKE %s
                OR i.zone LIKE %s
                OR b.code LIKE %s
                OR gm.code LIKE %s
                OR gm.name LIKE %s
                OR reviewer.username LIKE %s
                OR reviewer.full_name LIKE %s
            )
            """
        )

        params.extend([wildcard] * 8)

    if date_from:
        try:
            parsed_date = dt_datetime.strptime(
                date_from,
                "%Y-%m-%d",
            ).date()

            start_gt = dt_datetime.combine(
                parsed_date,
                dt_time.min,
                tzinfo=timezone_gt,
            )

            start_utc = (
                start_gt
                .astimezone(timezone_utc)
                .replace(tzinfo=None)
            )

            conditions.append("i.created_at >= %s")
            params.append(start_utc)

        except ValueError:
            flash(
                "La fecha inicial no es v\u00e1lida.",
                "error",
            )
            return redirect(url_for("records"))

    if date_to:
        try:
            parsed_date = dt_datetime.strptime(
                date_to,
                "%Y-%m-%d",
            ).date()

            end_gt = (
                dt_datetime.combine(
                    parsed_date,
                    dt_time.min,
                    tzinfo=timezone_gt,
                )
                + dt_timedelta(days=1)
            )

            end_utc = (
                end_gt
                .astimezone(timezone_utc)
                .replace(tzinfo=None)
            )

            conditions.append("i.created_at < %s")
            params.append(end_utc)

        except ValueError:
            flash(
                "La fecha final no es v\u00e1lida.",
                "error",
            )
            return redirect(url_for("records"))

    if batch_id:
        try:
            conditions.append("i.batch_id = %s")
            params.append(int(batch_id))
        except ValueError:
            batch_id = ""

    if model_id:
        try:
            conditions.append("b.garment_model_id = %s")
            params.append(int(model_id))
        except ValueError:
            model_id = ""

    if reviewer_id:
        try:
            conditions.append("i.reviewed_by = %s")
            params.append(int(reviewer_id))
        except ValueError:
            reviewer_id = ""

    if result_filter == "APTAS":
        conditions.append(passed_condition)

    elif result_filter == "RECHAZADAS":
        conditions.append(confirmed_condition)

    elif result_filter == "ALERTAS":
        conditions.append(alert_condition)

    elif result_filter == "PENDIENTES":
        conditions.append(pending_condition)

    where_sql = ""

    if conditions:
        where_sql = " AND " + " AND ".join(
            f"({condition})"
            for condition in conditions
        )

    from_sql = """
        FROM inspections i
        LEFT JOIN batches b
          ON b.id = i.batch_id
        LEFT JOIN garment_models gm
          ON gm.id = b.garment_model_id
        LEFT JOIN users reviewer
          ON reviewer.id = i.reviewed_by
        WHERE 1 = 1
    """

    summary = fetch_one(
        f"""
        SELECT
            COUNT(i.id) AS inspected,

            COUNT(
                DISTINCT i.batch_id
            ) AS batches_with_activity,

            COALESCE(
                SUM(
                    CASE
                        WHEN {passed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS passed,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS rejected,

            COALESCE(
                SUM(
                    CASE
                        WHEN {alert_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN {discarded_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded,

            COALESCE(
                SUM(
                    CASE
                        WHEN {pending_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending

        {from_sql}
        {where_sql}
        """,
        tuple(params),
    )

    inspected_count = int(summary["inspected"] or 0)
    passed_count = int(summary["passed"] or 0)
    rejected_count = int(summary["rejected"] or 0)

    resolved_count = passed_count + rejected_count

    summary["acceptance_rate"] = (
        round(
            passed_count / resolved_count * 100,
            1,
        )
        if resolved_count > 0
        else 0.0
    )

    lot_summary = []

    # No mostramos decenas de lotes al abrir la pantalla.
    # El resumen por lote aparece cuando el usuario filtra.
    if has_explicit_filters or show_mode:
        lot_summary = fetch_all(
            f"""
            SELECT
                b.id AS batch_id,
                COALESCE(
                    b.code,
                    'Sin lote'
                ) AS batch_code,

                gm.code AS garment_model_code,
                gm.name AS garment_model_name,

                COUNT(i.id) AS inspected,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {passed_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS passed,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {confirmed_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS rejected,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {alert_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS alerts,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {discarded_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS discarded,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {pending_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS pending

            {from_sql}
            {where_sql}

            GROUP BY
                b.id,
                b.code,
                gm.code,
                gm.name

            ORDER BY
                CASE
                    WHEN b.id IS NULL THEN 1
                    ELSE 0
                END,
                b.id DESC

            LIMIT 30
            """,
            tuple(params),
        )

    reviewer_summary = []

    if has_explicit_filters or show_mode:
        reviewer_conditions = list(conditions)

        reviewer_conditions.append(
            f"""
            (
                {confirmed_condition}
                OR {discarded_condition}
            )
            """
        )

        reviewer_where = " AND " + " AND ".join(
            f"({condition})"
            for condition in reviewer_conditions
        )

        reviewer_summary = fetch_all(
            f"""
            SELECT
                i.reviewed_by,

                COALESCE(
                    NULLIF(reviewer.full_name, ''),
                    reviewer.username,
                    'No registrado'
                ) AS reviewer_name,

                COUNT(i.id) AS decisions,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {confirmed_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS rejected,

                COALESCE(
                    SUM(
                        CASE
                            WHEN {discarded_condition}
                            THEN 1 ELSE 0
                        END
                    ),
                    0
                ) AS discarded

            {from_sql}
            {reviewer_where}

            GROUP BY
                i.reviewed_by,
                reviewer.full_name,
                reviewer.username

            ORDER BY decisions DESC
            """,
            tuple(params),
        )

    rows = []

    if show_mode in (
        "alerts",
        "reviewed",
        "rejected",
        "pending",
        "all",
        "accepted",
    ):
        detail_conditions = list(conditions)
        detail_params = list(params)

        if show_mode == "accepted":
            detail_conditions.append(
                passed_condition
            )

        elif show_mode == "alerts":
            detail_conditions.append(
                alert_condition
            )

        elif show_mode == "reviewed":
            detail_conditions.append(
                f"""
                (
                    {confirmed_condition}
                    OR {discarded_condition}
                )
                """
            )

        elif show_mode == "rejected":
            detail_conditions.append(
                confirmed_condition
            )

        elif show_mode == "pending":
            detail_conditions.append(
                pending_condition
            )

        detail_where = ""

        if detail_conditions:
            detail_where = (
                " AND "
                + " AND ".join(
                    f"({condition})"
                    for condition in detail_conditions
                )
            )

        rows = fetch_all(
            f"""
            SELECT
                i.id,
                i.code,
                i.batch_position,
                i.created_at,
                i.status,
                i.ai_decision,
                i.defect_type,
                i.confidence,
                i.zone,
                i.human_validation,
                i.review_status,
                i.reviewed_by,
                i.reviewed_at,
                i.review_notes,
                i.image_original,
                i.image_result,

                b.id AS batch_id,
                b.code AS batch_code,

                gm.code AS garment_model_code,
                gm.name AS garment_model_name,

                COALESCE(
                    NULLIF(reviewer.full_name, ''),
                    reviewer.username,
                    'No registrado'
                ) AS reviewer_name

            {from_sql}
            {detail_where}

            ORDER BY i.id DESC
            """,
            tuple(detail_params),
        )

    def convert_utc_to_gt(value):
        if value is None:
            return None

        value_utc = value.replace(
            tzinfo=timezone_utc
        )

        return value_utc.astimezone(
            timezone_gt
        )

    for row in rows:
        row["created_at_gt"] = convert_utc_to_gt(
            row.get("created_at")
        )

        row["reviewed_at_gt"] = convert_utc_to_gt(
            row.get("reviewed_at")
        )

        review_status = row.get("review_status")
        human_validation = row.get(
            "human_validation"
        )

        if (
            review_status == "DEFECTO_CONFIRMADO"
            or (
                review_status in (
                    None,
                    "",
                    "PENDIENTE",
                )
                and human_validation == "Correcto"
            )
        ):
            row["decision_label"] = (
                "Rechazada por defecto"
            )
            row["decision_class"] = "rejected"

        elif (
            review_status == "ALERTA_DESCARTADA"
            or (
                review_status in (
                    None,
                    "",
                    "PENDIENTE",
                )
                and human_validation == "Incorrecto"
            )
        ):
            row["decision_label"] = (
                "Apta - alerta descartada"
            )
            row["decision_class"] = "passed"

        else:
            row["decision_label"] = (
                "Pendiente de revision"
            )
            row["decision_class"] = "pending"

    batches_options = fetch_all(
        """
        SELECT id, code
        FROM batches
        ORDER BY id DESC
        LIMIT 200
        """
    )

    models_options = fetch_all(
        """
        SELECT id, code, name
        FROM garment_models
        ORDER BY code
        """
    )

    reviewers_options = fetch_all(
        """
        SELECT
            id,
            COALESCE(
                NULLIF(full_name, ''),
                username
            ) AS name
        FROM users
        ORDER BY name
        """
    )

    return render_template(
        "records.html",
        summary=summary,
        lot_summary=lot_summary,
        reviewer_summary=reviewer_summary,
        rows=rows,
        batches_options=batches_options,
        models_options=models_options,
        reviewers_options=reviewers_options,
        date_from=date_from,
        date_to=date_to,
        batch_id=batch_id,
        model_id=model_id,
        reviewer_id=reviewer_id,
        result_filter=result_filter,
        show_mode=show_mode,
        has_explicit_filters=has_explicit_filters,
    )



def get_informe_data():
    from datetime import datetime as dt_datetime
    from datetime import time as dt_time
    from datetime import timedelta as dt_timedelta
    from zoneinfo import ZoneInfo

    timezone_gt = ZoneInfo("America/Guatemala")
    timezone_utc = ZoneInfo("UTC")

    now_gt = dt_datetime.now(timezone_gt)
    today_gt = now_gt.date()

    requested_from = request.args.get(
        "date_from",
        "",
    ).strip()

    requested_to = request.args.get(
        "date_to",
        "",
    ).strip()

    batch_id = request.args.get(
        "batch_id",
        "",
    ).strip()

    model_id = request.args.get(
        "model_id",
        "",
    ).strip()

    try:
        start_date = (
            dt_datetime.strptime(
                requested_from,
                "%Y-%m-%d",
            ).date()
            if requested_from
            else today_gt
        )
    except ValueError:
        start_date = today_gt

    try:
        end_date = (
            dt_datetime.strptime(
                requested_to,
                "%Y-%m-%d",
            ).date()
            if requested_to
            else today_gt
        )
    except ValueError:
        end_date = today_gt

    if end_date < start_date:
        start_date, end_date = (
            end_date,
            start_date,
        )

    date_from = start_date.strftime(
        "%Y-%m-%d"
    )

    date_to = end_date.strftime(
        "%Y-%m-%d"
    )

    start_gt = dt_datetime.combine(
        start_date,
        dt_time.min,
        tzinfo=timezone_gt,
    )

    end_gt = (
        dt_datetime.combine(
            end_date,
            dt_time.min,
            tzinfo=timezone_gt,
        )
        + dt_timedelta(days=1)
    )

    start_utc = (
        start_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )

    end_utc = (
        end_gt
        .astimezone(timezone_utc)
        .replace(tzinfo=None)
    )

    alert_condition = """
        (
            i.ai_decision = 'ANOMALIA'
            OR (
                i.ai_decision IS NULL
                AND i.status = 'Defecto'
            )
        )
    """

    normal_condition = """
        (
            i.ai_decision = 'NORMAL'
            OR (
                i.ai_decision IS NULL
                AND i.status = 'Aprobado'
            )
        )
    """

    confirmed_condition = f"""
        (
            {alert_condition}
            AND (
                i.review_status = 'DEFECTO_CONFIRMADO'
                OR (
                    COALESCE(
                        i.review_status,
                        'PENDIENTE'
                    ) = 'PENDIENTE'
                    AND i.human_validation = 'Correcto'
                )
            )
        )
    """

    discarded_condition = f"""
        (
            {alert_condition}
            AND (
                i.review_status = 'ALERTA_DESCARTADA'
                OR (
                    COALESCE(
                        i.review_status,
                        'PENDIENTE'
                    ) = 'PENDIENTE'
                    AND i.human_validation = 'Incorrecto'
                )
            )
        )
    """

    pending_condition = f"""
        (
            {alert_condition}
            AND COALESCE(
                i.review_status,
                'PENDIENTE'
            ) = 'PENDIENTE'
            AND COALESCE(
                i.human_validation,
                'Pendiente'
            ) = 'Pendiente'
        )
    """

    passed_condition = f"""
        (
            {normal_condition}
            OR {discarded_condition}
        )
    """

    conditions = [
        "i.batch_id IS NOT NULL",
        "i.created_at >= %s",
        "i.created_at < %s",
    ]

    params = [
        start_utc,
        end_utc,
    ]

    if batch_id:
        try:
            conditions.append(
                "i.batch_id = %s"
            )
            params.append(
                int(batch_id)
            )
        except ValueError:
            batch_id = ""

    if model_id:
        try:
            conditions.append(
                "b.garment_model_id = %s"
            )
            params.append(
                int(model_id)
            )
        except ValueError:
            model_id = ""

    where_sql = (
        " WHERE "
        + " AND ".join(
            f"({condition})"
            for condition in conditions
        )
    )

    from_sql = """
        FROM inspections i

        LEFT JOIN batches b
          ON b.id = i.batch_id

        LEFT JOIN garment_models gm
          ON gm.id = b.garment_model_id

        LEFT JOIN users reviewer
          ON reviewer.id = i.reviewed_by
    """

    summary = fetch_one(
        f"""
        SELECT
            COUNT(i.id) AS inspected,

            COUNT(
                DISTINCT i.batch_id
            ) AS batches_with_activity,

            COUNT(
                DISTINCT gm.id
            ) AS models_with_activity,

            COALESCE(
                SUM(
                    CASE
                        WHEN {passed_condition}
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS passed,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS rejected,

            COALESCE(
                SUM(
                    CASE
                        WHEN {alert_condition}
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN {discarded_condition}
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS discarded,

            COALESCE(
                SUM(
                    CASE
                        WHEN {pending_condition}
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS pending

        {from_sql}
        {where_sql}
        """,
        tuple(params),
    )

    inspected = int(
        summary["inspected"] or 0
    )

    passed = int(
        summary["passed"] or 0
    )

    rejected = int(
        summary["rejected"] or 0
    )

    alerts = int(
        summary["alerts"] or 0
    )

    discarded = int(
        summary["discarded"] or 0
    )

    pending = int(
        summary["pending"] or 0
    )

    summary["acceptance_rate"] = (
        round(
            passed / inspected * 100,
            1,
        )
        if inspected
        else 0.0
    )

    summary["rejection_rate"] = (
        round(
            rejected / inspected * 100,
            1,
        )
        if inspected
        else 0.0
    )

    summary["alert_rate"] = (
        round(
            alerts / inspected * 100,
            1,
        )
        if inspected
        else 0.0
    )

    reviewed_alerts = (
        rejected
        + discarded
    )

    summary["reviewed_alerts"] = (
        reviewed_alerts
    )

    summary["review_completion_rate"] = (
        round(
            reviewed_alerts
            / alerts
            * 100,
            1,
        )
        if alerts
        else 0.0
    )

    lot_summary = fetch_all(
        f"""
        SELECT
            b.id,
            b.code,
            b.planned_quantity,
            b.status,

            gm.id AS garment_model_id,
            gm.code AS garment_model_code,
            gm.name AS garment_model_name,

            COUNT(i.id) AS inspected,

            (
                SELECT COUNT(*)
                FROM inspections ix
                WHERE ix.batch_id = b.id
            ) AS total_processed,

            COALESCE(
                SUM(
                    CASE
                        WHEN {passed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS passed,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS rejected,

            COALESCE(
                SUM(
                    CASE
                        WHEN {alert_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN {discarded_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded,

            COALESCE(
                SUM(
                    CASE
                        WHEN {pending_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending

        {from_sql}
        {where_sql}
          AND b.id IS NOT NULL

        GROUP BY
            b.id,
            b.code,
            b.planned_quantity,
            b.status,
            gm.id,
            gm.code,
            gm.name

        ORDER BY b.id DESC
        """,
        tuple(params),
    )

    for row in lot_summary:
        inspected_lot = int(
            row["inspected"] or 0
        )

        total_processed = int(
            row["total_processed"] or 0
        )

        planned = int(
            row["planned_quantity"] or 0
        )

        rejected_lot = int(
            row["rejected"] or 0
        )

        row["progress_rate"] = (
            round(
                total_processed
                / planned
                * 100,
                1,
            )
            if planned
            else 0.0
        )

        row["rejection_rate"] = (
            round(
                rejected_lot
                / inspected_lot
                * 100,
                1,
            )
            if inspected_lot
            else 0.0
        )

        row["acceptance_rate"] = (
            round(
                int(row["passed"] or 0)
                / inspected_lot
                * 100,
                1,
            )
            if inspected_lot
            else 0.0
        )

    model_summary = fetch_all(
        f"""
        SELECT
            gm.id,
            gm.code,
            gm.name,
            gm.color,

            COUNT(i.id) AS inspected,

            COUNT(
                DISTINCT i.batch_id
            ) AS batches,

            COALESCE(
                SUM(
                    CASE
                        WHEN {passed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS passed,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS rejected,

            COALESCE(
                SUM(
                    CASE
                        WHEN {alert_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN {discarded_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded,

            COALESCE(
                SUM(
                    CASE
                        WHEN {pending_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS pending

        {from_sql}
        {where_sql}
          AND gm.id IS NOT NULL

        GROUP BY
            gm.id,
            gm.code,
            gm.name,
            gm.color

        ORDER BY
            rejected DESC,
            inspected DESC
        """,
        tuple(params),
    )

    for row in model_summary:
        inspected_model = int(
            row["inspected"] or 0
        )

        rejected_model = int(
            row["rejected"] or 0
        )

        row["rejection_rate"] = (
            round(
                rejected_model
                / inspected_model
                * 100,
                1,
            )
            if inspected_model
            else 0.0
        )

        row["acceptance_rate"] = (
            round(
                int(row["passed"] or 0)
                / inspected_model
                * 100,
                1,
            )
            if inspected_model
            else 0.0
        )

    rejection_reasons = fetch_all(
        f"""
        SELECT
            COALESCE(
                NULLIF(i.defect_type, ''),
                'Sin especificar'
            ) AS defect_type,

            COUNT(i.id) AS total

        {from_sql}
        {where_sql}
          AND {confirmed_condition}

        GROUP BY
            COALESCE(
                NULLIF(i.defect_type, ''),
                'Sin especificar'
            )

        ORDER BY total DESC
        """,
        tuple(params),
    )

    rejection_zones = fetch_all(
        f"""
        SELECT
            COALESCE(
                NULLIF(i.zone, ''),
                'Sin especificar'
            ) AS zone,

            COUNT(i.id) AS total

        {from_sql}
        {where_sql}
          AND {confirmed_condition}

        GROUP BY
            COALESCE(
                NULLIF(i.zone, ''),
                'Sin especificar'
            )

        ORDER BY total DESC
        LIMIT 10
        """,
        tuple(params),
    )

    reviewer_summary = fetch_all(
        f"""
        SELECT
            i.reviewed_by,

            COALESCE(
                NULLIF(
                    reviewer.full_name,
                    ''
                ),
                reviewer.username,
                'No registrado'
            ) AS reviewer_name,

            COUNT(i.id) AS decisions,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS rejected,

            COALESCE(
                SUM(
                    CASE
                        WHEN {discarded_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS discarded

        {from_sql}
        {where_sql}
          AND (
              {confirmed_condition}
              OR {discarded_condition}
          )

        GROUP BY
            i.reviewed_by,
            reviewer.full_name,
            reviewer.username

        ORDER BY decisions DESC
        """,
        tuple(params),
    )

    active_batch = get_active_batch()

    if active_batch:
        active_processed = int(
            active_batch.get(
                "processed_quantity"
            )
            or 0
        )

        active_planned = int(
            active_batch.get(
                "planned_quantity"
            )
            or 0
        )

        active_alerts = int(
            active_batch.get(
                "alerts"
            )
            or 0
        )

        active_confirmed = int(
            active_batch.get(
                "confirmed_defects"
            )
            or 0
        )

        active_discarded = int(
            active_batch.get(
                "discarded_alerts"
            )
            or 0
        )

        active_auto = int(
            active_batch.get(
                "auto_approved"
            )
            or 0
        )

        active_batch["progress_rate"] = (
            round(
                active_processed
                / active_planned
                * 100,
                1,
            )
            if active_planned
            else 0.0
        )

        active_batch["passed_quantity"] = (
            active_auto
            + active_discarded
        )

        active_batch["pending_alerts"] = max(
            active_alerts
            - active_confirmed
            - active_discarded,
            0,
        )

        active_batch["rejection_rate"] = (
            round(
                active_confirmed
                / active_processed
                * 100,
                1,
            )
            if active_processed
            else 0.0
        )

    daily_goal = None

    if (
        start_date == end_date
        and not batch_id
        and not model_id
    ):
        daily_goal = fetch_one(
            """
            SELECT
                goal_date,
                target_batches,
                target_garments,
                shift_start,
                shift_end
            FROM daily_production_goals
            WHERE goal_date = %s
            """,
            (start_date,),
        )

        if daily_goal:
            target_garments = int(
                daily_goal[
                    "target_garments"
                ]
                or 0
            )

            daily_goal[
                "progress_rate"
            ] = (
                round(
                    inspected
                    / target_garments
                    * 100,
                    1,
                )
                if target_garments
                else 0.0
            )

    top_problem_lot = None

    rejected_lots = [
        row
        for row in lot_summary
        if int(
            row["rejected"] or 0
        ) > 0
    ]

    if rejected_lots:
        top_problem_lot = max(
            rejected_lots,
            key=lambda row: (
                int(
                    row["rejected"]
                    or 0
                ),
                float(
                    row["rejection_rate"]
                    or 0
                ),
            ),
        )

    top_problem_model = None

    rejected_models = [
        row
        for row in model_summary
        if int(
            row["rejected"] or 0
        ) > 0
    ]

    if rejected_models:
        top_problem_model = max(
            rejected_models,
            key=lambda row: (
                int(
                    row["rejected"]
                    or 0
                ),
                float(
                    row["rejection_rate"]
                    or 0
                ),
            ),
        )

    if start_date == end_date:
        period_label = (
            start_date.strftime(
                "%d/%m/%Y"
            )
        )
    else:
        period_label = (
            start_date.strftime(
                "%d/%m/%Y"
            )
            + " al "
            + end_date.strftime(
                "%d/%m/%Y"
            )
        )

    quick_7_from = (
        today_gt
        - dt_timedelta(days=6)
    ).strftime(
        "%Y-%m-%d"
    )

    quick_30_from = (
        today_gt
        - dt_timedelta(days=29)
    ).strftime(
        "%Y-%m-%d"
    )

    today_text = today_gt.strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # Compatibilidad con Excel/PDF actuales.
    # Las mismas claves siguen existiendo.
    # --------------------------------------------------------

    by_garment = fetch_all(
        f"""
        SELECT
            COALESCE(
                CONCAT(
                    gm.code,
                    ' - ',
                    gm.name
                ),
                i.garment_type,
                'Sin modelo'
            ) AS garment_type,

            COUNT(i.id) AS total,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS defects

        {from_sql}
        {where_sql}

        GROUP BY
            COALESCE(
                CONCAT(
                    gm.code,
                    ' - ',
                    gm.name
                ),
                i.garment_type,
                'Sin modelo'
            )

        ORDER BY total DESC
        """,
        tuple(params),
    )

    by_defect = rejection_reasons

    inspections = fetch_all(
        f"""
        SELECT
            i.id,
            i.code,
            i.created_at,
            i.garment_type,
            i.size,
            i.status,
            i.defect_type,
            i.confidence,
            i.zone,
            i.human_validation,
            i.ai_decision,
            i.review_status,
            i.batch_position,
            i.reviewed_by,
            i.reviewed_at,
            i.review_notes,
            i.image_result,

            b.code AS batch_code,

            gm.code AS garment_model_code,
            gm.name AS garment_model_name,

            COALESCE(
                NULLIF(
                    reviewer.full_name,
                    ''
                ),
                reviewer.username,
                'No registrado'
            ) AS reviewer_name

        {from_sql}
        {where_sql}

        ORDER BY i.id DESC
        """,
        tuple(params),
    )

    batches = fetch_all(
        f"""
        SELECT
            b.id,
            b.code,
            b.planned_quantity,
            b.status,
            b.created_at,
            b.started_at,
            b.inspection_completed_at,
            b.closed_at,

            COUNT(i.id) AS processed_quantity,

            COALESCE(
                SUM(
                    CASE
                        WHEN {normal_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS auto_approved,

            COALESCE(
                SUM(
                    CASE
                        WHEN {alert_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN {confirmed_condition}
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects

        {from_sql}
        {where_sql}

        GROUP BY
            b.id,
            b.code,
            b.planned_quantity,
            b.status,
            b.created_at,
            b.started_at,
            b.inspection_completed_at,
            b.closed_at

        ORDER BY b.id DESC
        """,
        tuple(params),
    )

    return {
        # Nuevos indicadores empresariales
        "summary": summary,
        "lot_summary": lot_summary,
        "model_summary": model_summary,
        "rejection_reasons": rejection_reasons,
        "rejection_zones": rejection_zones,
        "reviewer_summary": reviewer_summary,
        "active_batch": active_batch,
        "daily_goal": daily_goal,
        "top_problem_lot": top_problem_lot,
        "top_problem_model": top_problem_model,
        "period_label": period_label,
        "date_from": date_from,
        "date_to": date_to,
        "batch_id": batch_id,
        "model_id": model_id,
        "today_text": today_text,
        "quick_7_from": quick_7_from,
        "quick_30_from": quick_30_from,

        # Compatibilidad con exportaciones existentes
        "total": inspected,
        "approved": passed,
        "defects": rejected,
        "review": pending,
        "by_garment": by_garment,
        "by_defect": by_defect,
        "inspections": inspections,
        "batches": batches,
    }


def goal_time_value(value, default):
    """Convierte TIME de MySQL (timedelta o time) a datetime.time."""
    from datetime import time as dt_time, timedelta

    if isinstance(value, timedelta):
        total = int(value.total_seconds())
        return dt_time(
            (total // 3600) % 24,
            (total % 3600) // 60,
            total % 60,
        )

    if isinstance(value, dt_time):
        return value

    return default


def goal_shift_end(day, goal):
    """Momento local en que termina el turno de la meta de `day`."""
    from datetime import time as dt_time, timedelta

    timezone_gt = ZoneInfo("America/Guatemala")

    start_time = goal_time_value(
        goal.get("shift_start"),
        dt_time(8, 0),
    )
    end_time = goal_time_value(
        goal.get("shift_end"),
        dt_time(17, 0),
    )

    start_dt = datetime.combine(
        day,
        start_time,
        tzinfo=timezone_gt,
    )
    end_dt = datetime.combine(
        day,
        end_time,
        tzinfo=timezone_gt,
    )

    # Un turno que termina a medianoche o crusa el día siguiente.
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)

    return end_dt


def resolve_goal_history_period(args, today=None):
    """Resuelve fechas locales; 'hoy' consulta la jornada en curso."""
    from datetime import timedelta

    if today is None:
        today = datetime.now(ZoneInfo("America/Guatemala")).date()

    yesterday = today - timedelta(days=1)
    monday = today - timedelta(days=today.weekday())
    period = args.get("goal_period", "ayer").strip()
    result = {
        "period": period,
        "date_from": args.get("goal_from", "").strip(),
        "date_to": args.get("goal_to", "").strip(),
        "max_date": yesterday.isoformat(),
        "custom_date_from": yesterday.isoformat(),
        "custom_date_to": yesterday.isoformat(),
        "is_today": period == "hoy",
        "start_date": None,
        "end_date": None,
        "error": None,
        "notice": None,
    }
    presets = {
        "hoy": (today, today),
        "ayer": (yesterday, yesterday),
        "esta_semana": (monday, yesterday),
        "semana_pasada": (
            monday - timedelta(days=7),
            monday - timedelta(days=1),
        ),
        "este_mes": (today.replace(day=1), yesterday),
    }

    if period in presets:
        start_date, end_date = presets[period]
    elif period == "personalizado":
        try:
            start_date = datetime.strptime(
                result["date_from"], "%Y-%m-%d"
            ).date()
            end_date = datetime.strptime(
                result["date_to"], "%Y-%m-%d"
            ).date()
        except ValueError:
            result["error"] = "Ingrese fechas Desde y Hasta válidas."
            return result

        if end_date < start_date:
            result["error"] = "Desde no puede ser posterior a Hasta."
            return result
        if end_date > yesterday:
            end_date = yesterday
            result["notice"] = (
                "Se excluyeron hoy y las fechas futuras del período "
                "personalizado: use el acceso HOY para consultar la "
                "jornada en curso."
            )
    else:
        result["error"] = "Seleccione un período válido."
        return result

    result.update({
        "start_date": start_date,
        "end_date": end_date,
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "custom_date_from": (
            start_date.isoformat()
            if start_date <= yesterday
            else yesterday.isoformat()
        ),
        "custom_date_to": (
            end_date.isoformat()
            if end_date <= yesterday
            else yesterday.isoformat()
        ),
    })
    return result


def summarize_goal_history(rows):
    """Pondera por prendas; la producción sin meta no compensa otros días."""
    with_goal = [row for row in rows if row["has_goal"]]
    finished = [
        row for row in with_goal if row["status"] != "En curso"
    ]
    in_progress = [
        row for row in with_goal if row["status"] == "En curso"
    ]
    target = sum(row["target_garments"] for row in with_goal)
    inspected_with_goal = sum(row["inspected"] for row in with_goal)
    met = sum(row["status"] == "Cumplida" for row in finished)
    return {
        "target_garments": target,
        "inspected": sum(row["inspected"] for row in rows),
        "inspected_with_goal": inspected_with_goal,
        "progress": (
            round(inspected_with_goal / target * 100, 1)
            if target > 0 else None
        ),
        "days_with_goal": len(with_goal),
        "days_met": met,
        "days_unmet": len(finished) - met,
        "days_in_progress": len(in_progress),
    }



def get_goal_history_data(start_date, end_date):
    """Lee metas finales y eventos UTC en una misma fotografía de MySQL."""
    import json
    from datetime import time as dt_time, timedelta

    empty = {"rows": [], "summary": summarize_goal_history([])}
    if start_date is None or end_date is None or start_date > end_date:
        return empty

    timezone_gt = ZoneInfo("America/Guatemala")
    timezone_utc = ZoneInfo("UTC")
    days = []
    calendar = []
    day = start_date
    while day <= end_date:
        # Mismo contrato UTC que dashboard/get_informe_data. Los límites
        # se convierten por fecha, sin asumir un offset fijo ni depender
        # de las tablas de zonas horarias instaladas en MySQL.
        start = datetime.combine(day, dt_time.min, tzinfo=timezone_gt)
        end = datetime.combine(
            day + timedelta(days=1), dt_time.min, tzinfo=timezone_gt
        )
        days.append(day)
        calendar.append({
            "day": day.isoformat(),
            "start": start.astimezone(timezone_utc).strftime("%Y-%m-%d %H:%M:%S"),
            "end": end.astimezone(timezone_utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
        day += timedelta(days=1)

    # JSON_TABLE está disponible en MySQL 8 (docker-compose.yaml).
    # El calendario completo es un parámetro, nunca SQL interpolado.
    calendar_sql = """
        WITH calendar AS (
            SELECT day, start_utc, end_utc
            FROM JSON_TABLE(
                %s, '$[*]' COLUMNS (
                    day DATE PATH '$.day',
                    start_utc DATETIME PATH '$.start',
                    end_utc DATETIME PATH '$.end'
                )
            ) AS dates
        )
    """
    params = (
        json.dumps(calendar),
        calendar[0]["start"],
        calendar[-1]["end"],
    )
    conn = db()
    cur = None
    try:
        conn.start_transaction(
            isolation_level="REPEATABLE READ",
            consistent_snapshot=True,
            readonly=True,
        )
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT
                goal_date,
                target_garments,
                target_batches,
                shift_start,
                shift_end
            FROM daily_production_goals
            WHERE goal_date >= %s AND goal_date <= %s
            """,
            (start_date, end_date),
        )

        goals = {row["goal_date"]: row for row in cur.fetchall()}
        if any(int(goal["target_garments"]) <= 0 for goal in goals.values()):
            raise ValueError(
                "Hay metas guardadas con cantidad de prendas no válida. "
                "No se calculó el cumplimiento del período."
            )

        cur.execute(
            calendar_sql + """
            SELECT c.day, COUNT(*) AS inspected
            FROM inspections i
            JOIN calendar c
              ON i.created_at >= c.start_utc
             AND i.created_at < c.end_utc
            WHERE i.created_at >= %s AND i.created_at < %s
            GROUP BY c.day
            """,
            params,
        )
        inspected = {row["day"]: int(row["inspected"]) for row in cur.fetchall()}
        cur.execute(
            calendar_sql + """
            SELECT c.day, COUNT(*) AS completed_batches
            FROM batches b
            JOIN calendar c
              ON b.inspection_completed_at >= c.start_utc
             AND b.inspection_completed_at < c.end_utc
            WHERE b.inspection_completed_at >= %s
              AND b.inspection_completed_at < %s
              AND EXISTS (
                  SELECT 1 FROM inspections i WHERE i.batch_id = b.id
              )
            GROUP BY c.day
            """,
            params,
        )
        completed = {
            row["day"]: int(row["completed_batches"])
            for row in cur.fetchall()
        }
    finally:
        if cur is not None:
            cur.close()
        conn.close()

    rows = []
    now_gt = datetime.now(ZoneInfo("America/Guatemala"))
    local_today = now_gt.date()
    for day in days:
        goal = goals.get(day)
        actual = inspected.get(day, 0)
        target = int(goal["target_garments"]) if goal else None
        status = "Sin meta configurada"
        if goal:
            # La jornada en curso nunca se evalúa como incumplida:
            # solo los días terminados usan la regla histórica.
            shift_ends = goal_shift_end(day, goal)
            if day == local_today and now_gt < shift_ends:
                status = "En curso"
            else:
                status = "Cumplida" if actual >= target else "No cumplida"

        rows.append({
            "date": day,
            "has_goal": goal is not None,
            "target_garments": target,
            "inspected": actual,
            "progress": round(actual / target * 100, 1) if target else None,
            "target_batches": int(goal["target_batches"]) if goal else None,
            "completed_batches": completed.get(day, 0),
            "status": status,
        })
    return {"rows": rows, "summary": summarize_goal_history(rows)}


@app.route("/informes")
@login_required
def informes():
    data = get_informe_data()

    goal_history = resolve_goal_history_period(request.args)
    goal_history.update({"rows": [], "summary": summarize_goal_history([])})
    if not goal_history["error"]:
        try:
            goal_history.update(get_goal_history_data(
                goal_history["start_date"], goal_history["end_date"]
            ))
        except ValueError as error:
            goal_history["error"] = str(error)

    models_options = fetch_all(
        """
        SELECT
            id,
            code,
            name
        FROM garment_models
        ORDER BY code
        """
    )

    batches_options = fetch_all(
        """
        SELECT
            id,
            code
        FROM batches
        ORDER BY id DESC
        LIMIT 200
        """
    )

    return render_template(
        "informes.html",
        goal_history=goal_history,
        models_options=models_options,
        batches_options=batches_options,
        **data,
    )


@app.route("/reportes")
@login_required
def reportes_legacy():
    return redirect(
        url_for("informes")
    )



def report_period_title(data):
    from datetime import datetime as _dt

    months = (
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    )

    start = _dt.strptime(
        data["date_from"],
        "%Y-%m-%d",
    ).date()

    end = _dt.strptime(
        data["date_to"],
        "%Y-%m-%d",
    ).date()

    start_month = months[
        start.month - 1
    ]

    end_month = months[
        end.month - 1
    ]

    if (
        start.year == end.year
        and start.month == end.month
    ):
        return (
            start_month.capitalize()
            + " de "
            + str(start.year)
        )

    if start.year == end.year:
        return (
            start_month.capitalize()
            + " a "
            + end_month
            + " de "
            + str(start.year)
        )

    return (
        start_month.capitalize()
        + " de "
        + str(start.year)
        + " a "
        + end_month
        + " de "
        + str(end.year)
    )


def report_quality_state(
    inspected,
    rejected,
    pending,
):
    inspected = int(inspected or 0)
    rejected = int(rejected or 0)
    pending = int(pending or 0)

    if inspected <= 0:
        return "Sin datos suficientes"

    if pending > 0:
        return "Pendiente de revisi\u00f3n"

    if rejected <= 0:
        return "Sin defectos confirmados"

    return "Con incidencias confirmadas"


def report_quality_reading(
    inspected,
    rejected,
    pending,
):
    inspected = int(inspected or 0)
    rejected = int(rejected or 0)
    pending = int(pending or 0)

    if inspected <= 0:
        return (
            "No existen inspecciones suficientes "
            "para evaluar este resultado."
        )

    if pending > 0:
        noun = (
            "alerta"
            if pending == 1
            else "alertas"
        )

        return (
            f"{pending} {noun} "
            "pendiente de revisi\u00f3n. "
            "El resultado todav\u00eda es provisional."
        )

    if rejected <= 0:
        return (
            "No se confirmaron defectos "
            "en las prendas revisadas."
        )

    noun = (
        "defecto"
        if rejected == 1
        else "defectos"
    )

    return (
        f"{rejected} {noun} "
        f"confirmado de {inspected} "
        "prendas inspeccionadas."
    )


def build_report_insights(data):
    summary = data["summary"]

    inspected = int(
        summary["inspected"]
        or 0
    )

    rejected = int(
        summary["rejected"]
        or 0
    )

    pending = int(
        summary["pending"]
        or 0
    )

    batches = int(
        summary[
            "batches_with_activity"
        ]
        or 0
    )

    state = report_quality_state(
        inspected,
        rejected,
        pending,
    )

    state_detail = report_quality_reading(
        inspected,
        rejected,
        pending,
    )

    model_rows = []

    for row in data["model_summary"]:
        model_inspected = int(
            row["inspected"]
            or 0
        )

        model_rejected = int(
            row["rejected"]
            or 0
        )

        model_pending = int(
            row["pending"]
            or 0
        )

        model_rows.append({
            "code": row["code"],
            "name": row["name"],
            "color": row["color"],
            "batches": int(
                row["batches"]
                or 0
            ),
            "inspected": model_inspected,
            "passed": int(
                row["passed"]
                or 0
            ),
            "rejected": model_rejected,
            "alerts": int(
                row["alerts"]
                or 0
            ),
            "pending": model_pending,
            "rejection_rate": float(
                row["rejection_rate"]
                or 0
            ),
            "state": report_quality_state(
                model_inspected,
                model_rejected,
                model_pending,
            ),
            "reading": report_quality_reading(
                model_inspected,
                model_rejected,
                model_pending,
            ),
        })

    reviewed_models = [
        row
        for row in model_rows
        if (
            row["inspected"] > 0
            and row["pending"] == 0
        )
    ]

    best_model = None

    if reviewed_models:
        best_model = min(
            reviewed_models,
            key=lambda row: (
                row["rejection_rate"],
                -row["inspected"],
                str(row["code"]),
            ),
        )

    conclusions = [
        (
            f"Se inspeccionaron {inspected} "
            f"prendas distribuidas en "
            f"{batches} lotes con actividad."
        )
    ]

    if pending > 0:
        conclusions.append(
            (
                f"Existen {pending} alertas "
                "pendientes de revisi\u00f3n; "
                "los resultados de calidad "
                "del per\u00edodo son provisionales."
            )
        )

    elif inspected > 0 and rejected == 0:
        conclusions.append(
            (
                "No se confirmaron defectos "
                "durante el per\u00edodo evaluado."
            )
        )

    elif rejected > 0:
        conclusions.append(
            (
                f"Se confirmaron {rejected} "
                "prendas con defecto durante "
                "el per\u00edodo evaluado."
            )
        )

    if best_model is not None:
        if best_model["rejected"] == 0:
            conclusions.append(
                (
                    f"{best_model['code']} - "
                    f"{best_model['name']} "
                    "present\u00f3 el mejor desempe\u00f1o "
                    "entre los modelos completamente "
                    "revisados, sin defectos confirmados."
                )
            )
        else:
            conclusions.append(
                (
                    f"{best_model['code']} - "
                    f"{best_model['name']} "
                    "present\u00f3 la menor tasa "
                    "de rechazo entre los modelos "
                    "completamente revisados."
                )
            )

    elif model_rows:
        conclusions.append(
            (
                "Todav\u00eda no existe suficiente "
                "revisi\u00f3n cerrada para identificar "
                "un modelo con mejor desempe\u00f1o "
                "de forma definitiva."
            )
        )

    if data["top_problem_model"]:
        model = data[
            "top_problem_model"
        ]

        if int(
            model["rejected"]
            or 0
        ) > 0:
            conclusions.append(
                (
                    f"{model['code']} - "
                    f"{model['name']} "
                    "es el modelo con mayor cantidad "
                    "de rechazos confirmados "
                    "en el per\u00edodo."
                )
            )

    recommendations = []

    if pending > 0:
        recommendations.append(
            (
                "Completar la revisi\u00f3n de "
                "las alertas pendientes antes de "
                "cerrar conclusiones definitivas."
            )
        )

    if data["top_problem_model"]:
        model = data[
            "top_problem_model"
        ]

        if int(
            model["rejected"]
            or 0
        ) > 0:
            recommendations.append(
                (
                    f"Revisar el proceso de producci\u00f3n "
                    f"del modelo {model['code']} "
                    "para identificar la causa "
                    "de los defectos confirmados."
                )
            )

    if data["top_problem_lot"]:
        lot = data[
            "top_problem_lot"
        ]

        if int(
            lot["rejected"]
            or 0
        ) > 0:
            recommendations.append(
                (
                    f"Analizar el lote {lot['code']} "
                    "antes de repetir las mismas "
                    "condiciones de producci\u00f3n."
                )
            )

    if (
        inspected > 0
        and pending == 0
        and rejected == 0
    ):
        recommendations.append(
            (
                "Mantener las condiciones actuales "
                "del proceso y continuar el "
                "seguimiento preventivo."
            )
        )

    if not recommendations:
        recommendations.append(
            (
                "Continuar registrando inspecciones "
                "para disponer de mayor informaci\u00f3n "
                "para la toma de decisiones."
            )
        )

    return {
        "period_title":
            report_period_title(data),

        "state":
            state,

        "state_detail":
            state_detail,

        "models":
            model_rows,

        "best_model":
            best_model,

        "conclusions":
            conclusions,

        "recommendations":
            recommendations,
    }


@app.route("/informes/excel")
@login_required
def informe_excel():
    from flask import send_file

    from datetime import datetime as dt_datetime
    from zoneinfo import ZoneInfo

    from pathlib import Path

    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.drawing.spreadsheet_drawing import (
        OneCellAnchor,
        AnchorMarker,
    )
    from openpyxl.drawing.xdr import XDRPositiveSize2D
    from openpyxl.utils.units import pixels_to_EMU
    from openpyxl.styles import (
        Alignment,
        Border,
        Font,
        PatternFill,
        Side,
    )
    from openpyxl.utils import get_column_letter

    data = get_informe_data()

    # Trazabilidad del usuario que genera el informe Excel.
    # Solo agrega metadatos al archivo; no modifica métricas ni cálculos.
    excel_report_user = fetch_one(
        "SELECT username, full_name, role FROM users WHERE id = %s",
        (session.get("user_id"),),
    ) or {}

    excel_generated_by_username = (
        excel_report_user.get("username")
        or session.get("username")
        or "usuario"
    )

    excel_generated_by_name = (
        excel_report_user.get("full_name")
        or excel_generated_by_username
    )

    excel_generated_by_role = normalize_role(
        excel_report_user.get("role")
        or session.get("role")
    )

    excel_generated_by_role_label = {
        "ADMIN": "Administrador",
        "MODEL_MANAGER": "Encargado de modelos",
        "QUALITY_MANAGER": "Gestor de calidad",
    }.get(
        excel_generated_by_role,
        excel_generated_by_role or "Sin rol",
    )

    if excel_generated_by_name != excel_generated_by_username:
        excel_generated_by_display = (
            f"{excel_generated_by_name} "
            f"(@{excel_generated_by_username})"
        )
    else:
        excel_generated_by_display = excel_generated_by_username


    insights = build_report_insights(
        data
    )

    timezone_gt = ZoneInfo(
        "America/Guatemala"
    )

    timezone_utc = ZoneInfo(
        "UTC"
    )

    generated_at = dt_datetime.now(
        timezone_gt
    ).replace(
        tzinfo=None
    )

    def utc_to_gt(value):
        if value is None:
            return None

        return (
            value
            .replace(
                tzinfo=timezone_utc
            )
            .astimezone(
                timezone_gt
            )
            .replace(
                tzinfo=None
            )
        )

    def is_confirmed_rejection(row):
        alert = (
            row["ai_decision"] == "ANOMALIA"
            or (
                row["ai_decision"] is None
                and row["status"] == "Defecto"
            )
        )

        confirmed = (
            row["review_status"]
            == "DEFECTO_CONFIRMADO"
            or (
                row["review_status"]
                in (
                    None,
                    "",
                    "PENDIENTE",
                )
                and row["human_validation"]
                == "Correcto"
            )
        )

        return alert and confirmed

    workbook = Workbook()

    # --------------------------------------------------------
    # Portada institucional
    # --------------------------------------------------------
    cover_sheet = workbook.active
    cover_sheet.title = "Portada"
    cover_sheet.sheet_view.showGridLines = False

    summary_sheet = workbook.create_sheet(
        "Resumen ejecutivo"
    )

    summary_sheet.sheet_view.showGridLines = False

    logo_path = (
        Path(app.root_path)
        / "static"
        / "img"
        / "brand"
        / "astrid_beverly_logo.jpg"
    )

    # Dimensiones de la portada.
    for column in range(2, 9):
        cover_sheet.column_dimensions[
            get_column_letter(column)
        ].width = 16

    cover_sheet.row_dimensions[2].height = 165

    if logo_path.exists():
        excel_logo = XLImage(
            str(logo_path)
        )

        original_width = float(
            excel_logo.width or 1
        )
        original_height = float(
            excel_logo.height or 1
        )

        target_width = 340

        excel_logo.width = target_width
        excel_logo.height = (
            target_width
            * original_height
            / original_width
        )

        # Posicion precisa: entre las columnas D y E.
        # Esto permite centrar el logo sin moverlo una columna completa.
        excel_logo.anchor = OneCellAnchor(
            _from=AnchorMarker(
                col=3,
                colOff=pixels_to_EMU(5),
                row=1,
                rowOff=0,
            ),
            ext=XDRPositiveSize2D(
                cx=pixels_to_EMU(
                    int(excel_logo.width)
                ),
                cy=pixels_to_EMU(
                    int(excel_logo.height)
                ),
            ),
        )

        cover_sheet.add_image(
            excel_logo
        )

    cover_sheet.merge_cells(
        "B10:H10"
    )
    cover_sheet["B10"] = (
        "ASTRID & BEVERLY FASHION"
    )
    cover_sheet["B10"].font = Font(
        name="Times New Roman",
        size=22,
        bold=True,
    )
    cover_sheet["B10"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    cover_sheet.merge_cells(
        "B12:H12"
    )
    cover_sheet["B12"] = (
        "Informe de Producci\u00f3n y Control de Calidad"
    )
    cover_sheet["B12"].font = Font(
        name="Times New Roman",
        size=16,
        bold=True,
    )
    cover_sheet["B12"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    cover_sheet.merge_cells(
        "B14:H14"
    )
    cover_sheet["B14"] = (
        "Per\u00edodo: "
        + str(data["date_from"])
        + " al "
        + str(data["date_to"])
    )
    cover_sheet["B14"].font = Font(
        name="Times New Roman",
        size=12,
    )
    cover_sheet["B14"].alignment = Alignment(
        horizontal="center",
    )

    cover_sheet.merge_cells(
        "B15:H15"
    )
    cover_sheet["B15"] = (
        "Generado: "
        + generated_at.strftime(
            "%d/%m/%Y %H:%M"
        )
    )
    cover_sheet["B15"].font = Font(
        name="Times New Roman",
        size=10,
        italic=True,
    )
    cover_sheet["B15"].alignment = Alignment(

        horizontal="center",
    )

    # Responsable que generó el archivo.
    cover_sheet.merge_cells("B16:H16")
    cover_sheet["B16"] = (
        f"Generado por: {excel_generated_by_display} | "
        f"Rol: {excel_generated_by_role_label}"
    )
    cover_sheet["B16"].font = Font(
        name="Calibri",
        size=10,
        italic=True,
        color="666666",
    )
    cover_sheet["B16"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )
    cover_sheet.row_dimensions[16].height = 18

    cover_sheet.merge_cells(
        "B18:H18"
    )
    cover_sheet["B18"] = (
        "Sistema de Control de Calidad mediante Visi\u00f3n Artificial"
    )
    cover_sheet["B18"].font = Font(
        name="Times New Roman",
        size=10,
        italic=True,
    )
    cover_sheet["B18"].alignment = Alignment(
        horizontal="center",
    )

    cover_sheet.sheet_properties.pageSetUpPr.fitToPage = True
    cover_sheet.page_setup.fitToWidth = 1
    cover_sheet.page_setup.fitToHeight = 1

    black_fill = PatternFill(
        "solid",
        fgColor="111111",
    )

    beige_fill = PatternFill(
        "solid",
        fgColor="EDE6DD",
    )

    soft_fill = PatternFill(
        "solid",
        fgColor="F7F4EF",
    )

    green_fill = PatternFill(
        "solid",
        fgColor="EAF6ED",
    )

    yellow_fill = PatternFill(
        "solid",
        fgColor="FFF3CD",
    )

    red_fill = PatternFill(
        "solid",
        fgColor="FDECEC",
    )

    gray_fill = PatternFill(
        "solid",
        fgColor="F1F1F1",
    )

    thin = Side(
        style="thin",
        color="D9D0C5",
    )

    border = Border(
        left=thin,
        right=thin,
        top=thin,
        bottom=thin,
    )

    def state_fill(state):
        if state == "Pendiente de revisi\u00f3n":
            return yellow_fill

        if state == "Sin defectos confirmados":
            return green_fill

        if state == "Con incidencias confirmadas":
            return red_fill

        return gray_fill

    def apply_range_style(
        sheet,
        min_row,
        max_row,
        min_col,
        max_col,
        fill=None,
        border_value=None,
    ):
        for row in sheet.iter_rows(
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
        ):
            for cell in row:
                if fill is not None:
                    cell.fill = fill

                if border_value is not None:
                    cell.border = border_value

    def section_title(
        sheet,
        row,
        title,
    ):
        sheet.merge_cells(
            start_row=row,
            start_column=1,
            end_row=row,
            end_column=8,
        )

        cell = sheet.cell(
            row=row,
            column=1,
            value=title,
        )

        cell.font = Font(
            bold=True,
            size=12,
        )

        cell.fill = beige_fill

        cell.alignment = Alignment(
            vertical="center",
        )

        apply_range_style(
            sheet,
            row,
            row,
            1,
            8,
            fill=beige_fill,
            border_value=border,
        )

        sheet.row_dimensions[
            row
        ].height = 24

    def style_headers(
        sheet,
        row_number,
        max_column,
    ):
        for column in range(
            1,
            max_column + 1,
        ):
            cell = sheet.cell(
                row=row_number,
                column=column,
            )

            cell.font = Font(
                bold=True,
                color="FFFFFF",
            )

            cell.fill = black_fill

            cell.border = border

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

    def finish_sheet(
        sheet,
        widths,
        freeze=None,
        filter_range=None,
    ):
        for index, width in enumerate(
            widths,
            start=1,
        ):
            sheet.column_dimensions[
                get_column_letter(index)
            ].width = width

        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

        if freeze:
            sheet.freeze_panes = freeze

        if filter_range:
            sheet.auto_filter.ref = (
                filter_range
            )

        sheet.sheet_view.showGridLines = False

    # ========================================================
    # RESUMEN EJECUTIVO
    # ========================================================

    summary_sheet.sheet_view.showGridLines = False

    # Paleta ejecutiva:
    # gris carbon, azul grisaceo y colores de estado suaves.
    executive_fill = PatternFill(
        "solid",
        fgColor="20242A",
    )

    section_fill = PatternFill(
        "solid",
        fgColor="DCE4EC",
    )

    card_fill = PatternFill(
        "solid",
        fgColor="F6F8FA",
    )

    good_fill = PatternFill(
        "solid",
        fgColor="E3EFE6",
    )

    warning_fill = PatternFill(
        "solid",
        fgColor="F5EDD8",
    )

    danger_fill = PatternFill(
        "solid",
        fgColor="F1DFDD",
    )

    neutral_fill = PatternFill(
        "solid",
        fgColor="E9EDF1",
    )

    def executive_state_fill(state):
        if state == "Sin defectos confirmados":
            return good_fill

        if state == "Pendiente de revisi\u00f3n":
            return warning_fill

        if state == "Con incidencias confirmadas":
            return danger_fill

        return neutral_fill

    # --------------------------------------------------------
    # CALCULOS VISUALES
    # --------------------------------------------------------

    summary = data["summary"]

    inspected = int(
        summary["inspected"]
        or 0
    )

    passed = int(
        summary["passed"]
        or 0
    )

    rejected = int(
        summary["rejected"]
        or 0
    )

    alerts = int(
        summary["alerts"]
        or 0
    )

    pending = int(
        summary["pending"]
        or 0
    )

    discarded = int(
        summary["discarded"]
        or 0
    )

    batches_count = int(
        summary[
            "batches_with_activity"
        ]
        or 0
    )

    approval_rate = (
        passed / inspected
        if inspected > 0
        else 0
    )

    rejection_rate = (
        rejected / inspected
        if inspected > 0
        else 0
    )

    reviewed_rate = (
        float(
            summary[
                "review_completion_rate"
            ]
            or 0
        )
        / 100
    )

    pending_alert_rate = (
        pending / alerts
        if alerts > 0
        else 0
    )

    # --------------------------------------------------------
    # ENCABEZADO
    # --------------------------------------------------------

    summary_sheet.merge_cells(
        "A1:H2"
    )

    summary_sheet["A1"] = (
        "Informe de producci\u00f3n y calidad"
    )

    summary_sheet["A1"].font = Font(
        bold=True,
        color="FFFFFF",
        size=20,
    )

    summary_sheet["A1"].fill = executive_fill

    summary_sheet["A1"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    apply_range_style(
        summary_sheet,
        1,
        2,
        1,
        8,
        fill=executive_fill,
    )

    summary_sheet.row_dimensions[1].height = 30
    summary_sheet.row_dimensions[2].height = 13


    # --------------------------------------------------------
    # PERIODO
    # --------------------------------------------------------

    summary_sheet.merge_cells(
        "A4:H4"
    )

    summary_sheet["A4"] = (
        "Resumen de "
        + insights["period_title"]
    )

    summary_sheet["A4"].font = Font(
        bold=True,
        size=16,
        color="20242A",
    )

    summary_sheet["A4"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    summary_sheet.row_dimensions[4].height = 26

    summary_sheet.merge_cells(
        "A5:H5"
    )

    summary_sheet["A5"] = (
        "Per\u00edodo exacto: "
        + data["period_label"]
        + "   |   Generado: "
        + generated_at.strftime(
            "%d/%m/%Y %H:%M"
        )
    )

    summary_sheet["A5"].font = Font(
        size=9,
        color="69727A",
    )

    summary_sheet["A5"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    summary_sheet.row_dimensions[5].height = 20


    # --------------------------------------------------------
    # KPI PRINCIPALES
    # --------------------------------------------------------

    kpis = [
        (
            "PRENDAS INSPECCIONADAS",
            inspected,
            (
                f"{batches_count} "
                + (
                    "lote con actividad"
                    if batches_count == 1
                    else "lotes con actividad"
                )
            ),
            None,
        ),
        (
            "APTAS",
            passed,
            (
                f"{passed} de "
                f"{inspected} prendas"
            ),
            None,
        ),
        (
            "DEFECTOS CONFIRMADOS",
            rejected,
            (
                f"{rejected} de "
                f"{inspected} prendas"
            ),
            None,
        ),
        (
            "PENDIENTES DE REVISI\u00d3N",
            pending,
            (
                f"de {alerts} "
                "alertas IA"
            ),
            None,
        ),
    ]

    for index, (
        label,
        value,
        detail,
        number_format,
    ) in enumerate(kpis):

        start_col = (
            1 + index * 2
        )

        end_col = (
            start_col + 1
        )

        # Etiqueta
        summary_sheet.merge_cells(
            start_row=7,
            start_column=start_col,
            end_row=7,
            end_column=end_col,
        )

        # Valor
        summary_sheet.merge_cells(
            start_row=8,
            start_column=start_col,
            end_row=9,
            end_column=end_col,
        )

        # Explicacion
        summary_sheet.merge_cells(
            start_row=10,
            start_column=start_col,
            end_row=11,
            end_column=end_col,
        )

        label_cell = summary_sheet.cell(
            row=7,
            column=start_col,
            value=label,
        )

        value_cell = summary_sheet.cell(
            row=8,
            column=start_col,
            value=value,
        )

        detail_cell = summary_sheet.cell(
            row=10,
            column=start_col,
            value=detail,
        )

        label_cell.font = Font(
            bold=True,
            size=9,
            color="52606D",
        )

        label_cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        value_cell.font = Font(
            bold=True,
            size=22,
            color="20242A",
        )

        value_cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

        detail_cell.font = Font(
            size=9,
            color="69727A",
        )

        detail_cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        if number_format:
            value_cell.number_format = (
                number_format
            )

        apply_range_style(
            summary_sheet,
            7,
            11,
            start_col,
            end_col,
            fill=card_fill,
            border_value=border,
        )

    summary_sheet.row_dimensions[7].height = 28
    summary_sheet.row_dimensions[8].height = 26
    summary_sheet.row_dimensions[9].height = 22
    summary_sheet.row_dimensions[10].height = 21
    summary_sheet.row_dimensions[11].height = 21

    # Indicador secundario: la tasa no ocupa una de las cuatro
    # categorías principales (Inspeccionadas = Aptas + Confirmados
    # + Pendientes).
    summary_sheet.merge_cells(
        "A12:H12"
    )

    summary_sheet["A12"] = (
        "Tasa de rechazo confirmada: "
        + f"{rejection_rate * 100:.1f} %"
    )

    summary_sheet["A12"].font = Font(
        italic=True,
        size=10,
        color="69727A",
    )

    summary_sheet["A12"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    summary_sheet.row_dimensions[12].height = 18


    # --------------------------------------------------------
    # ESTADO GENERAL
    # --------------------------------------------------------

    summary_sheet.merge_cells(
        "A13:H13"
    )

    summary_sheet["A13"] = (
        "Estado general del per\u00edodo"
    )

    summary_sheet["A13"].font = Font(
        bold=True,
        size=12,
        color="20242A",
    )

    apply_range_style(
        summary_sheet,
        13,
        13,
        1,
        8,
        fill=section_fill,
        border_value=border,
    )

    summary_sheet.row_dimensions[13].height = 25

    summary_sheet.merge_cells(
        "A14:H15"
    )

    summary_sheet["A14"] = (
        insights["state"]
    )

    summary_sheet["A14"].font = Font(
        bold=True,
        size=16,
        color="20242A",
    )

    summary_sheet["A14"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    apply_range_style(
        summary_sheet,
        14,
        15,
        1,
        8,
        fill=executive_state_fill(
            insights["state"]
        ),
        border_value=border,
    )

    summary_sheet.merge_cells(
        "A16:H17"
    )

    summary_sheet["A16"] = (
        insights["state_detail"]
    )

    summary_sheet["A16"].font = Font(
        size=10,
        color="414A52",
    )

    summary_sheet["A16"].alignment = Alignment(
        horizontal="center",
        vertical="center",
        wrap_text=True,
    )

    summary_sheet.row_dimensions[16].height = 24
    summary_sheet.row_dimensions[17].height = 20


    # --------------------------------------------------------
    # SEGUIMIENTO DE CALIDAD
    # --------------------------------------------------------

    summary_sheet.merge_cells(
        "A19:H19"
    )

    summary_sheet["A19"] = (
        "Seguimiento de calidad"
    )

    summary_sheet["A19"].font = Font(
        bold=True,
        size=12,
        color="20242A",
    )

    apply_range_style(
        summary_sheet,
        19,
        19,
        1,
        8,
        fill=section_fill,
        border_value=border,
    )

    quick_cards = [
        (
            "ALERTAS IA",
            alerts,
            (
                f"de {inspected} "
                "prendas inspeccionadas"
            ),
        ),
        (
            "CONFIRMADAS",
            rejected,
            "defectos confirmados "
            "por revisi\u00f3n humana",
        ),
        (
            "DESCARTADAS",
            discarded,
            "alertas descartadas "
            "por revisi\u00f3n humana",
        ),
        (
            "PENDIENTES",
            pending,
            "alertas a\u00fan "
            "sin revisar",
        ),
    ]

    for index, (
        label,
        main,
        detail,
    ) in enumerate(
        quick_cards
    ):
        start_col = (
            1 + index * 2
        )

        end_col = (
            start_col + 1
        )

        summary_sheet.merge_cells(
            start_row=20,
            start_column=start_col,
            end_row=20,
            end_column=end_col,
        )

        summary_sheet.merge_cells(
            start_row=21,
            start_column=start_col,
            end_row=22,
            end_column=end_col,
        )

        summary_sheet.merge_cells(
            start_row=23,
            start_column=start_col,
            end_row=26,
            end_column=end_col,
        )

        label_cell = summary_sheet.cell(
            row=20,
            column=start_col,
            value=label,
        )

        main_cell = summary_sheet.cell(
            row=21,
            column=start_col,
            value=main,
        )

        detail_cell = summary_sheet.cell(
            row=23,
            column=start_col,
            value=detail,
        )

        label_cell.font = Font(
            bold=True,
            size=9,
            color="52606D",
        )

        label_cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        main_cell.font = Font(
            bold=True,
            size=14,
            color="20242A",
        )

        main_cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        detail_cell.font = Font(
            size=9,
            color="52606D",
        )

        detail_cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        apply_range_style(
            summary_sheet,
            20,
            26,
            start_col,
            end_col,
            fill=card_fill,
            border_value=border,
        )

    summary_sheet.row_dimensions[20].height = 25
    summary_sheet.row_dimensions[21].height = 25
    summary_sheet.row_dimensions[22].height = 21
    summary_sheet.row_dimensions[23].height = 21
    summary_sheet.row_dimensions[24].height = 21
    summary_sheet.row_dimensions[25].height = 21
    summary_sheet.row_dimensions[26].height = 21

    # Quinto indicador del seguimiento, como l\u00ednea secundaria.
    summary_sheet.merge_cells(
        "A27:H27"
    )

    summary_sheet["A27"] = (
        "% de alertas revisadas: "
        + f"{reviewed_rate * 100:.1f} %"
        + (
            f"  ({rejected + discarded} de {alerts} alertas)"
            if alerts
            else ""
        )
    )

    summary_sheet["A27"].font = Font(
        italic=True,
        size=10,
        color="69727A",
    )

    summary_sheet["A27"].alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    summary_sheet.row_dimensions[27].height = 18


    # --------------------------------------------------------
    # COMPARACION DE MODELOS
    # --------------------------------------------------------

    summary_sheet.merge_cells(
        "A28:H28"
    )

    summary_sheet["A28"] = (
        "Comparaci\u00f3n de modelos"
    )

    summary_sheet["A28"].font = Font(
        bold=True,
        size=12,
        color="20242A",
    )

    apply_range_style(
        summary_sheet,
        28,
        28,
        1,
        8,
        fill=section_fill,
        border_value=border,
    )

    # Los modelos con incidencias aparecen primero.
    comparison_models = sorted(
        insights["models"],
        key=lambda model: (
            -model["rejected"],
            -model["rejection_rate"],
            -model["pending"],
            -model["inspected"],
            str(model["code"]),
        ),
    )[:5]

    row_cursor = 29

    if comparison_models:

        # Cabecera agrupada
        headers = [
            (
                1,
                2,
                "MODELO",
            ),
            (
                3,
                4,
                "APROBACI\u00d3N",
            ),
            (
                5,
                6,
                "RECHAZO",
            ),
            (
                7,
                8,
                "PENDIENTE",
            ),
        ]

        for start_col, end_col, label in headers:

            summary_sheet.merge_cells(
                start_row=row_cursor,
                start_column=start_col,
                end_row=row_cursor,
                end_column=end_col,
            )

            cell = summary_sheet.cell(
                row=row_cursor,
                column=start_col,
                value=label,
            )

            cell.font = Font(
                bold=True,
                color="FFFFFF",
                size=9,
            )

            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
            )

            apply_range_style(
                summary_sheet,
                row_cursor,
                row_cursor,
                start_col,
                end_col,
                fill=executive_fill,
                border_value=border,
            )

        row_cursor += 1

        for model in comparison_models:

            model_inspected = int(
                model["inspected"]
                or 0
            )

            model_passed = int(
                model["passed"]
                or 0
            )

            model_rejected = int(
                model["rejected"]
                or 0
            )

            model_alerts = int(
                model["alerts"]
                or 0
            )

            model_pending = int(
                model["pending"]
                or 0
            )

            model_approval_rate = (
                model_passed
                / model_inspected
                if model_inspected > 0
                else 0
            )

            model_rejection_rate = (
                model_rejected
                / model_inspected
                if model_inspected > 0
                else 0
            )

            model_pending_rate = (
                model_pending
                / model_alerts
                if model_alerts > 0
                else 0
            )

            metrics_row = row_cursor

            values = [
                (
                    1,
                    2,
                    (
                        str(model["code"])
                        + " - "
                        + str(model["name"])
                    ),
                    None,
                ),
                (
                    3,
                    4,
                    model_approval_rate,
                    "0.0%",
                ),
                (
                    5,
                    6,
                    model_rejection_rate,
                    "0.0%",
                ),
                (
                    7,
                    8,
                    model_pending_rate,
                    "0.0%",
                ),
            ]

            for (
                start_col,
                end_col,
                value,
                number_format,
            ) in values:

                summary_sheet.merge_cells(
                    start_row=metrics_row,
                    start_column=start_col,
                    end_row=metrics_row,
                    end_column=end_col,
                )

                cell = summary_sheet.cell(
                    row=metrics_row,
                    column=start_col,
                    value=value,
                )

                cell.font = Font(
                    bold=(
                        start_col == 1
                    ),
                    size=10,
                    color="20242A",
                )

                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

                if number_format:
                    cell.number_format = (
                        number_format
                    )

                apply_range_style(
                    summary_sheet,
                    metrics_row,
                    metrics_row,
                    start_col,
                    end_col,
                    fill=card_fill,
                    border_value=border,
                )

            summary_sheet.row_dimensions[
                metrics_row
            ].height = 30

            row_cursor += 1

            # Estado y explicacion con espacio real.
            summary_sheet.merge_cells(
                start_row=row_cursor,
                start_column=1,
                end_row=row_cursor,
                end_column=8,
            )

            reading = (
                model["state"]
                + " | "
                + model["reading"]
            )

            cell = summary_sheet.cell(
                row=row_cursor,
                column=1,
                value=reading,
            )

            cell.font = Font(
                size=9,
                color="414A52",
            )

            cell.alignment = Alignment(
                horizontal="left",
                vertical="center",
                wrap_text=True,
            )

            apply_range_style(
                summary_sheet,
                row_cursor,
                row_cursor,
                1,
                8,
                fill=executive_state_fill(
                    model["state"]
                ),
                border_value=border,
            )

            summary_sheet.row_dimensions[
                row_cursor
            ].height = 34

            row_cursor += 2

    else:
        summary_sheet.merge_cells(
            start_row=row_cursor,
            start_column=1,
            end_row=row_cursor + 1,
            end_column=8,
        )

        summary_sheet.cell(
            row=row_cursor,
            column=1,
            value=(
                "No existen modelos con actividad "
                "en el per\u00edodo seleccionado."
            ),
        )

        summary_sheet.cell(
            row=row_cursor,
            column=1,
        ).alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

        row_cursor += 3


    # --------------------------------------------------------
    # CONCLUSIONES Y ACCIONES
    # --------------------------------------------------------

    row_cursor += 1

    summary_sheet.merge_cells(
        start_row=row_cursor,
        start_column=1,
        end_row=row_cursor,
        end_column=4,
    )

    summary_sheet.merge_cells(
        start_row=row_cursor,
        start_column=5,
        end_row=row_cursor,
        end_column=8,
    )

    summary_sheet.cell(
        row=row_cursor,
        column=1,
        value="Conclusiones",
    )

    summary_sheet.cell(
        row=row_cursor,
        column=5,
        value="Acciones sugeridas",
    )

    for col in (
        1,
        5,
    ):
        cell = summary_sheet.cell(
            row=row_cursor,
            column=col,
        )

        cell.font = Font(
            bold=True,
            size=12,
            color="20242A",
        )

        cell.alignment = Alignment(
            vertical="center",
        )

    apply_range_style(
        summary_sheet,
        row_cursor,
        row_cursor,
        1,
        4,
        fill=section_fill,
        border_value=border,
    )

    apply_range_style(
        summary_sheet,
        row_cursor,
        row_cursor,
        5,
        8,
        fill=section_fill,
        border_value=border,
    )

    summary_sheet.row_dimensions[
        row_cursor
    ].height = 26

    row_cursor += 1

    max_items = max(
        len(
            insights["conclusions"]
        ),
        len(
            insights[
                "recommendations"
            ]
        ),
        2,
    )

    for offset in range(
        max_items
    ):

        row = row_cursor + offset

        summary_sheet.merge_cells(
            start_row=row,
            start_column=1,
            end_row=row,
            end_column=4,
        )

        summary_sheet.merge_cells(
            start_row=row,
            start_column=5,
            end_row=row,
            end_column=8,
        )

        if offset < len(
            insights["conclusions"]
        ):
            summary_sheet.cell(
                row=row,
                column=1,
                value=(
                    "- "
                    + insights[
                        "conclusions"
                    ][offset]
                ),
            )

        if offset < len(
            insights["recommendations"]
        ):
            summary_sheet.cell(
                row=row,
                column=5,
                value=(
                    "- "
                    + insights[
                        "recommendations"
                    ][offset]
                ),
            )

        for col in (
            1,
            5,
        ):
            cell = summary_sheet.cell(
                row=row,
                column=col,
            )

            cell.font = Font(
                size=9,
                color="414A52",
            )

            cell.alignment = Alignment(
                horizontal="left",
                vertical="center",
                wrap_text=True,
            )

        apply_range_style(
            summary_sheet,
            row,
            row,
            1,
            4,
            fill=card_fill,
            border_value=border,
        )

        apply_range_style(
            summary_sheet,
            row,
            row,
            5,
            8,
            fill=card_fill,
            border_value=border,
        )

        summary_sheet.row_dimensions[
            row
        ].height = 42


    # --------------------------------------------------------
    # PIE
    # --------------------------------------------------------

    footer_row = (
        row_cursor
        + max_items
        + 1
    )

    summary_sheet.merge_cells(
        start_row=footer_row,
        start_column=1,
        end_row=footer_row,
        end_column=8,
    )

    summary_sheet.cell(
        row=footer_row,
        column=1,
        value=(
            "Para consultar el detalle completo, "
            "utilice las hojas Lotes, Modelos y Rechazos."
        ),
    )

    summary_sheet.cell(
        row=footer_row,
        column=1,
    ).font = Font(
        italic=True,
        color="69727A",
        size=9,
    )

    summary_sheet.cell(
        row=footer_row,
        column=1,
    ).alignment = Alignment(
        horizontal="center",
        vertical="center",
    )

    summary_sheet.row_dimensions[
        footer_row
    ].height = 23


    # --------------------------------------------------------
    # DIMENSIONES
    # --------------------------------------------------------

    finish_sheet(
        summary_sheet,
        [
            19,
            19,
            19,
            19,
            19,
            19,
            19,
            19,
        ],
    )

    summary_sheet.sheet_view.zoomScale = 85

    summary_sheet.freeze_panes = None

    summary_sheet.page_setup.orientation = (
        "landscape"
    )

    summary_sheet.page_setup.fitToWidth = 1
    summary_sheet.page_setup.fitToHeight = 0

    summary_sheet.sheet_properties.pageSetUpPr.fitToPage = True

    summary_sheet.print_area = (
        f"A1:H{footer_row}"
    )


    # ========================================================
    # LOTES
    # ========================================================

    lots_sheet = workbook.create_sheet(
        "Lotes"
    )

    lot_headers = [
        "Lote",
        "Modelo",
        "Planificadas",
        "Procesadas",
        "Avance",
        "Inspeccionadas",
        "Aptas",
        "Rechazadas",
        "Tasa de rechazo",
        "Alertas IA",
        "Pendientes",
        "Estado de calidad",
        "Lectura r\u00e1pida",
    ]

    lots_sheet.append(
        lot_headers
    )

    style_headers(
        lots_sheet,
        1,
        len(lot_headers),
    )

    for row in data[
        "lot_summary"
    ]:
        model_name = (
            (
                str(
                    row[
                        "garment_model_code"
                    ]
                )
                + " - "
                + str(
                    row[
                        "garment_model_name"
                    ]
                )
            )
            if row[
                "garment_model_code"
            ]
            else "Sin modelo asignado"
        )

        quality_state = report_quality_state(
            row["inspected"],
            row["rejected"],
            row["pending"],
        )

        quality_reading = report_quality_reading(
            row["inspected"],
            row["rejected"],
            row["pending"],
        )

        lots_sheet.append([
            row["code"],
            model_name,
            int(
                row[
                    "planned_quantity"
                ]
                or 0
            ),
            int(
                row[
                    "total_processed"
                ]
                or 0
            ),
            float(
                row[
                    "progress_rate"
                ]
                or 0
            ) / 100,
            int(
                row["inspected"]
                or 0
            ),
            int(
                row["passed"]
                or 0
            ),
            int(
                row["rejected"]
                or 0
            ),
            float(
                row[
                    "rejection_rate"
                ]
                or 0
            ) / 100,
            int(
                row["alerts"]
                or 0
            ),
            int(
                row["pending"]
                or 0
            ),
            quality_state,
            quality_reading,
        ])

        current_row = (
            lots_sheet.max_row
        )

        lots_sheet.cell(
            row=current_row,
            column=12,
        ).fill = state_fill(
            quality_state
        )

    for cell in lots_sheet[
        "E"
    ][1:]:
        cell.number_format = "0.0%"

    for cell in lots_sheet[
        "I"
    ][1:]:
        cell.number_format = "0.0%"

    finish_sheet(
        lots_sheet,
        [
            20,
            34,
            15,
            15,
            13,
            18,
            12,
            14,
            18,
            14,
            14,
            28,
            55,
        ],
        freeze="A2",
        filter_range=lots_sheet.dimensions,
    )

    # ========================================================
    # MODELOS
    # ========================================================

    models_sheet = workbook.create_sheet(
        "Modelos"
    )

    model_headers = [
        "Modelo",
        "Nombre",
        "Color",
        "Lotes",
        "Inspeccionadas",
        "Aptas",
        "Rechazadas",
        "Tasa de rechazo",
        "Alertas IA",
        "Pendientes",
        "Estado",
        "Lectura r\u00e1pida",
    ]

    models_sheet.append(
        model_headers
    )

    style_headers(
        models_sheet,
        1,
        len(model_headers),
    )

    for model in insights[
        "models"
    ]:
        models_sheet.append([
            model["code"],
            model["name"],
            model["color"] or "",
            model["batches"],
            model["inspected"],
            model["passed"],
            model["rejected"],
            (
                model[
                    "rejection_rate"
                ]
                / 100
            ),
            model["alerts"],
            model["pending"],
            model["state"],
            model["reading"],
        ])

        current_row = (
            models_sheet.max_row
        )

        models_sheet.cell(
            row=current_row,
            column=11,
        ).fill = state_fill(
            model["state"]
        )

    for cell in models_sheet[
        "H"
    ][1:]:
        cell.number_format = "0.0%"

    finish_sheet(
        models_sheet,
        [
            18,
            30,
            18,
            12,
            18,
            12,
            14,
            18,
            14,
            14,
            28,
            55,
        ],
        freeze="A2",
        filter_range=models_sheet.dimensions,
    )

    # ========================================================
    # RECHAZOS
    # ========================================================

    rejected_sheet = workbook.create_sheet(
        "Rechazos"
    )

    rejected_headers = [
        "Fecha de inspecci\u00f3n",
        "Lote",
        "Modelo",
        "Prenda",
        "C\u00f3digo de inspecci\u00f3n",
        "Defecto confirmado",
        "Zona",
        "Puntuaci\u00f3n IA (%)",
        "Responsable",
        "Fecha de revisi\u00f3n",
        "Motivo",
    ]

    rejected_sheet.append(
        rejected_headers
    )

    style_headers(
        rejected_sheet,
        1,
        len(rejected_headers),
    )

    for row in data["inspections"]:
        if not is_confirmed_rejection(
            row
        ):
            continue

        model_name = (
            (
                str(
                    row[
                        "garment_model_code"
                    ]
                )
                + " - "
                + str(
                    row[
                        "garment_model_name"
                    ]
                )
            )
            if row[
                "garment_model_code"
            ]
            else "Sin modelo asignado"
        )

        confidence = (
            None
            if row["confidence"] is None
            else float(
                row["confidence"]
            )
        )

        rejected_sheet.append([
            utc_to_gt(
                row["created_at"]
            ),
            row["batch_code"] or "",
            model_name,
            row[
                "batch_position"
            ] or "",
            row["code"] or "",
            row[
                "defect_type"
            ] or "Sin especificar",
            row["zone"] or "",
            confidence,
            row[
                "reviewer_name"
            ] or "No registrado",
            utc_to_gt(
                row["reviewed_at"]
            ),
            row[
                "review_notes"
            ] or "No registrado",
        ])

    for cell in rejected_sheet[
        "A"
    ][1:]:
        cell.number_format = (
            "dd/mm/yyyy hh:mm:ss"
        )

    for cell in rejected_sheet[
        "J"
    ][1:]:
        cell.number_format = (
            "dd/mm/yyyy hh:mm"
        )

    finish_sheet(
        rejected_sheet,
        [
            22,
            18,
            34,
            12,
            24,
            30,
            30,
            20,
            24,
            22,
            52,
        ],
        freeze="A2",
        filter_range=rejected_sheet.dimensions,
    )

    # --------------------------------------------------------
    # Configuraci?n de impresi?n
    # --------------------------------------------------------
    for sheet in workbook.worksheets:
        sheet.sheet_properties.pageSetUpPr.fitToPage = True

        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = (
            sheet.PAPERSIZE_LETTER
        )
        sheet.page_setup.fitToWidth = 1

        # La portada debe caber en una sola hoja.
        if sheet.title == "Portada":
            sheet.page_setup.fitToHeight = 1
            sheet.print_area = "A1:I22"
            sheet.print_options.horizontalCentered = True
            sheet.print_options.verticalCentered = True
        else:
            # Las tablas pueden ocupar varias p?ginas verticales,
            # pero nunca varias p?ginas a lo ancho.
            sheet.page_setup.fitToHeight = 0

        sheet.page_margins.left = 0.25
        sheet.page_margins.right = 0.25
        sheet.page_margins.top = 0.40
        sheet.page_margins.bottom = 0.40
        sheet.page_margins.header = 0.15
        sheet.page_margins.footer = 0.15

        sheet.sheet_view.showGridLines = False

    buffer = io.BytesIO()

    workbook.save(
        buffer
    )

    buffer.seek(0)

    filename = (
        "informe_produccion_calidad_"
        + data["date_from"]
        + "_"
        + data["date_to"]
        + ".xlsx"
    )

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


@app.route("/informes/pdf")
@login_required
def informe_pdf():
    from flask import send_file

    from pathlib import Path
    from datetime import datetime as dt_datetime
    from zoneinfo import ZoneInfo
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import (
        TA_CENTER,
        TA_LEFT,
    )
    from reportlab.lib.pagesizes import (
        letter,
        landscape,
    )
    from reportlab.lib.styles import (
        ParagraphStyle,
        getSampleStyleSheet,
    )
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        LongTable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    data = get_informe_data()

    insights = build_report_insights(
        data
    )

    timezone_gt = ZoneInfo(
        "America/Guatemala"
    )

    generated_at = dt_datetime.now(
        timezone_gt
    )


    # Trazabilidad del usuario que genera el informe PDF.
    # Se consulta el usuario autenticado sin modificar la sesión ni
    # ninguna métrica, filtro o cálculo del informe.
    current_report_user = fetch_one(
        """
        SELECT username, full_name, role
        FROM users
        WHERE id = %s
        """,
        (session.get("user_id"),),
    ) or {}

    generated_by_username = (
        current_report_user.get("username")
        or session.get("username")
        or "usuario"
    )

    generated_by_name = (
        current_report_user.get("full_name")
        or generated_by_username
    )

    generated_by_role = normalize_role(
        current_report_user.get("role")
        or session.get("role")
    )

    generated_by_role_label = {
        "ADMIN": "Administrador",
        "MODEL_MANAGER": "Encargado de modelos",
        "QUALITY_MANAGER": "Gestor de calidad",
    }.get(
        generated_by_role,
        generated_by_role or "Sin rol",
    )

    buffer = io.BytesIO()

    page_size = landscape(letter)

    margin_x = 22 * mm

    content_width = (
        page_size[0]
        - (2 * margin_x)
    )

    logo_path = (
        Path(app.root_path)
        / "static"
        / "img"
        / "brand"
        / "astrid_beverly_logo.jpg"
    )

    document = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=margin_x,
        leftMargin=margin_x,
        topMargin=25 * mm,
        bottomMargin=16 * mm,
        title=(
            "Informe de producci\u00f3n y calidad"
        ),
        author=(
            "Astrid y Beverly Fashion"
        ),
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor(
            "#5F574F"
        ),
        spaceAfter=3,
    )

    exact_style = ParagraphStyle(
        "ReportExact",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(
            "#82776D"
        ),
        spaceAfter=10,
    )

    cell_style = ParagraphStyle(
        "ReportCell",
        parent=styles["Normal"],
        fontSize=7,
        leading=9,
        alignment=TA_LEFT,
    )

    center_style = ParagraphStyle(
        "ReportCenter",
        parent=cell_style,
        alignment=TA_CENTER,
    )

    header_style = ParagraphStyle(
        "ReportHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    section_style = ParagraphStyle(
        "ReportSection",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
    )

    metric_value_style = ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
    )

    metric_label_style = ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor(
            "#6B625A"
        ),
    )

    secondary_metric_style = ParagraphStyle(
        "SecondaryMetric",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(
            "#6B625A"
        ),
    )

    note_style = ParagraphStyle(
        "ReportNote",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(
            "#5F574F"
        ),
    )

    def paragraph(
        value,
        style=None,
    ):
        return Paragraph(
            escape(
                str(
                    ""
                    if value is None
                    else value
                )
            ),
            style or cell_style,
        )

    def header(value):
        return paragraph(
            value,
            header_style,
        )

    def centered(value):
        return paragraph(
            value,
            center_style,
        )

    def section_header(value):
        table = Table(
            [[
                paragraph(
                    value,
                    section_style,
                )
            ]],
            colWidths=[
                content_width
            ],
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#F2EEE8"
                    ),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#D8D0C6"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ])
        )

        return table

    def metric_table(metrics):
        width = (
            content_width
            / len(metrics)
        )

        table = Table(
            [
                [
                    paragraph(
                        value,
                        metric_value_style,
                    )
                    for label, value
                    in metrics
                ],
                [
                    paragraph(
                        label,
                        metric_label_style,
                    )
                    for label, value
                    in metrics
                ],
            ],
            colWidths=[
                width
            ] * len(metrics),
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#FAF8F5"
                    ),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#DDD5CB"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, 0),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 1),
                    (-1, 1),
                    7,
                ),
            ])
        )

        return table

    def state_color(state):
        if state == "Pendiente de revisi\u00f3n":
            return colors.HexColor(
                "#FFF3CD"
            )

        if state == "Sin defectos confirmados":
            return colors.HexColor(
                "#EAF6ED"
            )

        if state == "Con incidencias confirmadas":
            return colors.HexColor(
                "#FDECEC"
            )

        return colors.HexColor(
            "#F1F1F1"
        )

    data_table_style = TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor(
                "#111111"
            ),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white,
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.35,
            colors.HexColor(
                "#DDD5CB"
            ),
        ),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [
                colors.white,
                colors.HexColor(
                    "#FAF8F5"
                ),
            ],
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE",
        ),
        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            4,
        ),
        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            4,
        ),
        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
    ])

    elements = []

    elements.append(
        Paragraph(
            "Informe de producci\u00f3n y calidad",
            title_style,
        )
    )

    elements.append(
        Paragraph(
            (
                "Resumen de "
                + escape(
                    insights[
                        "period_title"
                    ]
                )
            ),
            subtitle_style,
        )
    )

    elements.append(
        Paragraph(
            (
                "Astrid y Beverly Fashion"
                " | Per\u00edodo exacto: "
                + escape(
                    data[
                        "period_label"
                    ]
                )
                + " | Generado: "
                + generated_at.strftime(
                    "%d/%m/%Y %H:%M"
                )
            ),
            exact_style,
        )
    )

    generated_by_text = (
        "Generado por: "
        + escape(str(generated_by_name))
    )

    if (
        generated_by_username
        and str(generated_by_username) != str(generated_by_name)
    ):
        generated_by_text += (
            " (@"
            + escape(str(generated_by_username))
            + ")"
        )

    generated_by_text += (
        " | Rol: "
        + escape(str(generated_by_role_label))
    )

    elements.append(
        Paragraph(
            generated_by_text,
            exact_style,
        )
    )

    summary = data["summary"]

    elements.append(
        section_header(
            "Resumen del per\u00edodo"
        )
    )

    elements.append(
        Spacer(1, 5)
    )

    elements.append(
        metric_table([
            (
                "Inspeccionadas",
                str(
                    summary["inspected"]
                    or 0
                ),
            ),
            (
                "Aptas",
                str(
                    summary["passed"]
                    or 0
                ),
            ),
            (
                "Defecto confirmado",
                str(
                    summary["rejected"]
                    or 0
                ),
            ),
            (
                "Pendientes de revisi\u00f3n",
                str(
                    summary["pending"]
                    or 0
                ),
            ),
        ])
    )

    elements.append(
        Spacer(1, 5)
    )

    elements.append(
        paragraph(
            (
                "Tasa de rechazo confirmada: "
                + f"{float(summary['rejection_rate'] or 0):.1f}%"
            ),
            secondary_metric_style,
        )
    )

    elements.append(
        Spacer(1, 8)
    )

    state_box = Table(
        [
            [
                paragraph(
                    insights["state"],
                    ParagraphStyle(
                        "StateTitle",
                        parent=styles["Normal"],
                        fontName="Helvetica-Bold",
                        fontSize=12,
                        alignment=TA_CENTER,
                    ),
                )
            ],
            [
                paragraph(
                    insights[
                        "state_detail"
                    ],
                    ParagraphStyle(
                        "StateDetail",
                        parent=styles["Normal"],
                        fontSize=8,
                        leading=11,
                        alignment=TA_CENTER,
                    ),
                )
            ],
        ],
        colWidths=[
            content_width
        ],
    )

    state_box.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                state_color(
                    insights["state"]
                ),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#D8D0C6"
                ),
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
        ])
    )

    elements.append(
        state_box
    )

    elements.append(
        Spacer(1, 8)
    )

    elements.append(
        section_header(
            "Seguimiento de calidad"
        )
    )

    elements.append(
        Spacer(1, 5)
    )

    elements.append(
        metric_table([
            (
                "Alertas IA",
                str(
                    summary["alerts"]
                    or 0
                ),
            ),
            (
                "Confirmadas",
                str(
                    summary["rejected"]
                    or 0
                ),
            ),
            (
                "Descartadas",
                str(
                    summary["discarded"]
                    or 0
                ),
            ),
            (
                "Pendientes",
                str(
                    summary["pending"]
                    or 0
                ),
            ),
            (
                "% revisadas",
                (
                    f"{float(summary['review_completion_rate'] or 0):.1f}%"
                ),
            ),
        ])
    )

    if (
        data["daily_goal"]
        and data["date_from"]
            == data["date_to"]
    ):
        goal = data["daily_goal"]

        elements.append(
            Spacer(1, 8)
        )

        elements.append(
            section_header(
                "Objetivo de producci\u00f3n del d\u00eda"
            )
        )

        elements.append(
            Spacer(1, 5)
        )

        elements.append(
            metric_table([
                (
                    "Objetivo de prendas",
                    str(
                        goal[
                            "target_garments"
                        ]
                        or 0
                    ),
                ),
                (
                    "Inspeccionadas",
                    str(
                        summary[
                            "inspected"
                        ]
                        or 0
                    ),
                ),
                (
                    "Avance",
                    (
                        f"{float(goal['progress_rate'] or 0):.1f}%"
                    ),
                ),
                (
                    "Objetivo de lotes",
                    str(
                        goal[
                            "target_batches"
                        ]
                        or 0
                    ),
                ),
            ])
        )

    elements.append(
        Spacer(1, 8)
    )

    # Desempeno por modelo inicia en pagina nueva.
    elements.append(PageBreak())

    elements.append(
        section_header(
            "Desempe\u00f1o por modelo"
        )
    )

    elements.append(
        Spacer(1, 5)
    )

    if insights["models"]:
        rows = [[
            header("Modelo"),
            header("Nombre"),
            header("Inspeccionadas"),
            header("Aptas"),
            header("Rechazadas"),
            header("Pendientes"),
            header("Estado"),
            header("Lectura r\u00e1pida"),
        ]]

        for model in insights[
            "models"
        ][:10]:
            rows.append([
                paragraph(
                    model["code"]
                ),
                paragraph(
                    model["name"]
                ),
                centered(
                    model["inspected"]
                ),
                centered(
                    model["passed"]
                ),
                centered(
                    model["rejected"]
                ),
                centered(
                    model["pending"]
                ),
                paragraph(
                    model["state"]
                ),
                paragraph(
                    model["reading"]
                ),
            ])

        table = LongTable(
            rows,
            repeatRows=1,
            colWidths=[
                27 * mm,
                42 * mm,
                26 * mm,
                18 * mm,
                23 * mm,
                22 * mm,
                38 * mm,
                73 * mm,
            ],
        )

        table.setStyle(
            data_table_style
        )

        elements.append(
            table
        )

    else:
        elements.append(
            Paragraph(
                (
                    "No existen modelos con "
                    "actividad en el per\u00edodo."
                ),
                note_style,
            )
        )

    elements.append(
        Spacer(1, 8)
    )

    elements.append(
        section_header(
            "Conclusiones del per\u00edodo"
        )
    )

    elements.append(
        Spacer(1, 4)
    )

    for item in insights[
        "conclusions"
    ]:
        elements.append(
            Paragraph(
                "- " + escape(item),
                note_style,
            )
        )

        elements.append(
            Spacer(1, 2)
        )

    elements.append(
        Spacer(1, 5)
    )

    elements.append(
        section_header(
            "Acciones sugeridas"
        )
    )

    elements.append(
        Spacer(1, 4)
    )

    for item in insights[
        "recommendations"
    ]:
        elements.append(
            Paragraph(
                "- " + escape(item),
                note_style,
            )
        )

        elements.append(
            Spacer(1, 2)
        )

    # Los datos tecnicos continuan en paginas de detalle.
    elements.append(
        PageBreak()
    )

    elements.append(
        section_header(
            "Detalle de calidad por lote"
        )
    )

    elements.append(
        Spacer(1, 5)
    )

    if data["lot_summary"]:
        rows = [[
            header("Lote"),
            header("Modelo"),
            header("Avance"),
            header("Inspeccionadas"),
            header("Aptas"),
            header("Rechazadas"),
            header("% rechazo"),
            header("Pendientes"),
        ]]

        for row in data[
            "lot_summary"
        ][:20]:
            model_label = (
                (
                    str(
                        row[
                            "garment_model_code"
                        ]
                    )
                    + " - "
                    + str(
                        row[
                            "garment_model_name"
                        ]
                    )
                )
                if row[
                    "garment_model_code"
                ]
                else "Sin modelo"
            )

            rows.append([
                paragraph(
                    row["code"]
                ),
                paragraph(
                    model_label
                ),
                centered(
                    (
                        f"{row['total_processed']}/"
                        f"{row['planned_quantity']} "
                        f"({float(row['progress_rate'] or 0):.1f}%)"
                    )
                ),
                centered(
                    row["inspected"]
                ),
                centered(
                    row["passed"]
                ),
                centered(
                    row["rejected"]
                ),
                centered(
                    (
                        f"{float(row['rejection_rate'] or 0):.1f}%"
                    )
                ),
                centered(
                    row["pending"]
                ),
            ])

        table = LongTable(
            rows,
            repeatRows=1,
            colWidths=[
                38 * mm,
                58 * mm,
                40 * mm,
                31 * mm,
                24 * mm,
                28 * mm,
                27 * mm,
                23 * mm,
            ],
        )

        table.setStyle(
            data_table_style
        )

        elements.append(
            table
        )

    if data["rejection_reasons"]:
        elements.append(
            Spacer(1, 10)
        )

        elements.append(
            section_header(
                "Principales causas de rechazo"
            )
        )

        elements.append(
            Spacer(1, 5)
        )

        rows = [[
            header("Defecto confirmado"),
            header("Prendas"),
        ]]

        for row in data[
            "rejection_reasons"
        ][:8]:
            rows.append([
                paragraph(
                    row["defect_type"]
                ),
                centered(
                    row["total"]
                ),
            ])

        table = Table(
            rows,
            colWidths=[
                content_width
                - (35 * mm),
                35 * mm,
            ],
            repeatRows=1,
        )

        table.setStyle(
            data_table_style
        )

        elements.append(
            table
        )

    if data["rejection_zones"]:
        elements.append(
            Spacer(1, 10)
        )

        elements.append(
            section_header(
                "Zonas con mayor rechazo"
            )
        )

        elements.append(
            Spacer(1, 5)
        )

        rows = [[
            header("Zona"),
            header("Prendas"),
        ]]

        for row in data[
            "rejection_zones"
        ][:8]:
            rows.append([
                paragraph(
                    row["zone"]
                ),
                centered(
                    row["total"]
                ),
            ])

        table = Table(
            rows,
            colWidths=[
                content_width
                - (35 * mm),
                35 * mm,
            ],
            repeatRows=1,
        )

        table.setStyle(
            data_table_style
        )

        elements.append(
            table
        )

    elements.append(
        Spacer(1, 10)
    )

    elements.append(
        Paragraph(
            (
                "<b>Interpretaci\u00f3n:</b> "
                "las tasas de rechazo corresponden "
                "a defectos confirmados mediante "
                "revisi\u00f3n humana. "
                "Las alertas pendientes no deben "
                "considerarse rechazos definitivos."
            ),
            note_style,
        )
    )

    def add_page_footer(
        canvas,
        doc,
    ):
        canvas.saveState()

        canvas.setStrokeColor(
            colors.HexColor(
                "#D8D0C6"
            )
        )

        canvas.line(
            margin_x,
            11 * mm,
            page_size[0]
            - margin_x,
            11 * mm,
        )

        canvas.setFont(
            "Helvetica",
            7.5,
        )

        canvas.setFillColor(
            colors.HexColor(
                "#6B625A"
            )
        )

        canvas.drawString(
            margin_x,
            7 * mm,
            "Astrid y Beverly Fashion",
        )

        canvas.drawRightString(
            page_size[0]
            - margin_x,
            7 * mm,
            (
                "P\u00e1gina "
                f"{doc.page}"
            ),
        )

        canvas.restoreState()

    def add_first_page(
        canvas,
        doc,
    ):
        # Primero conserva el pie de pagina normal.
        add_page_footer(
            canvas,
            doc,
        )

        canvas.saveState()

        if logo_path.exists():
            canvas.drawImage(
                str(logo_path),
                margin_x,
                page_size[1] - (22 * mm),
                width=24 * mm,
                height=18 * mm,
                preserveAspectRatio=True,
                mask="auto",
            )

        canvas.setFillColor(
            colors.HexColor("#111111")
        )

        canvas.setFont(
            "Helvetica-Bold",
            9,
        )

        canvas.drawString(
            margin_x + (29 * mm),
            page_size[1] - (10 * mm),
            "Astrid y Beverly Fashion",
        )

        canvas.setFont(
            "Helvetica",
            7.5,
        )

        canvas.setFillColor(
            colors.HexColor("#6B625A")
        )

        canvas.drawString(
            margin_x + (29 * mm),
            page_size[1] - (15 * mm),
            "Informe de producci\u00f3n y control de calidad",
        )

        canvas.setStrokeColor(
            colors.HexColor("#D8D0C6")
        )

        canvas.line(
            margin_x,
            page_size[1] - (23 * mm),
            page_size[0] - margin_x,
            page_size[1] - (23 * mm),
        )

        canvas.restoreState()

    document.build(
        elements,
        onFirstPage=add_first_page,
        onLaterPages=add_page_footer,
    )

    buffer.seek(0)

    filename = (
        "informe_produccion_calidad_"
        + data["date_from"]
        + "_"
        + data["date_to"]
        + ".pdf"
    )

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/validar/<int:inspection_id>/<value>", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def validate(inspection_id, value):
    value = value if value in ["Correcto", "Incorrecto", "Pendiente"] else "Pendiente"

    execute(
        "UPDATE inspections SET human_validation = %s WHERE id = %s",
        (value, inspection_id),
    )

    return redirect(request.referrer or url_for("records"))



# ============================================================
# MODELOS DE PRENDA
# ============================================================

def build_patchcore_dataset_name(model):
    """Construye un identificador estable para el dataset de una prenda."""
    import re
    import unicodedata

    def token(value, fallback):
        value = str(value or fallback).strip()

        normalized = unicodedata.normalize("NFKD", value)
        ascii_value = "".join(
            char
            for char in normalized
            if not unicodedata.combining(char)
        )

        ascii_value = ascii_value.upper()
        ascii_value = re.sub(
            r"[^A-Z0-9]+",
            "_",
            ascii_value,
        ).strip("_")

        return ascii_value or fallback

    code = token(
        model.get("code"),
        f"MODEL_{model.get('id')}",
    )
    size = token(model.get("size"), "S")
    color = token(model.get("color"), "SIN_COLOR")

    side_raw = token(
        model.get("inspection_side"),
        "FRONT",
    )

    side_aliases = {
        "FRENTE": "FRONT",
        "FRONTAL": "FRONT",
        "FRONT": "FRONT",
        "ESPALDA": "BACK",
        "POSTERIOR": "BACK",
        "BACK": "BACK",
    }

    side = side_aliases.get(side_raw, side_raw)

    return f"{code}_{size}_{color}_{side}"


def get_garment_ai_versions(model_id):
    rows = fetch_all(
        """
        SELECT
            ai.*,
            COALESCE(
                creator.full_name,
                creator.username,
                ''
            ) AS creator_name,
            COALESCE(
                trainer.full_name,
                trainer.username,
                ''
            ) AS trainer_name,
            COALESCE(
                validator.full_name,
                validator.username,
                ''
            ) AS validator_name,
            COALESCE(
                activator.full_name,
                activator.username,
                ''
            ) AS activator_name
        FROM garment_ai_models ai
        LEFT JOIN users creator
          ON creator.id = ai.created_by
        LEFT JOIN users trainer
          ON trainer.id = ai.trained_by
        LEFT JOIN users validator
          ON validator.id = ai.validated_by
        LEFT JOIN users activator
          ON activator.id = ai.activated_by
        WHERE ai.garment_model_id = %s
        ORDER BY ai.id DESC
        """,
        (model_id,),
    )

    for row in rows:
        row["technically_invalidated"] = is_technically_invalidated(
            row.get("notes")
        )

    return rows


def get_garment_model(model_id):
    return fetch_one(
        """
        SELECT
            gm.*,
            COALESCE(
                creator.full_name,
                creator.username,
                'Sin usuario'
            ) AS creator_name,
            COALESCE(
                approver.full_name,
                approver.username,
                ''
            ) AS approver_name,
            (
                SELECT ai.version
                FROM garment_ai_models ai
                WHERE ai.garment_model_id = gm.id
                  AND ai.active = 1
                  AND ai.status = 'ACTIVO'
                ORDER BY ai.id DESC
                LIMIT 1
            ) AS ai_version,
            (
                SELECT ai.model_type
                FROM garment_ai_models ai
                WHERE ai.garment_model_id = gm.id
                  AND ai.active = 1
                  AND ai.status = 'ACTIVO'
                ORDER BY ai.id DESC
                LIMIT 1
            ) AS ai_model_type,
            EXISTS(
                SELECT 1
                FROM garment_ai_models ai
                WHERE ai.garment_model_id = gm.id
                  AND ai.active = 1
                  AND ai.status = 'ACTIVO'
            ) AS ai_ready
        FROM garment_models gm
        LEFT JOIN users creator
          ON creator.id = gm.created_by
        LEFT JOIN users approver
          ON approver.id = gm.approved_by
        WHERE gm.id = %s
        """,
        (model_id,),
    )


def can_manage_garment_model(model):
    role = normalize_role(session.get("role"))

    if role == ROLE_ADMIN:
        return True

    return (
        role == ROLE_MODEL_MANAGER
        and model
        and model.get("created_by") == session.get("user_id")
    )


def prepare_garment_reference_images(files):
    prepared = []
    allowed = {".jpg", ".jpeg", ".png", ".webp"}

    for uploaded in files[:8]:
        if not uploaded or not uploaded.filename:
            continue

        suffix = Path(uploaded.filename).suffix.lower()

        if suffix not in allowed:
            continue

        data = uploaded.read()

        if not data or len(data) > 10 * 1024 * 1024:
            continue

        image_array = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if decoded is None:
            continue

        prepared.append((suffix, data))

    return prepared


def save_garment_reference_images(model_id, images):
    from uuid import uuid4

    directory = (
        Path(app.root_path)
        / "static"
        / "garment_models"
        / str(model_id)
    )

    directory.mkdir(parents=True, exist_ok=True)

    for position, (suffix, data) in enumerate(images, start=1):
        filename = f"{uuid4().hex}{suffix}"
        full_path = directory / filename
        full_path.write_bytes(data)

        relative_path = (
            f"garment_models/{model_id}/{filename}"
        )

        execute(
            """
            INSERT INTO garment_model_images (
                garment_model_id,
                image_path,
                image_type,
                sort_order,
                created_by
            )
            VALUES (%s, %s, 'REFERENCIA', %s, %s)
            """,
            (
                model_id,
                relative_path,
                position,
                session.get("user_id"),
            ),
        )


@app.route("/modelos-prenda")
@login_required
def garment_models_page():
    role = normalize_role(session.get("role"))

    where = ""
    params = ()

    if role == ROLE_QUALITY_MANAGER:
        where = """
            WHERE gm.status = 'APROBADO'
              AND gm.active = 1
        """

    models = fetch_all(
        f"""
        SELECT
            gm.*,
            COALESCE(
                u.full_name,
                u.username,
                'Sin usuario'
            ) AS creator_name,
            EXISTS(
                SELECT 1
                FROM garment_ai_models ai
                WHERE ai.garment_model_id = gm.id
                  AND ai.active = 1
                  AND ai.status = 'ACTIVO'
            ) AS ai_ready,
            (
                SELECT ai.version
                FROM garment_ai_models ai
                WHERE ai.garment_model_id = gm.id
                  AND ai.active = 1
                  AND ai.status = 'ACTIVO'
                ORDER BY ai.id DESC
                LIMIT 1
            ) AS ai_version
        FROM garment_models gm
        LEFT JOIN users u
          ON u.id = gm.created_by
        {where}
        ORDER BY
            CASE gm.status
                WHEN 'PENDIENTE' THEN 1
                WHEN 'BORRADOR' THEN 2
                WHEN 'RECHAZADO' THEN 3
                WHEN 'APROBADO' THEN 4
                ELSE 5
            END,
            gm.id DESC
        """,
        params,
    )

    return render_template(
        "garment_models.html",
        models=models,
        role=role,
    )


def generate_garment_model_code():
    """
    Código de modelo nuevo, generado siempre en backend.

    La usuaria nunca lo escribe: la unicidad no depende del frontend.
    """
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        code = next_garment_model_code(cur)
        conn.commit()
        return code
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


@app.route(
    "/modelos-prenda/nuevo",
    methods=["GET", "POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def garment_model_create():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        color = request.form.get("color", "").strip()
        description = request.form.get(
            "description",
            "",
        ).strip()

        if not name:
            flash(
                "Ingrese el nombre del modelo de blusa.",
                "error",
            )
            return render_template(
                "garment_model_form.html"
            )

        if not color:
            flash(
                "Ingrese el color de la blusa.",
                "error",
            )
            return render_template(
                "garment_model_form.html"
            )

        try:
            code = generate_garment_model_code()
        except AIDomainError as error:
            flash(str(error), "error")
            return render_template(
                "garment_model_form.html"
            )
        except Exception:
            app.logger.exception(
                "No se pudo generar el código del modelo."
            )
            flash(
                "No se pudo generar el código del modelo. "
                "Intente guardar de nuevo.",
                "error",
            )
            return render_template(
                "garment_model_form.html"
            )

        images = prepare_garment_reference_images(
            request.files.getlist("reference_images")
        )

        if not images:
            flash(
                "Debe cargar al menos una imagen "
                "v\u00e1lida de referencia.",
                "error",
            )
            return render_template(
                "garment_model_form.html"
            )

        model_id = execute(
            """
            INSERT INTO garment_models (
                code,
                name,
                garment_type,
                color,
                size,
                inspection_side,
                description,
                status,
                created_by,
                active
            )
            VALUES (
                %s,
                %s,
                'Blusa',
                %s,
                'S',
                'Frente',
                %s,
                'BORRADOR',
                %s,
                1
            )
            """,
            (
                code,
                name,
                color,
                description or None,
                session.get("user_id"),
            ),
        )

        save_garment_reference_images(
            model_id,
            images,
        )

        flash(
            "Modelo de prenda creado correctamente. "
            "Revise la informaci\u00f3n antes de enviarlo "
            "a aprobaci\u00f3n.",
            "success",
        )

        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    return render_template(
        "garment_model_form.html"
    )


@app.route("/modelos-prenda/<int:model_id>")
@login_required
def garment_model_detail(model_id):
    model = get_garment_model(model_id)

    if not model:
        flash(
            "El modelo de prenda solicitado no existe.",
            "error",
        )
        return redirect(
            url_for("garment_models_page")
        )

    role = normalize_role(session.get("role"))

    if (
        role == ROLE_QUALITY_MANAGER
        and (
            model.get("status") != "APROBADO"
            or int(model.get("active") or 0) != 1
        )
    ):
        flash(
            "No tiene permisos para consultar ese modelo.",
            "error",
        )
        return redirect(
            url_for("garment_models_page")
        )

    images = fetch_all(
        """
        SELECT *
        FROM garment_model_images
        WHERE garment_model_id = %s
        ORDER BY sort_order ASC, id ASC
        """,
        (model_id,),
    )

    ai_versions = get_garment_ai_versions(model_id)

    pending_ai_statuses = {
        "PREPARACION",
        "ENTRENANDO",
        "ENTRENADO",
        "VALIDACION",
        "VALIDADO",
    }

    ai_in_progress = any(
        version.get("status") in pending_ai_statuses
        and not version.get("technically_invalidated")
        for version in ai_versions
    )

    # Solo quedan versiones invalidadas "en proceso": para la ficha eso
    # no es proceso, es historia (FASE 3A.3). Habilita el mensaje de
    # versión histórica y no tapa la preparación de una versión nueva.
    ai_historical_only = bool(ai_versions) and not ai_in_progress and any(
        version.get("technically_invalidated")
        for version in ai_versions
    )

    can_manage = can_manage_garment_model(model)

    can_prepare_ai = (
        role in {ROLE_ADMIN, ROLE_MODEL_MANAGER}
        and can_manage
        and model.get("status") == "APROBADO"
        and int(model.get("active") or 0) == 1
    )

    ai_capture_state = None

    if model.get("status") == "APROBADO":
        try:
            ai_capture_state = _json_sanitize(
                _ai_capture_status_payload(
                    garment_model_id=model_id,
                    allowed=can_prepare_ai,
                )
            )
        except Exception:
            app.logger.exception(
                "No se pudo leer el estado de captura IA "
                f"(garment_model_id={model_id})."
            )
            ai_capture_state = None

    ai_capture_visible = bool(
        can_prepare_ai
        and ai_capture_state
        and (
            ai_capture_state.get("has_preparation_version")
            or ai_capture_state.get("session_id")
        )
    )

    ai_training_state = None

    if model.get("status") == "APROBADO":
        try:
            ai_training_state = _ai_training_status_payload(
                model_id,
                allowed=can_prepare_ai,
            )
        except Exception:
            app.logger.exception(
                "No se pudo leer el estado del entrenamiento IA "
                f"(garment_model_id={model_id})."
            )
            ai_training_state = None

    return render_template(
        "garment_model_detail.html",
        model=model,
        images=images,
        role=role,
        can_manage=can_manage,
        ai_versions=ai_versions,
        ai_in_progress=ai_in_progress,
        ai_historical_only=ai_historical_only,
        can_prepare_ai=can_prepare_ai,
        ai_capture=ai_capture_state,
        ai_capture_visible=ai_capture_visible,
        ai_training=ai_training_state,
    )


@app.route(
    "/modelos-prenda/<int:model_id>/enviar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def garment_model_submit(model_id):
    model = get_garment_model(model_id)

    if not model:
        flash(
            "El modelo solicitado no existe.",
            "error",
        )
        return redirect(
            url_for("garment_models_page")
        )

    if not can_manage_garment_model(model):
        flash(
            "No tiene permisos para modificar este modelo.",
            "error",
        )
        return redirect(
            url_for("garment_models_page")
        )

    if model.get("status") not in {
        "BORRADOR",
        "RECHAZADO",
    }:
        flash(
            "El modelo no se encuentra en un estado "
            "que permita enviarlo a aprobaci\u00f3n.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    image_count = fetch_one(
        """
        SELECT COUNT(*) AS c
        FROM garment_model_images
        WHERE garment_model_id = %s
        """,
        (model_id,),
    )["c"]

    if int(image_count) == 0:
        flash(
            "El modelo necesita al menos una imagen "
            "de referencia.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    execute(
        """
        UPDATE garment_models
        SET
            status = 'PENDIENTE',
            rejection_reason = NULL
        WHERE id = %s
        """,
        (model_id,),
    )

    flash(
        "Modelo enviado a aprobaci\u00f3n.",
        "success",
    )

    return redirect(
        url_for(
            "garment_model_detail",
            model_id=model_id,
        )
    )


@app.route(
    "/modelos-prenda/<int:model_id>/ia/preparar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def garment_ai_prepare(model_id):
    model = get_garment_model(model_id)

    if not model:
        flash(
            "El modelo de prenda solicitado no existe.",
            "error",
        )
        return redirect(url_for("garment_models_page"))

    if not can_manage_garment_model(model):
        flash(
            "No tiene permisos para preparar IA para este modelo.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        flash(
            "La IA solo puede prepararse para modelos aprobados y activos.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()

        cur.execute(
            """
            SELECT
                id,
                code,
                color,
                size,
                inspection_side,
                status,
                active
            FROM garment_models
            WHERE id = %s
            FOR UPDATE
            """,
            (model_id,),
        )

        locked_model = cur.fetchone()

        if not locked_model:
            conn.rollback()
            flash(
                "El modelo de prenda ya no existe.",
                "error",
            )
            return redirect(
                url_for("garment_models_page")
            )

        if (
            locked_model.get("status") != "APROBADO"
            or int(locked_model.get("active") or 0) != 1
        ):
            conn.rollback()
            flash(
                "El modelo debe continuar aprobado y activo.",
                "error",
            )
            return redirect(
                url_for(
                    "garment_model_detail",
                    model_id=model_id,
                )
            )

        cur.execute(
            """
            SELECT
                id,
                version,
                status
            FROM garment_ai_models
            WHERE garment_model_id = %s
              AND status IN (
                  'PREPARACION',
                  'ENTRENANDO',
                  'ENTRENADO',
                  'VALIDACION',
                  'VALIDADO'
              )
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_id,),
        )

        existing_pending = cur.fetchone()

        if existing_pending:
            conn.rollback()
            flash(
                (
                    "Ya existe una version de IA en proceso: "
                    f"{existing_pending['version']} "
                    f"({existing_pending['status']})."
                ),
                "error",
            )
            return redirect(
                url_for(
                    "garment_model_detail",
                    model_id=model_id,
                )
            )

        cur.execute(
            """
            SELECT version
            FROM garment_ai_models
            WHERE garment_model_id = %s
            FOR UPDATE
            """,
            (model_id,),
        )

        versions = cur.fetchall()

        version_numbers = []

        for item in versions:
            value = str(
                item.get("version") or ""
            ).strip().lower()

            if (
                value.startswith("v")
                and value[1:].isdigit()
            ):
                version_numbers.append(
                    int(value[1:])
                )

        next_number = (
            max(version_numbers, default=0) + 1
        )

        version = f"v{next_number}"

        dataset_name = build_patchcore_dataset_name(
            locked_model
        )

        dataset_path = (
            f"anomaly_dataset/{dataset_name}"
        )

        cur.execute(
            """
            INSERT INTO garment_ai_models (
                garment_model_id,
                version,
                model_type,
                dataset_name,
                dataset_path,
                checkpoint_path,
                status,
                normal_images_count,
                notes,
                created_by,
                active
            )
            VALUES (
                %s,
                %s,
                'PatchCore',
                %s,
                %s,
                NULL,
                'PREPARACION',
                0,
                NULL,
                %s,
                0
            )
            """,
            (
                model_id,
                version,
                dataset_name,
                dataset_path,
                session.get("user_id"),
            ),
        )

        ai_id = cur.lastrowid

        conn.commit()

    except Exception as error:
        conn.rollback()

        print(
            "[IA] Error preparando version "
            f"para garment_model_id={model_id}: "
            f"{error}"
        )

        flash(
            "No se pudo preparar la nueva version de IA.",
            "error",
        )

        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    finally:
        cur.close()
        conn.close()

    flash(
        (
            f"Version {version} preparada correctamente. "
            f"Dataset: {dataset_name}."
        ),
        "success",
    )

    return redirect(
        url_for(
            "garment_model_detail",
            model_id=model_id,
        )
    )


@app.route(
    "/modelos-prenda/<int:model_id>/aprobar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN)
def garment_model_approve(model_id):
    model = get_garment_model(model_id)

    if not model:
        flash(
            "El modelo solicitado no existe.",
            "error",
        )
        return redirect(
            url_for("garment_models_page")
        )

    if model.get("status") != "PENDIENTE":
        flash(
            "Solo pueden aprobarse modelos pendientes.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    execute(
        """
        UPDATE garment_models
        SET
            status = 'APROBADO',
            approved_by = %s,
            approved_at = NOW(),
            rejection_reason = NULL,
            active = 1
        WHERE id = %s
        """,
        (
            session.get("user_id"),
            model_id,
        ),
    )

    flash(
        "Modelo de prenda aprobado correctamente.",
        "success",
    )

    return redirect(
        url_for(
            "garment_model_detail",
            model_id=model_id,
        )
    )


@app.route(
    "/modelos-prenda/<int:model_id>/rechazar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN)
def garment_model_reject(model_id):
    model = get_garment_model(model_id)

    if not model:
        flash(
            "El modelo solicitado no existe.",
            "error",
        )
        return redirect(
            url_for("garment_models_page")
        )

    if model.get("status") != "PENDIENTE":
        flash(
            "Solo pueden rechazarse modelos pendientes.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    reason = request.form.get(
        "rejection_reason",
        "",
    ).strip()

    if len(reason) < 5:
        flash(
            "Indique el motivo del rechazo.",
            "error",
        )
        return redirect(
            url_for(
                "garment_model_detail",
                model_id=model_id,
            )
        )

    execute(
        """
        UPDATE garment_models
        SET
            status = 'RECHAZADO',
            approved_by = NULL,
            approved_at = NULL,
            rejection_reason = %s
        WHERE id = %s
        """,
        (
            reason,
            model_id,
        ),
    )

    flash(
        "Modelo rechazado. El encargado podr\u00e1 "
        "revisar el motivo indicado.",
        "success",
    )

    return redirect(
        url_for(
            "garment_model_detail",
            model_id=model_id,
        )
    )


# ============================================================
# ADMINISTRACION DE USUARIOS
# ============================================================

@app.route("/usuarios", methods=["GET", "POST"])
@login_required
@role_required(ROLE_ADMIN)
def users_admin():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        role = normalize_role(request.form.get("role", ""))

        if not full_name:
            flash("Ingrese el nombre completo del usuario.", "error")

        elif len(username) < 3:
            flash(
                "El nombre de usuario debe contener al menos 3 caracteres.",
                "error",
            )

        elif len(password) < 8:
            flash(
                "La contrase\u00f1a debe contener al menos 8 caracteres.",
                "error",
            )

        elif role not in VALID_ROLES:
            flash("Seleccione un rol v\u00e1lido.", "error")

        elif fetch_one(
            "SELECT id FROM users WHERE username = %s",
            (username,),
        ):
            flash(
                "Ya existe una cuenta con ese nombre de usuario.",
                "error",
            )

        else:
            execute(
                """
                INSERT INTO users (
                    username,
                    full_name,
                    password_hash,
                    role,
                    active
                )
                VALUES (%s, %s, %s, %s, 1)
                """,
                (
                    username,
                    full_name,
                    generate_password_hash(password),
                    role,
                ),
            )

            flash(
                "Usuario creado correctamente.",
                "success",
            )

            return redirect(url_for("users_admin"))

    users = fetch_all(
        """
        SELECT
            id,
            username,
            full_name,
            role,
            active,
            created_at,
            updated_at
        FROM users
        ORDER BY active DESC, full_name, username
        """
    )

    return render_template(
        "users.html",
        users=users,
        role_labels=ROLE_LABELS,
        valid_roles=[
            ROLE_ADMIN,
            ROLE_MODEL_MANAGER,
            ROLE_QUALITY_MANAGER,
        ],
    )


@app.route("/usuarios/<int:user_id>/rol", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN)
def user_change_role(user_id):
    user = fetch_one(
        """
        SELECT id, username, role, active
        FROM users
        WHERE id = %s
        """,
        (user_id,),
    )

    if not user:
        flash("El usuario solicitado no existe.", "error")
        return redirect(url_for("users_admin"))

    new_role = normalize_role(
        request.form.get("role", "")
    )

    if new_role not in VALID_ROLES:
        flash("El rol seleccionado no es v\u00e1lido.", "error")
        return redirect(url_for("users_admin"))

    current_role = normalize_role(user.get("role"))

    if (
        user_id == session.get("user_id")
        and new_role != ROLE_ADMIN
    ):
        flash(
            "No puede retirar su propio rol de administrador.",
            "error",
        )
        return redirect(url_for("users_admin"))

    if (
        current_role == ROLE_ADMIN
        and new_role != ROLE_ADMIN
        and int(user.get("active") or 0) == 1
    ):
        admins = fetch_one(
            """
            SELECT COUNT(*) AS c
            FROM users
            WHERE role = %s
              AND active = 1
            """,
            (ROLE_ADMIN,),
        )["c"]

        if int(admins) <= 1:
            flash(
                "Debe existir al menos un administrador activo.",
                "error",
            )
            return redirect(url_for("users_admin"))

    execute(
        """
        UPDATE users
        SET role = %s
        WHERE id = %s
        """,
        (new_role, user_id),
    )

    flash("Rol actualizado correctamente.", "success")
    return redirect(url_for("users_admin"))


@app.route("/usuarios/<int:user_id>/estado", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN)
def user_toggle_status(user_id):
    user = fetch_one(
        """
        SELECT id, username, role, active
        FROM users
        WHERE id = %s
        """,
        (user_id,),
    )

    if not user:
        flash("El usuario solicitado no existe.", "error")
        return redirect(url_for("users_admin"))

    current_active = int(user.get("active") or 0)

    if (
        user_id == session.get("user_id")
        and current_active == 1
    ):
        flash(
            "No puede desactivar su propia cuenta.",
            "error",
        )
        return redirect(url_for("users_admin"))

    if (
        normalize_role(user.get("role")) == ROLE_ADMIN
        and current_active == 1
    ):
        admins = fetch_one(
            """
            SELECT COUNT(*) AS c
            FROM users
            WHERE role = %s
              AND active = 1
            """,
            (ROLE_ADMIN,),
        )["c"]

        if int(admins) <= 1:
            flash(
                "Debe existir al menos un administrador activo.",
                "error",
            )
            return redirect(url_for("users_admin"))

    new_active = 0 if current_active == 1 else 1

    execute(
        """
        UPDATE users
        SET active = %s
        WHERE id = %s
        """,
        (new_active, user_id),
    )

    if new_active:
        flash("Usuario activado correctamente.", "success")
    else:
        flash("Usuario desactivado correctamente.", "success")

    return redirect(url_for("users_admin"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/health")
def health():
    try:
        row = fetch_one("SELECT COUNT(*) AS c FROM inspections")
        model_ready = MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= 1024
        return jsonify({
            "status": "ok",
            "database": DB_NAME,
            "inspections": row["c"],
            "camera_source": (
            "RTSP configurado"
            if CAMERA_SOURCE.lower().startswith("rtsp://")
            else CAMERA_SOURCE
            ),
            "model_ready": model_ready,
            "detection_mode": "yolo" if model_ready else "prototype",
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500





@app.before_request
def discard_persistent_flash_messages():
    """
    Los mensajes flash no pueden mostrarse en respuestas JSON, así que
    se descartan ahí para que no aparezcan en una página siguiente.

    En respuestas HTML sí se conservan: es lo que permite que una
    redirección muestre "Captura completada" u otros avisos.
    """
    from flask import session as flask_session

    if request.path.startswith("/api/"):
        flask_session.pop(
            "_flashes",
            None,
        )


# ============================================================
# FASE 2A — CAPTURA DE DATASET IA (estación)
# Solo consume latest_frame; no abre RTSP ni crea inspections.
# ============================================================

def _ai_capture_encode_jpeg(frame):
    cfg = AI_CAPTURE_RUNTIME["config"] or get_ai_capture_config()
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            int(cfg["jpeg_quality"]),
        ],
    )
    if not ok:
        raise RuntimeError("No se pudo codificar el JPEG del candidato IA.")
    return buffer.tobytes()


def _ai_capture_evaluate(frame, coverage, sequence):
    state = AI_CAPTURE_RUNTIME
    return evaluate_candidate(
        frame,
        coverage=float(coverage),
        frame_sequence=int(sequence),
        known_sha256=set(state["known_sha256"]),
        known_dhashes=list(state["known_dhashes"]),
        get_roi_bounds=get_roi_bounds,
        cv2_mod=cv2,
        config=state["config"],
    )


def _ai_capture_persist(decision, jpeg_bytes, candidate):
    session_id = AI_CAPTURE_RUNTIME["session_id"]
    garment_model_id = AI_CAPTURE_RUNTIME["garment_model_id"]
    cfg = AI_CAPTURE_RUNTIME["config"] or get_ai_capture_config()
    sequence = decision.get("frame_sequence")
    accepted = bool(decision.get("accepted"))
    keep_file = accepted or bool(cfg.get("keep_rejected"))

    rel_path = None
    sha = decision.get("sha256")

    if keep_file and jpeg_bytes:
        ensure_capture_session_dirs(
            garment_model_id,
            session_id,
            keep_rejected=bool(cfg.get("keep_rejected")),
        )
        filename = f"frame_{int(sequence)}.jpg"
        rel_path = capture_image_relative_path(
            garment_model_id,
            session_id,
            filename,
            rejected=not accepted,
        )
        abs_path = resolve_under_root(get_ai_artifacts_root(), rel_path)
        atomic_write_bytes(abs_path, jpeg_bytes)
        sha = sha256_bytes(jpeg_bytes)
    else:
        # Solo metadata: ruta lógica controlada bajo AI_ARTIFACTS_ROOT.
        rel_path = capture_image_relative_path(
            garment_model_id,
            session_id,
            f"frame_{int(sequence)}_meta.jpg",
            rejected=not accepted,
        )
        if sha is None:
            sha = "0" * 64

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()

        claim_frame_sequence(cur, session_id, int(sequence))

        if accepted and sha:
            dup = find_accepted_sha256(cur, garment_model_id, sha)
            if dup is not None:
                conn.rollback()
                return {
                    "skipped": True,
                    "reason": "DUPLICATE_SHA256",
                    "existing_id": dup["id"],
                }

        status = "ACEPTADA" if accepted else "RECHAZADA"
        record = register_training_image(
            cur,
            garment_model_id=garment_model_id,
            image_path=rel_path,
            sha256=sha,
            capture_session_id=session_id,
            frame_sequence=int(sequence),
            coverage=decision.get("coverage"),
            quality_score=decision.get("quality_score"),
            status=status,
            reject_reason=decision.get("reject_reason"),
        )
        conn.commit()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    if accepted and decision.get("dhash") is not None:
        AI_CAPTURE_RUNTIME["known_dhashes"].append(
            int(decision["dhash"])
        )
        if sha:
            AI_CAPTURE_RUNTIME["known_sha256"].add(str(sha).lower())

    return {
        "skipped": False,
        "record": record,
        "accepted": accepted,
        "rel_path": rel_path,
        "garment_token": candidate.garment_token,
    }


def ai_capture_worker():
    """Consumidor de latest_frame para preparación de dataset IA."""
    global AI_CAPTURE_WORKER_ENABLED

    cfg = get_ai_capture_config()
    AI_CAPTURE_RUNTIME["config"] = cfg

    enter_coverage = float(
        os.getenv("AUTO_GARMENT_ENTER_COVERAGE", "0.28")
    )
    exit_coverage = float(
        os.getenv("AUTO_GARMENT_EXIT_COVERAGE", "0.12")
    )
    confirm_frames = int(
        os.getenv("AUTO_GARMENT_CONFIRM_FRAMES", "3")
    )
    exit_frames_needed = int(
        os.getenv("AUTO_GARMENT_EXIT_FRAMES", "3")
    )
    max_track_frames = int(
        os.getenv("AUTO_GARMENT_MAX_TRACK_FRAMES", "16")
    )

    print(
        "[AI_CAPTURE] Worker iniciado. "
        f"session_id={AI_CAPTURE_RUNTIME['session_id']} "
        f"target={cfg['target_images']}"
    )

    while (
        AI_CAPTURE_WORKER_ENABLED
        and is_ai_capture_mode_active()
        and not AI_GUIDED_ACTIVE
    ):
        try:
            frame, frame_timestamp, sequence = (
                get_latest_camera_frame()
            )

            if frame is None:
                time.sleep(0.5)
                continue

            if sequence == AI_CAPTURE_RUNTIME["last_sequence"]:
                time.sleep(0.05)
                continue

            AI_CAPTURE_RUNTIME["last_sequence"] = sequence
            log_camera_latency(
                source="ai_capture",
                sequence=sequence,
                timestamp_monotonic=frame_timestamp,
            )

            try:
                coverage = compute_roi_coverage(frame)
            except Exception:
                coverage = 0.0

            presence = AI_CAPTURE_RUNTIME["presence"]
            waiting_exit = AI_CAPTURE_RUNTIME["waiting_for_exit"]

            # -------------------------------------------------
            # Esperando salida de la prenda ya trackeada.
            # -------------------------------------------------
            if waiting_exit:
                if coverage <= exit_coverage:
                    AI_CAPTURE_RUNTIME["exit_frames"] += 1
                else:
                    AI_CAPTURE_RUNTIME["exit_frames"] = 0

                if (
                    AI_CAPTURE_RUNTIME["exit_frames"]
                    >= exit_frames_needed
                ):
                    candidate = AI_CAPTURE_RUNTIME["presence"]
                    if (
                        candidate is not None
                        and not candidate.finalized
                        and candidate.best_frame is not None
                        and candidate.best_coverage
                        >= cfg["min_coverage"]
                    ):
                        try:
                            result = ai_capture.persist_best_candidate(
                                candidate,
                                evaluate_fn=_ai_capture_evaluate,
                                encode_jpeg_fn=_ai_capture_encode_jpeg,
                                persist_fn=_ai_capture_persist,
                                config=cfg,
                            )
                            AI_CAPTURE_RUNTIME["last_result"] = result
                            AI_CAPTURE_RUNTIME["last_persisted_token"] = (
                                candidate.garment_token
                            )
                            AI_CAPTURE_RUNTIME["last_error"] = None
                            print(
                                "[AI_CAPTURE] Candidato persistido: "
                                f"token={candidate.garment_token} "
                                f"accepted={result['accepted']} "
                                f"reason={result['reject_reason']}"
                            )
                        except Exception as error:
                            AI_CAPTURE_RUNTIME["last_error"] = str(error)
                            print(
                                "[AI_CAPTURE] Error al persistir: "
                                f"{error}"
                            )
                    elif candidate is not None:
                        candidate.finalized = True
                        print(
                            "[AI_CAPTURE] Presencia descartada "
                            "(sin cobertura suficiente)."
                        )

                    AI_CAPTURE_RUNTIME["presence"] = None
                    AI_CAPTURE_RUNTIME["waiting_for_exit"] = False
                    AI_CAPTURE_RUNTIME["exit_frames"] = 0

                time.sleep(0.20)
                continue

            # -------------------------------------------------
            # Seguimiento de presencia (1 candidato por token).
            # -------------------------------------------------
            if coverage >= enter_coverage:
                if AI_CAPTURE_RUNTIME["presence"] is None:
                    token = next_garment_token()
                    AI_CAPTURE_RUNTIME["presence"] = (
                        reset_presence_candidate(token)
                    )
                    AI_CAPTURE_RUNTIME["exit_frames"] = 0
                    print(
                        "[AI_CAPTURE] Presencia detectada. "
                        f"token={token}"
                    )

                presence = AI_CAPTURE_RUNTIME["presence"]
                presence.observe(
                    frame,
                    coverage,
                    sequence,
                    enter_coverage=enter_coverage,
                    confirm_frames=confirm_frames,
                )

                coverage_dropping = (
                    presence.best_coverage >= cfg["min_coverage"]
                    and coverage < presence.best_coverage - 0.015
                )
                max_tracked = (
                    presence.tracked_frames >= max_track_frames
                )

                if coverage_dropping or max_tracked:
                    # Misma idea que producción: la prenda ya estuvo
                    # completa; esperar a que salga y persistir el mejor.
                    AI_CAPTURE_RUNTIME["waiting_for_exit"] = True
                    AI_CAPTURE_RUNTIME["exit_frames"] = 0
            else:
                presence = AI_CAPTURE_RUNTIME["presence"]
                if presence is not None:
                    if presence.best_coverage >= cfg["min_coverage"]:
                        # Salió del umbral de entrada con buen candidato.
                        AI_CAPTURE_RUNTIME["waiting_for_exit"] = True
                        if coverage <= exit_coverage:
                            AI_CAPTURE_RUNTIME["exit_frames"] += 1
                        else:
                            AI_CAPTURE_RUNTIME["exit_frames"] = 0
                    else:
                        # Nunca alcanzó cobertura válida: descartar.
                        presence.finalized = True
                        AI_CAPTURE_RUNTIME["presence"] = None
                        AI_CAPTURE_RUNTIME["exit_frames"] = 0

            time.sleep(0.20)

        except Exception as error:
            AI_CAPTURE_RUNTIME["last_error"] = str(error)
            print(f"[AI_CAPTURE] Error controlado: {error}")
            time.sleep(0.5)

    print("[AI_CAPTURE] Worker finalizado.")


def _ensure_ai_capture_worker():
    global AI_CAPTURE_THREAD
    global AI_CAPTURE_WORKER_ENABLED

    with AI_CAPTURE_THREAD_LOCK:
        if (
            AI_GUIDED_ACTIVE
            and AI_GUIDED_ENABLED
        ):
            # La estación guiada es el único motor mientras dura.
            return False

        if (
            AI_CAPTURE_THREAD is not None
            and AI_CAPTURE_THREAD.is_alive()
            and AI_CAPTURE_WORKER_ENABLED
        ):
            return True

        if cv2 is None:
            return False

        AI_CAPTURE_WORKER_ENABLED = True
        AI_CAPTURE_THREAD = threading.Thread(
            target=ai_capture_worker,
            daemon=True,
            name="ai-capture-worker",
        )
        AI_CAPTURE_THREAD.start()
        return True


def _stop_ai_capture_worker(timeout=5.0):
    global AI_CAPTURE_WORKER_ENABLED

    AI_CAPTURE_WORKER_ENABLED = False
    thread = AI_CAPTURE_THREAD
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)


def _ensure_ai_guided_worker():
    """Arranca el motor guiado (siempre a costa del worker simple)."""
    global AI_GUIDED_THREAD
    global AI_GUIDED_ENABLED
    global AI_GUIDED_ACTIVE
    global AI_CAPTURE_THREAD
    global AI_CAPTURE_WORKER_ENABLED

    with AI_CAPTURE_THREAD_LOCK:
        if (
            AI_GUIDED_THREAD is not None
            and AI_GUIDED_THREAD.is_alive()
            and AI_GUIDED_ENABLED
        ):
            return True

        if cv2 is None:
            return False

        AI_CAPTURE_WORKER_ENABLED = False
        plain = AI_CAPTURE_THREAD
        if plain is not None and plain.is_alive():
            plain.join(timeout=5.0)

        AI_GUIDED_ACTIVE = True
        AI_GUIDED_ENABLED = True
        AI_GUIDED_THREAD = threading.Thread(
            target=ai_guided_worker,
            daemon=True,
            name="ai-guided-worker",
        )
        AI_GUIDED_THREAD.start()
        return True


def _stop_ai_guided_worker(timeout=5.0):
    global AI_GUIDED_ENABLED
    global AI_GUIDED_ACTIVE

    AI_GUIDED_ENABLED = False
    thread = AI_GUIDED_THREAD
    if thread is not None and thread.is_alive():
        if threading.current_thread() is not thread:
            thread.join(timeout=timeout)
    AI_GUIDED_ACTIVE = False


# Un guide publicado hace más de esto se considera obsoleto: la UI
# muestra "Preparando la cámara" en vez de un estado congelado.
AI_GUIDED_STALE_SECONDS = 5.0
AI_GUIDED_FRAME_MAX_AGE_SECONDS = 2.5
AI_GUIDED_DIAGNOSTIC_INTERVAL_SECONDS = 2.0
AI_GUIDED_LAST_DIAGNOSTIC = 0.0


def _ai_guided_segment(frame, roi):
    """
    ÚNICA segmentación de un frame de la captura guiada.

    Devuelve (mask, bbox, coverage, diagnostics) con bbox y cobertura
    calculados sobre la misma máscara: dos ejecuciones del segmentador
    podían no coincidir y dejar detected=False con coverage > 0.
    """
    diagnostics = {}

    try:
        mask = create_guided_garment_mask(frame, diagnostics)
    except Exception as error:  # noqa: BLE001 - el motivo queda en diagnostics
        mask = None
        diagnostics["error"] = str(error)

    bbox = mask_bbox(mask, cv2) if mask is not None else None

    coverage = 0.0
    if mask is not None:
        try:
            x1, y1, x2, y2 = roi
            roi_mask = mask[y1:y2, x1:x2]
            coverage = float(
                cv2.countNonZero(roi_mask) / max(1, roi_mask.size)
            )
        except Exception:
            coverage = 0.0

    return mask, bbox, float(coverage), diagnostics


def _ai_guided_reset_state():
    AI_GUIDED_RUNTIME["guide"] = None
    AI_GUIDED_RUNTIME["tracker"] = None
    AI_GUIDED_RUNTIME["tracker_model_id"] = None
    AI_GUIDED_RUNTIME["last_error"] = None
    AI_GUIDED_RUNTIME["pending_capture"] = None


def _ai_camera_frame_status():
    """Edad y frescura del último frame compartido por la cámara."""
    with latest_frame_lock:
        timestamp = latest_frame_timestamp_monotonic
        sequence = latest_frame_sequence
    age = None
    if timestamp is not None:
        age = max(0.0, time.perf_counter() - float(timestamp))
    return {
        "sequence": int(sequence or 0),
        "frame_age_ms": None if age is None else round(age * 1000.0, 1),
        "frame_fresh": bool(
            age is not None and age <= AI_GUIDED_FRAME_MAX_AGE_SECONDS
        ),
    }


def _ai_guided_publish(tracker):
    """Reemplaza el estado visible (referencia atómica, sin candados)."""
    guide = tracker.snapshot()
    guide["published_at"] = time.time()
    AI_GUIDED_RUNTIME["guide"] = guide
    AI_GUIDED_RUNTIME["tracker"] = tracker
    AI_GUIDED_RUNTIME["tracker_model_id"] = (
        AI_GUIDED_RUNTIME.get("garment_model_id")
    )


def _ai_guided_guide_is_fresh(guide) -> bool:
    if not isinstance(guide, dict):
        return False
    published = guide.get("published_at")
    if not published:
        return False
    try:
        return (time.time() - float(published)) <= AI_GUIDED_STALE_SECONDS
    except (TypeError, ValueError):
        return False


def _ai_guided_ensure_alive():
    """
    Reaviva el motor guiado si murió sin avisar.

    Sin esto, la cámara sigue en vivo pero el estado de la prenda
    queda congelado (p. ej. "SIN PRENDA") aunque la prenda esté
    dentro del área.
    """
    if not (AI_GUIDED_ACTIVE and AI_GUIDED_ENABLED):
        return False

    thread = AI_GUIDED_THREAD
    if thread is not None and thread.is_alive():
        return True

    try:
        return bool(_ensure_ai_guided_worker())
    except Exception:
        app.logger.exception("No se pudo reavivar el motor guiado.")
        return False


def _ai_guided_payload(garment_model_id=None):
    """Estado guiado visible para la ficha y la pantalla de captura."""
    empty = {
        "active": False,
        "state": None,
        "state_label": None,
        "message": "Preparando la cámara",
        "aligned": False,
        "ready": False,
        "reason": None,
        "bbox": None,
        **_ai_camera_frame_status(),
    }

    if not (AI_GUIDED_ACTIVE and AI_GUIDED_ENABLED):
        return empty

    guide = AI_GUIDED_RUNTIME.get("guide")
    if guide is None:
        return empty

    guide_model = AI_GUIDED_RUNTIME.get("garment_model_id")
    if (
        garment_model_id is not None
        and guide_model is not None
        and int(guide_model) != int(garment_model_id)
    ):
        return empty

    if not _ai_guided_guide_is_fresh(guide):
        # Estado viejo: mejor decir "Preparando" que mentir.
        return empty

    payload = dict(guide)
    payload["active"] = True
    payload["ready"] = bool(guide.get("ready"))
    payload["last_error"] = AI_GUIDED_RUNTIME.get("last_error")
    payload.update(_ai_camera_frame_status())
    return payload


def _ai_guided_prepare_runtime(session_id, garment_model_id):
    """Monta el runtime de sesión para que el worker guiado pueda persistir."""
    set_ai_capture_mode(
        True,
        session_id=session_id,
        garment_model_id=garment_model_id,
    )

    cfg = get_ai_guided_config()
    capture_cfg = get_ai_capture_config()
    runtime_cfg = {**capture_cfg, **cfg}
    AI_CAPTURE_RUNTIME.update(
        {
            "session_id": session_id,
            "garment_model_id": garment_model_id,
            "config": runtime_cfg,
            "known_sha256": set(),
            "known_dhashes": [],
            "presence": None,
            "waiting_for_exit": False,
            "exit_frames": 0,
            "last_sequence": -1,
            "last_persisted_token": None,
            "last_error": None,
            "last_result": None,
        }
    )

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        hashes = load_session_accepted_hashes(cur, session_id)
        AI_CAPTURE_RUNTIME["known_sha256"] = set(hashes["sha256"])
    finally:
        cur.close()
        conn.close()

    AI_GUIDED_RUNTIME.update(
        {
            "garment_model_id": garment_model_id,
            "session_id": session_id,
            "last_error": None,
        }
    )
    _ai_guided_reset_state()
    return cfg


def _ai_guided_capture(tracker, *, manual=False):
    """
    Persiste el mejor frame del paso.

    - manual=False: captura automática (solo con AI_GUIDED_AUTO_CAPTURE).
    - manual=True: botón "CAPTURAR IMAGEN"; si falla se vuelve al paso
      de posicionamiento para poder reintentar sin retirar la prenda.
    """
    cfg = tracker.config
    AI_CAPTURE_RUNTIME["config"] = cfg

    def abort(reason):
        if manual:
            tracker.cancel_capture()
        else:
            tracker.release()
        _ai_guided_publish(tracker)
        return {"ok": False, "error": reason}

    tracker.begin_capture()
    _ai_guided_publish(tracker)

    candidate = tracker.best_candidate(next_garment_token())
    if candidate is None:
        return abort(GUIDED_MESSAGES["NO_GARMENT"])

    try:
        result = ai_capture.persist_best_candidate(
            candidate,
            evaluate_fn=_ai_capture_evaluate,
            encode_jpeg_fn=_ai_capture_encode_jpeg,
            persist_fn=_ai_capture_persist,
            config=cfg,
        )
    except Exception as error:
        AI_GUIDED_RUNTIME["last_error"] = str(error)
        print(f"[AI_GUIDED] Error al persistir: {error}")
        return abort("No se pudo guardar la captura. Intente de nuevo.")

    record = result.get("record") or {}

    if record.get("skipped"):
        # Imagen idéntica a una ya guardada: no cuenta ni se repite.
        return abort("Imagen repetida. Cambie la prenda o muévala un poco.")

    AI_GUIDED_RUNTIME["last_error"] = None
    tracker.confirm_capture(result)
    _ai_guided_publish(tracker)
    print(
        "[AI_GUIDED] Captura guardada: "
        f"accepted={result.get('accepted')} "
        f"reason={result.get('reject_reason')}"
    )
    return {
        "ok": True,
        "result": result,
        "record": record,
        "accepted": bool(result.get("accepted")),
    }


# Candado del botón manual: evita dos capturas simultáneas del mismo
# fotograma cuando la operadora presiona dos veces muy rápido.
AI_GUIDED_MANUAL_LOCK = threading.Lock()

# Anti-mismo-fotograma: separa tomas manuales para que dos capturas
# consecutivas no sean el mismo frame (o casi el mismo).
AI_GUIDED_MANUAL_COOLDOWN_DEFAULT = 0.6
AI_GUIDED_MANUAL_LAST = {"at": 0.0, "sequence": -1}


def _ai_guided_manual_cooldown():
    """
    Segundos mínimos entre tomas manuales.

    Se lee en cada llamada (no al importar) para poder ajustarlo por
    entorno y poder parchearlo desde los tests.
    """
    raw = os.environ.get("AI_GUIDED_MANUAL_COOLDOWN_SECONDS")

    try:
        value = (
            AI_GUIDED_MANUAL_COOLDOWN_DEFAULT
            if raw in (None, "")
            else float(raw)
        )
    except (TypeError, ValueError):
        value = AI_GUIDED_MANUAL_COOLDOWN_DEFAULT

    return max(0.0, min(value, 10.0))


def reset_for_next_manual_capture():
    """
    Libera la vista previa para poder capturar otra imagen.

    Único punto que limpia el estado de la toma pendiente: se usa
    cuando se acepta, se repite, se descarta o se cancela.
    """
    AI_GUIDED_RUNTIME["pending_capture"] = None


def _ai_guided_manual_capture(garment_model_id):
    """
    Congela una imagen fresca para revisión; no la persiste aún.

    Devuelve (payload, status_code). Nunca crea inspecciones ni toca
    la estación productiva: solo escribe en la sesión de captura IA.
    """
    with AI_GUIDED_MANUAL_LOCK:
        if AI_GUIDED_RUNTIME.get("pending_capture") is not None:
            return {
                "ok": False,
                "error": "Revise la imagen pendiente antes de capturar otra.",
            }, 409
        frame, timestamp, sequence = get_latest_camera_frame()
        age = (
            None
            if timestamp is None
            else max(0.0, time.perf_counter() - float(timestamp))
        )
        if frame is None or age is None or age > AI_GUIDED_FRAME_MAX_AGE_SECONDS:
            return {
                "ok": False,
                "error": "La cámara no tiene una imagen reciente. Espere y vuelva a intentar.",
            }, 409

        cooldown = _ai_guided_manual_cooldown()
        last_at = float(AI_GUIDED_MANUAL_LAST.get("at") or 0.0)
        elapsed = time.perf_counter() - last_at

        if last_at > 0.0:
            if int(sequence) == int(AI_GUIDED_MANUAL_LAST.get("sequence") or -1):
                return {
                    "ok": False,
                    "error": (
                        "La imagen es idéntica a la anterior: la cámara "
                        "debe entregar un fotograma nuevo. Repita en un "
                        "instante."
                    ),
                }, 409
            if elapsed < cooldown:
                return {
                    "ok": False,
                    "error": (
                        f"Espere {cooldown:.1f} s entre capturas para no "
                        "repetir la misma imagen."
                    ),
                }, 429


        try:
            x1, y1, x2, y2 = get_roi_bounds(frame)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                raise ValueError("ROI vacío")
        except Exception:
            return {"ok": False, "error": "El área de captura no es válida."}, 409

        _mask, bbox, coverage, segmentation = _ai_guided_segment(
            frame,
            (x1, y1, x2, y2),
        )

        try:
            sharpness = compute_sharpness(
                frame,
                get_roi_bounds=get_roi_bounds,
                cv2_mod=cv2,
            )
        except Exception:
            sharpness = 0.0

        decision = _ai_capture_evaluate(frame, coverage, sequence)
        warnings = []
        if segmentation.get("method") == "ninguno":
            warnings.append(
                "No se detectó la prenda dentro del área: el sistema "
                "no pudo confirmar una posición ideal. Colóquela entera "
                "dentro del rectángulo antes de aceptar."
            )
        elif bbox is None:
            warnings.append("El sistema no pudo confirmar una posición ideal. Revise la imagen antes de aceptarla.")
        if float(sharpness or 0.0) < float((AI_CAPTURE_RUNTIME.get("config") or get_ai_capture_config())["min_sharpness"]):
            warnings.append("Imagen posiblemente borrosa.")
        if float(coverage or 0.0) < float((AI_CAPTURE_RUNTIME.get("config") or get_ai_capture_config())["min_coverage"]):
            warnings.append("La prenda podría quedar parcial o poco visible dentro del área.")
        if decision.get("reason") in (
            "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
            "HIGH_COVERAGE",
        ):
            warnings.append("Parte de la prenda podría quedar fuera del área o descentrada.")
        if decision.get("reject_reason") in (
            "DUPLICATE_SHA256", "DUPLICATE_PERCEPTUAL"
        ):
            warnings.append("Esta captura es muy similar a una anterior.")
        if age > 1.0:
            warnings.append("La imagen tiene cierto retraso respecto a la cámara.")
        if not warnings:
            warnings.append("La posición parece adecuada. Confirme visualmente antes de aceptar.")

        try:
            source_jpeg = _ai_capture_encode_jpeg(frame)
            stored_frame = cv2.imdecode(
                np.frombuffer(source_jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            stored_crop = stored_frame[y1:y2, x1:x2].copy()
            preview_ok, preview_png = cv2.imencode(".png", stored_crop)
        except Exception:
            preview_ok = False
            source_jpeg = None
        if not preview_ok:
            return {"ok": False, "error": "No se pudo preparar la vista previa."}, 500

        token = uuid.uuid4().hex
        candidate = ai_capture.PresenceCandidate(garment_token=next_garment_token())
        candidate.best_frame = frame.copy()
        candidate.best_coverage = float(coverage or 0.0)
        candidate.best_sequence = int(sequence)
        pending = {
            "token": token,
            "candidate": candidate,
            "decision": dict(decision),
            "jpeg_bytes": source_jpeg,
            "roi": (x1, y1, x2, y2),
            "bbox": bbox,
            "sharpness": float(sharpness or 0.0),
            "frame_age_ms": round(age * 1000.0, 1),
            "warnings": warnings,
            "segmentation": dict(segmentation),
        }
        AI_GUIDED_RUNTIME["pending_capture"] = pending
        AI_GUIDED_MANUAL_LAST["at"] = time.perf_counter()
        AI_GUIDED_MANUAL_LAST["sequence"] = int(sequence)
        log_camera_latency(
            source="ai_guided_manual_preview",
            sequence=sequence,
            timestamp_monotonic=timestamp,
        )
        preview = base64.b64encode(preview_png.tobytes()).decode("ascii")
        return {
            "ok": True,
            "pending_token": token,
            "preview_data_url": "data:image/png;base64," + preview,
            "warnings": warnings,
            "metrics": {
                "sequence": int(sequence),
                "frame_age_ms": round(age * 1000.0, 1),
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                "roi": [x1, y1, x2, y2],
                "coverage": round(float(coverage or 0.0), 4),
                "sharpness": round(float(sharpness or 0.0), 2),
                "detected": bbox is not None,
                "bbox": list(bbox) if bbox else None,
                "mask_method": segmentation.get("method"),
                "mask_nonzero": segmentation.get("mask_nonzero"),
                "mask_pct": segmentation.get("mask_pct"),
                "roi_stats": segmentation.get("roi_stats"),
                "segmentation_reason": segmentation.get("reason"),
            },
            "guide": _ai_guided_payload(garment_model_id),
        }, 200


def ai_guided_worker():
    """
    Motor de la estación guiada: posicionamiento + captura automática.

    Lee el mismo latest_frame que producción; nunca escribe en
    inspecciones ni en la estación de calidad.
    """
    global AI_GUIDED_ENABLED

    cfg = get_ai_guided_config()
    tracker = GuidedTracker(cfg)
    AI_CAPTURE_RUNTIME["config"] = {
        **get_ai_capture_config(),
        **cfg,
    }
    _ai_guided_publish(tracker)

    last_sequence = -1

    print(
        "[AI_GUIDED] Worker iniciado: "
        f"model={AI_GUIDED_RUNTIME.get('garment_model_id')} "
        f"session={AI_GUIDED_RUNTIME.get('session_id')} "
        f"target={cfg['target_images']}"
    )

    while AI_GUIDED_ENABLED and is_ai_capture_mode_active():
        try:
            frame, frame_timestamp, sequence = get_latest_camera_frame()

            if frame is None:
                time.sleep(0.5)
                continue

            if sequence == last_sequence:
                time.sleep(0.05)
                continue

            last_sequence = sequence
            log_camera_latency(
                source="ai_guided",
                sequence=sequence,
                timestamp_monotonic=frame_timestamp,
            )

            try:
                roi = get_roi_bounds(frame)
            except Exception:
                time.sleep(0.20)
                continue

            _mask, bbox, coverage, segmentation = _ai_guided_segment(
                frame,
                roi,
            )

            try:
                sharpness = compute_sharpness(
                    frame,
                    get_roi_bounds=get_roi_bounds,
                    cv2_mod=cv2,
                )
            except Exception:
                sharpness = 0.0

            guide = tracker.step(
                roi=roi,
                bbox=bbox,
                coverage=coverage,
                sharpness=sharpness,
                sequence=sequence,
                frame=frame,
            )
            _ai_guided_publish(tracker)

            global AI_GUIDED_LAST_DIAGNOSTIC
            now = time.perf_counter()
            if now - AI_GUIDED_LAST_DIAGNOSTIC >= AI_GUIDED_DIAGNOSTIC_INTERVAL_SECONDS:
                AI_GUIDED_LAST_DIAGNOSTIC = now
                height, width = frame.shape[:2]
                age_ms = (
                    None if frame_timestamp is None
                    else max(0.0, now - float(frame_timestamp)) * 1000.0
                )
                print(
                    "[AI_GUIDE] "
                    f"seq={sequence} frame={width}x{height} "
                    f"roi={tuple(int(v) for v in roi)} "
                    f"detected={bool(bbox)} bbox={bbox} "
                    f"coverage={float(coverage):.4f} "
                    f"sharpness={float(sharpness):.2f} "
                    f"frame_age_ms={age_ms:.1f} "
                    f"state={guide.get('state')} "
                    f"method={segmentation.get('method')} "
                    f"mask_nonzero={segmentation.get('mask_nonzero')} "
                    f"mask_pct={segmentation.get('mask_pct')} "
                    f"roi_stats={segmentation.get('roi_stats')} "
                    f"reason={segmentation.get('reason')} "
                    f"seed={segmentation.get('seed')} "
                    f"worker={threading.current_thread().is_alive()}",
                    flush=True,
                )

            # Flujo MANUAL por defecto: la operadora decide cuándo
            # capturar con el botón "CAPTURAR IMAGEN".
            if guide.get("wants_capture") and cfg.get("auto_capture"):
                _ai_guided_capture(tracker)

            time.sleep(0.20)

        except Exception as error:
            AI_GUIDED_RUNTIME["last_error"] = str(error)
            print(f"[AI_GUIDED] Error controlado: {error}")
            time.sleep(0.5)

    print("[AI_GUIDED] Worker finalizado.")


def _ai_capture_recover_mode_from_db():
    """
    Si MySQL tiene una sesión ABIERTA pero el proceso se reinició,
    reconstruye el modo en memoria y relanza el worker.
    """
    if is_ai_capture_mode_active():
        return None

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        row = get_open_capture_session(cur)
    finally:
        cur.close()
        conn.close()

    if row is None:
        return None

    session_id = int(row["id"])
    garment_model_id = int(row["garment_model_id"])
    set_ai_capture_mode(
        True,
        session_id=session_id,
        garment_model_id=garment_model_id,
    )
    cfg = get_ai_capture_config()
    AI_CAPTURE_RUNTIME.update(
        {
            "session_id": session_id,
            "garment_model_id": garment_model_id,
            "config": cfg,
            "known_sha256": set(),
            "known_dhashes": [],
            "presence": None,
            "waiting_for_exit": False,
            "exit_frames": 0,
            "last_sequence": -1,
            "last_error": None,
        }
    )
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        hashes = load_session_accepted_hashes(cur, session_id)
        AI_CAPTURE_RUNTIME["known_sha256"] = set(hashes["sha256"])
    finally:
        cur.close()
        conn.close()
    ensure_camera_capture_worker()
    _ensure_ai_capture_worker()
    print(
        "[AI_CAPTURE] Sesión abierta recuperada tras reinicio: "
        f"session_id={session_id}"
    )
    return session_id


def _json_sanitize(value):
    """Convierte fechas y tipos de BD en texto seguro para el template."""
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _latest_capture_session_for_garment(cur, garment_model_id):
    """Última sesión del modelo (prefiere la que está ABIERTA)."""
    cur.execute(
        """
        SELECT id, garment_model_id, status, started_at, finished_at,
               created_by, notes
        FROM ai_capture_sessions
        WHERE garment_model_id = %s
        ORDER BY (status = 'ABIERTA') DESC, id DESC
        LIMIT 1
        """,
        (garment_model_id,),
    )
    return cur.fetchone()


def _has_preparation_version(cur, garment_model_id):
    if not garment_model_id:
        return False

    cur.execute(
        """
        SELECT id
        FROM garment_ai_models
        WHERE garment_model_id = %s
          AND status = 'PREPARACION'
        LIMIT 1
        """,
        (garment_model_id,),
    )
    return cur.fetchone() is not None


def _ai_capture_idle_payload(cur, cfg, garment_model_id=None, allowed=None):
    """Payload cuando el modelo todavía no tiene sesión de captura."""
    garment = {"id": garment_model_id, "code": None}

    if garment_model_id:
        cur.execute(
            "SELECT id, code FROM garment_models WHERE id = %s",
            (garment_model_id,),
        )
        row = cur.fetchone()
        if row:
            garment = {"id": row["id"], "code": row["code"]}

    has_prep = _has_preparation_version(cur, garment_model_id)

    station_row = get_open_capture_session(cur)
    station_busy = bool(
        station_row
        and (
            garment_model_id is None
            or int(station_row["garment_model_id"])
            != int(garment_model_id)
        )
    )
    station_code = None
    if station_busy:
        cur.execute(
            "SELECT code FROM garment_models WHERE id = %s",
            (station_row["garment_model_id"],),
        )
        station_row_model = cur.fetchone()
        station_code = (station_row_model or {}).get("code")

    payload = {
        "ok": True,
        "active": False,
        "session": None,
        "session_id": None,
        "status": None,
        "garment_model": garment,
        "accepted_count": 0,
        "rejected_count": 0,
        "target_count": int(cfg["target_images"]),
        "last_capture": None,
        "started_at": None,
        "finished_at": None,
        "mode": ai_capture.get_ai_capture_mode_state(),
        "config": {
            "min_coverage": cfg["min_coverage"],
            "min_sharpness": cfg["min_sharpness"],
            "duplicate_max_distance": cfg["duplicate_max_distance"],
            "keep_rejected": cfg["keep_rejected"],
        },
        "has_preparation_version": has_prep,
        "station_busy": station_busy,
        "station_busy_model_code": station_code,
    }
    return _ai_capture_enrich(
        payload,
        cur,
        allowed=allowed,
        config=cfg,
    )


def _ai_capture_enrich(payload, cur, allowed=None, config=None):
    """
    Agrega el estado visible y los mensajes comprensibles para la ficha.

    Nunca expone ids internos ni nombres técnicos a la usuaria.
    """
    cfg = dict(config or get_ai_capture_config())
    garment = payload.get("garment_model") or {}
    garment_id = garment.get("id")

    has_prep = payload.get("has_preparation_version")
    if has_prep is None:
        has_prep = _has_preparation_version(cur, garment_id)
    has_prep = bool(has_prep)
    payload["has_preparation_version"] = has_prep

    state = resolve_capture_ui_state(
        session_status=payload.get("status"),
        accepted_count=payload.get("accepted_count"),
        min_images=cfg.get("min_images"),
        has_preparation_version=has_prep,
    )
    payload["ui_state"] = state
    payload["ui_state_label"] = CAPTURE_UI_STATE_LABELS.get(state, state)

    target = int(payload.get("target_count") or 0)
    accepted = int(payload.get("accepted_count") or 0)
    payload["target_reached"] = bool(target > 0 and accepted >= target)

    # Finalizar y entrenar exigen el mínimo configurable; el objetivo
    # recomendado solo informa avance y nunca bloquea.
    gate = capture_progress_gate(
        accepted,
        min_images=cfg.get("min_images"),
        target_count=target or cfg.get("target_images"),
    )
    payload["min_count"] = gate["min_count"]
    payload["can_finalize"] = gate["can_finalize"]
    payload["can_train"] = gate["can_train"]
    payload["missing_to_train"] = gate["missing_to_train"]

    last = payload.get("last_capture")
    if last:
        accepted_last = (
            str(last.get("status") or "").strip().upper() == "ACEPTADA"
        )
        payload["last_capture_human"] = {
            "accepted": accepted_last,
            "label": "Aceptada" if accepted_last else "Descartada",
            "reason": (
                None
                if accepted_last
                else humanize_reject_reason(last.get("reject_reason"))
            ),
            "captured_at": str(last.get("captured_at") or ""),
        }
    else:
        payload["last_capture_human"] = None

    if allowed is not None:
        payload["allowed"] = bool(allowed)

    payload["guide"] = _ai_guided_payload(garment_id)

    # Estado del entrenamiento (FASE 3A): la ficha y la estación
    # consultan el mismo payload sin duplicar consultas en el front.
    try:
        payload["training"] = (
            training_status_payload(
                cur,
                garment_id,
                allowed=allowed,
            )
            if garment_id
            else None
        )
    except Exception:
        app.logger.exception(
            "No se pudo leer el estado del entrenamiento IA "
            f"(garment_model_id={garment_id})."
        )
        payload["training"] = None

    return payload


def _ai_capture_status_payload(
    session_id=None,
    garment_model_id=None,
    allowed=None,
):
    cfg = get_ai_capture_config()
    if session_id is None:
        _ai_capture_recover_mode_from_db()
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        if session_id is None:
            if garment_model_id is None:
                row = get_open_capture_session(cur)
            else:
                row = _latest_capture_session_for_garment(
                    cur,
                    garment_model_id,
                )
            if row is None:
                return _ai_capture_idle_payload(
                    cur,
                    cfg,
                    garment_model_id=garment_model_id,
                    allowed=allowed,
                )
            session_id = int(row["id"])

        cur.execute(
            """
            SELECT id, frame_sequence, coverage, quality_score,
                   status, reject_reason, captured_at
            FROM ai_training_images
            WHERE capture_session_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        last = cur.fetchone()
        last_capture = row_to_json(last) if last else None
        payload = capture_session_status_payload(
            cur,
            session_id,
            config=cfg,
            last_capture=last_capture,
        )
        payload["active"] = (
            payload["status"] == "ABIERTA"
            and is_ai_capture_mode_active()
        )
        payload["mode"] = ai_capture.get_ai_capture_mode_state()
        payload["ok"] = True
        payload.setdefault("session_id", int(session_id))
        payload["station_busy"] = False
        payload["station_busy_model_code"] = None
        payload = _ai_capture_enrich(
            payload,
            cur,
            allowed=allowed,
            config=cfg,
        )

        if payload.get("active"):
            # Si el motor guiado murió, se reaviva: si no, la UI
            # mostraría un estado de prenda congelado.
            _ai_guided_ensure_alive()

        return payload
    finally:
        cur.close()
        conn.close()


@app.route("/api/ai/capture/start", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_start():
    global AI_CAPTURE_WORKER_ENABLED

    body = request.get_json(silent=True) or {}
    raw_model = body.get("garment_model_id")

    try:
        garment_model_id = int(raw_model)
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "garment_model_id inválido.",
        }), 400

    if garment_model_id <= 0:
        return jsonify({
            "ok": False,
            "error": "garment_model_id debe ser positivo.",
        }), 400

    model = get_garment_model(garment_model_id)

    if not model:
        return jsonify({
            "ok": False,
            "error": "El modelo de prenda no existe.",
        }), 404

    if not can_manage_garment_model(model):
        return jsonify({
            "ok": False,
            "error": (
                "No tiene permisos para preparar la captura "
                "de este modelo."
            ),
        }), 403

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        return jsonify({
            "ok": False,
            "error": (
                "La captura solo puede iniciarse con el modelo "
                "aprobado y activo."
            ),
        }), 409

    try:
        ensure_ai_capture_can_start()
    except AIDomainError as error:
        return jsonify({
            "ok": False,
            "error": humanize_capture_error(error),
        }), 409

    actor_id = session.get("user_id")
    notes = str(body.get("notes") or "").strip()[:500] or None

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()
        opened = start_ai_capture_session(
            cur,
            garment_model_id,
            actor_id,
            notes=notes,
        )
        session_id = int(opened["id"])

        if not opened.get("already_open"):
            record_ai_event(
                cur,
                "CAPTURE_STARTED",
                actor_id=actor_id,
                capture_session_id=session_id,
                payload={
                    "garment_model_id": garment_model_id,
                },
            )

        cur.execute(
            "SELECT id, code FROM garment_models WHERE id = %s",
            (garment_model_id,),
        )
        garment = cur.fetchone()
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": humanize_capture_error(error),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            "No se pudo iniciar la sesión de captura IA "
            f"(garment_model_id={garment_model_id})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo iniciar la captura. "
                "Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    if garment is None:
        return jsonify({
            "ok": False,
            "error": "El modelo de prenda no existe.",
        }), 404

    # Arranque manual: NO se inicia solo sobre la cámara real.
    # Solo se activa el modo y el worker cuando el usuario llama a start.
    set_ai_capture_mode(
        True,
        session_id=session_id,
        garment_model_id=garment_model_id,
    )
    cfg = get_ai_capture_config()
    AI_CAPTURE_RUNTIME.update(
        {
            "session_id": session_id,
            "garment_model_id": garment_model_id,
            "config": cfg,
            "known_sha256": set(),
            "known_dhashes": [],
            "presence": None,
            "waiting_for_exit": False,
            "exit_frames": 0,
            "last_sequence": -1,
            "last_persisted_token": None,
            "last_error": None,
            "last_result": None,
        }
    )

    # Cargar hashes aceptados previos de la sesión (reinicio).
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        hashes = load_session_accepted_hashes(cur, session_id)
        AI_CAPTURE_RUNTIME["known_sha256"] = set(hashes["sha256"])
        # dhash no se persiste en Fase 2A: solo sha256 exacto entre reinicios.
    finally:
        cur.close()
        conn.close()

    ensure_camera_capture_worker()
    if body.get("guided"):
        # La pantalla de captura guiada arranca su propio motor
        # con /api/ai/capture/guide/start (1 motor a la vez).
        started_worker = False
    else:
        started_worker = _ensure_ai_capture_worker()

    status = _ai_capture_status_payload(session_id)
    return jsonify({
        "ok": True,
        "message": (
            "Sesión de captura IA iniciada."
            if not opened.get("already_open")
            else "Sesión de captura IA ya estaba activa."
        ),
        "worker_started": started_worker,
        "session": status,
        "garment_model": {
            "id": garment_model_id,
            "code": garment.get("code"),
        },
    }), 201 if not opened.get("already_open") else 200


@app.route("/api/ai/capture/status")
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_status():
    session_id = request.args.get("session_id")
    parsed_id = None
    if session_id not in (None, ""):
        try:
            parsed_id = int(session_id)
        except ValueError:
            return jsonify({
                "ok": False,
                "error": "session_id inválido.",
            }), 400

    garment_model_id = request.args.get("garment_model_id")
    parsed_garment = None
    if garment_model_id not in (None, ""):
        try:
            parsed_garment = int(garment_model_id)
        except ValueError:
            return jsonify({
                "ok": False,
                "error": "garment_model_id inválido.",
            }), 400

        if parsed_garment <= 0:
            return jsonify({
                "ok": False,
                "error": "garment_model_id debe ser positivo.",
            }), 400

    try:
        payload = _ai_capture_status_payload(
            parsed_id,
            garment_model_id=parsed_garment,
        )
    except AIDomainError as error:
        return jsonify({
            "ok": False,
            "error": str(error),
        }), 404
    except Exception:
        app.logger.exception(
            "Error consultando estado de captura IA."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo actualizar el estado de la captura. "
                "Intente de nuevo."
            ),
        }), 500

    return jsonify(payload)


@app.route("/api/ai/capture/stop", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_stop():
    return _ai_capture_finish("stop")


@app.route("/api/ai/capture/cancel", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_cancel():
    return _ai_capture_finish("cancel")


def _ai_capture_finish(action):
    body = request.get_json(silent=True) or {}
    raw_id = body.get("session_id")
    session_id = None

    if raw_id not in (None, ""):
        try:
            session_id = int(raw_id)
        except (TypeError, ValueError):
            return jsonify({
                "ok": False,
                "error": "session_id inválido.",
            }), 400

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()

        if session_id is None:
            row = get_open_capture_session(cur)
            if row is None:
                conn.rollback()
                return jsonify({
                    "ok": True,
                    "message": (
                        "No hay sesión de captura activa "
                        "(idempotente)."
                    ),
                    "session": None,
                })
            session_id = int(row["id"])

        if action == "stop":
            result = stop_ai_capture_session(cur, session_id)
            event_type = "CAPTURE_COMPLETED"
            message = "Sesión de captura finalizada."
        else:
            result = cancel_ai_capture_session(cur, session_id)
            event_type = "CAPTURE_CANCELLED"
            message = "Sesión de captura cancelada."

        already = result.get(
            "already_finished"
            if action == "stop"
            else "already_cancelled",
            False,
        )

        if not already:
            record_ai_event(
                cur,
                event_type,
                actor_id=session.get("user_id"),
                capture_session_id=session_id,
                payload={
                    "action": action,
                },
            )

        cfg = get_ai_capture_config()
        status_payload = capture_session_status_payload(
            cur,
            session_id,
            config=cfg,
        )
        status_payload["session_id"] = int(session_id)
        status_payload["active"] = False
        status_payload["ok"] = True
        status_payload["mode"] = ai_capture.get_ai_capture_mode_state()
        status_payload["station_busy"] = False
        status_payload["station_busy_model_code"] = None
        status_payload = _ai_capture_enrich(
            status_payload,
            cur,
            config=cfg,
        )
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": humanize_capture_error(error),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            "No se pudo cerrar la sesión de captura IA."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo cerrar la sesión de captura. "
                "Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    # Solo se apaga el modo si la estación quedó sin sesión abierta.
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        still_open = get_open_capture_session(cur)
    finally:
        cur.close()
        conn.close()

    if still_open is None:
        set_ai_capture_mode(False)
        _stop_ai_capture_worker()
        _stop_ai_guided_worker()
        _ai_guided_reset_state()

    status_payload["mode"] = ai_capture.get_ai_capture_mode_state()

    return jsonify({
        "ok": True,
        "message": message,
        "idempotent": bool(already),
        "session": status_payload,
    })


def _ai_training_status_payload(garment_model_id, allowed=None):
    """Estado del entrenamiento listo para JSON (con saneado)."""
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        payload = training_status_payload(
            cur,
            garment_model_id,
            allowed=allowed,
        )
    finally:
        cur.close()
        conn.close()

    # FASE 3B: enlace directo a la pantalla de validación de la versión.
    if payload.get("validation_available") and payload.get(
        "validation_ai_model_id"
    ):
        payload["validation_url"] = url_for(
            "garment_validation_page",
            model_id=garment_model_id,
            ai_model_id=payload["validation_ai_model_id"],
        )
    else:
        payload["validation_url"] = None

    return _json_sanitize(payload)


def _ai_training_guard(garment_model_id):
    """Valida modelo + permisos para entrenar. Devuelve (modelo, error)."""
    model = get_garment_model(garment_model_id)

    if not model:
        return None, (
            jsonify({
                "ok": False,
                "error": "El modelo de prenda no existe.",
            }),
            404,
        )

    if not can_manage_garment_model(model):
        return None, (
            jsonify({
                "ok": False,
                "error": (
                    "No tiene permisos para entrenar este modelo."
                ),
            }),
            403,
        )

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        return None, (
            jsonify({
                "ok": False,
                "error": (
                    "El entrenamiento solo puede iniciarse con el "
                    "modelo aprobado y activo."
                ),
            }),
            409,
        )

    return model, None


def _start_training_request(garment_model_id, *, as_new_version):
    """Encola un TRAINING (normal o como NUEVA versión). JSON + código."""
    actor_id = session.get("user_id")

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()
        created = request_training_job(
            cur,
            garment_model_id=garment_model_id,
            actor_id=actor_id,
            as_new_version=as_new_version,
        )
        job_id = int(created["job"]["id"])
        payload = training_status_payload(
            cur,
            garment_model_id,
            allowed=True,
        )
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": humanize_training_error(error),
            "training": _ai_training_status_payload(
                garment_model_id,
                allowed=True,
            ),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            "No se pudo solicitar el entrenamiento IA "
            f"(garment_model_id={garment_model_id}, "
            f"as_new_version={as_new_version})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo iniciar el entrenamiento. "
                "Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    message = (
        "Entrenamiento de una nueva versión encolado. El sistema lo "
        "ejecutará en segundo plano."
        if as_new_version
        else (
            "Entrenamiento encolado. El sistema lo ejecutará en "
            "segundo plano."
        )
    )

    retrain = created.get("retrain")
    retrain_summary = None

    if as_new_version and retrain:
        # Resumen legible para la UI/auditoría (sin notas internas).
        retrain_summary = {
            "version": created["ai_model"].get("version"),
            "next_version": retrain.get("next_version"),
            "superseded_versions": retrain.get("historical"),
            "images_reused": int(created["dataset"]["image_count"]),
        }

    return jsonify({
        "ok": True,
        "message": message,
        "job_id": job_id,
        "dataset_id": int(created["dataset"]["id"]),
        "ai_model_id": int(created["ai_model"]["id"]),
        "ai_model_version": created["ai_model"].get("version"),
        "retrain": _json_sanitize(retrain_summary),
        "training": _json_sanitize(payload),
    }), 201


def _parse_garment_model_id(body):
    """garment_model_id del cuerpo JSON o (None, respuesta 400)."""
    raw_model = body.get("garment_model_id")

    try:
        garment_model_id = int(raw_model)
    except (TypeError, ValueError):
        return None, jsonify({
            "ok": False,
            "error": "garment_model_id inválido.",
        }), 400

    if garment_model_id <= 0:
        return None, jsonify({
            "ok": False,
            "error": "garment_model_id debe ser positivo.",
        }), 400

    return garment_model_id, None


@app.route("/api/ai/training/start", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_training_start():
    body = request.get_json(silent=True) or {}
    garment_model_id, parse_error = _parse_garment_model_id(body)

    if parse_error is not None:
        return parse_error

    model, guard_error = _ai_training_guard(garment_model_id)

    if guard_error is not None:
        return guard_error

    return _start_training_request(
        garment_model_id,
        as_new_version=False,
    )


@app.route("/api/ai/training/retrain", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_training_retrain():
    """Reentrena como NUEVA versión (FASE 3A.3).

    Nunca reutiliza ni modifica la versión anterior: crea vN+1 y su
    propio directorio de artefactos. El quality gate corre antes de
    crear el job (mismo flujo que /start).
    """
    body = request.get_json(silent=True) or {}
    garment_model_id, parse_error = _parse_garment_model_id(body)

    if parse_error is not None:
        return parse_error

    model, guard_error = _ai_training_guard(garment_model_id)

    if guard_error is not None:
        return guard_error

    return _start_training_request(
        garment_model_id,
        as_new_version=True,
    )


@app.route("/api/ai/training/cancel", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_training_cancel():
    body = request.get_json(silent=True) or {}
    raw_job = body.get("job_id")

    try:
        job_id = int(raw_job)
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "job_id inválido.",
        }), 400

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()
        cur.execute(
            """
            SELECT j.id, j.status, m.garment_model_id
            FROM ai_jobs j
            JOIN garment_ai_models m ON m.id = j.ai_model_id
            WHERE j.id = %s
            """,
            (job_id,),
        )
        job = cur.fetchone()

        if job is None:
            conn.rollback()
            return jsonify({
                "ok": False,
                "error": "El trabajo de entrenamiento no existe.",
            }), 404

        garment_model_id = int(job["garment_model_id"])
        model = get_garment_model(garment_model_id)

        if model is None or not can_manage_garment_model(model):
            conn.rollback()
            return jsonify({
                "ok": False,
                "error": (
                    "No tiene permisos para cancelar este entrenamiento."
                ),
            }), 403

        result = cancel_pending_training_job(
            cur,
            job_id=job_id,
            actor_id=session.get("user_id"),
        )
        payload = training_status_payload(
            cur,
            garment_model_id,
            allowed=True,
        )
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": humanize_training_error(error),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            f"No se pudo cancelar el entrenamiento IA (job_id={job_id})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo cancelar el entrenamiento. "
                "Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "ok": True,
        "message": "Entrenamiento cancelado.",
        "job_id": int(result["id"]),
        "training": _json_sanitize(payload),
    })


@app.route("/api/ai/training/status")
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER, ROLE_QUALITY_MANAGER)
def ai_training_status():
    raw_model = request.args.get("garment_model_id")

    try:
        garment_model_id = int(raw_model)
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "garment_model_id inválido.",
        }), 400

    if garment_model_id <= 0:
        return jsonify({
            "ok": False,
            "error": "garment_model_id debe ser positivo.",
        }), 400

    model = get_garment_model(garment_model_id)

    if not model:
        return jsonify({
            "ok": False,
            "error": "El modelo de prenda no existe.",
        }), 404

    allowed = can_manage_garment_model(model) and model.get(
        "status"
    ) == "APROBADO"

    try:
        payload = _ai_training_status_payload(
            garment_model_id,
            allowed=allowed,
        )
    except Exception:
        app.logger.exception(
            "No se pudo leer el estado del entrenamiento IA "
            f"(garment_model_id={garment_model_id})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo consultar el entrenamiento. "
                "Intente de nuevo."
            ),
        }), 500

    return jsonify({
        "ok": True,
        "training": payload,
    })


# ============================================================
# FASE 3B — VALIDACIÓN CONTROLADA DEL MODELO
#
# Imágenes NUEVAS, banco propio (validations/) e inferencia
# aislada por versión: predict(ai_model_id=...) resuelve
# artefactos -> config -> checkpoint de ESA versión. El checkpoint
# productivo (PATCHCORE_CKPT) no se lee ni se modifica, y la
# validación nunca escribe en el dataset de entrenamiento.
# La activación pertenece a FASE 3C.
# ============================================================

VALIDATION_IMAGE_KINDS = ("original", "heatmap", "comparison")


def _resolve_validation_ai_model(garment_model_id, ai_model_id=None):
    """Versión a validar: la indicada o la última apta del modelo."""
    versions = get_garment_ai_versions(garment_model_id)

    if not versions:
        return None, (
            "Este modelo todavía no tiene versiones de IA registradas."
        )

    if ai_model_id is not None:
        for version in versions:
            if int(version.get("id")) == int(ai_model_id):
                if version.get("technically_invalidated"):
                    return None, (
                        "Esta versión está marcada NO VALIDADA / NO APTO "
                        "PARA ACTIVACIÓN: no puede validarse."
                    )
                return int(version["id"]), None

        return None, (
            "La versión solicitada no pertenece a este modelo."
        )

    for version in versions:
        if (
            str(version.get("status") or "").strip().upper()
            in VALIDATION_MODEL_STATUSES
            and not version.get("technically_invalidated")
        ):
            return int(version["id"]), None

    return None, (
        "Este modelo todavía no tiene una versión entrenada para validar."
    )


def _validation_api_guard(garment_model_id, ai_model_id=None):
    """Modelo + permisos + versión. Devuelve (model, ai_model_id, error)."""
    model = get_garment_model(garment_model_id)

    if not model:
        return None, None, (jsonify({
            "ok": False,
            "error": "El modelo de prenda no existe.",
        }), 404)

    if not can_manage_garment_model(model):
        return None, None, (jsonify({
            "ok": False,
            "error": "No tiene permisos para validar este modelo.",
        }), 403)

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        return None, None, (jsonify({
            "ok": False,
            "error": (
                "La validación requiere el modelo aprobado y activo."
            ),
        }), 409)

    resolved, error = _resolve_validation_ai_model(
        garment_model_id,
        ai_model_id,
    )

    if error:
        return None, None, (jsonify({
            "ok": False,
            "error": error,
        }), 404)

    return model, resolved, None


def _parse_optional_ai_model_id(body):
    """ai_model_id opcional del cuerpo JSON o (None, error 400)."""
    raw = body.get("ai_model_id")

    if raw in (None, ""):
        return None, None

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, (jsonify({
            "ok": False,
            "error": "ai_model_id inválido.",
        }), 400)

    if value <= 0:
        return None, (jsonify({
            "ok": False,
            "error": "ai_model_id debe ser positivo.",
        }), 400)

    return value, None


def _validation_image_bytes(body, files):
    """Imagen NUEVA: archivo, base64 o frame actual de la cámara."""
    upload = files.get("image") if files else None

    if upload is not None and getattr(upload, "filename", ""):
        data = upload.read()

        if data:
            return data, "upload"

    raw = body.get("image_base64")

    if raw:
        text = str(raw).strip()

        if "," in text and text.lower().startswith("data:"):
            text = text.split(",", 1)[1]

        try:
            data = base64.b64decode(text, validate=True)
        except Exception:
            raise AIDomainError(
                "La imagen enviada no es un base64 válido."
            )

        if data:
            return data, "upload"

    frame, _, _ = get_latest_camera_frame()

    if frame is None:
        raise AIDomainError(
            "No hay un frame de cámara disponible para validar. "
            "Intente de nuevo en unos segundos."
        )

    return _ai_capture_encode_jpeg(frame), "frame"


@app.route("/modelos-prenda/<int:model_id>/validacion-ia")
@app.route("/modelos-prenda/<int:model_id>/validacion-ia/<int:ai_model_id>")
@login_required
@role_required(
    ROLE_ADMIN,
    ROLE_MODEL_MANAGER,
    ROLE_QUALITY_MANAGER,
)
def garment_validation_page(model_id, ai_model_id=None):
    """Pantalla «Validar modelo» (FASE 3B)."""
    model = get_garment_model(model_id)

    if not model:
        flash("El modelo de prenda solicitado no existe.", "error")
        return redirect(url_for("garment_models_page"))

    role = normalize_role(session.get("role"))

    if (
        role == ROLE_QUALITY_MANAGER
        and (
            model.get("status") != "APROBADO"
            or int(model.get("active") or 0) != 1
        )
    ):
        flash("No tiene permisos para consultar ese modelo.", "error")
        return redirect(url_for("garment_models_page"))

    if not can_manage_garment_model(model):
        flash(
            "No tiene permisos para validar este modelo.",
            "error",
        )
        return redirect(url_for("garment_models_page"))

    resolved, error = _resolve_validation_ai_model(
        model_id,
        ai_model_id,
    )

    if error:
        flash(error, "error")
        return redirect(url_for("garment_model_detail", model_id=model_id))

    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        state = ai_validation.get_validation_state(cur, resolved)
    except AIDomainError as error:
        flash(str(error), "error")
        return redirect(url_for("garment_model_detail", model_id=model_id))
    except Exception:
        app.logger.exception(
            "No se pudo leer el estado de validación "
            f"(garment_model_id={model_id}, ai_model_id={resolved})."
        )
        flash(
            "No se pudo cargar la validación. Intente de nuevo.",
            "error",
        )
        return redirect(url_for("garment_model_detail", model_id=model_id))
    finally:
        cur.close()
        conn.close()

    return render_template(
        "validation_model.html",
        model=model,
        validation=_json_sanitize(state),
        role=role,
        can_manage=can_manage_garment_model(model),
        categories=VALIDATION_CATEGORIES,
    )


@app.route(
    "/modelos-prenda/<int:model_id>/validacion-ia/frame",
    methods=["GET"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def garment_validation_frame(model_id):
    """Frame actual de la cámara para la previsualización de validación."""
    model = get_garment_model(model_id)

    if not model or not can_manage_garment_model(model):
        return jsonify({
            "ok": False,
            "error": "No tiene permisos para ver esta cámara.",
        }), 403

    frame, _, _ = get_latest_camera_frame()

    if frame is None:
        return Response(
            "",
            status=204,
            headers={"Cache-Control": "no-store"},
        )

    try:
        jpeg = _ai_capture_encode_jpeg(frame)
    except Exception:
        app.logger.exception(
            f"No se pudo codificar el frame de validación ({model_id})."
        )
        return Response("", status=204)

    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.route(
    "/modelos-prenda/<int:model_id>/validacion-ia/caso/"
    "<int:case_id>/<kind>",
    methods=["GET"],
)
@login_required
@role_required(
    ROLE_ADMIN,
    ROLE_MODEL_MANAGER,
    ROLE_QUALITY_MANAGER,
)
def garment_validation_case_image(model_id, case_id, kind):
    """Sirve original/heatmap/comparación de un caso de validación."""
    return _serve_validation_case_image(model_id, case_id, kind)


@app.route(
    "/modelos-prenda/<int:model_id>/validacion-ia/imagen",
    methods=["GET"],
)
@login_required
@role_required(
    ROLE_ADMIN,
    ROLE_MODEL_MANAGER,
    ROLE_QUALITY_MANAGER,
)
def garment_validation_case_asset(model_id):
    """Mismo artefacto consultando ?case_id=&kind= (para la UI)."""
    raw_case = request.args.get("case_id")
    kind = request.args.get("kind") or "original"

    try:
        case_id = int(raw_case)
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "case_id inválido.",
        }), 400

    return _serve_validation_case_image(model_id, case_id, kind)


def _serve_validation_case_image(model_id, case_id, kind):
    if kind not in VALIDATION_IMAGE_KINDS:
        return jsonify({
            "ok": False,
            "error": "Artefacto de validación desconocido.",
        }), 404

    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute(
            """
            SELECT id, garment_model_id, image_path, heatmap_path,
                   comparison_path
            FROM ai_validation_cases
            WHERE id = %s
            """,
            (case_id,),
        )
        case = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if case is None or int(case["garment_model_id"]) != int(model_id):
        return jsonify({
            "ok": False,
            "error": "El caso de validación no existe.",
        }), 404

    column = {
        "original": "image_path",
        "heatmap": "heatmap_path",
        "comparison": "comparison_path",
    }[kind]

    relative = case.get(column)

    if not relative:
        return jsonify({
            "ok": False,
            "error": "Este caso no tiene ese artefacto.",
        }), 404

    try:
        path = resolve_under_root(get_ai_artifacts_root(), relative)
    except AIDomainError:
        return jsonify({
            "ok": False,
            "error": "Artefacto de validación inválido.",
        }), 404

    if not path.exists():
        return jsonify({
            "ok": False,
            "error": "El artefacto de validación no existe.",
        }), 404

    mimetype = (
        "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    )

    return send_file(path, mimetype=mimetype, conditional=True)


@app.route("/api/ai/validation/session/start", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_validation_session_start():
    """Abre la sesión de validación (ENTRENADO -> VALIDACION)."""
    body = request.get_json(silent=True) or {}
    garment_model_id, parse_error = _parse_garment_model_id(body)

    if parse_error is not None:
        return parse_error

    raw_ai, ai_error = _parse_optional_ai_model_id(body)

    if ai_error is not None:
        return ai_error

    model, resolved, guard_error = _validation_api_guard(
        garment_model_id,
        raw_ai,
    )

    if guard_error is not None:
        return guard_error

    actor_id = session.get("user_id")
    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()
        session_row = ai_validation.start_validation_session(
            cur,
            ai_model_id=resolved,
            actor_id=actor_id,
        )
        state = ai_validation.get_validation_state(cur, resolved)
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": str(error),
            "validation": _ai_validation_state(resolved),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            "No se pudo iniciar la validación "
            f"(ai_model_id={resolved})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo iniciar la validación. Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "ok": True,
        "message": "Validación iniciada. La imagen capturada se usará "
                   "solo para validar.",
        "session": _json_sanitize(session_row),
        "validation": _json_sanitize(state),
    }), 201


def _ai_validation_state(ai_model_id):
    """Estado de validación para respuestas de error (best effort)."""
    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        return _json_sanitize(
            ai_validation.get_validation_state(cur, ai_model_id)
        )
    except Exception:
        return None
    finally:
        cur.close()
        conn.close()


@app.route("/api/ai/validation/case", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_validation_case():
    """Registra un caso de validación con imagen NUEVA + score bruto."""
    body = request.get_json(silent=True) or {}
    garment_model_id, parse_error = _parse_garment_model_id(body)

    if parse_error is not None:
        return parse_error

    raw_ai, ai_error = _parse_optional_ai_model_id(body)

    if ai_error is not None:
        return ai_error

    model, resolved, guard_error = _validation_api_guard(
        garment_model_id,
        raw_ai,
    )

    if guard_error is not None:
        return guard_error

    category = body.get("category")
    observation = body.get("observation")
    actor_id = session.get("user_id")

    try:
        image_bytes, source = _validation_image_bytes(body, request.files)
    except AIDomainError as error:
        return jsonify({
            "ok": False,
            "error": str(error),
        }), 409

    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()
        case = ai_validation.register_validation_case(
            cur,
            ai_model_id=resolved,
            category=category,
            image_bytes=image_bytes,
            observation=observation,
            actor_id=actor_id,
            preprocess=True,
        )
        case["source"] = source
        state = ai_validation.get_validation_state(cur, resolved)
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": str(error),
            "validation": _ai_validation_state(resolved),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            "No se pudo registrar el caso de validación "
            f"(ai_model_id={resolved}, category={category})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo registrar el caso de validación. "
                "Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "ok": True,
        "message": (
            "Caso de validación registrado. "
            "Imagen de validación — no se utilizará para entrenamiento."
        ),
        "case": _json_sanitize(case),
        "validation": _json_sanitize(state),
    }), 201


@app.route("/api/ai/validation/evaluate", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_validation_evaluate():
    """«Evaluar validación»: métricas + thresholds candidatos.

    No cambia el estado del modelo (ni lo activa).
    """
    body = request.get_json(silent=True) or {}
    garment_model_id, parse_error = _parse_garment_model_id(body)

    if parse_error is not None:
        return parse_error

    raw_ai, ai_error = _parse_optional_ai_model_id(body)

    if ai_error is not None:
        return ai_error

    model, resolved, guard_error = _validation_api_guard(
        garment_model_id,
        raw_ai,
    )

    if guard_error is not None:
        return guard_error

    thresholds = body.get("thresholds")

    if thresholds is not None and not isinstance(thresholds, list):
        return jsonify({
            "ok": False,
            "error": "thresholds debe ser una lista numérica.",
        }), 400

    actor_id = session.get("user_id")
    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()
        result = ai_validation.evaluate_validation(
            cur,
            ai_model_id=resolved,
            actor_id=actor_id,
            thresholds=thresholds,
        )
        state = ai_validation.get_validation_state(cur, resolved)
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": str(error),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            f"No se pudo evaluar la validación (ai_model_id={resolved})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo evaluar la validación. Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "ok": True,
        "message": (
            "Validación evaluada. El umbral mostrado es un candidato: "
            "la activación es una acción posterior."
        ),
        "result": _json_sanitize(result),
        "validation": _json_sanitize(state),
    })


@app.route("/api/ai/validation/complete", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN)
def ai_validation_complete():
    """Cierra la validación (VALIDACION -> VALIDADO). Nunca ACTIVO."""
    body = request.get_json(silent=True) or {}
    garment_model_id, parse_error = _parse_garment_model_id(body)

    if parse_error is not None:
        return parse_error

    raw_ai, ai_error = _parse_optional_ai_model_id(body)

    if ai_error is not None:
        return ai_error

    model, resolved, guard_error = _validation_api_guard(
        garment_model_id,
        raw_ai,
    )

    if guard_error is not None:
        return guard_error

    actor_id = session.get("user_id")
    conn = db()
    cur = conn.cursor(dictionary=True)

    try:
        conn.start_transaction()
        result = ai_validation.complete_validation(
            cur,
            ai_model_id=resolved,
            actor_id=actor_id,
        )
        state = ai_validation.get_validation_state(cur, resolved)
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": str(error),
            "validation": _ai_validation_state(resolved),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            f"No se pudo cerrar la validación (ai_model_id={resolved})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo cerrar la validación. Intente de nuevo."
            ),
        }), 500
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "ok": True,
        "message": (
            "Validación completada. La versión quedó VALIDADA; "
            "la activación es una acción posterior (FASE 3C)."
        ),
        "result": _json_sanitize(result),
        "validation": _json_sanitize(state),
    })


# ============================================================
# FASE 2C — pantalla de captura guiada (video + posicionamiento)
# ============================================================


def _ascii_label(value):
    """cv2.putText no dibuja acentos: se translitera sin romper nada."""
    import unicodedata

    text = str(value or "")
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def draw_guided_overlay(frame, guide):
    """
    Overlay de posicionamiento sobre el video en vivo.

    El color NUNCA es la única señal: siempre acompaña a texto de
    estado, ícono de check/cruces y una línea gruesa de esquinas.
    """
    if cv2 is None or frame is None:
        return frame

    output = frame.copy()

    try:
        x1, y1, x2, y2 = get_roi_bounds(output)
    except Exception:
        return output

    height = output.shape[0]

    guide = guide or {}
    state = guide.get("state")
    color = GUIDED_STATE_COLORS.get(state, (150, 150, 150))
    # El visto aparece solo cuando la operadora YA puede capturar
    # (o cuando la captura está en curso/terminada).
    valid = bool(guide.get("ready")) or state in (
        "CAPTURANDO",
        "CAPTURADA",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- área objetivo ---
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

    # --- esquinas marcadoras (forma, además del color) ---
    corner = 30
    thick = 7
    cv2.line(output, (x1, y1), (x1 + corner, y1), (255, 255, 255), thick)
    cv2.line(output, (x1, y1), (x1, y1 + corner), (255, 255, 255), thick)
    cv2.line(output, (x2, y1), (x2 - corner, y1), (255, 255, 255), thick)
    cv2.line(output, (x2, y1), (x2, y1 + corner), (255, 255, 255), thick)
    cv2.line(output, (x1, y2), (x1 + corner, y2), (255, 255, 255), thick)
    cv2.line(output, (x1, y2), (x1, y2 - corner), (255, 255, 255), thick)
    cv2.line(output, (x2, y2), (x2 - corner, y2), (255, 255, 255), thick)
    cv2.line(output, (x2, y2), (x2, y2 - corner), (255, 255, 255), thick)

    # --- silueta detectada ---
    bbox = guide.get("bbox")
    if bbox:
        cv2.rectangle(
            output,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            2,
        )
        cv2.putText(
            output,
            "PRENDA",
            (int(bbox[0]), max(22, int(bbox[1]) - 8)),
            font,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # --- banner de estado + ícono ---
    label = _ascii_label(
        guide.get("state_label") or "PREPARANDO"
    )
    banner_height = 46
    banner_top = max(0, y1 - banner_height - 10)
    banner_right = min(output.shape[1] - 1, x1 + 380)
    cv2.rectangle(
        output,
        (x1, banner_top),
        (banner_right, banner_top + banner_height),
        color,
        -1,
    )
    cv2.putText(
        output,
        label,
        (x1 + 14, banner_top + 33),
        font,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    icon_x = banner_right - 44
    icon_y = banner_top + 23
    if valid:
        cv2.line(output, (icon_x - 14, icon_y), (icon_x - 4, icon_y + 10), (255, 255, 255), 4)
        cv2.line(output, (icon_x - 4, icon_y + 10), (icon_x + 14, icon_y - 12), (255, 255, 255), 4)
    else:
        cv2.line(output, (icon_x - 12, icon_y - 12), (icon_x + 12, icon_y + 12), (255, 255, 255), 4)
        cv2.line(output, (icon_x + 12, icon_y - 12), (icon_x - 12, icon_y + 12), (255, 255, 255), 4)

    # --- mensaje para la usuaria ---
    message = _ascii_label(guide.get("message"))
    if message:
        box_height = 46
        box_top = y2 + 8
        if box_top + box_height > height:
            box_top = max(0, height - box_height - 4)
        box_right = min(output.shape[1] - 1, x1 + 560)
        cv2.rectangle(
            output,
            (x1, box_top),
            (box_right, box_top + box_height),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            output,
            message.upper()[:64],
            (x1 + 14, box_top + 32),
            font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def generate_guided_video_feed(garment_model_id):
    """Stream MJPEG de la pantalla guiada, con overlay de posición."""
    while True:
        ensure_camera_capture_worker()

        frame, timestamp, sequence = get_latest_camera_frame()

        if frame is None:
            frame_to_send = make_camera_error_frame(
                "RECONECTANDO CAMARA"
            )
        else:
            log_camera_latency(
                source="ai_guided_video",
                sequence=sequence,
                timestamp_monotonic=timestamp,
            )
            guide = _ai_guided_payload(garment_model_id)
            frame_to_send = draw_guided_overlay(frame, guide)

        jpg = encode_jpeg(frame_to_send)

        if jpg is None:
            jpg = generate_placeholder_frame("ERROR DE VIDEO")

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg
            + b"\r\n"
        )

        time.sleep(0.08)


def _guarded_model_or_none(model_id):
    """Validación común de la pantalla guiada. Devuelve (modelo, respuesta)."""
    model = get_garment_model(model_id)

    if not model:
        flash("El modelo de prenda solicitado no existe.", "error")
        return None, redirect(url_for("garment_models_page"))

    if not can_manage_garment_model(model):
        flash(
            "No tiene permisos para preparar la captura de este modelo.",
            "error",
        )
        return None, redirect(url_for("garment_models_page"))

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        flash(
            "La captura solo puede iniciarse con el modelo "
            "aprobado y activo.",
            "error",
        )
        return None, redirect(
            url_for("garment_model_detail", model_id=model_id)
        )

    return model, None


@app.route("/modelos-prenda/<int:model_id>/captura-ia")
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def guided_capture_page(model_id):
    model, blocked = _guarded_model_or_none(model_id)
    if blocked is not None:
        return blocked

    try:
        payload = _ai_capture_status_payload(
            garment_model_id=model_id,
            allowed=True,
        )
    except Exception:
        app.logger.exception(
            "No se pudo leer el estado de la captura guiada "
            f"(garment_model_id={model_id})."
        )
        payload = None

    return render_template(
        "guided_capture.html",
        model=model,
        ai_capture=_json_sanitize(payload) if payload else None,
    )


@app.route(
    "/modelos-prenda/<int:model_id>/captura-ia/video",
    methods=["GET"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def guided_capture_video(model_id):
    model = get_garment_model(model_id)

    if not model or not can_manage_garment_model(model):
        return jsonify({
            "ok": False,
            "error": "No tiene permisos para ver esta cámara.",
        }), 403

    return Response(
        generate_guided_video_feed(model_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/ai/capture/guide/start", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_guide_start():
    body = request.get_json(silent=True) or {}
    raw_model = body.get("garment_model_id")

    try:
        garment_model_id = int(raw_model)
    except (TypeError, ValueError):
        return jsonify({
            "ok": False,
            "error": "garment_model_id inválido.",
        }), 400

    if garment_model_id <= 0:
        return jsonify({
            "ok": False,
            "error": "garment_model_id debe ser positivo.",
        }), 400

    model = get_garment_model(garment_model_id)

    if not model:
        return jsonify({
            "ok": False,
            "error": "El modelo de prenda no existe.",
        }), 404

    if not can_manage_garment_model(model):
        return jsonify({
            "ok": False,
            "error": (
                "No tiene permisos para preparar la captura "
                "de este modelo."
            ),
        }), 403

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        return jsonify({
            "ok": False,
            "error": (
                "La captura solo puede iniciarse con el modelo "
                "aprobado y activo."
            ),
        }), 409

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        row = get_open_capture_session(cur)
        has_prep = _has_preparation_version(cur, garment_model_id)
    finally:
        cur.close()
        conn.close()

    if row is None or int(row["garment_model_id"]) != int(garment_model_id):
        return jsonify({
            "ok": False,
            "error": (
                "Primero inicie la captura de este modelo "
                "desde su ficha."
            ),
        }), 409

    if not has_prep:
        return jsonify({
            "ok": False,
            "error": (
                "Este modelo aún no tiene una sesión de preparación."
            ),
        }), 409

    session_id = int(row["id"])

    try:
        _ai_guided_prepare_runtime(session_id, garment_model_id)
    except AIDomainError as error:
        return jsonify({
            "ok": False,
            "error": humanize_capture_error(error),
        }), 409
    except Exception:
        app.logger.exception(
            "No se pudo preparar la estación guiada "
            f"(garment_model_id={garment_model_id})."
        )
        return jsonify({
            "ok": False,
            "error": (
                "No se pudo iniciar la captura guiada. "
                "Intente de nuevo."
            ),
        }), 500

    ensure_camera_capture_worker()
    started = _ensure_ai_guided_worker()

    status = _ai_capture_status_payload(
        session_id=session_id,
        garment_model_id=garment_model_id,
        allowed=True,
    )
    status = _json_sanitize(status)

    return jsonify({
        "ok": True,
        "worker_started": bool(started),
        "message": "Estación de captura lista.",
        "guide": status.get("guide"),
        "session": status,
    })


@app.route("/api/ai/capture/guide/reset", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_guide_reset():
    """Vuelve a empezar el paso actual (por ejemplo, si la prenda se movió)."""
    if not is_ai_capture_mode_active():
        return jsonify({
            "ok": False,
            "error": "No hay una captura activa.",
        }), 409

    _ai_guided_reset_state()

    return jsonify({
        "ok": True,
        "message": "Posición reiniciada.",
        "guide": _ai_guided_payload(
            AI_GUIDED_RUNTIME.get("garment_model_id")
        ),
    })


def _ai_capture_request_model():
    """
    Body común de los endpoints manuales de la pantalla guiada.

    Devuelve (garment_model_id, model, body, error_response).
    """
    body = request.get_json(silent=True) or {}
    raw_model = body.get("garment_model_id")

    try:
        garment_model_id = int(raw_model)
    except (TypeError, ValueError):
        return None, None, body, (
            jsonify({"ok": False, "error": "garment_model_id inválido."}),
            400,
        )

    if garment_model_id <= 0:
        return None, None, body, (
            jsonify({
                "ok": False,
                "error": "garment_model_id debe ser positivo.",
            }),
            400,
        )

    model = get_garment_model(garment_model_id)

    if not model:
        return None, None, body, (
            jsonify({"ok": False, "error": "El modelo de prenda no existe."}),
            404,
        )

    if not can_manage_garment_model(model):
        return None, None, body, (
            jsonify({
                "ok": False,
                "error": (
                    "No tiene permisos para preparar la captura "
                    "de este modelo."
                ),
            }),
            403,
        )

    if (
        model.get("status") != "APROBADO"
        or int(model.get("active") or 0) != 1
    ):
        return None, None, body, (
            jsonify({
                "ok": False,
                "error": (
                    "La captura solo puede iniciarse con el modelo "
                    "aprobado y activo."
                ),
            }),
            409,
        )

    return garment_model_id, model, body, None


def _ai_open_session_for_model(garment_model_id):
    """Sesión ABIERTA del modelo en la estación (o None)."""
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        row = get_open_capture_session(cur)
    finally:
        cur.close()
        conn.close()

    if row is None:
        return None
    if int(row["garment_model_id"]) != int(garment_model_id):
        return None
    return row


@app.route("/api/ai/capture/guide/manual", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_guide_manual():
    """
    Botón "CAPTURAR IMAGEN": guarda la captura actual.

    Solo responde si la posición es válida ("Lista para capturar");
    en cualquier otro caso devuelve un mensaje accionable.
    """
    garment_model_id, _model, _body, error = _ai_capture_request_model()
    if error is not None:
        return error

    if not is_ai_capture_mode_active():
        return jsonify({
            "ok": False,
            "error": (
                "Primero inicie la preparación de este modelo "
                "desde su ficha."
            ),
        }), 409

    if _ai_open_session_for_model(garment_model_id) is None:
        return jsonify({
            "ok": False,
            "error": (
                "La sesión de preparación no está activa. "
                "Vuelva a iniciarla desde la ficha del modelo."
            ),
        }), 409

    payload, status_code = _ai_guided_manual_capture(garment_model_id)
    return jsonify(_json_sanitize(payload)), status_code


# Decisiones de revisión de la última captura (confirmación manual).
#
# REPETIR: en el flujo activo (pending_token) NO persiste la imagen y
# no cuenta como válida ni como descartada (los contadores no cambian).
# Solo la ruta legacy (image_id, sobre una imagen ya guardada) la marca
# RECHAZADA/MANUAL_REPEAT para que deje de contar como válida.
GUIDED_REVIEW_DECISIONS = {
    "accept": ("ACEPTADA", None),
    "repeat": ("RECHAZADA", "MANUAL_REPEAT"),
    "discard": ("RECHAZADA", "MANUAL_DISCARD"),
}


@app.route("/api/ai/capture/guide/review", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def ai_capture_guide_review():
    """
    Marca la última captura como Aceptada / Repetir / Descartada.

    Los contadores de la sesión son derivados de la BD, así que se
    actualizan solos al cambiar el estado de la imagen.
    """
    garment_model_id, _model, body, error = _ai_capture_request_model()
    if error is not None:
        return error

    decision = str(body.get("decision") or "").strip().lower()
    if decision == "accepted":
        decision = "accept"

    if decision not in GUIDED_REVIEW_DECISIONS:
        return jsonify({
            "ok": False,
            "error": "Decisión de revisión no reconocida.",
        }), 400

    if not is_ai_capture_mode_active():
        return jsonify({
            "ok": False,
            "error": (
                "La sesión de preparación ya no está activa. "
                "No se puede revisar esta imagen."
            ),
        }), 409

    session_row = _ai_open_session_for_model(garment_model_id)
    if session_row is None:
        return jsonify({
            "ok": False,
            "error": (
                "La sesión de preparación ya no está activa. "
                "No se puede revisar esta imagen."
            ),
        }), 409

    session_id = int(session_row["id"])

    pending_token = str(body.get("pending_token") or "").strip()
    if pending_token:
        pending = AI_GUIDED_RUNTIME.get("pending_capture")
        if not pending or pending.get("token") != pending_token:
            return jsonify({
                "ok": False,
                "error": "La vista previa expiró. Capture otra imagen.",
            }), 409

        if decision == "repeat":
            reset_for_next_manual_capture()
            return jsonify({
                "ok": True,
                "decision": "repeat",
                "session": _json_sanitize(
                    _ai_capture_status_payload(
                        garment_model_id=garment_model_id,
                        allowed=True,
                    )
                ),
            })

        candidate = pending["candidate"]
        metrics = dict(pending["decision"])
        metrics["accepted"] = decision == "accept"
        metrics["reject_reason"] = (
            None if decision == "accept" else "MANUAL_DISCARD"
        )
        metrics["quality_score"] = metrics.get("quality_score", 0.0)
        cfg = AI_CAPTURE_RUNTIME["config"] or get_ai_capture_config()
        jpeg_bytes = (
            pending.get("jpeg_bytes")
            if metrics["accepted"] or cfg.get("keep_rejected")
            else None
        )
        record = _ai_capture_persist(metrics, jpeg_bytes, candidate)
        if record.get("skipped"):
            return jsonify({
                "ok": False,
                "error": "La imagen ya existe en la preparación y no se guardó otra vez.",
            }), 409

        # Auditoría conserva que la decisión humana aceptó una toma con
        # advertencias, y las métricas usadas por la guía asistencial.
        conn = db()
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            record_ai_event(
                cur,
                "CAPTURE_IMAGE_REVIEWED",
                actor_id=session.get("user_id"),
                capture_session_id=session_id,
                payload={
                    "decision": decision,
                    "image_id": (record.get("record") or {}).get("id"),
                    "warnings": pending.get("warnings") or [],
                    "frame_sequence": candidate.best_sequence,
                    "frame_age_ms": pending.get("frame_age_ms"),
                    "roi": pending.get("roi"),
                    "bbox": pending.get("bbox"),
                    "coverage": metrics.get("coverage"),
                    "sharpness": metrics.get("sharpness"),
                    "segmentation": pending.get("segmentation"),
                },
            )
            conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            app.logger.exception("No se pudo registrar la revisión humana.")
        finally:
            cur.close()
            conn.close()

        reset_for_next_manual_capture()
        if decision == "discard":
            status = _ai_capture_status_payload(
                garment_model_id=garment_model_id,
                allowed=True,
            )
            return jsonify({
                "ok": True,
                "decision": decision,
                "image_id": (record.get("record") or {}).get("id"),
                "session": _json_sanitize(status),
            })

        status = _ai_capture_status_payload(
            garment_model_id=garment_model_id,
            allowed=True,
        )
        return jsonify({
            "ok": True,
            "decision": decision,
            "image_id": (record.get("record") or {}).get("id"),
            "preview_url": url_for(
                "guided_capture_image",
                model_id=garment_model_id,
                image_id=(record.get("record") or {}).get("id"),
            ),
            "session": _json_sanitize(status),
        })

    raw_image = body.get("image_id")
    try:
        image_id = int(raw_image)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "image_id inválido."}), 400
    if image_id <= 0:
        return jsonify({"ok": False, "error": "image_id debe ser positivo."}), 400

    new_status, new_reason = GUIDED_REVIEW_DECISIONS[decision]

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        conn.start_transaction()
        cur.execute(
            """
            SELECT id, capture_session_id, status, reject_reason, sha256
            FROM ai_training_images
            WHERE id = %s AND garment_model_id = %s
            FOR UPDATE
            """,
            (image_id, garment_model_id),
        )
        image = cur.fetchone()

        if image is None:
            conn.rollback()
            return jsonify({
                "ok": False,
                "error": "La imagen ya no existe en esta sesión.",
            }), 404

        if (
            image["capture_session_id"] is None
            or int(image["capture_session_id"]) != session_id
        ):
            conn.rollback()
            return jsonify({
                "ok": False,
                "error": (
                    "La imagen no pertenece a la sesión activa "
                    "de preparación."
                ),
            }), 409

        cur.execute(
            """
            UPDATE ai_training_images
            SET status = %s, reject_reason = %s
            WHERE id = %s
            """,
            (new_status, new_reason, image_id),
        )
        conn.commit()
    except AIDomainError as error:
        if conn.in_transaction:
            conn.rollback()
        return jsonify({
            "ok": False,
            "error": humanize_capture_error(error),
        }), 409
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        app.logger.exception(
            "No se pudo revisar la captura "
            f"(image_id={image_id})."
        )
        return jsonify({
            "ok": False,
            "error": "No se pudo guardar la revisión. Intente de nuevo.",
        }), 500
    finally:
        cur.close()
        conn.close()

    # Los hashes aceptados mandan para detectar duplicados: se
    # recargan desde la verdad primaria (la tabla).
    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        hashes = load_session_accepted_hashes(cur, session_id)
        AI_CAPTURE_RUNTIME["known_sha256"] = set(hashes["sha256"])
    finally:
        cur.close()
        conn.close()

    status = _ai_capture_status_payload(
        garment_model_id=garment_model_id,
        allowed=True,
    )

    return jsonify({
        "ok": True,
        "decision": decision,
        "image_id": image_id,
        "status": new_status,
        "preview_url": url_for(
            "guided_capture_image",
            model_id=garment_model_id,
            image_id=image_id,
        ),
        "guide": _ai_guided_payload(garment_model_id),
        "session": _json_sanitize(status),
    })


@app.route(
    "/modelos-prenda/<int:model_id>/captura-ia/imagen/<int:image_id>",
    methods=["GET"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def guided_capture_image(model_id, image_id):
    """Vista previa de una captura de la sesión (para revisarla)."""
    model = get_garment_model(model_id)

    if not model or not can_manage_garment_model(model):
        return jsonify({
            "ok": False,
            "error": "No tiene permisos para ver esta imagen.",
        }), 403

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT image_path
            FROM ai_training_images
            WHERE id = %s AND garment_model_id = %s
            """,
            (image_id, model_id),
        )
        row = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if row is None:
        return jsonify({
            "ok": False,
            "error": "La imagen ya no existe.",
        }), 404

    try:
        path = resolve_under_root(
            get_ai_artifacts_root(),
            row["image_path"],
        )
    except AIDomainError:
        return jsonify({
            "ok": False,
            "error": "Ruta de imagen no válida.",
        }), 404

    if not path.is_file():
        return jsonify({
            "ok": False,
            "error": "La imagen no está disponible en este momento.",
        }), 404

    return send_file(path, mimetype="image/jpeg", conditional=True)


def _ai_guided_close(model_id, action):
    model, blocked = _guarded_model_or_none(model_id)
    if blocked is not None:
        return blocked

    conn = db()
    cur = conn.cursor(dictionary=True)
    try:
        row = get_open_capture_session(cur)
        counts = None
        if row is not None and int(row["garment_model_id"]) == int(
            model_id
        ):
            counts = count_session_images(cur, int(row["id"]))
    finally:
        cur.close()
        conn.close()

    if row is None or int(row["garment_model_id"]) != int(model_id):
        flash("No hay una captura activa de este modelo.", "message")
        return redirect(
            url_for("garment_model_detail", model_id=model_id)
        )

    cfg = get_ai_capture_config()
    counts = counts or {}
    accepted = int(counts.get("accepted_images") or 0)
    rejected = int(counts.get("rejected_images") or 0)

    # Finalizar requiere el mínimo configurable; no basta una sola toma.
    min_images = int(cfg["min_images"])
    if action == "stop" and accepted < min_images:
        missing = min_images - accepted
        flash(
            f"Faltan {missing} imágenes para alcanzar el mínimo de "
            f"{min_images} imágenes válidas antes de finalizar.",
            "error",
        )
        return redirect(
            url_for("guided_capture_page", model_id=model_id)
        )

    # _ai_capture_finish trabaja sobre la sesión abierta; aquí ya se
    # verificó que es la de este modelo.
    result = _ai_capture_finish(action)
    if isinstance(result, tuple):
        response, status_code = result
    else:
        response, status_code = result, 200

    try:
        payload = json.loads(response.get_data(as_text=True))
    except Exception:
        payload = {}

    _stop_ai_guided_worker()
    _ai_guided_reset_state()

    if not payload.get("ok"):
        flash(
            payload.get("error")
            or "No se pudo cerrar la captura. Intente de nuevo.",
            "error",
        )
        return redirect(
            url_for("garment_model_detail", model_id=model_id)
        )

    if action == "stop":
        session_payload = payload.get("session") or {}
        accepted = int(session_payload.get("accepted_count") or 0)
        rejected = int(session_payload.get("rejected_count") or 0)
        min_images = int(
            session_payload.get("min_count") or cfg["min_images"]
        )
        missing = max(0, min_images - accepted)
        target = int(session_payload.get("target_count") or cfg["target_images"])
        state_label = (
            session_payload.get("ui_state_label") or "COMPLETADA"
        )
        training = (
            "Objetivo recomendado alcanzado."
            if accepted >= target
            else "Se alcanzó el mínimo. Se recomiendan "
            f"{target} imágenes."
            if missing == 0
            else (
                "No listo para entrenamiento: faltan "
                f"{missing} imágenes para alcanzar el mínimo de "
                f"{min_images}."
            )
        )
        flash(
            f"Captura completada. {accepted} imágenes normales. "
            f"{rejected} descartadas. Sesión: {state_label}. "
            f"{training}",
            "success",
        )
    else:
        flash("Sesión de captura cancelada.", "message")

    return redirect(
        url_for("garment_model_detail", model_id=model_id)
    )


@app.route(
    "/modelos-prenda/<int:model_id>/captura-ia/finalizar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def guided_capture_finish(model_id):
    return _ai_guided_close(model_id, "stop")


@app.route(
    "/modelos-prenda/<int:model_id>/captura-ia/cancelar",
    methods=["POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def guided_capture_cancel(model_id):
    return _ai_guided_close(model_id, "cancel")


if __name__ == "__main__":
    init_db()
    # Exactamente un lector RTSP continuo (no uno por request).
    ensure_camera_capture_worker()
    # app.run(debug=True, host="127.0.0.1", port=5000)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
