from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import quote
from patchcore_inference import PatchCoreInspector
import os
import time
import uuid
import threading
from functools import wraps
import io
from zoneinfo import ZoneInfo

import mysql.connector


load_dotenv()
# RTSP mediante TCP: más estable que UDP para OpenCV
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"
)

PATCHCORE_CKPT = os.getenv("PATCHCORE_CKPT", "").strip()

PATCHCORE_PIXEL_THRESHOLD = float(
    os.getenv("PATCHCORE_PIXEL_THRESHOLD", "0.72")
)

PATCHCORE_MAX_BOX_RATIO = float(
    os.getenv("PATCHCORE_MAX_BOX_RATIO", "0.15")
)

patchcore_inspector = None

if PATCHCORE_CKPT:
    try:
        patchcore_inspector = PatchCoreInspector(PATCHCORE_CKPT)
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

camera_lock = threading.Lock()
latest_camera_frame = None
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

AUTO_INSPECTION_ENABLED = False
AUTO_THREAD = None
AUTO_LAST_RESULT = None
AUTO_LAST_ERROR = None
AUTO_LAST_CAPTURE_TIME = 0

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


def get_camera():
    """
    Devuelve una unica instancia de camara.

    La direccion RTSP se construye usando la IP
    localizada actualmente por el sistema.
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

        camera_capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

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


def read_camera_frame():
    """
    Lee un frame y mantiene el estado operativo de la cámara.

    Una pérdida de señal nunca registra una inspección.
    Si ocurre durante el modo automático, este se detiene
    después de varios fallos consecutivos.
    """
    global camera_capture
    global latest_camera_frame

    global CAMERA_CONNECTED
    global CAMERA_LAST_OK_AT
    global CAMERA_LAST_ERROR
    global CAMERA_FAILURE_COUNT

    global AUTO_INSPECTION_ENABLED
    global AUTO_LAST_ERROR

    if cv2 is None:
        CAMERA_CONNECTED = False
        CAMERA_LAST_ERROR = "OpenCV no está disponible."
        return False, None

    with camera_lock:
        cap = get_camera()

        if cap is None:
            CAMERA_CONNECTED = False
            CAMERA_FAILURE_COUNT += 1

            if not CAMERA_LAST_ERROR:
                CAMERA_LAST_ERROR = "Cámara sin señal."

            if (
                AUTO_INSPECTION_ENABLED
                and CAMERA_FAILURE_COUNT >= CAMERA_FAILURE_LIMIT
            ):
                AUTO_INSPECTION_ENABLED = False
                AUTO_LAST_ERROR = (
                    "Inspección automática detenida: "
                    "la cámara perdió la señal."
                )

            return False, None

        ok, frame = cap.read()

        if not ok or frame is None:
            try:
                cap.release()
            except Exception:
                pass

            camera_capture = None
            CAMERA_CONNECTED = False
            CAMERA_FAILURE_COUNT += 1
            CAMERA_LAST_ERROR = (
                "No se está recibiendo video de la cámara."
            )

            if (
                AUTO_INSPECTION_ENABLED
                and CAMERA_FAILURE_COUNT >= CAMERA_FAILURE_LIMIT
            ):
                AUTO_INSPECTION_ENABLED = False
                AUTO_LAST_ERROR = (
                    "Inspección automática detenida: "
                    "la cámara perdió la señal."
                )

            return False, None

        latest_camera_frame = frame.copy()

        CAMERA_CONNECTED = True
        CAMERA_FAILURE_COUNT = 0
        CAMERA_LAST_ERROR = None
        CAMERA_LAST_OK_AT = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return True, frame



def reset_camera_connection():
    """
    Libera la conexion RTSP actual para que la siguiente lectura
    abra una sesion nueva con la camara.
    No modifica lotes ni registra inspecciones.
    """
    global camera_capture
    global latest_camera_frame
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
        latest_camera_frame = None

        CAMERA_CONNECTED = False
        CAMERA_LAST_ERROR = (
            "Reconectando con la c\u00e1mara de inspecci\u00f3n."
        )
        CAMERA_LAST_CONNECT_ATTEMPT = 0.0
        CAMERA_FAILURE_COUNT = 0



def camera_reconnect_worker():
    """
    Recuperacion persistente.

    Primero intenta la IP conocida. Si falla varias
    veces, busca automaticamente la camara en la LAN
    y adopta la nueva IP cuando encuentra video real.
    """
    global CAMERA_RECONNECTING
    global CAMERA_CURRENT_IP
    global CAMERA_LAST_CONNECT_ATTEMPT

    app.logger.warning(
        "[CAMARA] Reconexion automatica iniciada."
    )

    direct_failures = 0

    try:
        reset_camera_connection()

        while True:
            ok, frame = read_camera_frame()

            if (
                ok
                and frame is not None
            ):
                app.logger.info(
                    "[CAMARA] Conexion recuperada "
                    "automaticamente en "
                    f"{CAMERA_CURRENT_IP}."
                )
                return

            direct_failures += 1

            # No escanear toda la LAN en cada intento.
            # Tras 2 fallos directos se realiza busqueda
            # y luego se repite periodicamente.
            if (
                CAMERA_DISCOVERY_ENABLED
                and (
                    direct_failures == 2
                    or direct_failures % 5 == 0
                )
            ):
                app.logger.warning(
                    "[CAMARA] IP actual sin respuesta. "
                    "Buscando camara en la red local."
                )

                discovered_ip = (
                    discover_camera_ip()
                )

                if (
                    discovered_ip
                    and discovered_ip
                    != CAMERA_CURRENT_IP
                ):
                    old_ip = (
                        CAMERA_CURRENT_IP
                    )

                    CAMERA_CURRENT_IP = (
                        discovered_ip
                    )

                    CAMERA_LAST_CONNECT_ATTEMPT = (
                        0.0
                    )

                    reset_camera_connection()

                    app.logger.warning(
                        "[CAMARA] Cambio automatico "
                        f"de IP: {old_ip} -> "
                        f"{CAMERA_CURRENT_IP}"
                    )

                    direct_failures = 0

                    continue

            time.sleep(
                max(
                    float(
                        CAMERA_RETRY_SECONDS
                    ),
                    1.0,
                )
            )

    finally:
        with CAMERA_RECONNECT_LOCK:
            CAMERA_RECONNECTING = False


def ensure_camera_reconnect_worker():
    """
    Inicia la recuperacion automatica solamente
    cuando no existe otro worker activo.
    """
    global CAMERA_RECONNECTING

    if CAMERA_CONNECTED:
        return False

    with CAMERA_RECONNECT_LOCK:
        if CAMERA_RECONNECTING:
            return False

        CAMERA_RECONNECTING = True

        thread = threading.Thread(
            target=camera_reconnect_worker,
            daemon=True,
            name="camera-reconnect-worker",
        )

        thread.start()

    return True


def get_camera_status():
    """
    Devuelve el estado operativo de la camara.
    Si esta desconectada, garantiza que exista
    un worker de recuperacion en segundo plano.
    """
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

    ensure_camera_reconnect_worker()

    if CAMERA_RECONNECTING:
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
    Stream MJPEG.

    Cuando la camara esta desconectada, solamente el
    worker de reconexion intenta abrir RTSP. El stream
    muestra un placeholder y evita crear conexiones
    paralelas que interfieran con la recuperacion.
    """
    while True:

        if not CAMERA_CONNECTED:
            ensure_camera_reconnect_worker()

            frame_to_send = (
                make_camera_error_frame(
                    "RECONECTANDO CAMARA"
                )
            )

        else:
            ok, frame = (
                read_camera_frame()
            )

            if ok:
                frame_to_send = (
                    draw_inspection_overlay(
                        frame
                    )
                )

            else:
                ensure_camera_reconnect_worker()

                frame_to_send = (
                    make_camera_error_frame(
                        "RECONECTANDO CAMARA"
                    )
                )

        jpg = encode_jpeg(
            frame_to_send
        )

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

    ok, frame = read_camera_frame()

    if not ok:
        return None, None

    return save_frame_to_static(frame)


def register_inspection_from_frame(
    frame,
    notes="Registro generado por estación de inspección.",
):
    """
    Ejecuta la detección, guarda la evidencia y registra la blusa
    dentro del lote activo.

    Los contadores del lote se sincronizan con las inspecciones
    realmente almacenadas para evitar inconsistencias.
    """
    global AUTO_INSPECTION_ENABLED

    if frame is None:
        raise ValueError(
            "No se recibió imagen de cámara para registrar la inspección."
        )

    img_path, img_rel = save_frame_to_static(frame)

    status, defect, conf, zone, result_rel = detect_defect(img_path)

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

    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

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

    print(
        "[AUTO] Modo automático iniciado. "
        "Esperando entrada de una blusa."
    )

    while AUTO_INSPECTION_ENABLED:
        try:
            ok, frame = read_camera_frame()

            if not ok or frame is None:
                AUTO_LAST_ERROR = (
                    "Cámara no disponible."
                )
                time.sleep(0.5)
                continue

            # -------------------------------------------------
            # Medir presencia de prenda.
            #
            # Se reutiliza EXACTAMENTE la segmentación de la
            # blusa que usa PatchCore.
            # -------------------------------------------------
            try:
                garment_mask = create_garment_mask(
                    frame
                )

                roi_x1, roi_y1, roi_x2, roi_y2 = (
                    get_roi_bounds(frame)
                )

                roi_mask = garment_mask[
                    roi_y1:roi_y2,
                    roi_x1:roi_x2,
                ]

                roi_area = float(
                    max(
                        1,
                        roi_mask.shape[0]
                        * roi_mask.shape[1],
                    )
                )

                coverage = (
                    cv2.countNonZero(roi_mask)
                    / roi_area
                )

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

                    print(
                        "[AUTO] Entrada de prenda detectada."
                    )

                tracked_frames += 1

                # Conservar siempre el frame donde la prenda
                # ocupa mayor superficie dentro del ROI.
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_frame = frame.copy()

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
                        f"({best_coverage * 100:.2f}%)."
                    )

                    result = (
                        register_inspection_from_frame(
                            best_frame,
                            notes=(
                                "Registro automático generado "
                                "por presencia de prenda en "
                                "cinta transportadora."
                            ),
                        )
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
                        f"{result.get('defect_type')}"
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
    Obtiene la silueta completa de la blusa rosa talla S.

    El color rosa se utiliza solamente como semilla para encontrar
    la prenda. Después se rellena su contorno exterior para conservar
    cualquier alteración visual situada sobre la tela, aunque tenga
    un color diferente.
    """
    if image is None:
        raise ValueError(
            "No se recibió una imagen válida."
        )

    height, width = image.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = (
        get_roi_bounds(image)
    )

    roi = image[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]

    if roi is None or roi.size == 0:
        raise ValueError(
            "El ROI de inspección está vacío."
        )

    roi_height, roi_width = roi.shape[:2]
    roi_area = float(
        roi_height * roi_width
    )

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB,
    )

    saturation = hsv[:, :, 1]
    lab_a = lab[:, :, 1]

    # Semilla: tela rosa.
    seed = np.where(
        (saturation >= 30)
        & (lab_a >= 136),
        255,
        0,
    ).astype(np.uint8)

    # Cerrar discontinuidades producidas por manchas,
    # reflejos, costuras y pequeños huecos en la tela.
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (31, 31),
        ),
        iterations=1,
    )

    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        ),
        iterations=1,
    )

    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        np.ones(
            (5, 5),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            seed,
            connectivity=8,
        )
    )

    if count <= 1:
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    candidates = []

    for label in range(1, count):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < roi_area * 0.04:
            continue

        cx, cy = centroids[label]

        dx = abs(
            cx - roi_width / 2.0
        ) / max(
            1.0,
            roi_width,
        )

        dy = abs(
            cy - roi_height / 2.0
        ) / max(
            1.0,
            roi_height,
        )

        score = (
            area / roi_area
            - dx * 0.20
            - dy * 0.10
        )

        candidates.append(
            (
                score,
                label,
            )
        )

    if not candidates:
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected_label = (
        candidates[0][1]
    )

    component = np.where(
        labels == selected_label,
        255,
        0,
    ).astype(np.uint8)

    # Recuperar el CONTORNO EXTERIOR de la prenda.
    #
    # Es la diferencia fundamental respecto del algoritmo anterior:
    # los huecos internos de otro color ya no desaparecen.
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    garment_contour = max(
        contours,
        key=cv2.contourArea,
    )

    garment_roi = np.zeros(
        (roi_height, roi_width),
        dtype=np.uint8,
    )

    cv2.drawContours(
        garment_roi,
        [garment_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    # Ligero cierre del borde exterior sin convertirlo
    # en un rectángulo ni en un convex hull.
    garment_roi = cv2.morphologyEx(
        garment_roi,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        ),
        iterations=1,
    )

    garment_roi = cv2.dilate(
        garment_roi,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    coverage = (
        cv2.countNonZero(
            garment_roi
        )
        / roi_area
    )

    print(
        "[SEGMENTACION] "
        f"Cobertura silueta: "
        f"{coverage * 100:.2f}%"
    )

    if coverage < 0.20:
        raise RuntimeError(
            "La silueta detectada es demasiado pequeña."
        )

    if coverage > 0.75:
        raise RuntimeError(
            "La silueta detectada es demasiado grande."
        )

    garment_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    garment_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ] = garment_roi

    return garment_mask




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


def detect_defect(image_path):
    # ========================================================
    # 1. PATCHCORE: detector principal
    # ========================================================
    if patchcore_inspector is not None and cv2 is not None:
        try:
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
                raise RuntimeError(
                    "No se pudo separar la blusa del fondo."
                )

            roi_x1, roi_y1, roi_x2, roi_y2 = (
                get_roi_bounds(image)
            )

            roi_image = image[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]

            roi_garment_mask = garment_mask_full[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]

            # Fondo blanco uniforme.
            # PatchCore recibe ?nicamente la blusa.
            patchcore_input = np.full_like(
                roi_image,
                255,
            )

            patchcore_input[
                roi_garment_mask > 0
            ] = roi_image[
                roi_garment_mask > 0
            ]

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

            try:
                print(
                    "[PATCHCORE] Esperando acceso exclusivo "
                    "al motor de inferencia."
                )

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
            finally:
                try:
                    roi_path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            is_anomaly = bool(
                prediction["is_anomaly"]
            )

            raw_score = float(
                prediction["score"]
            )

            # Algunos modelos devuelven 0-1 y otros una escala mayor.
            confidence = (
                raw_score * 100
                if raw_score <= 1
                else raw_score
            )

            confidence = round(
                max(0.0, min(confidence, 100.0)),
                2,
            )

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

            print(
                f"[PATCHCORE] Estado={status}, "
                f"pred_label={int(is_anomaly)}, "
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
        cv2.imwrite(str(result_path), annotated)

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
            return status, defect_type, round(confidence, 2), zone, f"results/{result_name}"

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

    cv2.imwrite(str(result_path), original)
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

    return render_template(
        "batch_review.html",
        batch=batch,
        alerts=alerts,
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

        ok, frame = read_camera_frame()

        if not ok:
            return jsonify({
                "ok": False,
                "message": (
                    "No se pudo leer la cámara. "
                    "No se guardó ningún registro."
                ),
            }), 503

        result = register_inspection_from_frame(
            frame,
            notes=(
                "Registro manual generado desde "
                "estación de inspección."
            ),
        )

        AUTO_LAST_RESULT = result
        AUTO_LAST_ERROR = None

        return jsonify({
            "ok": True,
            "message": (
                "Inspección manual registrada correctamente."
            ),
            "result": result,
        })

    except Exception as e:
        AUTO_LAST_ERROR = str(e)

        return jsonify({
            "ok": False,
            "message": (
                "Error al ejecutar la inspección manual."
            ),
            "detail": str(e),
        }), 500




@app.route("/api/station/camera/reconnect", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_camera_reconnect():
    started = (
        ensure_camera_reconnect_worker()
    )

    if not started:
        return jsonify({
            "ok": True,
            "message": (
                "La reconexion de la camara "
                "ya esta en curso."
            ),
            "camera": get_camera_status(),
        }), 202

    return jsonify({
        "ok": True,
        "message": (
            "Reconexion de camara iniciada."
        ),
        "camera": get_camera_status(),
    }), 202


@app.route("/api/station/auto/start", methods=["POST"])
@login_required
@role_required(ROLE_ADMIN, ROLE_QUALITY_MANAGER)
def station_auto_start():
    global AUTO_INSPECTION_ENABLED
    global AUTO_THREAD

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
    })



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

        status, defect, conf, zone, result_rel = detect_defect(img_path)

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
    show_mode = request.args.get("show", "").strip().lower()

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
    ):
        detail_conditions = list(conditions)
        detail_params = list(params)

        if show_mode == "alerts":
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
            LIMIT 100
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


@app.route("/informes")
@login_required
def informes():
    data = get_informe_data()

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

    from openpyxl import Workbook
    from openpyxl.styles import (
        Alignment,
        Border,
        Font,
        PatternFill,
        Side,
    )
    from openpyxl.utils import get_column_letter

    data = get_informe_data()

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

    summary_sheet = workbook.active

    summary_sheet.title = (
        "Resumen ejecutivo"
    )

    summary_sheet.sheet_view.showGridLines = False

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
            "APTAS CONFIRMADAS",
            approval_rate,
            (
                f"{passed} de "
                f"{inspected} prendas"
            ),
            "0.0%",
        ),
        (
            "RECHAZO CONFIRMADO",
            rejection_rate,
            (
                f"{rejected} de "
                f"{inspected} prendas"
            ),
            "0.0%",
        ),
        (
            "ALERTAS REVISADAS",
            reviewed_rate,
            (
                f"{pending} pendientes "
                f"de {alerts} alertas"
            ),
            "0.0%",
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
    # LECTURA RAPIDA
    # --------------------------------------------------------

    summary_sheet.merge_cells(
        "A19:H19"
    )

    summary_sheet["A19"] = (
        "Lectura r\u00e1pida"
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

    production_main = (
        f"{inspected} prendas"
    )

    production_detail = (
        f"{batches_count} "
        + (
            "lote con actividad."
            if batches_count == 1
            else "lotes con actividad."
        )
    )

    quality_main = (
        f"{approval_rate:.1%} aptas"
    )

    quality_detail = (
        f"{rejection_rate:.1%} de rechazo "
        "confirmado."
    )

    review_main = (
        f"{pending_alert_rate:.1%} pendiente"
    )

    review_detail = (
        f"{pending} de {alerts} alertas "
        "todav\u00eda requieren revisi\u00f3n."
    )

    best_model = insights[
        "best_model"
    ]

    if best_model:
        model_main = (
            best_model["code"]
        )

        model_detail = (
            f"{best_model['name']} | "
            f"{best_model['rejection_rate']:.1f}% "
            "de rechazo confirmado."
        )

    else:
        model_main = (
            "A\u00fan no definido"
        )

        model_detail = (
            "Se necesita completar la revisi\u00f3n "
            "antes de comparar el rendimiento "
            "definitivo de los modelos."
        )

    quick_cards = [
        (
            "PRODUCCI\u00d3N",
            production_main,
            production_detail,
        ),
        (
            "CALIDAD",
            quality_main,
            quality_detail,
        ),
        (
            "REVISI\u00d3N",
            review_main,
            review_detail,
        ),
        (
            "MODELO DESTACADO",
            model_main,
            model_detail,
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

    from datetime import datetime as dt_datetime
    from zoneinfo import ZoneInfo
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import (
        TA_CENTER,
        TA_LEFT,
    )
    from reportlab.lib.pagesizes import (
        A4,
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

    buffer = io.BytesIO()

    page_size = landscape(A4)

    margin_x = 14 * mm

    content_width = (
        page_size[0]
        - (2 * margin_x)
    )

    document = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=margin_x,
        leftMargin=margin_x,
        topMargin=12 * mm,
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
                "Defectos confirmados",
                str(
                    summary["rejected"]
                    or 0
                ),
            ),
            (
                "Tasa de rechazo",
                (
                    f"{float(summary['rejection_rate'] or 0):.1f}%"
                ),
            ),
        ])
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

    document.build(
        elements,
        onFirstPage=add_page_footer,
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
    return fetch_all(
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


@app.route(
    "/modelos-prenda/nuevo",
    methods=["GET", "POST"],
)
@login_required
@role_required(ROLE_ADMIN, ROLE_MODEL_MANAGER)
def garment_model_create():
    if request.method == "POST":
        code = request.form.get("code", "").strip().upper()
        name = request.form.get("name", "").strip()
        color = request.form.get("color", "").strip()
        description = request.form.get(
            "description",
            "",
        ).strip()

        if not code:
            code = datetime.now().strftime(
                "BLUSA-%Y%m%d-%H%M%S"
            )

        code = code.replace(" ", "-")

        if not all(
            character.isalnum()
            or character in {"-", "_"}
            for character in code
        ):
            flash(
                "El c\u00f3digo solo puede contener letras, "
                "n\u00fameros, guiones y guion bajo.",
                "error",
            )
            return render_template(
                "garment_model_form.html"
            )

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

        existing = fetch_one(
            """
            SELECT id
            FROM garment_models
            WHERE code = %s
            """,
            (code,),
        )

        if existing:
            flash(
                "Ya existe un modelo con ese c\u00f3digo.",
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
        for version in ai_versions
    )

    can_manage = can_manage_garment_model(model)

    can_prepare_ai = (
        role in {ROLE_ADMIN, ROLE_MODEL_MANAGER}
        and can_manage
        and model.get("status") == "APROBADO"
        and int(model.get("active") or 0) == 1
    )

    return render_template(
        "garment_model_detail.html",
        model=model,
        images=images,
        role=role,
        can_manage=can_manage,
        ai_versions=ai_versions,
        ai_in_progress=ai_in_progress,
        can_prepare_ai=can_prepare_ai,
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
    from flask import session as flask_session

    flask_session.pop(
        "_flashes",
        None,
    )


if __name__ == "__main__":
    init_db()
    # app.run(debug=True, host="127.0.0.1", port=5000)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
