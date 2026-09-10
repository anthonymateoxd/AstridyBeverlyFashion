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
app.secret_key = os.environ.get("SECRET_KEY", "***REDACTED***")
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
            role VARCHAR(50) NOT NULL DEFAULT 'operario',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

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
        "batch_position",
        "batch_position INT NULL"
    )
    ensure_inspection_column(
        "ai_decision",
        "ai_decision VARCHAR(30) NULL"
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
        "idx_batch_position",
        "INDEX idx_batch_position (batch_id, batch_position)"
    )

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

    cur.execute("SELECT id FROM users WHERE username = %s", (admin_username,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
            (admin_username, generate_password_hash(admin_password), "administrador"),
        )

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


def get_active_batch():
    """
    Devuelve el lote actualmente en inspección y sus contadores.
    Solo puede existir un lote operativo a la vez.
    """
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
        WHERE b.status = 'EN_INSPECCION'
        GROUP BY b.id
        ORDER BY b.id DESC
        LIMIT 1
        """
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
            COUNT(i.id) AS processed_quantity,
            COALESCE(
                SUM(CASE WHEN i.ai_decision = 'NORMAL' THEN 1 ELSE 0 END),
                0
            ) AS auto_approved,
            COALESCE(
                SUM(CASE WHEN i.ai_decision = 'ANOMALIA' THEN 1 ELSE 0 END),
                0
            ) AS alerts
        FROM batches b
        LEFT JOIN inspections i ON i.batch_id = b.id
        GROUP BY b.id
        ORDER BY b.id DESC
        """
    )



# ============================================================
# AUTENTICACIÓN
# ============================================================

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


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
# Detecci?n autom?tica de presencia de la blusa.
#
# La blusa completa suele ocupar aproximadamente 50-52 % del ROI.
# El autom?tico espera que entre suficientemente antes de capturar.
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
    Devuelve una única instancia de cámara para todo el sistema.

    Cuando la cámara no está disponible, limita los intentos de
    reconexión para evitar abrir RTSP continuamente.
    """
    global camera_capture
    global CAMERA_LAST_CONNECT_ATTEMPT
    global CAMERA_LAST_ERROR

    if cv2 is None:
        CAMERA_LAST_ERROR = "OpenCV no está disponible."
        return None

    if camera_capture is not None and camera_capture.isOpened():
        return camera_capture

    now = time.time()

    if (
        CAMERA_LAST_CONNECT_ATTEMPT > 0
        and now - CAMERA_LAST_CONNECT_ATTEMPT < CAMERA_RETRY_SECONDS
    ):
        return None

    CAMERA_LAST_CONNECT_ATTEMPT = now

    source = parse_camera_source(CAMERA_SOURCE)

    try:
        if isinstance(source, int):
            if os.name == "nt":
                camera_capture = cv2.VideoCapture(
                    source,
                    cv2.CAP_DSHOW,
                )

                if not camera_capture.isOpened():
                    camera_capture.release()
                    camera_capture = cv2.VideoCapture(source)
            else:
                camera_capture = cv2.VideoCapture(source)

        else:
            capture_params = []

            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture_params.extend([
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    CAMERA_OPEN_TIMEOUT_MS,
                ])

            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
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

        camera_capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )
        camera_capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )
        camera_capture.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS,
        )

        if not camera_capture.isOpened():
            try:
                camera_capture.release()
            except Exception:
                pass

            camera_capture = None
            CAMERA_LAST_ERROR = (
                "No se pudo establecer conexión con la cámara."
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
            f"Error al conectar con la cámara: {error}"
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
    Reconecta RTSP en segundo plano para no bloquear
    ninguna solicitud HTTP ni la interfaz.
    """
    global CAMERA_RECONNECTING

    try:
        reset_camera_connection()

        # Intenta recuperar una primera imagen.
        # Los timeouts evitan una espera indefinida.
        read_camera_frame()

    finally:
        CAMERA_RECONNECTING = False


def get_camera_status():
    """
    Estado de cámara consumido por la interfaz de estación.
    """
    if CAMERA_CONNECTED:
        return {
            "connected": True,
            "reconnecting": CAMERA_RECONNECTING,
            "state": "CONNECTED",
            "message": "Cámara conectada y transmitiendo.",
            "last_ok_at": CAMERA_LAST_OK_AT,
        }

    if CAMERA_LAST_CONNECT_ATTEMPT == 0:
        return {
            "connected": False,
            "reconnecting": CAMERA_RECONNECTING,
            "state": "CHECKING",
            "message": "Comprobando conexión con la cámara.",
            "last_ok_at": CAMERA_LAST_OK_AT,
        }

    return {
        "connected": False,
        "reconnecting": CAMERA_RECONNECTING,
        "state": "DISCONNECTED",
        "message": (
            CAMERA_LAST_ERROR
            or "No se recibe señal de la cámara."
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
    Stream MJPEG usado por dashboard, inspección y estación automática.
    Si la cámara falla, mantiene la página viva mostrando un placeholder.
    """
    while True:
        ok, frame = read_camera_frame()

        if ok:
            frame_to_send = draw_inspection_overlay(frame)
        else:
            frame_to_send = make_camera_error_frame()

        jpg = encode_jpeg(frame_to_send)

        if jpg is None:
            jpg = generate_placeholder_frame("ERROR DE VIDEO")

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
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
    notes="Registro generado por estaci?n de inspecci?n.",
):
    """
    Ejecuta la detecci?n, guarda la evidencia y registra la blusa
    dentro del lote activo.

    Los contadores del lote se sincronizan con las inspecciones
    realmente almacenadas para evitar inconsistencias.
    """
    global AUTO_INSPECTION_ENABLED

    if frame is None:
        raise ValueError(
            "No se recibi? imagen de c?mara para registrar la inspecci?n."
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
                batch_position,
                ai_decision,
                review_status,
                audit_selected
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s
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
    Inspecci?n autom?tica orientada a cinta transportadora.

    Flujo:
    1. Espera el ?rea vac?a.
    2. Detecta entrada de una blusa mediante su m?scara.
    3. Sigue varios frames mientras entra.
    4. Conserva el frame con mayor cobertura.
    5. Ejecuta exactamente el mismo pipeline de IA que el manual.
    6. No vuelve a registrar hasta que esa blusa haya salido.
    """
    global AUTO_INSPECTION_ENABLED
    global AUTO_LAST_RESULT
    global AUTO_LAST_ERROR
    global AUTO_LAST_CAPTURE_TIME

    # Estado de la prenda que actualmente atraviesa la estaci?n.
    tracking = False
    waiting_for_exit = False

    present_frames = 0
    exit_frames = 0
    tracked_frames = 0

    best_frame = None
    best_coverage = 0.0

    print(
        "[AUTO] Modo autom?tico iniciado. "
        "Esperando entrada de una blusa."
    )

    while AUTO_INSPECTION_ENABLED:
        try:
            ok, frame = read_camera_frame()

            if not ok or frame is None:
                AUTO_LAST_ERROR = (
                    "C?mara no disponible."
                )
                time.sleep(0.5)
                continue

            # -------------------------------------------------
            # Medir presencia de prenda.
            #
            # Se reutiliza EXACTAMENTE la segmentaci?n de la
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
                # Una silueta demasiado peque?a normalmente
                # significa que la prenda apenas est? entrando,
                # saliendo, o que el ?rea est? vac?a.
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
                # Si apenas comenz? una detecci?n pero no lleg?
                # a ser una prenda v?lida, descartarla.
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
            # Si empieza a salir, usamos el m?ximo ya observado.
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
                # algunos frames: eso hac?a que la blusa se
                # inspeccionara cuando todav?a estaba entrando.
                #
                # Se captura cuando:
                # 1. ya alcanz? cobertura de prenda completa y
                #    comienza a salir, o
                # 2. lleva suficientes frames pr?cticamente
                #    centrada en el ?rea.
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
                                "Registro autom?tico generado "
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
                        "[AUTO] Inspecci?n registrada: "
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
    la prenda. Despu?s se rellena su contorno exterior para conservar
    cualquier alteraci?n visual situada sobre la tela, aunque tenga
    un color diferente.
    """
    if image is None:
        raise ValueError(
            "No se recibi? una imagen v?lida."
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
            "El ROI de inspecci?n est? vac?o."
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
    # reflejos, costuras y peque?os huecos en la tela.
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
    # en un rect?ngulo ni en un convex hull.
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
            "La silueta detectada es demasiado peque?a."
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
    Localiza la anomal?a principal dentro de la blusa.

    No depende de una posici?n fija. Combina el mapa PatchCore con
    informaci?n de contraste visual y aplica solamente una
    penalizaci?n suave a los bordes.
    """
    if image is None:
        raise ValueError(
            "No se recibi? una imagen v?lida."
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
            "Mapa PatchCore inv?lido: "
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
            "No existen p?xeles v?lidos en la prenda."
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
    # SEGUNDA SE?AL:
    # cambio visual respecto del entorno local.
    #
    # No clasifica por color amarillo ni por una posici?n fija.
    # Ayuda a reforzar una regi?n cuya apariencia difiere
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

    # PatchCore contin?a siendo la se?al dominante.
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
    # IMPORTANTE: se usa como penalizaci?n, no como exclusi?n.
    # Una mancha pr?xima al borde sigue siendo candidata.
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

        # Suave penalizaci?n de borde: 0.72 .. 1.00.
        # No elimina anomal?as que est?n en un extremo.
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
    # AGRUPACI?N MULTIMANCHA
    #
    # Una misma mancha puede generar varios hotspots PatchCore.
    # Primero agrupamos fragmentos cercanos y DESPU?S contamos
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
        # uno de los ejes y estar pr?ximos en el otro.
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

        # Fragmentos muy pr?ximos en ambos ejes.
        if (
            gap_x <= horizontal_join * 0.55
            and gap_y <= vertical_join * 0.55
        ):
            return True

        return False

    # Union-Find: garantiza agrupaci?n transitiva.
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

        # Peque?o refuerzo cuando varios hotspots coherentes
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
    # Se filtra DESPU?S de agrupar, para no contar cada hotspot
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

    # No existe un m?ximo funcional de 2.
    # 10 es solamente una protecci?n defensiva ante un mapa
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
# Flask puede atender varias solicitudes simult?neamente, por lo que
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
            # la eliminaci?n del fondo.
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
            # Si no existe una blusa v?lida, nunca ejecutar el
            # detector provisional ni registrar el panel vac?o.
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
                    "[PATCHCORE] Inspecci?n cancelada: "
                    "no hay una blusa completa en el ?rea."
                )

                raise RuntimeError(
                    "No se detect? una blusa completa "
                    "en el ?rea de inspecci?n."
                ) from error

            # Solamente un fallo t?cnico real de PatchCore
            # puede utilizar el m?todo de respaldo.
            print(
                "[PATCHCORE] Error t?cnico durante "
                f"la inferencia: {error}. "
                "Se usar? el m?todo alternativo."
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
        ORDER BY batch_position ASC, id ASC
        """,
        (batch_id,),
    )



# ============================================================
# RUTAS
# ============================================================

@app.route("/video_feed")
@login_required
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
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))

        flash("Usuario o contraseña incorrectos.", "error")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    inspections = fetch_all("SELECT * FROM inspections ORDER BY id DESC LIMIT 8")

    alerts = fetch_all("""
        SELECT * FROM inspections
        WHERE status IN ('Defecto', 'Revisar')
        ORDER BY id DESC
        LIMIT 5
    """)

    total = fetch_one("SELECT COUNT(*) AS c FROM inspections")["c"]
    defects = fetch_one("SELECT COUNT(*) AS c FROM inspections WHERE status = 'Defecto'")["c"]
    approved = fetch_one("SELECT COUNT(*) AS c FROM inspections WHERE status = 'Aprobado'")["c"]
    pending = fetch_one("SELECT COUNT(*) AS c FROM inspections WHERE human_validation = 'Pendiente'")["c"]

    return render_template(
        "dashboard.html",
        inspections=inspections,
        alerts=alerts,
        total=total,
        defects=defects,
        approved=approved,
        pending=pending,
    )



@app.route("/lotes", methods=["GET", "POST"])
@login_required
def batches():
    if request.method == "POST":
        code = request.form.get("code", "").strip()
        quantity_raw = request.form.get(
            "planned_quantity",
            "",
        ).strip()

        try:
            planned_quantity = int(quantity_raw)
        except (TypeError, ValueError):
            planned_quantity = 0

        if planned_quantity < 1 or planned_quantity > 5000:
            flash(
                "La cantidad del lote debe estar entre 1 y 5000 blusas.",
                "error",
            )
            return redirect(url_for("batches"))

        if not code:
            code = (
                f"LOT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                f"{uuid.uuid4().hex[:4].upper()}"
            )

        existing = fetch_one(
            "SELECT id FROM batches WHERE code = %s",
            (code,),
        )

        if existing:
            flash(
                "Ya existe un lote con ese código.",
                "error",
            )
            return redirect(url_for("batches"))

        batch_id = execute(
            """
            INSERT INTO batches (
                code,
                planned_quantity,
                status
            )
            VALUES (%s, %s, 'PREPARACION')
            """,
            (code, planned_quantity),
        )

        flash(
            f"Lote {code} creado correctamente.",
            "success",
        )

        return redirect(
            url_for("batches", created=batch_id)
        )

    return render_template(
        "batches.html",
        batches=get_batches(),
        active_batch=get_active_batch(),
    )


@app.route("/lotes/<int:batch_id>/iniciar", methods=["POST"])
@login_required
def start_batch(batch_id):
    global AUTO_INSPECTION_ENABLED

    batch = get_batch(batch_id)

    if batch is None:
        flash("El lote no existe.", "error")
        return redirect(url_for("batches"))

    if batch["status"] != "PREPARACION":
        flash(
            "Solo pueden iniciarse lotes en preparación.",
            "error",
        )
        return redirect(url_for("batches"))

    active = get_active_batch()

    if active is not None:
        flash(
            f"Ya existe un lote activo: {active['code']}.",
            "error",
        )
        return redirect(url_for("batches"))

    AUTO_INSPECTION_ENABLED = False

    execute(
        """
        UPDATE batches
        SET status = 'EN_INSPECCION',
            started_at = %s
        WHERE id = %s
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_id,
        ),
    )

    flash(
        f"Lote {batch['code']} iniciado.",
        "success",
    )

    return redirect(url_for("station"))



@app.route("/lotes/<int:batch_id>/revision")
@login_required
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


@app.route(
    "/lotes/<int:batch_id>/revision/"
    "<int:inspection_id>/<decision>",
    methods=["POST"],
)
@login_required
def review_batch_alert(
    batch_id,
    inspection_id,
    decision,
):
    batch = get_batch_review_summary(batch_id)

    if batch is None:
        flash(
            "El lote solicitado no existe.",
            "error",
        )
        return redirect(url_for("batches"))

    if batch["status"] != "REVISION_PENDIENTE":
        flash(
            "Este lote no se encuentra en revisi\u00f3n.",
            "error",
        )
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
        flash(
            "La inspecci\u00f3n no pertenece a este lote.",
            "error",
        )
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    if inspection["ai_decision"] != "ANOMALIA":
        flash(
            "Solo las alertas de anomal\u00eda "
            "pueden revisarse desde esta pantalla.",
            "error",
        )
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    if decision == "confirmar":
        review_status = "DEFECTO_CONFIRMADO"
        human_validation = "Correcto"
        message = (
            "Defecto confirmado para la blusa "
            f"#{inspection['batch_position']}."
        )

    elif decision == "descartar":
        review_status = "ALERTA_DESCARTADA"
        human_validation = "Incorrecto"
        message = (
            "Alerta descartada para la blusa "
            f"#{inspection['batch_position']}."
        )

    else:
        flash(
            "Decisi\u00f3n de revisi\u00f3n no v\u00e1lida.",
            "error",
        )
        return redirect(
            url_for(
                "batch_review",
                batch_id=batch_id,
            )
        )

    execute(
        """
        UPDATE inspections
        SET review_status = %s,
            human_validation = %s
        WHERE id = %s
          AND batch_id = %s
        """,
        (
            review_status,
            human_validation,
            inspection_id,
            batch_id,
        ),
    )

    flash(
        message,
        "success",
    )

    updated = get_batch_review_summary(batch_id)

    if int(updated["pending_alerts"] or 0) == 0:
        flash(
            "Todas las alertas del lote fueron revisadas.",
            "success",
        )

    return redirect(
        url_for(
            "batch_review",
            batch_id=batch_id,
        )
    )


@app.route("/estacion")
@login_required
def station():
    return render_template(
        "station.html",
        active_batch=get_active_batch(),
    )


@app.route("/api/station/manual", methods=["POST"])
@login_required
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
                    "No se pudo leer la c?mara. "
                    "No se guard? ning?n registro."
                ),
            }), 503

        result = register_inspection_from_frame(
            frame,
            notes=(
                "Registro manual generado desde "
                "estaci?n de inspecci?n."
            ),
        )

        AUTO_LAST_RESULT = result
        AUTO_LAST_ERROR = None

        return jsonify({
            "ok": True,
            "message": (
                "Inspecci?n manual registrada correctamente."
            ),
            "result": result,
        })

    except Exception as e:
        AUTO_LAST_ERROR = str(e)

        return jsonify({
            "ok": False,
            "message": (
                "Error al ejecutar la inspecci?n manual."
            ),
            "detail": str(e),
        }), 500




@app.route("/api/station/camera/reconnect", methods=["POST"])
@login_required
def station_camera_reconnect():
    global CAMERA_RECONNECTING

    if CAMERA_RECONNECTING:
        return jsonify({
            "ok": True,
            "message": "La reconexi\u00f3n de la c\u00e1mara ya est\u00e1 en curso.",
            "camera": get_camera_status(),
        }), 202

    CAMERA_RECONNECTING = True

    thread = threading.Thread(
        target=camera_reconnect_worker,
        daemon=True,
    )
    thread.start()

    return jsonify({
        "ok": True,
        "message": "Reconexi\u00f3n de c\u00e1mara iniciada.",
        "camera": get_camera_status(),
    }), 202


@app.route("/api/station/auto/start", methods=["POST"])
@login_required
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
def station_auto_stop():
    global AUTO_INSPECTION_ENABLED

    AUTO_INSPECTION_ENABLED = False

    return jsonify({
        "ok": True,
        "message": "Modo automático detenido.",
    })


@app.route("/api/station/auto/status")
@login_required
def station_auto_status():
    active_batch = get_active_batch()

    last_result = AUTO_LAST_RESULT

    # Evita mostrar el resultado de un lote anterior mientras
    # un lote nuevo todav?a no tiene inspecciones.
    if (
        last_result is not None
        and active_batch is not None
        and last_result.get("batch_id") != active_batch["id"]
    ):
        last_result = None

    # Persistencia: si Flask se reinici? o la p?gina se recarg?,
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
    q = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip()
    garment = request.args.get("garment_type", "").strip()

    sql = "SELECT * FROM inspections WHERE 1=1"
    params = []

    if q:
        sql += " AND (code LIKE %s OR defect_type LIKE %s OR zone LIKE %s)"
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])

    if status:
        sql += " AND status = %s"
        params.append(status)

    if garment:
        sql += " AND garment_type = %s"
        params.append(garment)

    sql += " ORDER BY id DESC LIMIT 200"

    rows = fetch_all(sql, tuple(params))
    return render_template("records.html", rows=rows, garments=GARMENTS)



def get_informe_data():
    total = fetch_one(
        "SELECT COUNT(*) AS c FROM inspections"
    )["c"]

    defects = fetch_one(
        """
        SELECT COUNT(*) AS c
        FROM inspections
        WHERE status = 'Defecto'
        """
    )["c"]

    approved = fetch_one(
        """
        SELECT COUNT(*) AS c
        FROM inspections
        WHERE status = 'Aprobado'
        """
    )["c"]

    review = fetch_one(
        """
        SELECT COUNT(*) AS c
        FROM inspections
        WHERE status = 'Revisar'
        """
    )["c"]

    by_garment = fetch_all(
        """
        SELECT
            garment_type,
            COUNT(*) AS total,
            SUM(
                CASE
                    WHEN status = 'Defecto'
                    THEN 1
                    ELSE 0
                END
            ) AS defects
        FROM inspections
        GROUP BY garment_type
        ORDER BY garment_type
        """
    )

    by_defect = fetch_all(
        """
        SELECT
            defect_type,
            COUNT(*) AS total
        FROM inspections
        GROUP BY defect_type
        ORDER BY total DESC
        """
    )

    inspections = fetch_all(
        """
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
            b.code AS batch_code
        FROM inspections i
        LEFT JOIN batches b
          ON b.id = i.batch_id
        ORDER BY i.id DESC
        """
    )

    batches = fetch_all(
        """
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
                        WHEN i.ai_decision = 'NORMAL'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS auto_approved,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.ai_decision = 'ANOMALIA'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS alerts,

            COALESCE(
                SUM(
                    CASE
                        WHEN i.review_status = 'DEFECTO_CONFIRMADO'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS confirmed_defects

        FROM batches b
        LEFT JOIN inspections i
          ON i.batch_id = b.id

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
        """
    )

    return {
        "total": total,
        "defects": defects,
        "approved": approved,
        "review": review,
        "by_garment": by_garment,
        "by_defect": by_defect,
        "inspections": inspections,
        "batches": batches,
    }


@app.route("/informes")
@login_required
def informes():
    data = get_informe_data()

    return render_template(
        "informes.html",
        **data,
    )


@app.route("/reportes")
@login_required
def reportes_legacy():
    return redirect(
        url_for("informes")
    )


@app.route("/informes/excel")
@login_required
def informe_excel():
    from flask import send_file

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

    workbook = Workbook()

    summary = workbook.active
    summary.title = "Resumen"

    title_fill = PatternFill(
        "solid",
        fgColor="111111",
    )

    header_fill = PatternFill(
        "solid",
        fgColor="EDE6DD",
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

    summary.merge_cells("A1:D1")
    summary["A1"] = (
        "Informe de control de calidad - "
        "Astrid y Beverly Fashion"
    )

    summary["A1"].font = Font(
        bold=True,
        color="FFFFFF",
        size=16,
    )

    summary["A1"].fill = title_fill
    summary["A1"].alignment = Alignment(
        horizontal="center",
    )

    summary["A3"] = "Fecha de generaci\u00f3n"
    summary["B3"] = datetime.now()

    summary["A5"] = "Indicador"
    summary["B5"] = "Cantidad"

    for cell in summary["5:5"]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.border = border

    indicators = [
        ("Total de inspecciones", data["total"]),
        ("Aprobadas", data["approved"]),
        ("Con defecto", data["defects"]),
        ("Para revisar", data["review"]),
    ]

    for row_index, (label, value) in enumerate(
        indicators,
        start=6,
    ):
        summary.cell(
            row=row_index,
            column=1,
            value=label,
        )

        summary.cell(
            row=row_index,
            column=2,
            value=value,
        )

    defect_start = 12

    summary.cell(
        row=defect_start,
        column=1,
        value="Resumen por defecto",
    ).font = Font(
        bold=True,
        size=13,
    )

    summary.cell(
        row=defect_start + 1,
        column=1,
        value="Defecto",
    )

    summary.cell(
        row=defect_start + 1,
        column=2,
        value="Total",
    )

    for cell in summary[defect_start + 1]:
        if cell.column <= 2:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.border = border

    for index, row in enumerate(
        data["by_defect"],
        start=defect_start + 2,
    ):
        summary.cell(
            row=index,
            column=1,
            value=row["defect_type"] or "Sin especificar",
        )

        summary.cell(
            row=index,
            column=2,
            value=int(row["total"] or 0),
        )

    note_row = (
        defect_start
        + 3
        + len(data["by_defect"])
    )

    summary.merge_cells(
        start_row=note_row,
        start_column=1,
        end_row=note_row + 2,
        end_column=4,
    )

    summary.cell(
        row=note_row,
        column=1,
        value=(
            "Nota metodol\u00f3gica: este informe contiene "
            "registros hist\u00f3ricos del prototipo. "
            "Las cifras no representan por s\u00ed solas "
            "la exactitud final del modelo sin una "
            "validaci\u00f3n humana suficiente."
        ),
    )

    summary.cell(
        row=note_row,
        column=1,
    ).alignment = Alignment(
        wrap_text=True,
        vertical="top",
    )

    summary.column_dimensions["A"].width = 48
    summary.column_dimensions["B"].width = 18
    summary.column_dimensions["C"].width = 18
    summary.column_dimensions["D"].width = 18

    summary["B3"].number_format = (
        "yyyy-mm-dd hh:mm:ss"
    )

    # ========================================================
    # HOJA DE INSPECCIONES
    # ========================================================

    inspections_sheet = workbook.create_sheet(
        "Inspecciones"
    )

    inspection_headers = [
        "ID",
        "C\u00f3digo",
        "Fecha",
        "Lote",
        "Posici\u00f3n lote",
        "Prenda",
        "Estado",
        "Decisi\u00f3n IA",
        "Defecto",
        "Confianza (%)",
        "Zona",
        "Validaci\u00f3n humana",
        "Estado de revisi\u00f3n",
    ]

    inspections_sheet.append(
        inspection_headers
    )

    for cell in inspections_sheet[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF",
        )
        cell.fill = title_fill
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    for row in data["inspections"]:
        confidence = row["confidence"]

        if confidence is not None:
            confidence = float(confidence)

        inspections_sheet.append([
            row["id"],
            row["code"],
            row["created_at"],
            row["batch_code"] or "",
            row["batch_position"] or "",
            row["garment_type"] or "",
            row["status"] or "",
            row["ai_decision"] or "",
            row["defect_type"] or "",
            confidence,
            row["zone"] or "",
            row["human_validation"] or "",
            row["review_status"] or "",
        ])

    inspections_sheet.freeze_panes = "A2"
    inspections_sheet.auto_filter.ref = (
        inspections_sheet.dimensions
    )

    if inspections_sheet.max_row >= 2:
        for cell in inspections_sheet["C"][1:]:
            cell.number_format = (
                "yyyy-mm-dd hh:mm:ss"
            )

    inspection_widths = {
        1: 8,
        2: 20,
        3: 20,
        4: 18,
        5: 14,
        6: 24,
        7: 16,
        8: 18,
        9: 42,
        10: 16,
        11: 22,
        12: 22,
        13: 24,
    }

    for column, width in inspection_widths.items():
        inspections_sheet.column_dimensions[
            get_column_letter(column)
        ].width = width

    for row in inspections_sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    # ========================================================
    # HOJA DE LOTES
    # ========================================================

    batches_sheet = workbook.create_sheet(
        "Lotes"
    )

    batch_headers = [
        "ID",
        "C\u00f3digo",
        "Cantidad planificada",
        "Procesadas",
        "Aprobadas autom\u00e1ticamente",
        "Alertas",
        "Defectos confirmados",
        "Estado",
        "Fecha de creaci\u00f3n",
        "Inicio",
        "Fin de inspecci\u00f3n",
        "Cierre",
    ]

    batches_sheet.append(
        batch_headers
    )

    for cell in batches_sheet[1]:
        cell.font = Font(
            bold=True,
            color="FFFFFF",
        )
        cell.fill = title_fill
        cell.alignment = Alignment(
            horizontal="center",
        )

    for row in data["batches"]:
        batches_sheet.append([
            row["id"],
            row["code"],
            row["planned_quantity"],
            row["processed_quantity"],
            row["auto_approved"],
            row["alerts"],
            row["confirmed_defects"],
            row["status"],
            row["created_at"],
            row["started_at"],
            row["inspection_completed_at"],
            row["closed_at"],
        ])

    batches_sheet.freeze_panes = "A2"
    batches_sheet.auto_filter.ref = (
        batches_sheet.dimensions
    )

    for column in range(
        1,
        batches_sheet.max_column + 1,
    ):
        batches_sheet.column_dimensions[
            get_column_letter(column)
        ].width = 22

    for row in batches_sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    buffer = io.BytesIO()

    workbook.save(buffer)
    buffer.seek(0)

    filename = (
        "informe_control_calidad_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
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

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
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
    from xml.sax.saxutils import escape

    data = get_informe_data()

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Informe de control de calidad",
        author="Astrid y Beverly Fashion",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "InformeTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "InformeSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.HexColor("#6B625A"),
        spaceAfter=14,
    )

    cell_style = ParagraphStyle(
        "InformeCell",
        parent=styles["Normal"],
        fontSize=6.5,
        leading=8,
    )

    header_style = ParagraphStyle(
        "InformeHeader",
        parent=cell_style,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )

    elements = []

    elements.append(
        Paragraph(
            "Informe de control de calidad",
            title_style,
        )
    )

    elements.append(
        Paragraph(
            (
                "Astrid y Beverly Fashion - "
                "prototipo de inspecci\u00f3n visual automatizada"
            ),
            subtitle_style,
        )
    )

    elements.append(
        Paragraph(
            (
                "Fecha de generaci\u00f3n: "
                + datetime.now().strftime(
                    "%d/%m/%Y %H:%M:%S"
                )
            ),
            styles["Normal"],
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    kpi_data = [
        [
            "Total de inspecciones",
            "Aprobadas",
            "Con defecto",
            "Para revisar",
        ],
        [
            str(data["total"]),
            str(data["approved"]),
            str(data["defects"]),
            str(data["review"]),
        ],
    ]

    kpi_table = Table(
        kpi_data,
        colWidths=[62 * mm] * 4,
    )

    kpi_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111111"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER",
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#DDD5CB"),
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
        ])
    )

    elements.append(kpi_table)
    elements.append(Spacer(1, 16))

    elements.append(
        Paragraph(
            "Resumen por defecto",
            styles["Heading2"],
        )
    )

    defect_rows = [
        ["Defecto", "Total"]
    ]

    for row in data["by_defect"]:
        defect_rows.append([
            row["defect_type"] or "Sin especificar",
            str(row["total"]),
        ])

    defect_table = Table(
        defect_rows,
        colWidths=[205 * mm, 35 * mm],
        repeatRows=1,
    )

    defect_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#EDE6DD"),
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#DDD5CB"),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6,
            ),
        ])
    )

    elements.append(defect_table)
    elements.append(Spacer(1, 14))

    elements.append(
        Paragraph(
            (
                "<b>Nota metodol\u00f3gica:</b> "
                "este informe contiene registros hist\u00f3ricos "
                "del prototipo. Las cifras no representan por "
                "s\u00ed solas la exactitud final del modelo sin "
                "una validaci\u00f3n humana suficiente."
            ),
            styles["Normal"],
        )
    )

    elements.append(PageBreak())

    elements.append(
        Paragraph(
            "Lotes de producci\u00f3n",
            styles["Heading2"],
        )
    )

    batch_rows = [[
        "Lote",
        "Plan",
        "Procesadas",
        "Auto aprobadas",
        "Alertas",
        "Defectos",
        "Estado",
    ]]

    for row in data["batches"]:
        batch_rows.append([
            str(row["code"]),
            str(row["planned_quantity"]),
            str(row["processed_quantity"]),
            str(row["auto_approved"]),
            str(row["alerts"]),
            str(row["confirmed_defects"]),
            str(row["status"]),
        ])

    batch_table = LongTable(
        batch_rows,
        repeatRows=1,
        colWidths=[
            38 * mm,
            25 * mm,
            30 * mm,
            36 * mm,
            25 * mm,
            28 * mm,
            48 * mm,
        ],
    )

    batch_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111111"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#DDD5CB"),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
        ])
    )

    elements.append(batch_table)
    elements.append(PageBreak())

    elements.append(
        Paragraph(
            "Detalle de inspecciones",
            styles["Heading2"],
        )
    )

    inspection_rows = [[
        Paragraph("C\u00f3digo", header_style),
        Paragraph("Fecha", header_style),
        Paragraph("Lote", header_style),
        Paragraph("Estado", header_style),
        Paragraph("Defecto", header_style),
        Paragraph("Conf.", header_style),
        Paragraph("Zona", header_style),
        Paragraph("Validaci\u00f3n", header_style),
    ]]

    for row in data["inspections"]:
        confidence = (
            ""
            if row["confidence"] is None
            else f"{float(row['confidence']):.2f}%"
        )

        values = [
            row["code"] or "",
            row["created_at"] or "",
            row["batch_code"] or "",
            row["status"] or "",
            row["defect_type"] or "",
            confidence,
            row["zone"] or "",
            row["human_validation"] or "",
        ]

        inspection_rows.append([
            Paragraph(
                escape(str(value)),
                cell_style,
            )
            for value in values
        ])

    inspection_table = LongTable(
        inspection_rows,
        repeatRows=1,
        colWidths=[
            33 * mm,
            31 * mm,
            28 * mm,
            25 * mm,
            62 * mm,
            18 * mm,
            31 * mm,
            32 * mm,
        ],
    )

    inspection_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111111"),
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor("#D8D0C6"),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                3,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                3,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                3,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                3,
            ),
        ])
    )

    elements.append(
        inspection_table
    )

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont(
            "Helvetica",
            8,
        )

        canvas.drawRightString(
            landscape(A4)[0] - 12 * mm,
            7 * mm,
            f"P\u00e1gina {document.page}",
        )

        canvas.restoreState()

    document.build(
        elements,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    buffer.seek(0)

    filename = (
        "informe_control_calidad_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
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
def validate(inspection_id, value):
    value = value if value in ["Correcto", "Incorrecto", "Pendiente"] else "Pendiente"

    execute(
        "UPDATE inspections SET human_validation = %s WHERE id = %s",
        (value, inspection_id),
    )

    return redirect(request.referrer or url_for("records"))


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


if __name__ == "__main__":
    init_db()
    # app.run(debug=True, host="127.0.0.1", port=5000)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
