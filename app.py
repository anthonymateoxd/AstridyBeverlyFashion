from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from garment_service import identify_garment
import os
import time
import uuid
import threading
from functools import wraps

import mysql.connector


load_dotenv(override=True)

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
SIZES = ["S", "M"]

camera_lock = threading.Lock()
latest_camera_frame = None
yolo_model = None

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "tesis-control-calidad-dev")
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
            size VARCHAR(20) NOT NULL DEFAULT 'N/A',
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

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", os.getenv("CAMERA_INDEX", "0")).strip()
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "15"))

AUTO_INSPECTION_ENABLED = False
AUTO_THREAD = None
AUTO_LAST_RESULT = None
AUTO_LAST_ERROR = None
AUTO_LAST_CAPTURE_TIME = 0

AUTO_COOLDOWN_SECONDS = float(os.getenv("AUTO_COOLDOWN_SECONDS", "4"))
AUTO_MOTION_THRESHOLD = int(os.getenv("AUTO_MOTION_THRESHOLD", "18000"))
AUTO_STABILIZATION_SECONDS = float(os.getenv("AUTO_STABILIZATION_SECONDS", "0.6"))

ROI_X1 = float(os.getenv("ROI_X1", "0.10"))
ROI_Y1 = float(os.getenv("ROI_Y1", "0.10"))
ROI_X2 = float(os.getenv("ROI_X2", "0.90"))
ROI_Y2 = float(os.getenv("ROI_Y2", "0.90"))

YOLO_INFERENCE_CONF = float(os.getenv("YOLO_INFERENCE_CONF", "0.40"))
YOLO_DEFECT_THRESHOLD = float(os.getenv("YOLO_DEFECT_THRESHOLD", "0.70"))

camera_capture = None


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
    Abre una sola instancia de cámara para todo el sistema.
    Evita que /video_feed, inspección manual y modo automático abran cámaras separadas.
    """
    global camera_capture

    if cv2 is None:
        return None

    if camera_capture is not None and camera_capture.isOpened():
        return camera_capture

    source = parse_camera_source(CAMERA_SOURCE)

    if isinstance(source, int) and os.name == "nt":
        camera_capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not camera_capture.isOpened():
            camera_capture.release()
            camera_capture = cv2.VideoCapture(source)
    else:
        camera_capture = cv2.VideoCapture(source)

    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera_capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not camera_capture.isOpened():
        try:
            camera_capture.release()
        except Exception:
            pass

        camera_capture = None
        return None

    return camera_capture


def read_camera_frame():
    """
    Lee un frame de la cámara activa.
    Devuelve: (True, frame) si funciona; (False, None) si falla.
    """
    global camera_capture
    global latest_camera_frame

    if cv2 is None:
        return False, None

    with camera_lock:
        cap = get_camera()

        if cap is None:
            return False, None

        ok, frame = cap.read()

        if not ok or frame is None:
            try:
                cap.release()
            except Exception:
                pass

            camera_capture = None
            return False, None

        latest_camera_frame = frame.copy()
        return True, frame


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
    notes="Registro generado por estación de inspección.",
    identify_first=False,
):
    """
    Guarda la captura y, cuando identify_first=True,
    identifica primero el modelo de prenda.
    """
    if frame is None:
        raise ValueError(
            "No se recibió imagen de cámara para registrar la inspección."
        )

    img_path, img_rel = save_frame_to_static(frame)

    identity_result = None
    garment_type = "Prenda inspeccionada"

    if identify_first:
        identity_result = identify_garment(img_path)

        if not identity_result["identified"]:
            garment_type = "Prenda no identificada"
            status = "Revisar"
            defect = "Prenda no identificada"
            conf = round(identity_result["similarity"] * 100, 2)
            zone = "No aplica"
            result_rel = None

        else:
            garment_type = identity_result["result"]

            status, defect, conf, zone, result_rel = detect_defect(
                img_path
            )

    else:
        status, defect, conf, zone, result_rel = detect_defect(
            img_path
        )

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = (
        f"INS-{datetime.now().strftime('%H%M%S')}-"
        f"{uuid.uuid4().hex[:4].upper()}"
    )

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
            created_at,
            garment_type,
            "N/A",
            status,
            defect,
            conf,
            zone,
            img_rel,
            result_rel,
            notes,
        ),
    )

    return {
        "code": code,
        "garment_type": garment_type,
        "status": status,
        "defect_type": defect,
        "confidence": float(conf) if conf is not None else None,
        "zone": zone,
        "image_original": img_rel,
        "image_result": result_rel,
        "created_at": created_at,
        "identity_identified": (
            identity_result["identified"]
            if identity_result is not None
            else None
        ),
        "identity_similarity": (
            identity_result["similarity"]
            if identity_result is not None
            else None
        ),
        "identity_threshold": (
            identity_result["threshold"]
            if identity_result is not None
            else None
        ),
    }

def auto_inspection_worker():
    """
    Modo automático: detecta movimiento/cambio en la escena.
    Cuando la prenda pasa por la cámara, espera una fracción de segundo,
    captura una imagen estable, ejecuta IA y registra la inspección.
    """
    global AUTO_INSPECTION_ENABLED
    global AUTO_LAST_RESULT
    global AUTO_LAST_ERROR
    global AUTO_LAST_CAPTURE_TIME

    previous_gray = None

    while AUTO_INSPECTION_ENABLED:
        try:
            ok, frame = read_camera_frame()

            if not ok:
                AUTO_LAST_ERROR = "Cámara no disponible."
                time.sleep(1)
                continue

            roi_frame = crop_inspection_roi(frame)
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if previous_gray is None:
                previous_gray = gray
                time.sleep(0.5)
                continue

            diff = cv2.absdiff(previous_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            motion_pixels = cv2.countNonZero(thresh)
            now = time.time()

            if (
                motion_pixels > AUTO_MOTION_THRESHOLD
                and now - AUTO_LAST_CAPTURE_TIME >= AUTO_COOLDOWN_SECONDS
            ):
                time.sleep(AUTO_STABILIZATION_SECONDS)

                ok_stable, stable_frame = read_camera_frame()

                if ok_stable:
                    result = register_inspection_from_frame(
                        stable_frame,
                        notes="Registro automático generado por detección de movimiento.",
                    )

                    AUTO_LAST_RESULT = result
                    AUTO_LAST_ERROR = None
                    AUTO_LAST_CAPTURE_TIME = time.time()
                    previous_gray = None
                    continue

            previous_gray = gray

        except Exception as e:
            AUTO_LAST_ERROR = str(e)

        time.sleep(0.5)


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


def detect_defect(image_path):
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


@app.route("/estacion")
@login_required
def station():
    return render_template("station.html")


@app.route("/api/station/manual", methods=["POST"])
@login_required
def station_manual_inspect():
    try:
        ok, frame = read_camera_frame()

        if not ok:
            return jsonify({
                "ok": False,
                "message": "No se pudo leer la cámara. No se guardó ningún registro.",
            }), 503

        result = register_inspection_from_frame(
            frame,
            notes="Registro manual generado desde estación de inspección.",
            identify_first=True,
        )

        return jsonify({
            "ok": True,
            "message": "Inspección manual registrada correctamente.",
            "result": result,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "message": "Error al ejecutar la inspección manual.",
            "detail": str(e),
        }), 500


@app.route("/api/station/auto/start", methods=["POST"])
@login_required
def station_auto_start():
    global AUTO_INSPECTION_ENABLED
    global AUTO_THREAD

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
    return jsonify({
        "ok": True,
        "automatic": AUTO_INSPECTION_ENABLED,
        "last_result": AUTO_LAST_RESULT,
        "last_error": AUTO_LAST_ERROR,
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
        size = "N/A"
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


@app.route("/reportes")
@login_required
def reports():
    total = fetch_one("SELECT COUNT(*) AS c FROM inspections")["c"]
    defects = fetch_one("SELECT COUNT(*) AS c FROM inspections WHERE status = 'Defecto'")["c"]
    approved = fetch_one("SELECT COUNT(*) AS c FROM inspections WHERE status = 'Aprobado'")["c"]
    review = fetch_one("SELECT COUNT(*) AS c FROM inspections WHERE status = 'Revisar'")["c"]

    by_garment = fetch_all("""
        SELECT garment_type, COUNT(*) AS total,
               SUM(CASE WHEN status = 'Defecto' THEN 1 ELSE 0 END) AS defects
        FROM inspections
        GROUP BY garment_type
        ORDER BY garment_type
    """)

    by_defect = fetch_all("""
        SELECT defect_type, COUNT(*) AS total
        FROM inspections
        GROUP BY defect_type
        ORDER BY total DESC
    """)

    return render_template(
        "reports.html",
        total=total,
        defects=defects,
        approved=approved,
        review=review,
        by_garment=by_garment,
        by_defect=by_defect,
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
            "camera_source": CAMERA_SOURCE,
            "model_ready": model_ready,
            "detection_mode": "yolo" if model_ready else "prototype",
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    init_db()
    # app.run(debug=True, host="127.0.0.1", port=5000)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)