from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import os
import time
import uuid
import random
import threading
from datetime import datetime

import mysql.connector
from mysql.connector import Error


load_dotenv()

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
    "Costura irregular",
    "Sin defecto",
]

GARMENTS = ["Blusa", "Top corto", "Camisa cropped"]
SIZES = ["S", "M"]

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
    Inicializa MySQL:
    1. Crea la base textile_quality_db si no existe.
    2. Crea tablas compatibles con las plantillas actuales.
    3. Crea usuario admin/admin123.
    """
def init_db():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    cur.execute("SELECT id FROM users WHERE username = %s", ("admin",))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
            ("admin", generate_password_hash("admin123"), "administrador"),
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
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    return wrapper


# ============================================================
# CÁMARA
# ============================================================

def get_camera_index():
    return int(os.environ.get("CAMERA_INDEX", "0"))


def generate_placeholder_frame(message="CAMARA NO DISPONIBLE"):
    if cv2 is not None and np is not None:
        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        frame[:] = (25, 25, 25)
        cv2.putText(
            frame,
            message,
            (90, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            return buffer.tobytes()

    from PIL import Image, ImageDraw
    import io

    img = Image.new("RGB", (800, 480), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    draw.text((120, 230), message, fill=(255, 255, 255))
    bio = io.BytesIO()
    img.save(bio, format="JPEG")
    return bio.getvalue()


def generate_camera_frames():
    global latest_camera_frame

    if cv2 is None:
        placeholder = generate_placeholder_frame("OPENCV NO INSTALADO")
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"

    camera_index = get_camera_index()

    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        placeholder = generate_placeholder_frame("NO SE PUDO ABRIR LA CAMARA")
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()

        if not success:
            placeholder = generate_placeholder_frame("ERROR LEYENDO CAMARA")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            continue

        with camera_lock:
            latest_camera_frame = frame.copy()

        h, w = frame.shape[:2]
        cv2.rectangle(
            frame,
            (int(w * 0.08), int(h * 0.08)),
            (int(w * 0.92), int(h * 0.92)),
            (0, 180, 255),
            2,
        )
        cv2.putText(
            frame,
            "AREA DE INSPECCION",
            (int(w * 0.08), max(25, int(h * 0.08) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
        )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def save_image(file_storage=None):
    global latest_camera_frame

    name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    path = CAPTURE_DIR / name

    if file_storage and file_storage.filename:
        file_storage.save(path)
        return path, f"captures/{name}"

    if cv2 is not None:
        with camera_lock:
            frame = latest_camera_frame.copy() if latest_camera_frame is not None else None

        if frame is not None:
            cv2.imwrite(str(path), frame)
            return path, f"captures/{name}"

    if cv2 is not None:
        camera_index = get_camera_index()
        if os.name == "nt":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        ok, frame = cap.read()
        cap.release()

        if ok:
            cv2.imwrite(str(path), frame)
            return path, f"captures/{name}"

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 720), (245, 242, 235))
    d = ImageDraw.Draw(img)
    d.rectangle((180, 90, 1100, 630), outline=(210, 190, 150), width=6)
    d.text((420, 330), "IMAGEN DE PRUEBA - SIN CAMARA", fill=(70, 70, 70))
    img.save(path)

    return path, f"captures/{name}"


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
        results = model(str(image_path), conf=0.40, verbose=False)
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
            defect_type = model.names.get(class_id, "Defecto visible")

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

            status = "Defecto" if confidence >= 70 else "Revisar"
            return status, defect_type, round(confidence, 2), zone, f"results/{result_name}"

        return "Aprobado", "Sin defecto", 90.0, "Centro", f"results/{result_name}"

    if cv2 is None or np is None:
        status = random.choices(["Aprobado", "Defecto", "Revisar"], [0.55, 0.25, 0.20])[0]
        defect = "Sin defecto" if status == "Aprobado" else random.choice(DEFECT_TYPES[:-1])
        confidence = round(random.uniform(65, 94), 2)
        zone = random.choice(["Centro", "Zona frontal", "Lateral derecho", "Lateral izquierdo"])
        return status, defect, confidence, zone, None

    img = cv2.imread(str(image_path))

    if img is None:
        return "Revisar", "Imagen no válida", 0.0, "No definida", None

    original = img.copy()
    h, w = img.shape[:2]

    x1, y1 = int(w * 0.08), int(h * 0.08)
    x2, y2 = int(w * 0.92), int(h * 0.92)

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

        defect_type = "Posible mancha o anomalía visible"
        status = "Defecto" if confidence >= 70 else "Revisar"

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
        confidence = 90.0
        defect_type = "Sin defecto"
        status = "Aprobado"
        zone = "Centro"

        cv2.putText(
            original,
            f"Aprobado {confidence}%",
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
    return Response(generate_camera_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


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


@app.route("/inspeccion", methods=["GET", "POST"])
@login_required
def inspection():
    result = None

    if request.method == "POST":
        garment = "Prenda inspeccionada"
        size = "N/A"
        notes = "Registro generado automáticamente por el sistema de inspección."

        img_path, img_rel = save_image(request.files.get("image"))
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
        return jsonify({"status": "ok", "database": DB_NAME, "inspections": row["c"]})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    init_db()
    # app.run(debug=True, host="127.0.0.1", port=5000)
    app.run(debug=True, host="0.0.0.0", port=5000)