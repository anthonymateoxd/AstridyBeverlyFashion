from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def check_imports() -> bool:
    modules = {
        "Flask": "flask",
        "OpenCV": "cv2",
        "NumPy": "numpy",
        "MySQL Connector": "mysql.connector",
        "Ultralytics": "ultralytics",
    }
    ok = True
    for label, module_name in modules.items():
        try:
            __import__(module_name)
            print(f"[OK] {label}")
        except Exception as exc:
            ok = False
            print(f"[ERROR] {label}: {exc}")
    return ok


def parse_camera_source(value: str):
    value = value.strip()
    return int(value) if value.isdigit() else value


def check_camera() -> bool:
    try:
        import cv2
    except ImportError as exc:
        print(f"[ERROR] Cámara: OpenCV no está instalado: {exc}")
        return False

    source = parse_camera_source(os.getenv("CAMERA_SOURCE", "0"))
    backend = cv2.CAP_DSHOW if isinstance(source, int) and os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)

    if not cap.isOpened() and isinstance(source, int) and os.name == "nt":
        cap.release()
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir CAMERA_SOURCE={source}")
        return False

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[ERROR] La cámara {source} abrió, pero no entregó un frame.")
        return False

    output = ROOT / "static" / "captures" / "camera_test.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output), frame)
    if not saved:
        print(f"[ERROR] No se pudo guardar la prueba en {output}")
        return False

    height, width = frame.shape[:2]
    print(f"[OK] Cámara {source}: {width}x{height}. Captura: {output}")
    return True


def check_model() -> bool:
    model_path = ROOT / "models" / "best.pt"
    if not model_path.exists() or model_path.stat().st_size < 1024:
        print("[PENDIENTE] models/best.pt todavía no existe. Flask usará modo prototipo.")
        return False

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"[OK] Modelo YOLO cargado. Clases: {model.names}")
        return True
    except Exception as exc:
        print(f"[ERROR] best.pt no pudo cargarse: {exc}")
        return False


def check_database() -> bool:
    try:
        import mysql.connector
    except ImportError as exc:
        print(f"[ERROR] MySQL Connector no está instalado: {exc}")
        return False

    config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
    }

    try:
        conn = mysql.connector.connect(**config)
        conn.close()
        print(f"[OK] MySQL {config['host']}:{config['port']}")
        return True
    except Exception as exc:
        print(f"[ERROR] MySQL: {exc}")
        return False


def main() -> None:
    print("=== COMPROBACIÓN DEL SISTEMA TEXTIL ===")
    check_imports()
    check_camera()
    check_database()
    check_model()


if __name__ == "__main__":
    main()
