from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GARMENTS = {
    ord("b"): "blusa",
    ord("t"): "top_corto",
    ord("c"): "camisa_cropped",
}

DEFECT_KEYS = {
    ord("0"): (None, "sin_defecto"),
    ord("1"): (0, "mancha"),
    ord("2"): (1, "rotura"),
    ord("3"): (2, "agujero"),
    ord("4"): (3, "variacion_color"),
}


def parse_camera_source(value: str):
    value = value.strip()
    return int(value) if value.isdigit() else value


def open_camera(source):
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("CAMERA_WIDTH", "1280")))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("CAMERA_HEIGHT", "720")))
    cap.set(cv2.CAP_PROP_FPS, int(os.getenv("CAMERA_FPS", "15")))
    return cap


def yolo_box(x: int, y: int, width: int, height: int, image_width: int, image_height: int):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return x_center, y_center, norm_width, norm_height


def save_sample(frame, split: str, garment: str, class_id: int | None, defect_name: str) -> bool:
    images_dir = ROOT / "dataset" / "images" / split
    labels_dir = ROOT / "dataset" / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{garment}_{defect_name}_{timestamp}_{uuid.uuid4().hex[:6]}"
    image_path = images_dir / f"{stem}.jpg"
    label_path = labels_dir / f"{stem}.txt"

    labels: list[str] = []

    if class_id is not None:
        selection = cv2.selectROI(
            "Marque el defecto y presione ENTER",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("Marque el defecto y presione ENTER")
        x, y, width, height = map(int, selection)

        if width <= 0 or height <= 0:
            print("[CANCELADO] No se seleccionó una caja válida.")
            return False

        image_height, image_width = frame.shape[:2]
        xc, yc, nw, nh = yolo_box(x, y, width, height, image_width, image_height)
        labels.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    if not cv2.imwrite(str(image_path), frame):
        raise IOError(f"No se pudo guardar {image_path}")

    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    print(f"[GUARDADO] {image_path.name} | {garment} | {defect_name} | split={split}")
    return True


def draw_help(frame, split: str, garment: str, defect_name: str):
    output = frame.copy()
    lines = [
        f"SPLIT: {split.upper()} | PRENDA: {garment} | CLASE: {defect_name}",
        "Prenda: B=blusa  T=top corto  C=camisa cropped",
        "Clase: 0=limpia  1=mancha  2=rotura  3=agujero  4=variacion color",
        "S=guardar/etiquetar   Q=salir",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            output,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura y etiqueta imágenes desde la cámara local.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()

    source = parse_camera_source(os.getenv("CAMERA_SOURCE", "0"))
    cap = open_camera(source)

    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir CAMERA_SOURCE={source}")

    garment = "blusa"
    class_id: int | None = None
    defect_name = "sin_defecto"

    print("Controles: B/T/C prenda, 0-4 clase, S guardar, Q salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] La cámara dejó de entregar imágenes.")
                break

            preview = draw_help(frame, args.split, garment, defect_name)
            cv2.imshow("Recolector de dataset textil", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in GARMENTS:
                garment = GARMENTS[key]
            elif key in DEFECT_KEYS:
                class_id, defect_name = DEFECT_KEYS[key]
            elif key == ord("s"):
                save_sample(frame.copy(), args.split, garment, class_id, defect_name)
            elif key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
