from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from dotenv import load_dotenv


# ============================================================
# CONFIGURACIÓN DEL DATASET
# ============================================================

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DEFAULT_DATASET_NAME = "BLUSA_002_S_ROSA_FRONT"

DATASET_NAME = (
    os.getenv(
        "PATCHCORE_DATASET_NAME",
        DEFAULT_DATASET_NAME,
    ).strip()
    or DEFAULT_DATASET_NAME
)

TOTAL_IMAGES = 40
TRAIN_IMAGES = 32
TEST_IMAGES = TOTAL_IMAGES - TRAIN_IMAGES

INITIAL_DELAY_SECONDS = 5
INTERVAL_SECONDS = 3

# Valor mínimo recomendado para evitar fotos borrosas.
# Si rechaza demasiadas capturas, bájalo a 55 o 45.
MIN_SHARPNESS = 65.0

JPEG_QUALITY = 95

TRAIN_DIR = (
    ROOT
    / "anomaly_dataset"
    / DATASET_NAME
    / "train"
    / "good"
)

TEST_DIR = (
    ROOT
    / "anomaly_dataset"
    / DATASET_NAME
    / "test"
    / "good"
)


# ============================================================
# CONFIGURACIÓN DE CÁMARA RTSP DESDE .env
# ============================================================

CAMERA_IP = os.getenv("CAMERA_IP", "10.145.208.30").strip()
CAMERA_USER = os.getenv("CAMERA_USER", "admin").strip()
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "").strip()

CAMERA_RTSP_PORT = int(os.getenv("CAMERA_RTSP_PORT", "554"))
CAMERA_CHANNEL = int(os.getenv("CAMERA_CHANNEL", "1"))
CAMERA_SUBTYPE = int(os.getenv("CAMERA_SUBTYPE", "0"))

CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "15"))

ROI_X1 = float(os.getenv("ROI_X1", "0.16"))
ROI_Y1 = float(os.getenv("ROI_Y1", "0.05"))
ROI_X2 = float(os.getenv("ROI_X2", "0.84"))
ROI_Y2 = float(os.getenv("ROI_Y2", "0.88"))


# Fuerza RTSP sobre TCP para mayor estabilidad.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

CAMERA_USER_ENCODED = quote(CAMERA_USER, safe="")
CAMERA_PASSWORD_ENCODED = quote(CAMERA_PASSWORD, safe="")

CAMERA_SOURCE = (
    f"rtsp://{CAMERA_USER_ENCODED}:{CAMERA_PASSWORD_ENCODED}"
    f"@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={CAMERA_CHANNEL}&subtype={CAMERA_SUBTYPE}"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def count_images(folder: Path) -> int:
    """Cuenta imágenes JPG ya guardadas."""
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg"}
    )


def calculate_sharpness(frame) -> float:
    """Calcula nitidez mediante varianza del Laplaciano."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_camera():
    """Abre la cámara RTSP."""
    capture = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not capture.isOpened():
        capture.release()
        return None

    return capture



def preprocess_patchcore_frame(frame):
    """
    ROI + segmentaci?n crom?tica de blusa rosa + fondo blanco.
    Debe coincidir con el preprocesamiento utilizado por app.py.
    """
    if frame is None:
        raise ValueError(
            "No se recibi? un frame v?lido."
        )

    height, width = frame.shape[:2]

    x1 = int(ROI_X1 * width)
    y1 = int(ROI_Y1 * height)
    x2 = int(ROI_X2 * width)
    y2 = int(ROI_Y2 * height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "El ROI configurado no es v?lido."
        )

    roi = frame[y1:y2, x1:x2]

    if roi is None or roi.size == 0:
        raise ValueError(
            "El ROI est? vac?o."
        )

    h, w = roi.shape[:2]

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

    mask = np.where(
        (saturation >= 30)
        & (lab_a >= 136),
        255,
        0,
    ).astype(np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), dtype=np.uint8),
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if count <= 1:
        raise RuntimeError(
            "No se detect? la blusa rosa."
        )

    roi_area = float(h * w)

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
            cx - w / 2.0
        ) / max(1.0, w)

        dy = abs(
            cy - h / 2.0
        ) / max(1.0, h)

        score = (
            area / roi_area
            - dx * 0.20
            - dy * 0.10
        )

        candidates.append(
            (score, label)
        )

    if not candidates:
        raise RuntimeError(
            "No se encontr? una silueta v?lida de la blusa."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected_label = candidates[0][1]

    garment_mask = np.where(
        labels == selected_label,
        255,
        0,
    ).astype(np.uint8)

    garment_mask = cv2.dilate(
        garment_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    coverage = (
        cv2.countNonZero(garment_mask)
        / roi_area
    )

    print(
        f"[SEGMENTACION ROSA] "
        f"Cobertura: {coverage * 100:.2f}%"
    )

    # Control de calidad exclusivo para captura del dataset.
    # Evita guardar segmentaciones incompletas o contaminadas.
    if coverage < 0.40:
        raise RuntimeError(
            f"Segmentaci?n incompleta: {coverage * 100:.2f}%"
        )

    if coverage > 0.65:
        raise RuntimeError(
            f"Segmentaci?n excesiva: {coverage * 100:.2f}%"
        )

    processed = np.full_like(
        roi,
        255,
    )

    processed[
        garment_mask > 0
    ] = roi[
        garment_mask > 0
    ]

    return processed


def next_destination(train_count: int, test_count: int) -> tuple[Path, str]:
    """Decide si la próxima imagen va a train/good o test/good."""
    if train_count < TRAIN_IMAGES:
        return TRAIN_DIR, "train/good"

    if test_count < TEST_IMAGES:
        return TEST_DIR, "test/good"

    raise RuntimeError("El dataset ya alcanzó la cantidad configurada.")


def save_capture(frame, destination: Path, category: str, number: int) -> Path:
    """Guarda una captura con nombre ordenado y fecha."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{DATASET_NAME.lower()}_good_"
        f"{number:04d}_{timestamp}.jpg"
    )

    output_path = destination / filename

    processed_frame = preprocess_patchcore_frame(frame)

    saved = cv2.imwrite(
        str(output_path),
        processed_frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )

    if not saved:
        raise IOError(f"No se pudo guardar la imagen: {output_path}")

    print(
        f"[GUARDADA] {number}/{TOTAL_IMAGES} "
        f"→ {category}/{filename}"
    )

    return output_path


def draw_status(
    frame,
    total_count: int,
    remaining_seconds: float,
    destination_name: str,
    paused: bool,
    last_message: str,
):
    """Dibuja contador y estado en la vista previa."""
    preview = frame.copy()

    overlay = preview.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (preview.shape[1], 150),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    preview = cv2.addWeighted(overlay, 0.58, preview, 0.42, 0)

    status_text = "PAUSADO" if paused else "CAPTURA AUTOMATICA ACTIVA"
    countdown_text = (
        "Pausado"
        if paused
        else f"Siguiente captura en: {max(0, int(remaining_seconds) + 1)} s"
    )

    cv2.putText(
        preview,
        status_text,
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255) if paused else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        f"Capturas: {total_count}/{TOTAL_IMAGES} | Destino: {destination_name}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        countdown_text,
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        last_message,
        (25, preview.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "Q: salir | P: pausar/reanudar | C: capturar ahora",
        (25, preview.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_count = count_images(TRAIN_DIR)
    test_count = count_images(TEST_DIR)
    total_count = train_count + test_count

    print("=" * 68)
    print("CAPTURA AUTOMÁTICA PARA PATCHCORE")
    print("=" * 68)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Train/good: {train_count}/{TRAIN_IMAGES}")
    print(f"Test/good:  {test_count}/{TEST_IMAGES}")
    print(f"Total:      {total_count}/{TOTAL_IMAGES}")
    print(f"Intervalo:  {INTERVAL_SECONDS} segundos")
    print(f"Cámara:     {CAMERA_IP}")
    print("=" * 68)
    print("Importante: detén app.py antes de ejecutar este archivo.")
    print("Mueve la blusa después de cada captura y retira tus manos.")
    print("=" * 68)

    if total_count >= TOTAL_IMAGES:
        print("El dataset ya contiene las 150 imágenes configuradas.")
        return

    capture = open_camera()

    if capture is None:
        raise RuntimeError(
            "No se pudo abrir la cámara RTSP. "
            "Verifica que app.py esté detenido y que la cámara esté disponible."
        )

    window_name = f"Captura PatchCore - {DATASET_NAME}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    last_message = (
        f"Inicio en {INITIAL_DELAY_SECONDS} segundos. "
        "Acomoda la blusa y retira tus manos."
    )

    next_capture_time = time.monotonic() + INITIAL_DELAY_SECONDS

    try:
        while total_count < TOTAL_IMAGES:
            ok, frame = capture.read()

            if not ok or frame is None:
                last_message = "Reconectando cámara..."
                capture.release()
                time.sleep(1)
                capture = open_camera()

                if capture is None:
                    time.sleep(2)
                    continue

                next_capture_time = time.monotonic() + INTERVAL_SECONDS
                continue

            destination, destination_name = next_destination(
                train_count,
                test_count,
            )

            now = time.monotonic()
            remaining = next_capture_time - now

            preview = draw_status(
                frame=frame,
                total_count=total_count,
                remaining_seconds=remaining,
                destination_name=destination_name,
                paused=paused,
                last_message=last_message,
            )

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("\nCaptura detenida por el usuario.")
                break

            if key == ord("p"):
                paused = not paused

                if paused:
                    last_message = "Captura pausada."
                else:
                    last_message = (
                        f"Captura reanudada. Próxima foto en "
                        f"{INTERVAL_SECONDS} segundos."
                    )
                    next_capture_time = (
                        time.monotonic() + INTERVAL_SECONDS
                    )

            force_capture = key == ord("c")

            if paused:
                continue

            if now >= next_capture_time or force_capture:
                sharpness = calculate_sharpness(frame)

                if sharpness < MIN_SHARPNESS:
                    last_message = (
                        f"Foto rechazada por desenfoque "
                        f"({sharpness:.1f} < {MIN_SHARPNESS:.1f})."
                    )
                    print(f"[RECHAZADA] {last_message}")
                    next_capture_time = time.monotonic() + 3
                    continue

                total_number = total_count + 1

                save_capture(
                    frame=frame,
                    destination=destination,
                    category=destination_name,
                    number=total_number,
                )

                if destination_name == "train/good":
                    train_count += 1
                else:
                    test_count += 1

                total_count += 1

                last_message = (
                    f"Captura {total_count}/{TOTAL_IMAGES} guardada. "
                    "Ya puedes mover la blusa."
                )

                next_capture_time = (
                    time.monotonic() + INTERVAL_SECONDS
                )

        if total_count >= TOTAL_IMAGES:
            print("\n" + "=" * 68)
            print("DATASET COMPLETADO")
            print("=" * 68)
            print(f"Train/good: {train_count}")
            print(f"Test/good:  {test_count}")
            print(f"Ruta: {ROOT / 'anomaly_dataset' / DATASET_NAME}")
            print("=" * 68)

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
