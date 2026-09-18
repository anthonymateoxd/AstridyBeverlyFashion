from __future__ import annotations

import csv
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
from dotenv import load_dotenv

from preprocess_patchcore_v2 import preprocess


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(
    "/data/BLUSA_002_S_ROSA_FRONT_V2"
)

RAW_DIR = (
    DATA_ROOT
    / "raw"
    / "train"
    / "good_manual"
)

PROCESSED_DIR = (
    DATA_ROOT
    / "processed"
    / "train"
    / "good_manual_candidate"
)

METADATA = (
    DATA_ROOT
    / "manual_capture_metadata.csv"
)

TARGET_TOTAL = 26

CAMERA_IP = os.getenv("CAMERA_IP", "").strip()
CAMERA_USER = os.getenv("CAMERA_USER", "admin").strip()
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
CAMERA_RTSP_PORT = int(
    os.getenv("CAMERA_RTSP_PORT", "554")
)
CAMERA_CHANNEL = int(
    os.getenv("CAMERA_CHANNEL", "1")
)
CAMERA_SUBTYPE = int(
    os.getenv("CAMERA_SUBTYPE", "0")
)

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

user = quote(CAMERA_USER, safe="")
password = quote(CAMERA_PASSWORD, safe="")

CAMERA_SOURCE = (
    f"rtsp://{user}:{password}"
    f"@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?"
    f"channel={CAMERA_CHANNEL}"
    f"&subtype={CAMERA_SUBTYPE}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def image_count(folder: Path) -> int:
    return len(
        list(folder.glob("*.jpg"))
    )


def open_camera():
    capture = cv2.VideoCapture(
        CAMERA_SOURCE,
        cv2.CAP_FFMPEG,
    )

    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not capture.isOpened():
        raise RuntimeError(
            "No se pudo abrir RTSP."
        )

    return capture


def fresh_frame(capture):
    frame = None

    for _ in range(6):
        ok, candidate = capture.read()

        if ok and candidate is not None:
            frame = candidate

    if frame is None:
        raise RuntimeError(
            "No se obtuvo un frame valido."
        )

    return frame


def append_metadata(row):
    exists = METADATA.exists()

    with METADATA.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    current = image_count(RAW_DIR)

    print("=" * 68)
    print("ASTRID QUALITY CONTROL - CAPTURA MANUAL V2")
    print("=" * 68)
    print(f"Camara:       {CAMERA_IP}")
    print(f"Capturadas:   {current}/{TARGET_TOTAL}")
    print()
    print("Reglas:")
    print("- Blusa normal, talla S.")
    print("- Sin manchas ni agujeros.")
    print("- Sin manos dentro del encuadre.")
    print("- Orientacion realista de produccion.")
    print("- Enter = capturar.")
    print("- q + Enter = terminar.")
    print("=" * 68)

    capture = open_camera()

    try:
        for _ in range(10):
            capture.read()

        while current < TARGET_TOTAL:

            action = input(
                f"\n[{current}/{TARGET_TOTAL}] "
                "Acomoda la blusa y presiona Enter "
                "(q para salir): "
            ).strip().lower()

            if action == "q":
                break

            print(
                "Retira completamente las manos. "
                "Captura en 5 segundos..."
            )

            time.sleep(5)

            frame = fresh_frame(capture)

            try:
                processed, coverage = preprocess(
                    frame
                )
            except Exception as exc:
                print()
                print(
                    "RECHAZADA AUTOMATICAMENTE:"
                )
                print(exc)
                print(
                    "No se guardo en el dataset."
                )
                continue

            sequence = current + 1

            timestamp = (
                datetime.now()
                .astimezone()
            )

            suffix = (
                timestamp.strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )

            raw_name = (
                f"manual_good_"
                f"{sequence:04d}_"
                f"{suffix}.jpg"
            )

            raw_path = RAW_DIR / raw_name
            processed_path = (
                PROCESSED_DIR
                / raw_name
            )

            if not cv2.imwrite(
                str(raw_path),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    97,
                ],
            ):
                raise RuntimeError(
                    "No se pudo guardar RAW."
                )

            if not cv2.imwrite(
                str(processed_path),
                processed,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    97,
                ],
            ):
                raw_path.unlink(
                    missing_ok=True
                )

                raise RuntimeError(
                    "No se pudo guardar procesada."
                )

            h, w = frame.shape[:2]

            append_metadata({
                "filename": raw_name,
                "sequence": sequence,
                "timestamp": (
                    timestamp.isoformat()
                ),
                "width_px": w,
                "height_px": h,
                "garment_coverage": (
                    f"{coverage:.6f}"
                ),
                "raw_sha256": (
                    sha256_file(raw_path)
                ),
                "processed_sha256": (
                    sha256_file(
                        processed_path
                    )
                ),
            })

            current += 1

            print(
                f"OK: {raw_name}"
            )
            print(
                f"Cobertura: "
                f"{coverage * 100:.2f}%"
            )
            print(
                f"Total: "
                f"{current}/{TARGET_TOTAL}"
            )

    finally:
        capture.release()

    print()
    print("=" * 68)
    print(
        f"Sesion terminada. "
        f"Capturas validas: "
        f"{current}/{TARGET_TOTAL}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
