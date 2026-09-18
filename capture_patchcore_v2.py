from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATASET_NAME = "BLUSA_002_S_ROSA_FRONT_V2"

PHASES = {
    "scale_calibration": ("scale_calibration", "calibration"),
    "scale_calibration_vertical": ("scale_calibration_vertical", "calibration"),
    "train_good": ("train/good", "good"),
    "calibration_good": ("calibration/good", "good"),
    "calibration_mancha": ("calibration/mancha", "mancha"),
    "calibration_agujero": ("calibration/agujero", "agujero"),
    "test_good": ("test/good", "good"),
    "test_mancha": ("test/mancha", "mancha"),
    "test_agujero": ("test/agujero", "agujero"),
}

DEFAULT_TARGETS = {
    "scale_calibration": 1,
    "scale_calibration_vertical": 1,
    "train_good": 80,
    "calibration_good": 10,
    "calibration_mancha": 10,
    "calibration_agujero": 10,
    "test_good": 20,
    "test_mancha": 20,
    "test_agujero": 20,
}

CAMERA_IP = os.getenv("CAMERA_IP", "").strip()
CAMERA_USER = os.getenv("CAMERA_USER", "admin").strip()
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")
CAMERA_RTSP_PORT = int(os.getenv("CAMERA_RTSP_PORT", "554"))
CAMERA_CHANNEL = int(os.getenv("CAMERA_CHANNEL", "1"))
CAMERA_SUBTYPE = int(os.getenv("CAMERA_SUBTYPE", "0"))

ROI_X1 = float(os.getenv("ROI_X1", "0.16"))
ROI_Y1 = float(os.getenv("ROI_Y1", "0.05"))
ROI_X2 = float(os.getenv("ROI_X2", "0.84"))
ROI_Y2 = float(os.getenv("ROI_Y2", "0.88"))

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp",
)

user = quote(CAMERA_USER, safe="")
password = quote(CAMERA_PASSWORD, safe="")

CAMERA_SOURCE = (
    f"rtsp://{user}:{password}"
    f"@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor?channel={CAMERA_CHANNEL}"
    f"&subtype={CAMERA_SUBTYPE}"
)


def image_count(folder: Path) -> int:
    if not folder.exists():
        return 0

    return sum(
        1
        for item in folder.iterdir()
        if item.is_file()
        and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


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
        capture.release()
        raise RuntimeError(
            "No se pudo abrir el flujo RTSP."
        )

    return capture


def get_fresh_frame(capture):
    frame = None

    # Vaciar algunos frames del buffer.
    for _ in range(5):
        ok, candidate = capture.read()

        if ok and candidate is not None:
            frame = candidate

    if frame is None:
        raise RuntimeError(
            "No se recibio un frame valido."
        )

    return frame


def append_metadata(
    metadata_path: Path,
    data: dict,
):
    exists = metadata_path.exists()

    with metadata_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(data.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(data)


def main():
    parser = argparse.ArgumentParser(
        description="Captura RAW para PatchCore v2."
    )

    parser.add_argument(
        "phase",
        choices=sorted(PHASES),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Cantidad a capturar en esta sesion.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Segundos entre capturas.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Espera antes de iniciar.",
    )

    parser.add_argument(
        "--data-root",
        default="/data",
    )

    args = parser.parse_args()

    relative_dir, label = PHASES[args.phase]

    dataset_root = (
        Path(args.data_root)
        / DATASET_NAME
        / "raw"
    )

    destination = (
        dataset_root
        / relative_dir
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        Path(args.data_root)
        / DATASET_NAME
        / "capture_metadata.csv"
    )

    existing = image_count(destination)

    target = DEFAULT_TARGETS[args.phase]

    remaining = max(
        0,
        target - existing,
    )

    requested = (
        remaining
        if args.count is None
        else min(
            args.count,
            remaining,
        )
    )

    print("=" * 64)
    print("ASTRID QUALITY CONTROL - CAPTURA IA V2")
    print("=" * 64)
    print(f"Fase:       {args.phase}")
    print(f"Etiqueta:   {label}")
    print(f"Camara:     {CAMERA_IP}")
    print(f"Existentes: {existing}")
    print(f"Objetivo:   {target}")
    print(f"Sesion:     {requested}")
    print(f"Destino:    {destination}")
    print("=" * 64)

    if requested <= 0:
        print("Esta fase ya alcanzo su objetivo.")
        return

    print(
        f"\nInicio en {args.delay:.0f} segundos."
    )
    time.sleep(args.delay)

    capture = open_camera()

    try:
        # Warm-up inicial.
        for _ in range(10):
            capture.read()

        saved_count = 0

        while saved_count < requested:
            sequence = (
                existing
                + saved_count
                + 1
            )

            frame = get_fresh_frame(capture)

            h, w = frame.shape[:2]

            sharp = sharpness(frame)
            bright = brightness(frame)

            timestamp = datetime.now().astimezone()

            filename = (
                f"{args.phase}_"
                f"{sequence:04d}_"
                f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )

            output = (
                destination
                / filename
            )

            ok = cv2.imwrite(
                str(output),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    97,
                ],
            )

            if not ok:
                raise IOError(
                    f"No se pudo guardar {output}"
                )

            digest = file_sha256(output)

            append_metadata(
                metadata_path,
                {
                    "filename": filename,
                    "phase": args.phase,
                    "label": label,
                    "sequence": sequence,
                    "timestamp": timestamp.isoformat(),
                    "camera_ip": CAMERA_IP,
                    "width_px": w,
                    "height_px": h,
                    "sharpness": f"{sharp:.4f}",
                    "brightness": f"{bright:.4f}",
                    "roi_x1": ROI_X1,
                    "roi_y1": ROI_Y1,
                    "roi_x2": ROI_X2,
                    "roi_y2": ROI_Y2,
                    "sha256": digest,
                },
            )

            saved_count += 1

            print(
                f"[{saved_count}/{requested}] "
                f"{filename} | "
                f"{w}x{h} | "
                f"nitidez={sharp:.1f} | "
                f"brillo={bright:.1f}"
            )

            if saved_count < requested:
                time.sleep(
                    max(
                        0.1,
                        args.interval,
                    )
                )

    finally:
        capture.release()

    print()
    print("OK: sesion finalizada.")
    print(
        f"Total actual en fase: "
        f"{image_count(destination)}/{target}"
    )


if __name__ == "__main__":
    main()
