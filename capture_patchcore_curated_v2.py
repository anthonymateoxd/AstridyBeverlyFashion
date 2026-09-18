from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from dotenv import load_dotenv

from preprocess_patchcore_v2 import preprocess


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(
    "/data/BLUSA_002_S_ROSA_FRONT_V2"
)

CURATED_DIR = (
    DATA_ROOT
    / "processed"
    / "train"
    / "good_v2_curated"
)

RAW_ACCEPTED_DIR = (
    DATA_ROOT
    / "raw"
    / "train"
    / "good_v2_curated_new"
)

REJECTED_DIR = (
    DATA_ROOT
    / "diagnostics"
    / "curated_rejected"
)

MANIFEST = (
    DATA_ROOT
    / "train_curated_v2.csv"
)

TARGET_UNIQUE = 18

CORR_LIMIT = 0.995
MAD_LIMIT = 3.5


CAMERA_IP = os.getenv(
    "CAMERA_IP",
    "",
).strip()

CAMERA_USER = os.getenv(
    "CAMERA_USER",
    "admin",
).strip()

CAMERA_PASSWORD = os.getenv(
    "CAMERA_PASSWORD",
    "",
)

CAMERA_RTSP_PORT = int(
    os.getenv(
        "CAMERA_RTSP_PORT",
        "554",
    )
)

CAMERA_CHANNEL = int(
    os.getenv(
        "CAMERA_CHANNEL",
        "1",
    )
)

CAMERA_SUBTYPE = int(
    os.getenv(
        "CAMERA_SUBTYPE",
        "0",
    )
)


def build_rtsp_url():
    user = quote(
        CAMERA_USER,
        safe="",
    )

    password = quote(
        CAMERA_PASSWORD,
        safe="",
    )

    return (
        f"rtsp://{user}:{password}"
        f"@{CAMERA_IP}:{CAMERA_RTSP_PORT}"
        f"/cam/realmonitor?"
        f"channel={CAMERA_CHANNEL}"
        f"&subtype={CAMERA_SUBTYPE}"
    )


def open_camera():
    cap = cv2.VideoCapture(
        build_rtsp_url(),
        cv2.CAP_FFMPEG,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir el flujo RTSP."
        )

    return cap


def fresh_frame(cap):
    frame = None

    for _ in range(8):
        ok, candidate = cap.read()

        if ok and candidate is not None:
            frame = candidate

    if frame is None:
        raise RuntimeError(
            "No se obtuvo un frame valido."
        )

    return frame


def feature(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.resize(
        gray,
        (480, 270),
        interpolation=cv2.INTER_AREA,
    )


def compare(a, b):
    mad = float(
        cv2.absdiff(
            a,
            b,
        ).mean()
    )

    corr = float(
        np.corrcoef(
            a.astype(
                np.float32
            ).ravel(),
            b.astype(
                np.float32
            ).ravel(),
        )[0, 1]
    )

    return corr, mad


def load_curated():
    items = []

    for path in sorted(
        CURATED_DIR.glob("*.jpg")
    ):
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"No se pudo leer {path.name}"
            )

        items.append(
            (
                path,
                feature(image),
            )
        )

    return items


def next_curated_index():
    """
    Devuelve un identificador historico unico.

    No depende de la cantidad de muestras activas porque algunas
    pueden haber sido archivadas durante QC. Se inspeccionan tanto
    activas como archivadas dentro de DATA_ROOT para no reutilizar
    indices anteriores.
    """

    highest = 0

    for path in DATA_ROOT.rglob("normal_v2_*.jpg"):
        name = path.name

        parts = name.split("_", 3)

        if len(parts) < 4:
            continue

        if parts[0] != "normal" or parts[1] != "v2":
            continue

        try:
            index = int(parts[2])
        except ValueError:
            continue

        highest = max(
            highest,
            index,
        )

    return highest + 1


def append_manifest(
    index,
    raw_name,
    curated_name,
):
    exists = MANIFEST.exists()

    fieldnames = [
        "index",
        "source",
        "original_filename",
        "curated_filename",
    ]

    with MANIFEST.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            "index": index,
            "source": "MANUAL_CURATED",
            "original_filename": raw_name,
            "curated_filename": curated_name,
        })


def save_rejected(
    frame,
    reason,
):
    REJECTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"rejected_{reason}_{stamp}.jpg"
    )

    path = REJECTED_DIR / filename

    cv2.imwrite(
        str(path),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            97,
        ],
    )

    return path


def main():
    CURATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_ACCEPTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    curated = load_curated()

    if len(curated) < 1:
        raise RuntimeError(
            "El dataset curado esta vacio."
        )

    print("=" * 72)
    print(
        "ASTRID QUALITY CONTROL "
        "- CAPTURA CURADA PATCHCORE V2"
    )
    print("=" * 72)
    print(
        f"Unicas actuales: "
        f"{len(curated)}/{TARGET_UNIQUE}"
    )
    print(
        f"Camara: {CAMERA_IP}"
    )
    print()
    print(
        "Enter = evaluar nueva posicion"
    )
    print(
        "q + Enter = terminar"
    )
    print()
    print(
        "Solo se guardara si pasa "
        "segmentacion Y diversidad."
    )
    print("=" * 72)

    cap = open_camera()

    try:
        for _ in range(10):
            cap.read()

        while len(curated) < TARGET_UNIQUE:

            action = input(
                f"\n[{len(curated)}/{TARGET_UNIQUE}] "
                "Acomoda la blusa y presiona Enter "
                "(q para salir): "
            ).strip().lower()

            if action == "q":
                break

            print(
                "Retira las manos. "
                "Estabilizando 5 segundos..."
            )

            time.sleep(5)

            # IMPORTANTE:
            # Para cada evaluacion se abre una sesion RTSP nueva.
            # Esto evita procesar frames antiguos que hayan quedado
            # almacenados en buffers de OpenCV/FFmpeg.
            try:
                cap.release()
            except Exception:
                pass

            time.sleep(0.25)

            try:
                cap = open_camera()
                frame = fresh_frame(cap)
            except RuntimeError as exc:
                print()
                print("FALLO TRANSITORIO RTSP:")
                print(exc)
                print("Reconectando camara...")

                try:
                    cap.release()
                except Exception:
                    pass

                time.sleep(2)

                try:
                    cap = open_camera()

                    for _ in range(12):
                        cap.read()

                    print(
                        "OK: camara reconectada. "
                        "La captura no fue contabilizada."
                    )

                except Exception as reconnect_error:
                    print(
                        "ERROR: no se pudo reconectar: "
                        f"{reconnect_error}"
                    )

                    time.sleep(2)

                continue

            try:
                processed, coverage = (
                    preprocess(frame)
                )
            except Exception as exc:
                rejected = save_rejected(
                    frame,
                    "segmentacion",
                )

                print()
                print(
                    "RECHAZADA POR SEGMENTACION"
                )
                print(exc)
                print(
                    f"Diagnostico: {rejected}"
                )
                continue

            current_feature = feature(
                processed
            )

            duplicate = None
            nearest_corr = -1.0
            nearest_mad = 999.0

            for path, previous_feature in curated:
                corr, mad = compare(
                    current_feature,
                    previous_feature,
                )

                if corr > nearest_corr:
                    nearest_corr = corr
                    nearest_mad = mad

                if (
                    corr >= CORR_LIMIT
                    and mad < MAD_LIMIT
                ):
                    duplicate = path
                    break

            if duplicate is not None:
                rejected = save_rejected(
                    frame,
                    "redundancia",
                )

                print()
                print(
                    "RECHAZADA POR REDUNDANCIA"
                )
                print(
                    f"Similar a: "
                    f"{duplicate.name}"
                )
                print(
                    f"corr={nearest_corr:.6f}"
                )
                print(
                    f"MAD={nearest_mad:.3f}"
                )
                print(
                    f"Diagnostico: {rejected}"
                )
                continue

            next_index = next_curated_index()

            stamp = (
                datetime.now()
                .astimezone()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )

            raw_name = (
                f"curated_good_"
                f"{next_index:02d}_"
                f"{stamp}.jpg"
            )

            curated_name = (
                f"normal_v2_"
                f"{next_index:02d}_"
                f"{raw_name}"
            )

            raw_path = (
                RAW_ACCEPTED_DIR
                / raw_name
            )

            curated_path = (
                CURATED_DIR
                / curated_name
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
                str(curated_path),
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

            append_manifest(
                next_index,
                raw_name,
                curated_name,
            )

            curated.append(
                (
                    curated_path,
                    current_feature,
                )
            )

            print()
            print(
                "ACEPTADA COMO NUEVA NORMAL"
            )
            print(
                f"Indice:     {next_index}"
            )
            print(
                f"Cobertura:  "
                f"{coverage * 100:.2f}%"
            )
            print(
                f"Corr max:   "
                f"{nearest_corr:.6f}"
            )
            print(
                f"MAD:        "
                f"{nearest_mad:.3f}"
            )
            print(
                f"Total:      "
                f"{len(curated)}/{TARGET_UNIQUE}"
            )

    finally:
        cap.release()

    print()
    print("=" * 72)
    print(
        f"Dataset curado actual: "
        f"{len(curated)}/{TARGET_UNIQUE}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
