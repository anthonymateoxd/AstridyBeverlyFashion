from pathlib import Path
from datetime import datetime
import argparse
import csv
import time

import cv2

from capture_patchcore_curated_v2 import (
    open_camera,
    fresh_frame,
)
from preprocess_patchcore_v2 import preprocess


DATA_ROOT = Path(
    "/data/BLUSA_002_S_ROSA_FRONT_V2"
)

VALID_SPLITS = {
    "validation",
    "test",
}

VALID_CLASSES = {
    "good",
    "mancha",
    "agujero",
}


def next_index(directory: Path, split: str, label: str) -> int:
    """
    Devuelve un indice historico unico.

    Busca tanto muestras activas como archivadas para evitar
    reutilizar identificadores despues del control de calidad.
    """

    # Se conserva el parametro por compatibilidad con las llamadas
    # existentes, pero el indice debe ser global para split/clase.
    _ = directory

    highest = 0
    prefix = f"{split}_{label}_"

    for path in DATA_ROOT.rglob(
        f"{prefix}*.jpg"
    ):
        rest = path.stem[len(prefix):]

        index_text = rest.split(
            "_",
            1,
        )[0]

        try:
            index = int(index_text)
        except ValueError:
            continue

        highest = max(
            highest,
            index,
        )

    return highest + 1




def append_manifest(
    manifest: Path,
    split: str,
    label: str,
    index: int,
    raw_name: str,
    processed_name: str,
    coverage: float,
):
    exists = manifest.exists()

    fieldnames = [
        "split",
        "class",
        "index",
        "raw_filename",
        "processed_filename",
        "garment_coverage",
        "captured_at",
    ]

    with manifest.open(
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
            "split": split,
            "class": label,
            "index": index,
            "raw_filename": raw_name,
            "processed_filename": processed_name,
            "garment_coverage": f"{coverage:.6f}",
            "captured_at": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
        })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(VALID_SPLITS),
    )

    parser.add_argument(
        "--class",
        dest="label",
        required=True,
        choices=sorted(VALID_CLASSES),
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit(
            "ERROR: --count debe ser mayor que cero."
        )

    raw_dir = (
        DATA_ROOT
        / "raw"
        / args.split
        / args.label
    )

    processed_dir = (
        DATA_ROOT
        / "processed"
        / args.split
        / args.label
    )

    manifest = (
        DATA_ROOT
        / f"{args.split}_manifest_v2.csv"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = len(
        list(
            processed_dir.glob("*.jpg")
        )
    )

    target_total = (
        existing + args.count
    )

    print("=" * 72)
    print(
        "ASTRID QUALITY CONTROL - "
        "CAPTURA EVALUACION V2"
    )
    print("=" * 72)
    print(f"Split:       {args.split}")
    print(f"Clase:       {args.label}")
    print(f"Existentes:  {existing}")
    print(f"Nuevas:      {args.count}")
    print(f"Objetivo:    {target_total}")
    print()
    print("Enter = capturar")
    print("q + Enter = terminar")
    print()
    print(
        "IMPORTANTE: estas imagenes NO pertenecen "
        "al entrenamiento."
    )
    print("=" * 72)

    cap = None

    try:
        while (
            len(
                list(
                    processed_dir.glob("*.jpg")
                )
            )
            < target_total
        ):
            current = len(
                list(
                    processed_dir.glob("*.jpg")
                )
            )

            action = input(
                f"\n[{current}/{target_total}] "
                "Acomoda la blusa y presiona Enter "
                "(q para salir): "
            ).strip().lower()

            if action == "q":
                break

            print(
                f"Retira las manos. "
                f"Estabilizando {args.delay:.0f} segundos..."
            )

            time.sleep(
                args.delay
            )

            # Apertura RTSP nueva en cada captura:
            # evita procesar frames antiguos del buffer.
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

                cap = None

            time.sleep(0.25)

            try:
                cap = open_camera()
                frame = fresh_frame(cap)

            except Exception as exc:
                print()
                print("FALLO TRANSITORIO RTSP:")
                print(exc)
                print(
                    "La captura no fue contabilizada."
                )
                continue

            try:
                processed, coverage = preprocess(
                    frame
                )

            except Exception as exc:
                print()
                print(
                    "RECHAZADA POR SEGMENTACION:"
                )
                print(exc)
                print(
                    "La captura no fue contabilizada."
                )
                continue

            index = next_index(
                processed_dir,
                args.split,
                args.label,
            )

            stamp = (
                datetime.now()
                .astimezone()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )

            raw_name = (
                f"{args.split}_{args.label}_"
                f"{index:03d}_{stamp}.jpg"
            )

            processed_name = raw_name

            raw_path = (
                raw_dir
                / raw_name
            )

            processed_path = (
                processed_dir
                / processed_name
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

            append_manifest(
                manifest,
                args.split,
                args.label,
                index,
                raw_name,
                processed_name,
                coverage,
            )

            current = len(
                list(
                    processed_dir.glob("*.jpg")
                )
            )

            print()
            print("CAPTURA ACEPTADA")
            print(f"Indice:     {index}")
            print(
                f"Cobertura:  "
                f"{coverage * 100:.2f}%"
            )
            print(
                f"Total:      "
                f"{current}/{target_total}"
            )

    finally:
        if cap is not None:
            cap.release()

    final_count = len(
        list(
            processed_dir.glob("*.jpg")
        )
    )

    print()
    print("=" * 72)
    print(
        f"Resultado {args.split}/{args.label}: "
        f"{final_count}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
