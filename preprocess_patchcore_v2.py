from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


ROI_X1 = 0.16
ROI_Y1 = 0.05
ROI_X2 = 0.84
ROI_Y2 = 0.88

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_roi_bounds(image):
    h, w = image.shape[:2]

    x1 = int(
        max(0.0, min(ROI_X1, 0.99))
        * w
    )
    y1 = int(
        max(0.0, min(ROI_Y1, 0.99))
        * h
    )
    x2 = int(
        max(0.01, min(ROI_X2, 1.0))
        * w
    )
    y2 = int(
        max(0.01, min(ROI_Y2, 1.0))
        * h
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "ROI invalido."
        )

    return x1, y1, x2, y2


def create_garment_mask(image):
    """
    Replica la segmentacion actual de produccion.

    El rosa se utiliza como semilla y luego se
    rellena el contorno exterior completo de la prenda.
    """

    if image is None:
        raise ValueError(
            "Imagen invalida."
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
            "ROI vacio."
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

    seed = np.where(
        (saturation >= 30)
        & (lab_a >= 136),
        255,
        0,
    ).astype(np.uint8)

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
        raise RuntimeError(
            "No se detecto la prenda."
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
        raise RuntimeError(
            "No se encontro una silueta valida."
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

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "No se encontro el contorno."
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

    if coverage < 0.20:
        raise RuntimeError(
            f"Silueta demasiado pequena: "
            f"{coverage * 100:.2f}%"
        )

    if coverage > 0.75:
        raise RuntimeError(
            f"Silueta demasiado grande: "
            f"{coverage * 100:.2f}%"
        )

    garment_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    garment_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ] = garment_roi

    return garment_mask, coverage


def preprocess(image):
    garment_mask, coverage = (
        create_garment_mask(image)
    )

    x1, y1, x2, y2 = (
        get_roi_bounds(image)
    )

    roi = image[
        y1:y2,
        x1:x2,
    ]

    roi_mask = garment_mask[
        y1:y2,
        x1:x2,
    ]

    processed = np.full_like(
        roi,
        255,
    )

    processed[
        roi_mask > 0
    ] = roi[
        roi_mask > 0
    ]

    return processed, coverage


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="/data/BLUSA_002_S_ROSA_FRONT_V2",
    )

    args = parser.parse_args()

    root = Path(args.data_root)

    source = (
        root
        / "selected"
        / "train"
        / "good"
    )

    destination = (
        root
        / "processed"
        / "train"
        / "good"
    )

    manifest = (
        root
        / "preprocessing_manifest_v2.csv"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        p
        for p in source.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if len(images) != 28:
        raise RuntimeError(
            f"Se esperaban 28 imagenes "
            f"seleccionadas y hay {len(images)}."
        )

    # Esta carpeta es generada.
    for old in destination.iterdir():
        if (
            old.is_file()
            and old.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            old.unlink()

    rows = []
    failures = []

    for index, src in enumerate(
        images,
        start=1,
    ):
        try:
            image = cv2.imread(
                str(src)
            )

            if image is None:
                raise RuntimeError(
                    "cv2.imread devolvio None."
                )

            processed, coverage = (
                preprocess(image)
            )

            dst = (
                destination
                / src.name
            )

            ok = cv2.imwrite(
                str(dst),
                processed,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    97,
                ],
            )

            if not ok:
                raise IOError(
                    "cv2.imwrite fallo."
                )

            ph, pw = (
                processed.shape[:2]
            )

            rows.append({
                "filename": src.name,
                "source_sha256": (
                    sha256_file(src)
                ),
                "processed_sha256": (
                    sha256_file(dst)
                ),
                "processed_width": pw,
                "processed_height": ph,
                "garment_coverage": (
                    f"{coverage:.6f}"
                ),
                "status": "OK",
            })

            print(
                f"[{index:02d}/"
                f"{len(images):02d}] "
                f"OK | "
                f"cobertura="
                f"{coverage * 100:.2f}% | "
                f"{pw}x{ph} | "
                f"{src.name}"
            )

        except Exception as exc:
            failures.append(
                (
                    src.name,
                    str(exc),
                )
            )

            print(
                f"[{index:02d}/"
                f"{len(images):02d}] "
                f"ERROR | "
                f"{src.name} | "
                f"{exc}"
            )

    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filename",
                "source_sha256",
                "processed_sha256",
                "processed_width",
                "processed_height",
                "garment_coverage",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "===== RESUMEN "
        "PREPROCESAMIENTO ====="
    )
    print(
        f"Seleccionadas: {len(images)}"
    )
    print(
        f"Procesadas OK: {len(rows)}"
    )
    print(
        f"Fallidas:      {len(failures)}"
    )

    if rows:
        coverages = np.asarray(
            [
                float(
                    row[
                        "garment_coverage"
                    ]
                )
                for row in rows
            ],
            dtype=float,
        )

        print(
            "Cobertura minima: "
            f"{coverages.min() * 100:.2f}%"
        )
        print(
            "Cobertura media:  "
            f"{coverages.mean() * 100:.2f}%"
        )
        print(
            "Cobertura maxima: "
            f"{coverages.max() * 100:.2f}%"
        )

    if failures:
        print()
        print("===== FALLOS =====")

        for filename, error in failures:
            print(
                f"{filename}: {error}"
            )

        raise SystemExit(
            "ERROR: existen imagenes "
            "que no pudieron procesarse."
        )

    if len(rows) != 28:
        raise SystemExit(
            "ERROR: el resultado no "
            "contiene exactamente 28 "
            "imagenes."
        )

    print()
    print(
        "OK: dataset normal v2 "
        "preprocesado."
    )
    print(
        f"Manifest: {manifest}"
    )


if __name__ == "__main__":
    main()
