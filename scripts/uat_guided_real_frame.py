"""Inspección óptica de una foto de cámara: solo lectura; sin lotes ni BD.

Ejemplo:
  python scripts/uat_guided_real_frame.py --image FOTO.jpg --roi 0.16 0.05 0.84 0.88
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluar segmentación sin registrar inspecciones")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
        help="ROI normalizado, tomado de la configuración real del servidor",
    )
    parser.add_argument(
        "--force-adaptive", action="store_true",
        help="Forzar diagnóstico del método adaptativo sin activar la semilla cromática",
    )
    arguments = parser.parse_args()

    if not arguments.image.is_file():
        parser.error("No existe el fotograma solicitado")
    if arguments.roi:
        x1, y1, x2, y2 = arguments.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            parser.error("El ROI normalizado está fuera de rango")
        for key, value in zip(("ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2"), arguments.roi):
            os.environ[key] = str(value)

    os.environ.setdefault("SECRET_KEY", "uat-guided-frame-test-only")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cv2  # noqa: PLC0415
    import app as application  # noqa: PLC0415

    frame = cv2.imread(str(arguments.image))
    if frame is None:
        parser.error("No fue posible decodificar el fotograma")
    if arguments.force_adaptive:
        application.create_garment_mask = lambda _frame: None

    diagnostics = {}
    mask = application.create_guided_garment_mask(frame, diagnostics)
    roi = application.get_roi_bounds(frame)
    x1, y1, x2, y2 = roi
    region = mask[y1:y2, x1:x2]
    coverage = cv2.countNonZero(region) / max(1, region.size)
    result = {
        "status": "detected" if diagnostics.get("mask_bbox") else "not_detected",
        "method": diagnostics.get("method"),
        "reason": diagnostics.get("reason"),
        "coverage_ratio": round(coverage, 5),
        "bbox": diagnostics.get("mask_bbox"),
        "adaptive_candidate_ratio": diagnostics.get("adaptive", {}).get("adaptive_coverage"),
        "frame_dimensions": [frame.shape[1], frame.shape[0]],
        "roi_pixels": list(roi),
        "readonly": True,
    }
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
