from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from anomalib.engine import Engine
from anomalib.models import Patchcore


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def scalar(value: Any) -> float:
    if value is None:
        return 0.0
    array = np.asarray(to_numpy(value)).reshape(-1)
    return float(array[0]) if array.size else 0.0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def extract_map(value: Any, width: int, height: int) -> np.ndarray:
    anomaly_map = np.squeeze(to_numpy(value)).astype(np.float32)
    if anomaly_map.ndim != 2:
        raise ValueError(f"Forma inesperada del anomaly_map: {anomaly_map.shape}")

    return cv2.resize(
        anomaly_map,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def build_garment_mask(
    image_rgb: np.ndarray,
    gray_value: int = 205,
    tolerance: int = 8,
) -> np.ndarray:
    background = np.full_like(image_rgb, gray_value, dtype=np.int16)
    difference = np.max(
        np.abs(image_rgb.astype(np.int16) - background),
        axis=2,
    )

    mask = np.where(difference > tolerance, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.full(mask.shape, 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    garment_mask = np.zeros_like(mask)
    cv2.drawContours(
        garment_mask,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED,
    )
    return garment_mask


def localize_strongest_regions(
    anomaly_map: np.ndarray,
    garment_mask: np.ndarray,
    min_area: int,
    max_regions: int,
    percentile: float,
    absolute_floor: float,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    garment_pixels = anomaly_map[garment_mask > 0]

    if garment_pixels.size == 0:
        return np.zeros_like(garment_mask), [], absolute_floor

    percentile_threshold = float(np.percentile(garment_pixels, percentile))
    relative_threshold = float(np.max(garment_pixels)) * 0.84

    dynamic_threshold = max(
        absolute_floor,
        percentile_threshold,
        relative_threshold,
    )

    candidate = np.where(
        (anomaly_map >= dynamic_threshold) & (garment_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    distance_to_edge = cv2.distanceTransform(
        np.where(garment_mask > 0, 255, 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[dict[str, Any]] = []

    for contour in contours:
        area = int(round(cv2.contourArea(contour)))
        if area < min_area:
            continue

        component = np.zeros_like(candidate)
        cv2.drawContours(
            component,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        pixels = component > 0
        scores = anomaly_map[pixels]
        edge_distances = distance_to_edge[pixels]

        if scores.size == 0:
            continue

        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        median_edge_distance = (
            float(np.median(edge_distances))
            if edge_distances.size
            else 0.0
        )

        interior_bonus = min(median_edge_distance / 18.0, 1.0)
        ranking_score = (
            0.55 * max_score
            + 0.35 * mean_score
            + 0.10 * interior_bonus
        )

        x, y, width, height = cv2.boundingRect(contour)

        candidates.append(
            {
                "contour": contour,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": area,
                "mean_score": round(mean_score, 6),
                "max_score": round(max_score, 6),
                "median_edge_distance": round(median_edge_distance, 2),
                "ranking_score": round(float(ranking_score), 6),
            }
        )

    candidates.sort(
        key=lambda region: region["ranking_score"],
        reverse=True,
    )

    selected = candidates[:max_regions]
    filtered_mask = np.zeros_like(candidate)
    public_regions: list[dict[str, Any]] = []

    for region in selected:
        cv2.drawContours(
            filtered_mask,
            [region["contour"]],
            -1,
            255,
            thickness=cv2.FILLED,
        )
        public_regions.append(
            {
                key: value
                for key, value in region.items()
                if key != "contour"
            }
        )

    return filtered_mask, public_regions, dynamic_threshold


def annotate(
    image_rgb: np.ndarray,
    pred_score: float,
    has_anomaly: bool,
    regions: list[dict[str, Any]],
) -> np.ndarray:
    output = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    status = (
        f"DEFECTO DETECTADO | SCORE {pred_score:.3f}"
        if has_anomaly
        else f"PRENDA SIN DEFECTO | SCORE {pred_score:.3f}"
    )
    status_color = (0, 0, 255) if has_anomaly else (0, 170, 0)

    cv2.putText(
        output,
        status,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        status_color,
        2,
        cv2.LINE_AA,
    )

    for index, region in enumerate(regions, start=1):
        x = max(0, region["x"] - 8)
        y = max(0, region["y"] - 8)
        x2 = min(output.shape[1] - 1, region["x"] + region["width"] + 8)
        y2 = min(output.shape[0] - 1, region["y"] + region["height"] + 8)

        cv2.rectangle(
            output,
            (x, y),
            (x2, y2),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            output,
            f"ANOMALIA {index}",
            (x, max(58, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localiza solo las zonas más fuertes del mapa PatchCore."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "patchcore_blusa_modelo_01_ready" / "model.ckpt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("patchcore_localized_results"),
    )
    parser.add_argument("--image-threshold", type=float, default=0.50)
    parser.add_argument("--absolute-floor", type=float, default=0.20)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-regions", type=int, default=1)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"No existe la entrada: {args.data}")

    if not args.checkpoint.exists():
        raise SystemExit(f"No existe el modelo: {args.checkpoint}")

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.10,
        num_neighbors=9,
        visualizer=False,
    )

    engine = Engine(
        default_root_dir=args.output / "runtime",
        accelerator="cpu",
        devices=1,
        logger=False,
    )

    predictions = engine.predict(
        model=model,
        data_path=args.data,
        ckpt_path=args.checkpoint,
        return_predictions=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []

    for batch in predictions:
        for item in batch:
            image_path = Path(item.image_path)
            image_rgb = load_rgb(image_path)
            height, width = image_rgb.shape[:2]

            pred_score = scalar(item.pred_score)
            has_anomaly = pred_score >= args.image_threshold

            anomaly_map = extract_map(item.anomaly_map, width, height)
            garment_mask = build_garment_mask(image_rgb)

            if has_anomaly:
                filtered_mask, regions, threshold_used = localize_strongest_regions(
                    anomaly_map=anomaly_map,
                    garment_mask=garment_mask,
                    min_area=args.min_area,
                    max_regions=args.max_regions,
                    percentile=args.percentile,
                    absolute_floor=args.absolute_floor,
                )
            else:
                filtered_mask = np.zeros((height, width), dtype=np.uint8)
                regions = []
                threshold_used = 0.0

            annotated = annotate(
                image_rgb=image_rgb,
                pred_score=pred_score,
                has_anomaly=has_anomaly,
                regions=regions,
            )

            stem = image_path.stem
            result_path = args.output / f"{stem}_resultado.png"
            mask_path = args.output / f"{stem}_mascara.png"

            cv2.imwrite(str(result_path), annotated)
            cv2.imwrite(str(mask_path), filtered_mask)

            row = {
                "image": str(image_path),
                "pred_score": round(pred_score, 6),
                "has_anomaly": bool(has_anomaly),
                "threshold_used": round(float(threshold_used), 6),
                "region_count": len(regions),
                "regions": regions,
                "result_path": str(result_path),
                "mask_path": str(mask_path),
            }
            report.append(row)

            print("\n=== LOCALIZACIÓN PATCHCORE ===")
            print(f"Imagen: {image_path.name}")
            print(f"pred_score: {pred_score:.6f}")
            print(f"Anomalía global: {has_anomaly}")
            print(f"Umbral local usado: {threshold_used:.6f}")
            print(f"Regiones conservadas: {len(regions)}")

            for index, region in enumerate(regions, start=1):
                print(
                    f"Zona {index}: "
                    f"area={region['area']}, "
                    f"media={region['mean_score']}, "
                    f"max={region['max_score']}, "
                    f"distancia_borde={region['median_edge_distance']}, "
                    f"ranking={region['ranking_score']}"
                )

            print(f"Resultado: {result_path}")

    report_path = args.output / "reporte_localizacion.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReporte: {report_path}")


if __name__ == "__main__":
    main()
