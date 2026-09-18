from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


ROOT = Path("/data/BLUSA_002_S_ROSA_FRONT_V2")

MODEL_DIR = (
    ROOT
    / "models/patchcore_v22_ar_352x512"
    / "Patchcore"
    / "BLUSA_002_S_ROSA_FRONT_V22_AR_352X512"
)

CHECKPOINTS = sorted(
    MODEL_DIR.glob("v*/weights/lightning/model.ckpt")
)

if len(CHECKPOINTS) != 1:
    raise RuntimeError(
        f"Esperaba exactamente 1 checkpoint y encontré "
        f"{len(CHECKPOINTS)}: {CHECKPOINTS}"
    )

CKPT = CHECKPOINTS[0]

GOOD_DIR = ROOT / "processed/validation/good"
MANCHA_DIR = ROOT / "processed/validation/mancha"

OUTPUT_DIR = (
    ROOT
    / "evaluation"
    / "patchcore_v22_ar_352x512"
)

HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
CSV_PATH = OUTPUT_DIR / "validation_scores.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def images_in(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())

    return float(value)


def map_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise RuntimeError(
            f"anomaly_map inesperado: shape={arr.shape}"
        )

    return arr


def save_heatmap(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"No se pudo leer imagen: {image_path}"
        )

    h, w = image.shape[:2]

    amap = cv2.resize(
        anomaly_map,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    minimum = float(amap.min())
    maximum = float(amap.max())

    if maximum > minimum:
        normalized = (
            (amap - minimum)
            / (maximum - minimum)
            * 255.0
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(
            amap,
            dtype=np.uint8,
        )

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image,
        0.60,
        colored,
        0.40,
        0,
    )

    # Panel: original | heatmap | overlay
    panel = np.hstack(
        (
            image,
            colored,
            overlay,
        )
    )

    if not cv2.imwrite(str(output_path), panel):
        raise RuntimeError(
            f"No se pudo guardar: {output_path}"
        )


def main() -> None:
    if not CKPT.is_file():
        raise SystemExit(
            f"Checkpoint no encontrado: {CKPT}"
        )

    good = images_in(GOOD_DIR)
    mancha = images_in(MANCHA_DIR)

    print("=" * 72)
    print("PATCHCORE V2 - EVALUACIÓN VALIDATION")
    print("=" * 72)
    print("Checkpoint:", CKPT)
    print("GOOD:      ", len(good))
    print("MANCHA:    ", len(mancha))
    print("=" * 72)

    if len(good) != 5:
        raise SystemExit(
            f"ERROR: esperaba 5 GOOD; encontré {len(good)}"
        )

    if len(mancha) != 5:
        raise SystemExit(
            f"ERROR: esperaba 5 MANCHA; encontré {len(mancha)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HEATMAP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(352, 512)
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=False,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
        pre_processor=pre_processor,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir="/tmp/anomalib_results",
    )

    rows = []

    samples = (
        [("GOOD", p) for p in good]
        + [("MANCHA", p) for p in mancha]
    )

    for index, (ground_truth, image_path) in enumerate(
        samples,
        start=1,
    ):
        print()
        print(
            f"[{index:02d}/10] "
            f"{ground_truth}: {image_path.name}"
        )

        dataset = PredictDataset(
            path=image_path,
            image_size=(352, 512),
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(CKPT),
        )

        if not predictions:
            raise RuntimeError(
                f"Sin predicción para {image_path.name}"
            )

        pred = predictions[0]

        score = scalar(pred.pred_score)
        automatic_label = bool(
            scalar(pred.pred_label)
        )

        anomaly_map = map_to_numpy(
            pred.anomaly_map
        )

        map_min = float(anomaly_map.min())
        map_max = float(anomaly_map.max())
        map_mean = float(anomaly_map.mean())

        output_name = (
            f"{ground_truth.lower()}__"
            f"{image_path.stem}__heatmap.jpg"
        )

        save_heatmap(
            image_path=image_path,
            anomaly_map=anomaly_map,
            output_path=HEATMAP_DIR / output_name,
        )

        rows.append(
            {
                "ground_truth": ground_truth,
                "filename": image_path.name,
                "score": score,
                "anomalib_pred_label": int(
                    automatic_label
                ),
                "anomaly_map_min": map_min,
                "anomaly_map_max": map_max,
                "anomaly_map_mean": map_mean,
                "heatmap": output_name,
            }
        )

        print(f"  score:    {score:.6f}")
        print(
            "  auto:     "
            f"{automatic_label}"
        )
        print(
            "  map min:  "
            f"{map_min:.6f}"
        )
        print(
            "  map max:  "
            f"{map_max:.6f}"
        )
        print(
            "  map mean: "
            f"{map_mean:.6f}"
        )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ground_truth",
                "filename",
                "score",
                "anomalib_pred_label",
                "anomaly_map_min",
                "anomaly_map_max",
                "anomaly_map_mean",
                "heatmap",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    good_scores = np.array(
        [
            row["score"]
            for row in rows
            if row["ground_truth"] == "GOOD"
        ],
        dtype=np.float64,
    )

    stain_scores = np.array(
        [
            row["score"]
            for row in rows
            if row["ground_truth"] == "MANCHA"
        ],
        dtype=np.float64,
    )

    max_good = float(good_scores.max())
    min_stain = float(stain_scores.min())

    print()
    print("=" * 72)
    print("RESUMEN DE SCORES")
    print("=" * 72)

    print(
        "GOOD:   ",
        ", ".join(
            f"{value:.6f}"
            for value in good_scores
        ),
    )

    print(
        "MANCHA: ",
        ", ".join(
            f"{value:.6f}"
            for value in stain_scores
        ),
    )

    print()
    print(
        f"GOOD min/max:   "
        f"{good_scores.min():.6f} / "
        f"{max_good:.6f}"
    )

    print(
        f"MANCHA min/max: "
        f"{min_stain:.6f} / "
        f"{stain_scores.max():.6f}"
    )

    print()

    if min_stain > max_good:
        candidate = (
            max_good + min_stain
        ) / 2.0

        print(
            "SEPARACIÓN COMPLETA EN VALIDATION: SÍ"
        )
        print(
            "Margen: "
            f"{min_stain - max_good:.6f}"
        )
        print(
            "Threshold candidato intermedio: "
            f"{candidate:.6f}"
        )
    else:
        print(
            "SEPARACIÓN COMPLETA EN VALIDATION: NO"
        )
        print(
            "Solapamiento/margen: "
            f"{min_stain - max_good:.6f}"
        )

    print()
    print("CSV:", CSV_PATH)
    print("Heatmaps:", HEATMAP_DIR)

    print()
    print(
        "IMPORTANTE: todavía NO se ha usado TEST "
        "y NO se ha modificado producción."
    )


if __name__ == "__main__":
    main()
