from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNTIME_DATA_YAML = DATASET_DIR / "data.runtime.yaml"
RUN_DIR = ROOT / "training_runs" / "ropa_defectos"
FINAL_MODEL = ROOT / "models" / "best.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "mancha",
    1: "rotura",
    2: "agujero",
    3: "variacion_color",
}




def build_runtime_data_yaml() -> Path:
    """Crea un YAML con ruta absoluta para evitar diferencias entre Windows y Linux."""
    content = DATA_YAML.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    path_replaced = False

    for line in content:
        if line.strip().startswith("path:") and not path_replaced:
            output.append(f'path: "{DATASET_DIR.as_posix()}"')
            path_replaced = True
        else:
            output.append(line)

    if not path_replaced:
        output.insert(0, f'path: "{DATASET_DIR.as_posix()}"')

    RUNTIME_DATA_YAML.write_text("\n".join(output) + "\n", encoding="utf-8")
    return RUNTIME_DATA_YAML

def validate_label_file(label_path: Path) -> list[str]:
    """Valida una etiqueta YOLO. Un archivo vacío representa una imagen sin defecto."""
    errors: list[str] = []
    content = label_path.read_text(encoding="utf-8").strip()

    if not content:
        return errors

    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{label_path}: línea {line_number}: se esperaban 5 valores y hay {len(parts)}."
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path}: línea {line_number}: contiene valores no numéricos.")
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"{label_path}: línea {line_number}: clase {class_id} fuera del rango 0-3."
            )

        for name, value in {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }.items():
            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"{label_path}: línea {line_number}: {name}={value} debe estar entre 0 y 1."
                )

        if width <= 0 or height <= 0:
            errors.append(
                f"{label_path}: línea {line_number}: ancho y alto deben ser mayores que cero."
            )

    return errors


def validate_dataset() -> dict[str, dict[str, int]]:
    """Comprueba estructura, correspondencia imagen-etiqueta y formato YOLO."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"No existe {DATA_YAML}")

    all_errors: list[str] = []
    summary: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        images_dir = DATASET_DIR / "images" / split
        labels_dir = DATASET_DIR / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        labeled_images = 0
        negative_images = 0

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                all_errors.append(
                    f"Falta etiqueta para {image_path.name}: debe existir {label_path.name}; "
                    "si la prenda no tiene defecto, el archivo debe estar vacío."
                )
                continue

            if label_path.read_text(encoding="utf-8").strip():
                labeled_images += 1
            else:
                negative_images += 1

            all_errors.extend(validate_label_file(label_path))

        summary[split] = {
            "images": len(images),
            "with_defects": labeled_images,
            "without_defects": negative_images,
        }

    if summary["train"]["images"] == 0:
        all_errors.append("La carpeta dataset/images/train está vacía.")
    if summary["val"]["images"] == 0:
        all_errors.append("La carpeta dataset/images/val está vacía.")

    if all_errors:
        preview = "\n".join(f"- {error}" for error in all_errors[:50])
        extra = len(all_errors) - 50
        if extra > 0:
            preview += f"\n- ... y {extra} errores adicionales."
        raise ValueError(f"El dataset no está listo:\n{preview}")

    return summary


def main() -> None:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Ejecute: pip install -r requirements.txt"
        ) from exc

    summary = validate_dataset()
    runtime_data_yaml = build_runtime_data_yaml()
    print("Dataset validado:")
    for split, values in summary.items():
        print(
            f"  {split}: {values['images']} imágenes, "
            f"{values['with_defects']} con defectos y "
            f"{values['without_defects']} sin defectos"
        )

    base_model = os.getenv("TRAIN_BASE_MODEL", "yolov8n.pt")
    epochs = int(os.getenv("TRAIN_EPOCHS", "100"))
    image_size = int(os.getenv("TRAIN_IMGSZ", "640"))
    batch = int(os.getenv("TRAIN_BATCH", "8"))
    workers = int(os.getenv("TRAIN_WORKERS", "2"))
    device = os.getenv("TRAIN_DEVICE", "").strip()

    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Modelo base: {base_model}")
    print(f"Dispositivo: {device}")

    model = YOLO(base_model)
    model.train(
        data=str(runtime_data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        workers=workers,
        device=device,
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=True,
        patience=25,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    trained_best = RUN_DIR / "weights" / "best.pt"
    if not trained_best.exists():
        raise FileNotFoundError(
            f"El entrenamiento terminó, pero no se encontró el modelo esperado en {trained_best}"
        )

    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_best, FINAL_MODEL)
    print(f"Modelo listo para Flask: {FINAL_MODEL}")

    best_model = YOLO(str(FINAL_MODEL))
    metrics = best_model.val(data=str(runtime_data_yaml), imgsz=image_size, device=device)
    print(f"mAP50-95: {float(metrics.box.map):.4f}")
    print(f"mAP50: {float(metrics.box.map50):.4f}")


if __name__ == "__main__":
    main()
