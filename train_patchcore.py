from __future__ import annotations

import shutil
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


ROOT = Path(__file__).resolve().parent
# NORMAL_DIR = ROOT / "dataset_identidad" / "blusa_modelo_01"
# OUTPUT_DIR = ROOT / "models" / "patchcore_blusa_modelo_01"


NORMAL_DIR = ROOT / "dataset_patchcore_ready" / "train" / "good"
OUTPUT_DIR = ROOT / "models" / "patchcore_blusa_modelo_01_ready"

FINAL_CHECKPOINT = OUTPUT_DIR / "model.ckpt"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def count_images(folder: Path) -> int:
    return sum(
        1
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def main() -> None:
    if not NORMAL_DIR.exists():
        raise SystemExit(f"No existe la carpeta: {NORMAL_DIR}")

    image_count = count_images(NORMAL_DIR)

    if image_count < 10:
        raise SystemExit(
            f"Solo se encontraron {image_count} imágenes limpias. "
            "Usa al menos 10."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ENTRENAMIENTO PATCHCORE ===")
    print(f"Imágenes limpias: {image_count}")
    print(f"Carpeta: {NORMAL_DIR}")
    print("Dispositivo: CPU\n")

    datamodule = Folder(
        name="blusa_modelo_01",
        root=ROOT,
        normal_dir=NORMAL_DIR.relative_to(ROOT),
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        test_split_mode=TestSplitMode.SYNTHETIC,
        test_split_ratio=0.20,
        val_split_ratio=0.50,
        seed=42,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.10,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=OUTPUT_DIR,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
    )

    engine.fit(model=model, datamodule=datamodule)

    best_model_path = engine.best_model_path

    if best_model_path:
        source = Path(best_model_path)
        if source.resolve() != FINAL_CHECKPOINT.resolve():
            shutil.copy2(source, FINAL_CHECKPOINT)
    else:
        engine.trainer.save_checkpoint(str(FINAL_CHECKPOINT))

    print("\n=== ENTRENAMIENTO TERMINADO ===")
    print(f"Modelo guardado en: {FINAL_CHECKPOINT}")


if __name__ == "__main__":
    main()
