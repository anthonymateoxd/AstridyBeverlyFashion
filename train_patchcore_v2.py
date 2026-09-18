from __future__ import annotations

import os
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


DATA_ROOT = Path(
    os.getenv(
        "PATCHCORE_V2_DATA_ROOT",
        "/data/BLUSA_002_S_ROSA_FRONT_V2",
    )
).resolve()

TRAIN_GOOD = DATA_ROOT / "processed" / "train" / "good_v2_curated"

RESULTS_ROOT = Path(
    os.getenv(
        "PATCHCORE_V2_RESULTS_ROOT",
        "/data/BLUSA_002_S_ROSA_FRONT_V2/models/patchcore_v2_baseline_256",
    )
).resolve()

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

EXPECTED_TRAIN_IMAGES = 18


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_training_data() -> list[Path]:
    files = image_files(TRAIN_GOOD)

    print("=" * 70)
    print("PATCHCORE V2 - AUDITORÍA PREVIA")
    print("=" * 70)
    print(f"Dataset V2:     {DATA_ROOT}")
    print(f"Train normal:   {TRAIN_GOOD}")
    print(f"Imágenes:       {len(files)}")
    print(f"Resultados:     {RESULTS_ROOT}")
    print("=" * 70)

    if len(files) != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(
            "SEGURIDAD V2: se esperaban exactamente "
            f"{EXPECTED_TRAIN_IMAGES} imágenes normales y se encontraron "
            f"{len(files)}. NO SE ENTRENA."
        )

    return files


def main() -> None:
    files = validate_training_data()

    # PatchCore debe aprender exclusivamente de imágenes normales.
    #
    # No usamos validation ni test durante fit.
    # Las imágenes validation/good y validation/mancha se evaluarán
    # posteriormente con un script separado para calibrar el threshold.
    datamodule = Folder(
        name="BLUSA_002_S_ROSA_FRONT_V2_BASELINE_256",
        root=DATA_ROOT,
        normal_dir="processed/train/good_v2_curated",
        abnormal_dir=None,
        normal_test_dir=None,
        train_batch_size=4,
        eval_batch_size=1,
        num_workers=0,
        test_split_mode=TestSplitMode.NONE,
        val_split_mode=ValSplitMode.NONE,
        seed=42,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir=str(RESULTS_ROOT),
        logger=False,
        deterministic=True,
        limit_val_batches=0,
    )

    print()
    print("Configuración:")
    print("  Backbone: wide_resnet50_2")
    print("  Layers: layer2, layer3")
    print("  Resolución: 256x256")
    print("  Coreset ratio: 0.05")
    print("  Vecinos: 9")
    print("  Pretrained: sí")
    print("  Accelerator: CPU")
    print()
    print(f"Se usarán {len(files)} imágenes normales.")
    print("Validation NO participa en el entrenamiento.")
    print("Test NO participa en el entrenamiento.")
    print()
    print("INICIANDO PATCHCORE V2...")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    checkpoint = engine.best_model_path

    print()
    print("=" * 70)
    print("PATCHCORE V2 FINALIZADO")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint}")
    print(f"Resultados: {RESULTS_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
