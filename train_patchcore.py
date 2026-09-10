from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from anomalib.data.utils import TestSplitMode, ValSplitMode

ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    ROOT
    / "anomaly_dataset"
    / "BLUSA_002_S_ROSA_FRONT"
)

RESULTS_ROOT = ROOT / "patchcore_results"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def count_images(folder: Path) -> int:
    """Cuenta imágenes válidas dentro de una carpeta."""
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset() -> list[str]:
    """Valida las carpetas y devuelve las rutas anómalas disponibles."""
    train_good = DATASET_ROOT / "train" / "good"
    test_good = DATASET_ROOT / "test" / "good"

    if not train_good.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de entrenamiento: {train_good}"
        )

    train_count = count_images(train_good)
    test_good_count = count_images(test_good)

    if train_count == 0:
        raise RuntimeError(
            "La carpeta train/good no contiene imágenes."
        )

    abnormal_dirs: list[str] = []

    for folder_name in ("mancha", "agujero", "anomaly"):
        folder = DATASET_ROOT / "test" / folder_name

        if count_images(folder) > 0:
            abnormal_dirs.append(f"test/{folder_name}")

    if not abnormal_dirs:
        raise RuntimeError(
            "No se encontraron imágenes en test/mancha, "
            "test/agujero o test/anomaly."
        )

    print("=" * 60)
    print("DATASET DETECTADO")
    print("=" * 60)
    print(f"Raíz: {DATASET_ROOT}")
    print(f"Entrenamiento normal: {train_count}")
    print(f"Prueba normal: {test_good_count}")

    for relative_path in abnormal_dirs:
        image_count = count_images(DATASET_ROOT / relative_path)
        print(f"Prueba {relative_path}: {image_count}")

    print("=" * 60)

    return abnormal_dirs


def main() -> None:
    abnormal_dirs = validate_dataset()

    test_good = DATASET_ROOT / "test" / "good"

    datamodule_args = {
        "name": "BLUSA_002_S_ROSA_FRONT",
        "root": DATASET_ROOT,
        "normal_dir": "train/good",
        "abnormal_dir": abnormal_dirs,
        "train_batch_size": 4,
        "eval_batch_size": 1,
        "num_workers": 0,

        # Las imágenes de prueba ya están organizadas en carpetas.
        "test_split_mode": TestSplitMode.FROM_DIR,

        # No dividir el conjunto de prueba porque actualmente
        # solo existe una imagen por categoría.
        "val_split_mode": ValSplitMode.SAME_AS_TEST,

        "seed": 42,
    }
    
    if count_images(test_good) > 0:
        datamodule_args["normal_test_dir"] = "test/good"

    datamodule = Folder(**datamodule_args)

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir=str(RESULTS_ROOT),
        logger=False,
        deterministic=True,
    )

    print("\nIniciando entrenamiento PatchCore...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nEntrenamiento finalizado.")
    print(f"Checkpoint: {engine.best_model_path}")

    print("\nEjecutando prueba del modelo...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nResultados:")
    print(test_results)

    print("\nArchivos guardados en:")
    print(RESULTS_ROOT)


if __name__ == "__main__":
    main()