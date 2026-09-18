from pathlib import Path
import os

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore


DATA_ROOT = Path(
    os.environ.get(
        "ASTRID_V2_ROOT",
        "/data/BLUSA_002_S_ROSA_FRONT_V2",
    )
)

TRAIN_GOOD = DATA_ROOT / "processed/train/good_v2_curated"

RESULTS_ROOT = Path(
    os.environ.get(
        "ASTRID_V21_RESULTS",
        "/data/BLUSA_002_S_ROSA_FRONT_V2/models/patchcore_v21_baseline_384",
    )
)

EXPECTED_TRAIN = 18


def image_files(path: Path):
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def main():
    print("=" * 72)
    print("PATCHCORE V2.1 - EXPERIMENTO 384x384")
    print("=" * 72)
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Train GOOD   : {TRAIN_GOOD}")
    print(f"Resultados   : {RESULTS_ROOT}")
    print()

    if not TRAIN_GOOD.exists():
        raise RuntimeError(f"No existe TRAIN_GOOD: {TRAIN_GOOD}")

    train_images = image_files(TRAIN_GOOD)

    print(f"Imágenes TRAIN encontradas: {len(train_images)}")

    if len(train_images) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"TRAIN congelado inválido. "
            f"Esperaba {EXPECTED_TRAIN} imágenes y encontré {len(train_images)}."
        )

    print("\nTRAIN congelado:")
    for image in train_images:
        print(f"  - {image.name}")

    print("\nConfiguración experimental:")
    print("  backbone: wide_resnet50_2")
    print("  layers: layer2, layer3")
    print("  image_size: 384x384")
    print("  pretrained: True")
    print("  coreset_sampling_ratio: 0.05")
    print("  num_neighbors: 9")
    print("  validation durante fit: NO")
    print("  test durante fit: NO")
    print()

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(384, 384)
    )

    datamodule = Folder(
        name="BLUSA_002_S_ROSA_FRONT_V21_BASELINE_384",
        root=DATA_ROOT,
        normal_dir=TRAIN_GOOD,
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
        pre_processor=pre_processor,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        deterministic=True,
        default_root_dir=RESULTS_ROOT,
        limit_val_batches=0,
    )

    print("=" * 72)
    print("INICIANDO FIT V2.1 384")
    print("=" * 72)

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print()
    print("=" * 72)
    print("ENTRENAMIENTO V2.1 384 FINALIZADO")
    print("=" * 72)
    print(f"Resultados: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
