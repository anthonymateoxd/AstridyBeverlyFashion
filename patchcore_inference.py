from pathlib import Path

import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def _to_numpy(value) -> np.ndarray:
    """Convierte tensores o arreglos de Anomalib a NumPy."""
    if value is None:
        raise ValueError("La predicción no contiene mapa de anomalías.")

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    return np.asarray(value).squeeze()


def _to_scalar(value) -> float:
    """Convierte un tensor o valor numérico en float."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    return float(np.asarray(value).squeeze())


class PatchCoreInspector:
    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
    ):
        checkpoint = Path(checkpoint_path).resolve()

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint PatchCore: {checkpoint}"
            )

        self.checkpoint_path = str(checkpoint)

        image_size = int(image_size)

        if image_size <= 0:
            raise ValueError(
                "La resolución PatchCore debe ser mayor que cero."
            )

        self.image_size = (image_size, image_size)

        # Debe conservar la misma configuración del entrenamiento.
        self.model = Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=False,
            coreset_sampling_ratio=0.05,
            num_neighbors=9,
        )

        self.engine = Engine(
            accelerator="cpu",
            devices=1,
            logger=False,
        )

        print(f"[PATCHCORE] Checkpoint configurado: {self.checkpoint_path}")
        print(f"[PATCHCORE] Resolución configurada: {self.image_size}")

    def inspect(self, image_path: str) -> dict:
        image = Path(image_path).resolve()

        if not image.exists():
            raise FileNotFoundError(
                f"No existe la imagen que se quiere inspeccionar: {image}"
            )

        dataset = PredictDataset(
            path=image,
            image_size=self.image_size,
        )

        predictions = self.engine.predict(
            model=self.model,
            dataset=dataset,
            ckpt_path=self.checkpoint_path,
        )

        if not predictions:
            raise RuntimeError("PatchCore no devolvió ninguna predicción.")

        prediction = predictions[0]

        score = _to_scalar(prediction.pred_score)
        label = int(_to_scalar(prediction.pred_label))
        anomaly_map = _to_numpy(prediction.anomaly_map)

        return {
            "is_anomaly": label == 1,
            "score": score,
            "anomaly_map": anomaly_map,
        }