from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "models" / "garment_identity" / "blusa_modelo_01.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_model = None
_transform = None
_profile_cache: dict[str, tuple[str, np.ndarray, float]] = {}


def pad_to_square(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(width, height)

    canvas = Image.new("RGB", (side, side), (220, 220, 220))
    x = (side - width) // 2
    y = (side - height) // 2
    canvas.paste(image, (x, y))

    return canvas


def build_transform():
    return transforms.Compose(
        [
            transforms.Lambda(pad_to_square),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_feature_model():
    global _model
    global _transform

    if _model is None:
        print(f"[INFO] Dispositivo: {DEVICE}")
        print("[INFO] Cargando ResNet18 preentrenada...")

        _model = resnet18(weights=ResNet18_Weights.DEFAULT)
        _model.fc = nn.Identity()
        _model.eval()
        _model.to(DEVICE)

        _transform = build_transform()

    return _model, _transform


def load_profile(profile_path: str | Path):
    profile_path = Path(profile_path)

    if not profile_path.exists():
        raise FileNotFoundError(f"No existe el perfil: {profile_path}")

    cache_key = str(profile_path.resolve())

    if cache_key not in _profile_cache:
        data = np.load(profile_path, allow_pickle=False)

        model_name = str(data["model_name"])
        embeddings = data["embeddings"].astype(np.float32)
        threshold = float(data["threshold"])

        _profile_cache[cache_key] = (model_name, embeddings, threshold)

    return _profile_cache[cache_key]


@torch.inference_mode()
def extract_embedding(
    image: Image.Image,
    model,
    transform,
) -> np.ndarray:
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    vector = model(tensor).squeeze(0)
    vector = torch.nn.functional.normalize(vector, dim=0)

    return vector.cpu().numpy().astype(np.float32)


def calculate_similarity(
    query: np.ndarray,
    references: np.ndarray,
    top_k: int = 3,
) -> float:
    similarities = references @ query

    k = min(top_k, len(similarities))
    best = np.partition(similarities, -k)[-k:]

    return float(best.mean())


def identify_garment(
    image_path: str | Path,
    profile_path: str | Path = DEFAULT_PROFILE,
) -> dict[str, Any]:
    image_path = Path(image_path)
    profile_path = Path(profile_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    model_name, references, threshold = load_profile(profile_path)
    model, transform = load_feature_model()

    with Image.open(image_path) as image:
        query = extract_embedding(image, model, transform)

    similarity = calculate_similarity(query, references)
    identified = similarity >= threshold

    return {
        "identified": bool(identified),
        "model_name": model_name if identified else None,
        "result": model_name if identified else "prenda_no_identificada",
        "similarity": round(similarity, 4),
        "threshold": round(threshold, 4),
    }
