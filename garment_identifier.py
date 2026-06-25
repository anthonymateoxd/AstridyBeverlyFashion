from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "models" / "garment_identity" / "blusa_modelo_01.npz"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9áéíóúñ]+", "_", value)
    return value.strip("_") or "modelo_prenda"


def pad_to_square(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (220, 220, 220))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def load_model():
    print(f"[INFO] Dispositivo: {DEVICE}")
    print("[INFO] Cargando ResNet18 preentrenada...")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(DEVICE)

    transform = transforms.Compose(
        [
            transforms.Lambda(pad_to_square),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return model, transform


@torch.inference_mode()
def embedding(image: Image.Image, model, transform) -> np.ndarray:
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    vector = model(tensor).squeeze(0)
    vector = torch.nn.functional.normalize(vector, dim=0)
    return vector.cpu().numpy().astype(np.float32)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def similarity_score(query: np.ndarray, references: np.ndarray, top_k: int = 3) -> float:
    similarities = references @ query
    k = min(top_k, len(similarities))
    best = np.partition(similarities, -k)[-k:]
    return float(best.mean())


def estimate_threshold(embeddings: np.ndarray) -> tuple[float, np.ndarray]:
    scores = []

    for index in range(len(embeddings)):
        references = np.delete(embeddings, index, axis=0)
        scores.append(similarity_score(embeddings[index], references))

    scores = np.asarray(scores, dtype=np.float32)
    threshold = float(np.percentile(scores, 5) - 0.04)
    threshold = float(np.clip(threshold, 0.72, 0.90))
    return threshold, scores


def build_profile(folder: Path, output: Path, model_name: str):
    if not folder.exists():
        raise SystemExit(f"No existe la carpeta: {folder}")

    files = list_images(folder)

    if len(files) < 10:
        raise SystemExit(
            f"Solo se encontraron {len(files)} imágenes. Usa al menos 10; idealmente 25–40."
        )

    model, transform = load_model()
    vectors = []
    valid_names = []

    for number, path in enumerate(files, start=1):
        try:
            with Image.open(path) as image:
                vectors.append(embedding(image, model, transform))
            valid_names.append(path.name)
            print(f"[{number}/{len(files)}] {path.name}")
        except Exception as error:
            print(f"[OMITIDA] {path.name}: {error}")

    if len(vectors) < 10:
        raise SystemExit("No quedaron suficientes imágenes válidas.")

    matrix = np.stack(vectors).astype(np.float32)
    threshold, internal_scores = estimate_threshold(matrix)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_name=model_name,
        embeddings=matrix,
        threshold=np.float32(threshold),
        image_names=np.asarray(valid_names),
    )

    print("\n=== PERFIL VISUAL CREADO ===")
    print(f"Modelo: {model_name}")
    print(f"Imágenes usadas: {len(matrix)}")
    print(f"Similitud interna mínima: {internal_scores.min():.4f}")
    print(f"Similitud interna media: {internal_scores.mean():.4f}")
    print(f"Umbral inicial: {threshold:.4f}")
    print(f"Guardado en: {output}")


def load_profile(profile_path: Path):
    if not profile_path.exists():
        raise SystemExit(f"No existe el perfil: {profile_path}")

    data = np.load(profile_path, allow_pickle=False)
    return (
        str(data["model_name"]),
        data["embeddings"].astype(np.float32),
        float(data["threshold"]),
    )


def test_image(image_path: Path, profile_path: Path):
    if not image_path.exists():
        raise SystemExit(f"No existe la imagen: {image_path}")

    model_name, references, threshold = load_profile(profile_path)
    model, transform = load_model()

    with Image.open(image_path) as image:
        query = embedding(image, model, transform)

    score = similarity_score(query, references)
    result = model_name if score >= threshold else "prenda_no_identificada"

    print("\n=== RESULTADO ===")
    print(f"Resultado: {result}")
    print(f"Similitud: {score:.4f}")
    print(f"Umbral: {threshold:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Identificador de un modelo de prenda mediante similitud visual."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Crear perfil de la prenda.")
    build_parser.add_argument("--folder", required=True, type=Path)
    build_parser.add_argument("--name", default=None)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE)

    test_parser = subparsers.add_parser("test", help="Probar una fotografía.")
    test_parser.add_argument("--image", required=True, type=Path)
    test_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)

    args = parser.parse_args()

    if args.command == "build":
        model_name = args.name or safe_name(args.folder.name)
        build_profile(args.folder, args.output, model_name)
    elif args.command == "test":
        test_image(args.image, args.profile)


if __name__ == "__main__":
    main()
