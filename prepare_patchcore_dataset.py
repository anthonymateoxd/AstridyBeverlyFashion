from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        return np.asarray(image)


def crop_to_alpha(image_rgba: np.ndarray, margin_ratio: float = 0.04) -> np.ndarray:
    alpha = image_rgba[:, :, 3]
    coords = np.argwhere(alpha > 10)

    if coords.size == 0:
        raise ValueError("La imagen no contiene un objeto visible en el canal alfa.")

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    height = y2 - y1 + 1
    width = x2 - x1 + 1
    margin = int(round(max(height, width) * margin_ratio))

    y1 = max(0, y1 - margin)
    x1 = max(0, x1 - margin)
    y2 = min(image_rgba.shape[0], y2 + margin + 1)
    x2 = min(image_rgba.shape[1], x2 + margin + 1)

    return image_rgba[y1:y2, x1:x2]


def composite_on_gray(image_rgba: np.ndarray, gray_value: int = 205) -> np.ndarray:
    rgb = image_rgba[:, :, :3].astype(np.float32)
    alpha = (image_rgba[:, :, 3].astype(np.float32) / 255.0)[..., None]
    background = np.full_like(rgb, gray_value, dtype=np.float32)
    result = rgb * alpha + background * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def resize_letterbox(image_rgb: np.ndarray, size: int = 512, gray_value: int = 205) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("La imagen tiene dimensiones inválidas.")

    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=interpolation)

    canvas = np.full((size, size, 3), gray_value, dtype=np.uint8)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"No se pudo guardar la imagen: {path}")


def process_image(input_path: Path, output_path: Path, size: int, gray_value: int) -> None:
    rgba = load_rgba(input_path)
    cropped = crop_to_alpha(rgba)
    rgb = composite_on_gray(cropped, gray_value=gray_value)
    normalized = resize_letterbox(rgb, size=size, gray_value=gray_value)
    save_rgb(output_path, normalized)


def process_folder(input_dir: Path, output_dir: Path, size: int, gray_value: int) -> None:
    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    if not files:
        raise SystemExit(f"No se encontraron imágenes en: {input_dir}")

    success = 0
    failed = 0

    for index, input_path in enumerate(files, start=1):
        relative = input_path.relative_to(input_dir)
        output_path = (output_dir / relative).with_suffix(".png")

        try:
            process_image(input_path, output_path, size=size, gray_value=gray_value)
            success += 1
            print(f"[{index}/{len(files)}] OK: {input_path.name}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(files)}] ERROR: {input_path.name} -> {exc}")

    print("\n=== PREPARACIÓN TERMINADA ===")
    print(f"Correctas: {success}")
    print(f"Fallidas: {failed}")
    print(f"Salida: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centra y normaliza imágenes segmentadas para PatchCore."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gray", type=int, default=205)
    args = parser.parse_args()

    if not 0 <= args.gray <= 255:
        raise SystemExit("--gray debe estar entre 0 y 255.")

    if args.input.is_file():
        output_path = args.output
        if output_path.suffix == "":
            output_path = output_path / f"{args.input.stem}.png"

        process_image(args.input, output_path, size=args.size, gray_value=args.gray)
        print(f"Imagen preparada: {output_path}")
        return

    if not args.input.exists():
        raise SystemExit(f"No existe la entrada: {args.input}")

    process_folder(args.input, args.output, size=args.size, gray_value=args.gray)


if __name__ == "__main__":
    main()
