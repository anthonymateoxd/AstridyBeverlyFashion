from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path.home() / "astrid_ai_data" / "BLUSA_002_S_ROSA_FRONT_V2"
IMAGE_DIR = ROOT / "raw" / "train" / "good"
OUTPUT = ROOT / "train_selection_v2.csv"

RX1, RY1, RX2, RY2 = 0.16, 0.05, 0.84, 0.88

CORR_LIMIT = 0.995
MAD_LIMIT = 3.5


def load_roi(path):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"No se pudo leer: {path}")

    h, w = image.shape[:2]

    roi = image[
        int(RY1 * h):int(RY2 * h),
        int(RX1 * w):int(RX2 * w),
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    reduced = cv2.resize(
        gray,
        (480, 270),
        interpolation=cv2.INTER_AREA,
    )

    return reduced


files = sorted(IMAGE_DIR.glob("*.jpg"))
accepted = []
rows = []

for path in files:
    current = load_roi(path)

    duplicate_of = ""
    best_corr = -1.0
    best_mad = 999.0

    for previous_path, previous in accepted:
        diff = cv2.absdiff(previous, current)
        mad = float(diff.mean())

        a = previous.astype(np.float32).ravel()
        b = current.astype(np.float32).ravel()

        corr = float(
            np.corrcoef(a, b)[0, 1]
        )

        if corr > best_corr:
            best_corr = corr
            best_mad = mad

        if (
            corr >= CORR_LIMIT
            and mad < MAD_LIMIT
        ):
            duplicate_of = previous_path.name
            break

    if duplicate_of:
        decision = "MUY_SIMILAR"
    else:
        decision = "ACEPTADA"
        accepted.append((path, current))

    rows.append({
        "filename": path.name,
        "decision": decision,
        "duplicate_of": duplicate_of,
        "best_corr": (
            f"{best_corr:.6f}"
            if best_corr >= 0
            else ""
        ),
        "best_mad": (
            f"{best_mad:.3f}"
            if best_corr >= 0
            else ""
        ),
    })


with OUTPUT.open(
    "w",
    encoding="utf-8",
    newline="",
) as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "filename",
            "decision",
            "duplicate_of",
            "best_corr",
            "best_mad",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print("===== SELECCION V2 =====")
print(f"RAW totales:       {len(files)}")
print(f"Aceptadas únicas:  {len(accepted)}")
print(f"Muy similares:     {len(files) - len(accepted)}")

print()
print("===== ACEPTADAS 51-60 =====")

accepted_names = {
    path.name
    for path, _ in accepted
}

for path in files:
    try:
        number = int(path.name.split("_")[2])
    except Exception:
        continue

    if 51 <= number <= 60 and path.name in accepted_names:
        print(path.name)

print()
print(f"CSV: {OUTPUT}")
