from pathlib import Path
import csv

import cv2
import numpy as np

from preprocess_patchcore_v2 import (
    preprocess,
    get_roi_bounds,
)

ROOT = Path(
    "/data/BLUSA_002_S_ROSA_FRONT_V2"
)

RAW_DIR = ROOT / "raw" / "train" / "good"
SELECTION = ROOT / "train_selection_v2.csv"

CORR_LIMIT = 0.995
MAD_LIMIT = 3.5


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(
            f"No se pudo leer {path.name}"
        )

    return image


def comparison_roi(image):
    x1, y1, x2, y2 = get_roi_bounds(image)

    roi = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.resize(
        gray,
        (480, 270),
        interpolation=cv2.INTER_AREA,
    )


def compare(a, b):
    diff = cv2.absdiff(a, b)
    mad = float(diff.mean())

    aa = a.astype(np.float32).ravel()
    bb = b.astype(np.float32).ravel()

    corr = float(
        np.corrcoef(aa, bb)[0, 1]
    )

    return corr, mad


with SELECTION.open(
    "r",
    encoding="utf-8",
    newline="",
) as fh:
    rows = list(csv.DictReader(fh))

accepted_names = [
    row["filename"]
    for row in rows
    if row["decision"] == "ACEPTADA"
]

valid = []
failed = []

print("===== REVALIDANDO SELECCIONADAS =====")

for name in accepted_names:
    path = RAW_DIR / name
    image = load_image(path)

    try:
        _, coverage = preprocess(image)

        valid.append({
            "name": name,
            "feature": comparison_roi(image),
            "coverage": coverage,
        })

    except Exception as exc:
        failed.append(
            (name, str(exc))
        )


print(f"Seleccionadas originales: {len(accepted_names)}")
print(f"Validas produccion:       {len(valid)}")
print(f"Fallidas produccion:      {len(failed)}")

for name, error in failed:
    print(f"FALLO: {name} -> {error}")


all_files = sorted(
    RAW_DIR.glob("*.jpg")
)

current_names = {
    item["name"]
    for item in valid
}

failed_names = {
    name
    for name, _ in failed
}

eligible = []

print()
print("===== BUSCANDO SUSTITUTAS =====")

for path in all_files:

    if (
        path.name in current_names
        or path.name in failed_names
    ):
        continue

    image = load_image(path)

    try:
        _, coverage = preprocess(image)
    except Exception:
        continue

    feature = comparison_roi(image)

    nearest_corr = -1.0
    nearest_mad = 999.0
    too_similar = False

    for current in valid:
        corr, mad = compare(
            feature,
            current["feature"],
        )

        if corr > nearest_corr:
            nearest_corr = corr
            nearest_mad = mad

        if (
            corr >= CORR_LIMIT
            and mad < MAD_LIMIT
        ):
            too_similar = True
            break

    if too_similar:
        continue

    eligible.append({
        "name": path.name,
        "feature": feature,
        "coverage": coverage,
        "nearest_corr": nearest_corr,
        "nearest_mad": nearest_mad,
    })


# Más diversidad primero:
# menor correlación máxima con las actuales.
eligible.sort(
    key=lambda item: (
        item["nearest_corr"],
        -item["nearest_mad"],
    )
)


# Selección greedy de hasta 2 reemplazos,
# verificando diversidad también entre ellos.
recommended = []

for candidate in eligible:

    similar_to_replacement = False

    for chosen in recommended:
        corr, mad = compare(
            candidate["feature"],
            chosen["feature"],
        )

        if (
            corr >= CORR_LIMIT
            and mad < MAD_LIMIT
        ):
            similar_to_replacement = True
            break

    if similar_to_replacement:
        continue

    recommended.append(candidate)

    if len(recommended) == 2:
        break


print(f"Candidatas nuevas validas: {len(eligible)}")

print()
print("===== RECOMENDADAS =====")

if not recommended:
    print("NINGUNA")
else:
    for index, item in enumerate(
        recommended,
        start=1,
    ):
        print(
            f"{index}. {item['name']} | "
            f"cobertura={item['coverage'] * 100:.2f}% | "
            f"corr_cercana={item['nearest_corr']:.6f} | "
            f"MAD={item['nearest_mad']:.3f}"
        )

print()
print("===== TOP 10 CANDIDATAS =====")

for item in eligible[:10]:
    print(
        f"{item['name']} | "
        f"cobertura={item['coverage'] * 100:.2f}% | "
        f"corr={item['nearest_corr']:.6f} | "
        f"MAD={item['nearest_mad']:.3f}"
    )
