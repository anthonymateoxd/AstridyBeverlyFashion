from pathlib import Path
import csv
import shutil

root = Path.home() / "astrid_ai_data" / "BLUSA_002_S_ROSA_FRONT_V2"

raw_dir = root / "raw" / "train" / "good"
selection_csv = root / "train_selection_v2.csv"

# RAW seleccionadas. Aún NO es el dataset procesado.
out_dir = root / "selected" / "train" / "good"
out_dir.mkdir(parents=True, exist_ok=True)

with selection_csv.open(
    "r",
    encoding="utf-8",
    newline="",
) as fh:
    reader = csv.DictReader(fh)

    print("Columnas CSV:", reader.fieldnames)

    selected_rows = [
        row
        for row in reader
        if (row.get("decision") or "").strip().upper()
        == "ACEPTADA"
    ]

print(f"Seleccionadas en CSV: {len(selected_rows)}")

if len(selected_rows) != 28:
    raise SystemExit(
        f"ERROR: se esperaban 28 aceptadas y se encontraron "
        f"{len(selected_rows)}. NO SE COPIO NADA."
    )

# Limpiar únicamente materialización previa de esta carpeta generada.
for path in out_dir.glob("*.jpg"):
    path.unlink()

missing = []
copied = 0

for row in selected_rows:
    filename = row["filename"].strip()

    src = raw_dir / filename
    dst = out_dir / filename

    if not src.exists():
        missing.append(filename)
        continue

    shutil.copy2(src, dst)
    copied += 1

if missing:
    print()
    print("ERROR: faltan archivos RAW:")
    for filename in missing:
        print(filename)

    raise SystemExit("NO CONTINUAR.")

print()
print("===== MATERIALIZACION RAW SELECCIONADA =====")
print(f"Seleccionadas: {len(selected_rows)}")
print(f"Copiadas:      {copied}")
print(f"Faltantes:     {len(missing)}")
print(f"Destino:       {out_dir}")
