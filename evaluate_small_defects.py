"""Offline sensitivity study of human-labeled camera scores.

Usage:
  python evaluate_small_defects.py labeled_scores.csv

CSV header:
  sample_id,position,actual,score
  batch94-01,inferior-centro,DEFECTO,58.94
  clean-control-01,centro,BUENA,23.50

"actual" must be supplied by a person inspecting the ORIGINAL photograph,
not inferred from PatchCore's decision. All samples must belong to one model
and a matching, frozen preprocessing profile. No database access or mutation.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

POSITIONS = (
    "superior-izquierda", "superior-centro", "superior-derecha",
    "media-izquierda", "media-centro", "media-derecha",
    "inferior-izquierda", "inferior-centro", "inferior-derecha",
    "control",
)
THRESHOLDS = (40.0, 44.0, 45.0, 48.0, 50.0, 52.0, 54.0, 55.2)
CATEGORIES = {"DEFECTO", "BUENA"}


def load_cases(filename):
    rows = []
    ids = set()
    with Path(filename).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        mandatory = {"sample_id", "position", "actual", "score"}
        if not mandatory.issubset(set(reader.fieldnames or ())):
            raise ValueError("Faltan columnas: sample_id, position, actual, score")
        for line_no, row in enumerate(reader, start=2):
            name = str(row["sample_id"] or "").strip()
            where = str(row["position"] or "").strip().lower()
            actual = str(row["actual"] or "").strip().upper()
            if not actual:
                continue  # Pending human annotation; never guess ground truth.
            if not name or name in ids or where not in POSITIONS or actual not in CATEGORIES:
                raise ValueError(f"Datos inválidos o duplicados en línea {line_no}")
            try:
                score = float(row["score"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Score inválido en línea {line_no}") from error
            if not 0 <= score <= 100:
                raise ValueError(f"Score fuera de rango en línea {line_no}")
            ids.add(name)
            rows.append((name, where, actual, score))
    return rows


def metrics(rows, threshold):
    good = [r for r in rows if r[2] == "BUENA"]
    defects = [r for r in rows if r[2] == "DEFECTO"]
    tp = sum(r[3] >= threshold for r in defects)
    fp = sum(r[3] >= threshold for r in good)
    return dict(
        threshold=threshold,
        tp=tp, fn=len(defects)-tp, fp=fp, tn=len(good)-fp,
        recall=tp/len(defects) if defects else None,
        specificity=(len(good)-fp)/len(good) if good else None,
    )


def report(rows, thresholds=THRESHOLDS):
    positives = sum(r[2] == "DEFECTO" for r in rows)
    negatives = sum(r[2] == "BUENA" for r in rows)
    lines = [
        f"Pruebas etiquetadas manualmente: {len(rows)} | "
        f"DEFECTO={positives} | BUENA={negatives}",
        "Este análisis NO modifica el umbral de producción.",
        "Umbral | Detectadas | Omitidas | Falsas alertas | "
        "Sensibilidad | Especificidad",
    ]
    for threshold in thresholds:
        m=metrics(rows, threshold)
        sensitivity = f"{100*m['recall']:.1f}%" if m["recall"] is not None else "N/D"
        specificity = (
            f"{100*m['specificity']:.1f}%"
            if m["specificity"] is not None else "N/D"
        )
        lines.append(
            f"{threshold:6.2f} | {m['tp']:9d} | {m['fn']:7d} | "
            f"{m['fp']:13d} | {sensitivity:12} | {specificity}"
        )
    lines.append("DEFECTOS POR POSICIÓN (umbral congelado = 55.20)")
    by_position = defaultdict(list)
    for row in rows:
        if row[2]=="DEFECTO":
            by_position[row[1]].append(row)
    for position in POSITIONS:
        cases=by_position.get(position,[])
        if cases:
            detected=sum(r[3]>=55.2 for r in cases)
            lines.append(f"  {position}: {detected}/{len(cases)}")
    if not positives or not negatives:
        lines.append(
            "INCOMPLETO: se requieren imágenes etiquetadas de ambas clases "
            "para evaluar detección y falsas alarmas."
        )
    if negatives < 10:
        lines.append(
            "ADVERTENCIA: menos de 10 prendas buenas; no se debe elegir "
            "un umbral de producción con este conjunto."
        )
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file",help="CSV con etiquetas verificadas manualmente")
    args=parser.parse_args()
    print(report(load_cases(args.csv_file)))


if __name__=="__main__":
    main()
