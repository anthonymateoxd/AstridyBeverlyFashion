#!/usr/bin/env python3
"""
Benchmark offline de candidatos de cobertura 40/45/50.

No importa app.py ni arranca Flask/MySQL. Reutiliza
PatchCoreInspector y PATCHCORE_SCORE_THRESHOLD desde .env.

Entrada: benchmark_frames/token_N/ con metadata.json y
coverage_*.jpg (generados con FRAME_SELECTION_BENCHMARK=true).

Salida: CSV por token, una fila por nivel de cobertura.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from patchcore_inference import PatchCoreInspector


ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "benchmark_frames"
DEFAULT_CSV = ROOT / "frame_benchmark_results.csv"

PATCHCORE_CKPT = os.getenv("PATCHCORE_CKPT", "").strip()
PATCHCORE_IMAGE_SIZE = int(os.getenv("PATCHCORE_IMAGE_SIZE", "384"))
PATCHCORE_SCORE_THRESHOLD = float(
    os.getenv("PATCHCORE_SCORE_THRESHOLD", "55.20")
)

if not 0.0 <= PATCHCORE_SCORE_THRESHOLD <= 100.0:
    raise ValueError(
        "PATCHCORE_SCORE_THRESHOLD debe estar entre 0 y 100."
    )


def normalize_score(raw_score: float) -> float:
    confidence = (
        raw_score * 100.0 if raw_score <= 1.0 else float(raw_score)
    )
    return round(max(0.0, min(confidence, 100.0)), 2)


def decision_from_score(confidence: float) -> str:
    return "ANOMALIA" if confidence >= PATCHCORE_SCORE_THRESHOLD else "NORMAL"


def production_result_label(metadata: dict) -> str:
    raw = metadata.get("production_result")
    if raw is None:
        return ""

    text = str(raw).strip().upper()
    if text in ("ANOMALIA", "NORMAL"):
        return text
    if text in ("DEFECTO", "ANOMALÍA", "ANOMALIA"):
        return "ANOMALIA"
    if text in ("APROBADO", "NORMAL", "OK"):
        return "NORMAL"
    if "DEFECT" in text or "ANOMAL" in text:
        return "ANOMALIA"
    return text


def iter_token_dirs(benchmark_dir: Path):
    if not benchmark_dir.exists():
        return

    for path in sorted(
        benchmark_dir.iterdir(),
        key=lambda item: item.name,
    ):
        if not path.is_dir():
            continue
        if not path.name.startswith("token_"):
            continue
        if (path / "metadata.json").exists():
            yield path


def coverage_level_from_name(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("coverage_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def load_metadata(token_dir: Path) -> dict:
    meta_path = token_dir / "metadata.json"
    try:
        return json.loads(
            meta_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        print(
            f"[WARN] metadata inválida en {meta_path}: {error}",
            file=sys.stderr,
        )
        return {}


def analyze_token(
    inspector: PatchCoreInspector,
    token_dir: Path,
    csv_rows: list[dict],
) -> int:
    metadata = load_metadata(token_dir)
    token = metadata.get("token")
    if token is None:
        name = token_dir.name
        if name.startswith("token_"):
            token = name[len("token_"):]

    batch_id = metadata.get("batch_id")
    batch_position = metadata.get("batch_position")
    production_result = production_result_label(metadata)
    coverages_meta = metadata.get("coverages") or {}

    frames = sorted(
        token_dir.glob("coverage_*.jpg"),
        key=lambda p: coverage_level_from_name(p) or 0,
    )

    rows_added = 0

    for frame_path in frames:
        level = coverage_level_from_name(frame_path)
        if level is None:
            continue

        try:
            prediction = inspector.inspect(str(frame_path))
        except Exception as error:
            print(
                f"[ERROR] inspect {frame_path}: {error}",
                file=sys.stderr,
            )
            continue

        confidence = normalize_score(float(prediction["score"]))
        candidate_result = decision_from_score(confidence)
        matches = ""
        if production_result:
            matches = "true" if candidate_result == production_result else "false"

        meta_key = str(level)
        actual_coverage = coverages_meta.get(meta_key, "")

        csv_rows.append(
            {
                "token": token,
                "batch_id": batch_id,
                "batch_position": batch_position,
                "production_result": production_result,
                "coverage_level": level,
                "actual_coverage": actual_coverage,
                "candidate_score": confidence,
                "candidate_result": candidate_result,
                "matches_production_result": matches,
            }
        )
        rows_added += 1

        print(
            f"[BENCH] token={token} coverage={level}% "
            f"actual={actual_coverage} score={confidence} "
            f"result={candidate_result} match={matches or 'n/a'}"
        )

    return rows_added


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "token",
        "batch_id",
        "batch_position",
        "production_result",
        "coverage_level",
        "actual_coverage",
        "candidate_score",
        "candidate_result",
        "matches_production_result",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Re-evalúa candidatos 40/45/50 con PatchCore "
            "y escribe un CSV offline."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        default=str(BENCHMARK_DIR),
        help="Directorio raíz de benchmark_frames.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CSV),
        help="Ruta del CSV de salida.",
    )
    args = parser.parse_args()

    if not PATCHCORE_CKPT:
        print(
            "[ERROR] PATCHCORE_CKPT no está configurado en .env.",
            file=sys.stderr,
        )
        return 1

    if not 0.0 <= PATCHCORE_SCORE_THRESHOLD <= 100.0:
        print(
            "[ERROR] PATCHCORE_SCORE_THRESHOLD inválido.",
            file=sys.stderr,
        )
        return 1

    benchmark_dir = Path(args.benchmark_dir).resolve()
    if not benchmark_dir.exists():
        print(
            f"[ERROR] No existe el directorio: {benchmark_dir}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[PATCHCORE] threshold={PATCHCORE_SCORE_THRESHOLD:.2f} "
        f"image_size={PATCHCORE_IMAGE_SIZE} "
        f"ckpt={PATCHCORE_CKPT}"
    )

    inspector = PatchCoreInspector(
        PATCHCORE_CKPT,
        image_size=PATCHCORE_IMAGE_SIZE,
    )

    rows: list[dict] = []
    total = 0

    for token_dir in iter_token_dirs(benchmark_dir):
        total += analyze_token(inspector, token_dir, rows)

    if not rows:
        print(
            "[ERROR] No se procesó ningún candidato. "
            "¿FRAME_SELECTION_BENCHMARK=true y hay tokens en "
            f"{benchmark_dir}?",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.output).resolve()
    write_csv(out_path, rows)

    matches = sum(
        1
        for row in rows
        if str(row.get("matches_production_result")).lower() == "true"
    )

    print(
        f"[DONE] Filas={len(rows)} tokens_procesados_ok "
        f"match_production={matches}/{len(rows)} csv={out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
