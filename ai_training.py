"""FASE 3A — Entrenamiento PatchCore genérico, persistente y asíncrono.

Este módulo NO entrena dentro de una petición Flask. Expone:

- configuración efectiva versionable (config.json);
- materialización idempotente de datasets inmutables;
- verificación de integridad previa al entrenamiento;
- creación del job de entrenamiento (con todas las barreras);
- ejecución del job por etapas reales con progreso y log;
- payload de estado para la UI.

El worker (ai_worker.py) es quien ejecuta `execute_training_job`.
La inferencia productiva (PATCHCORE_CKPT) no se toca nunca aquí.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import time
from datetime import datetime
from pathlib import Path

from ai_domain import (
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_ENTRENANDO,
    AI_MODEL_STATUS_FALLIDO,
    AI_MODEL_STATUS_PREPARACION,
    AI_MODEL_STATUS_VALIDADO,
    AI_MODEL_STATUS_VALIDACION,
    AIDomainError,
    DATASET_STATUS_ABIERTO,
    DATASET_STATUS_CERRADO,
    JOB_KIND_TRAINING,
    JOB_STATUS_CANCELADO,
    JOB_STATUS_COMPLETADO,
    JOB_STATUS_EN_CURSO,
    JOB_STATUS_FALLIDO,
    JOB_STATUS_PENDIENTE,
    add_images_to_dataset,
    close_ai_dataset,
    compute_manifest_hash,
    compute_manifest_path,
    create_ai_dataset,
    create_ai_job,
    create_next_ai_model_version,
    ensure_safe_relative_path,
    evaluate_retrain_availability,
    get_ai_artifacts_root,
    get_ai_capture_config,
    is_technically_invalidated,
    next_version_label,
    normalize_sha256,
    record_ai_event,
    resolve_under_root,
    sha256_file,
    transition_ai_job_status,
    transition_ai_model_status,
    update_job_progress,
)
import patchcore_preprocess


# ============================================================
# CONFIGURACIÓN EFECTIVA DEL ENTRENAMIENTO
#
# Valores históricos verificados (train_patchcore_v2*.py + ckpt
# productivo): wide_resnet50_2 / layer2+layer3 / coreset 0.05 /
# num_neighbors 9 / batch 4 / seed 42 / pre_trained / CPU.
#
# input_size: PRODUCCIÓN aplica Resize(256,256) porque
# PatchCoreInspector construye Patchcore(...) sin pre_processor
# (anomalib 2.5.0 usa 256 por defecto) y PredictDataset ignora
# image_size cuando no recibe transform. Se entrena a 256 para que
# entrenamiento e inferencia compartan representación. El entrenamiento
# histórico V2.1 usó 384; se conserva como override explícito.
# ============================================================

PRODUCTION_INFERENCE_INPUT_SIZE = 256

TRAINING_CONFIG_DEFAULTS = {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "input_size": PRODUCTION_INFERENCE_INPUT_SIZE,
    "coreset_ratio": 0.05,
    "num_neighbors": 9,
    "seed": 42,
    "batch_size": 4,
    "num_workers": 0,
    "pre_trained": True,
    "accelerator": "cpu",
    "devices": 1,
    "deterministic": True,
    "keep_staging": True,
}

_ENV_OVERRIDES = {
    "PATCHCORE_BACKBONE": ("backbone", str),
    "PATCHCORE_LAYERS": ("layers", "layers"),
    "PATCHCORE_INPUT_SIZE": ("input_size", int),
    "PATCHCORE_CORESET_RATIO": ("coreset_ratio", float),
    "PATCHCORE_NUM_NEIGHBORS": ("num_neighbors", int),
    "PATCHCORE_SEED": ("seed", int),
    "PATCHCORE_BATCH_SIZE": ("batch_size", int),
    "PATCHCORE_NUM_WORKERS": ("num_workers", int),
    "PATCHCORE_PRETRAINED": ("pre_trained", "bool"),
    "PATCHCORE_ACCELERATOR": ("accelerator", str),
    "PATCHCORE_DEVICES": ("devices", int),
    "AI_TRAINING_KEEP_STAGING": ("keep_staging", "bool"),
}


def _coerce(kind, raw):
    if kind == "bool":
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if kind == "layers":
        values = [
            item.strip()
            for item in str(raw).replace(";", ",").split(",")
            if item.strip()
        ]
        if not values:
            raise AIDomainError("PATCHCORE_LAYERS no puede estar vacío.")
        return values
    return kind(raw)


def get_training_config(env=None) -> dict:
    """Configuración efectiva (defaults históricos + override por env)."""
    source = os.environ if env is None else env
    config = dict(TRAINING_CONFIG_DEFAULTS)

    for name, (key, kind) in _ENV_OVERRIDES.items():
        raw = source.get(name)

        if raw is None or str(raw).strip() == "":
            continue

        try:
            config[key] = _coerce(kind, raw)
        except (TypeError, ValueError) as error:
            raise AIDomainError(
                f"Configuración inválida en {name}: {raw!r}"
            ) from error

    config["input_size"] = int(config["input_size"])

    if config["input_size"] < 64:
        raise AIDomainError(
            "PATCHCORE_INPUT_SIZE debe ser mayor o igual a 64."
        )
    if not 0.0 < float(config["coreset_ratio"]) <= 1.0:
        raise AIDomainError(
            "PATCHCORE_CORESET_RATIO debe estar en (0, 1]."
        )
    if int(config["batch_size"]) < 1:
        raise AIDomainError("PATCHCORE_BATCH_SIZE debe ser >= 1.")
    if int(config["seed"]) < 0:
        raise AIDomainError("PATCHCORE_SEED debe ser >= 0.")
    if int(config["num_neighbors"]) < 1:
        raise AIDomainError("PATCHCORE_NUM_NEIGHBORS debe ser >= 1.")
    if not isinstance(config["layers"], (list, tuple)) or not config["layers"]:
        raise AIDomainError("PATCHCORE_LAYERS debe listar al menos una capa.")

    config["layers"] = [str(item) for item in config["layers"]]
    config["production_inference_input_size"] = (
        PRODUCTION_INFERENCE_INPUT_SIZE
    )
    config["input_size_consistent_with_production"] = (
        int(config["input_size"]) == PRODUCTION_INFERENCE_INPUT_SIZE
    )
    config["roi_source"] = "ROI_X1,ROI_Y1,ROI_X2,ROI_Y2 (mismas fracciones)"
    config["preprocessing"] = (
        "frame completo -> mascara de prenda -> recorte ROI -> fondo blanco"
    )

    return config


def get_roi_fractions(env=None) -> tuple:
    """Mismas fracciones ROI que usa la inferencia productiva."""
    source = os.environ if env is None else env
    return (
        float(source.get("ROI_X1", "0.10")),
        float(source.get("ROI_Y1", "0.10")),
        float(source.get("ROI_X2", "0.90")),
        float(source.get("ROI_Y2", "0.90")),
    )


def library_versions() -> dict:
    """Versiones relevantes registradas en metadata.json."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    for name in ("anomalib", "torch", "lightning", "cv2", "numpy"):
        try:
            module = __import__(name)
            versions[name] = str(
                getattr(module, "__version__", "desconocida")
            )
        except Exception:
            versions[name] = "no disponible"

    return versions


# ============================================================
# RUTAS DE ARTEFACTOS (siempre relativas bajo AI_ARTIFACTS_ROOT)
# ============================================================


def model_artifact_paths(garment_model_id: int, ai_model_id: int) -> dict:
    """Estructura preferida de FASE 3A bajo AI_ARTIFACTS_ROOT."""
    root = get_ai_artifacts_root()
    model_dir = (
        root
        / f"garment_{int(garment_model_id)}"
        / "models"
        / f"ai_model_{int(ai_model_id)}"
    )
    training_dir = model_dir / "training"

    return {
        "root": root,
        "model_dir": model_dir,
        "training_dir": training_dir,
        "staging_dir": training_dir / "staging",
        "checkpoint_dir": model_dir / "checkpoint",
        "config_path": training_dir / "config.json",
        "metadata_path": training_dir / "metadata.json",
        "log_path": training_dir / "training.log",
    }


def relative_to_root(path, root=None) -> str:
    """Ruta relativa (posix) bajo la raíz de artefactos IA."""
    base = Path(root) if root is not None else get_ai_artifacts_root()
    resolved = Path(path).expanduser().resolve()
    base_resolved = base.expanduser().resolve()

    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise AIDomainError(
            "La ruta del artefacto queda fuera de AI_ARTIFACTS_ROOT."
        )

    return ensure_safe_relative_path(
        resolved.relative_to(base_resolved).as_posix()
    )


# ============================================================
# LOG DEL JOB (append: nunca sobrescribe intentos anteriores)
# ============================================================

_SECRET_ENV_KEYS = (
    "MYSQL_PASSWORD",
    "MYSQL_ROOT_PASSWORD",
    "CAMERA_PASSWORD",
    "SECRET_KEY",
    "RTSP_PASSWORD",
    "API_KEY",
    "TOKEN",
)


def redact_secrets(text) -> str:
    """Elimina valores de secretos conocidos de cualquier texto de log."""
    raw = str(text)

    for key in _SECRET_ENV_KEYS:
        value = os.environ.get(key)

        if value and len(str(value)) >= 4:
            raw = raw.replace(str(value), "***")

    raw = re.sub(r"(?i)(password\s*[=:]\s*)\S+", r"\1***", raw)
    raw = re.sub(r"(?i)(rtsp://[^:\s]+):[^@\s]+@", r"\1***@", raw)

    return raw


def append_training_log(paths: dict, lines) -> None:
    """Escribe (append) en training.log con marca de tiempo."""
    training_dir = Path(paths["training_dir"])
    training_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = []

    for line in lines if isinstance(lines, (list, tuple)) else [lines]:
        payload.append(f"[{stamp}] {redact_secrets(str(line))}")

    with Path(paths["log_path"]).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(payload) + "\n")


# ============================================================
# MATERIALIZACIÓN IDEMPOTENTE DEL DATASET
# ============================================================


def _lock_garment(cur, garment_model_id: int) -> None:
    cur.execute(
        "SELECT id FROM garment_models WHERE id = %s FOR UPDATE",
        (int(garment_model_id),),
    )

    if cur.fetchone() is None:
        raise AIDomainError("El modelo de prenda no existe.")


def _accepted_rows_from_completed_sessions(cur, garment_model_id: int) -> list:
    """Imágenes ACEPTADAS de sesiones COMPLETADA del modelo (orden id)."""
    cur.execute(
        """
        SELECT ti.id, ti.sha256, ti.image_path, ti.garment_model_id,
               ti.status, ti.capture_session_id
        FROM ai_training_images ti
        JOIN ai_capture_sessions s
          ON s.id = ti.capture_session_id
        WHERE ti.garment_model_id = %s
          AND ti.status = 'ACEPTADA'
          AND s.status = 'COMPLETADA'
        ORDER BY ti.id ASC
        """,
        (int(garment_model_id),),
    )
    return [dict(row) for row in cur.fetchall()]


def materialize_training_dataset(
    cur,
    garment_model_id: int,
    actor_id: int | None,
    *,
    artifacts_root=None,
) -> dict:
    """Cierre idempotente del dataset a partir de la captura completada.

    Reglas:
    - solo imágenes ACEPTADAS de sesiones COMPLETADA del mismo modelo;
    - si el dataset más reciente CERRADO ya representa ese conjunto, se reutiliza;
    - si hay un dataset ABIERTO, se completa y se cierra;
    - si cambió el conjunto, se crea una versión nueva (dN).
    """
    _lock_garment(cur, garment_model_id)

    rows = _accepted_rows_from_completed_sessions(cur, garment_model_id)
    desired_hash = compute_manifest_hash(rows)

    cur.execute(
        """
        SELECT id, garment_model_id, version, status, image_count,
               manifest_path, manifest_hash
        FROM ai_datasets
        WHERE garment_model_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(garment_model_id),),
    )
    latest = cur.fetchone()
    created_event = False

    if latest is not None and latest["status"] == DATASET_STATUS_CERRADO:
        same_snapshot = (
            str(latest["manifest_hash"] or "").lower() == desired_hash
            and int(latest["image_count"] or 0) == len(rows)
        )

        if same_snapshot:
            return {
                "id": int(latest["id"]),
                "garment_model_id": int(garment_model_id),
                "version": latest["version"],
                "status": DATASET_STATUS_CERRADO,
                "image_count": int(latest["image_count"] or 0),
                "manifest_path": latest["manifest_path"],
                "manifest_hash": latest["manifest_hash"],
                "created": False,
            }

        dataset = create_ai_dataset(cur, garment_model_id, actor_id)
    elif latest is not None and latest["status"] == DATASET_STATUS_ABIERTO:
        dataset = {
            "id": int(latest["id"]),
            "garment_model_id": int(garment_model_id),
            "version": latest["version"],
            "status": DATASET_STATUS_ABIERTO,
            "image_count": int(latest["image_count"] or 0),
        }
    else:
        dataset = create_ai_dataset(cur, garment_model_id, actor_id)

    if not rows:
        raise AIDomainError(
            "No hay imágenes normales aceptadas de capturas finalizadas "
            "para preparar el dataset."
        )

    dataset_id = int(dataset["id"])

    cur.execute(
        "SELECT image_id FROM ai_dataset_images WHERE dataset_id = %s",
        (dataset_id,),
    )
    linked = {int(row["image_id"]) for row in cur.fetchall()}

    pending = [
        int(row["id"]) for row in rows if int(row["id"]) not in linked
    ]

    if pending:
        added = add_images_to_dataset(cur, dataset_id, pending)
        if added:
            created_event = True

    closed = close_ai_dataset(
        cur,
        dataset_id,
        actor_id,
        artifacts_root=artifacts_root,
    )
    closed["created"] = bool(created_event)

    if created_event:
        record_ai_event(
            cur,
            "DATASET_CLOSED",
            actor_id=actor_id,
            dataset_id=dataset_id,
            payload={
                "garment_model_id": int(garment_model_id),
                "image_count": int(closed["image_count"]),
                "manifest_hash": closed["manifest_hash"],
            },
        )

    return closed


# ============================================================
# VERIFICACIÓN DE INTEGRIDAD PREVIA AL ENTRENAMIENTO
# ============================================================


def verify_training_dataset(
    cur,
    *,
    dataset_id: int,
    ai_model_id: int | None = None,
    artifacts_root=None,
    min_images: int | None = None,
) -> dict:
    """Comprueba todas las condiciones antes de permitir entrenar.

    Devuelve {"ok": bool, "errors": [humanos], "details": {...}}.
    Nunca lanza: quien decide es el llamador (request_training_job).
    """
    errors = []
    details = {}
    root = Path(artifacts_root) if artifacts_root else get_ai_artifacts_root()

    minimum = int(
        min_images
        if min_images is not None
        else get_ai_capture_config()["min_images"]
    )

    cur.execute(
        """
        SELECT id, garment_model_id, version, status, image_count,
               manifest_path, manifest_hash
        FROM ai_datasets
        WHERE id = %s
        """,
        (int(dataset_id),),
    )
    dataset = cur.fetchone()

    if dataset is None:
        return {
            "ok": False,
            "errors": ["El dataset indicado no existe."],
            "details": {},
        }

    dataset = dict(dataset)
    details["dataset"] = {
        "id": int(dataset["id"]),
        "version": dataset["version"],
        "status": dataset["status"],
        "image_count": int(dataset["image_count"] or 0),
        "manifest_path": dataset["manifest_path"],
        "manifest_hash": dataset["manifest_hash"],
    }

    if dataset["status"] != DATASET_STATUS_CERRADO:
        errors.append("El dataset todavía no está cerrado.")

    if int(dataset["image_count"] or 0) < minimum:
        errors.append(
            f"El dataset tiene {int(dataset['image_count'] or 0)} imágenes "
            f"y se requieren al menos {minimum}."
        )

    if ai_model_id is not None:
        cur.execute(
            """
            SELECT id, garment_model_id, version, status
            FROM garment_ai_models
            WHERE id = %s
            """,
            (int(ai_model_id),),
        )
        model = cur.fetchone()

        if model is None:
            errors.append("La versión de IA indicada no existe.")
        else:
            model = dict(model)
            details["ai_model"] = model

            if int(model["garment_model_id"]) != int(
                dataset["garment_model_id"]
            ):
                errors.append(
                    "La versión de IA pertenece a otro modelo de prenda."
                )

    cur.execute(
        """
        SELECT
            di.image_id,
            ti.image_path,
            ti.sha256,
            ti.status,
            ti.garment_model_id
        FROM ai_dataset_images di
        JOIN ai_training_images ti ON ti.id = di.image_id
        WHERE di.dataset_id = %s
        ORDER BY ti.id ASC
        """,
        (int(dataset_id),),
    )
    links = [dict(row) for row in cur.fetchall()]
    details["image_rows"] = len(links)

    seen = set()

    for row in links:
        image_id = int(row["image_id"])

        if image_id in seen:
            errors.append(
                f"La imagen {image_id} está duplicada en el dataset."
            )
            continue

        seen.add(image_id)

        if int(row["garment_model_id"]) != int(dataset["garment_model_id"]):
            errors.append(
                f"La imagen {image_id} pertenece a otro modelo de prenda."
            )
            continue

        if row["status"] != "ACEPTADA":
            errors.append(f"La imagen {image_id} no está aceptada.")
            continue

        try:
            relative = ensure_safe_relative_path(row["image_path"])
            absolute = resolve_under_root(root, relative)
        except AIDomainError:
            errors.append(f"La imagen {image_id} tiene una ruta no segura.")
            continue

        if not absolute.is_file():
            errors.append(
                f"Falta el archivo de la imagen {image_id} en el dataset."
            )
            continue

        try:
            digest = sha256_file(absolute)
        except OSError:
            errors.append(
                f"No se pudo leer la imagen {image_id} del dataset."
            )
            continue

        if digest != normalize_sha256(str(row["sha256"])):
            errors.append(
                f"El hash de la imagen {image_id} no coincide con el "
                "registrado."
            )

    if int(dataset["image_count"] or 0) != len(links):
        errors.append(
            "El número de imágenes del dataset no coincide con su "
            "relación de imágenes."
        )

    if not dataset["manifest_path"]:
        errors.append("El dataset no tiene manifest.")
    else:
        try:
            manifest_abs = resolve_under_root(
                root,
                ensure_safe_relative_path(dataset["manifest_path"]),
            )
        except AIDomainError:
            manifest_abs = None
            errors.append("La ruta del manifest no es segura.")

        if manifest_abs is not None:
            if not manifest_abs.is_file():
                errors.append("No existe el manifest del dataset.")
            else:
                try:
                    manifest = json.loads(
                        manifest_abs.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    manifest = None
                    errors.append("El manifest del dataset no es legible.")

                if manifest is not None:
                    if int(manifest.get("image_count") or 0) != len(links):
                        errors.append(
                            "El manifest no coincide con las imágenes "
                            "del dataset."
                        )

                    if (
                        str(manifest.get("manifest_hash") or "").lower()
                        != str(dataset["manifest_hash"] or "").lower()
                    ):
                        errors.append(
                            "El hash del manifest no coincide con el "
                            "registrado en base de datos."
                        )

                    recomputed = compute_manifest_hash(
                        [
                            {
                                "id": item.get("image_id"),
                                "sha256": item.get("sha256"),
                            }
                            for item in manifest.get("images", [])
                        ]
                    )

                    if recomputed != str(dataset["manifest_hash"] or ""):
                        errors.append(
                            "El contenido del manifest no reproduce su "
                            "hash declarado."
                        )

    if ai_model_id is not None and details.get("ai_model"):
        model = details["ai_model"]

        if model["status"] not in (
            AI_MODEL_STATUS_PREPARACION,
            AI_MODEL_STATUS_ENTRENANDO,
            AI_MODEL_STATUS_FALLIDO,
        ):
            errors.append(
                "La versión de IA no está en estado que permita entrenar."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "details": details,
    }


def humanize_training_error(message) -> str:
    """Mensaje comprensible para la dueña (sin traceback ni ids SQL)."""
    text = str(message or "").strip()

    if not text:
        return "El entrenamiento no pudo completarse."

    lowered = text.lower()

    # Barreras del reentrenamiento como nueva versión (FASE 3A.3):
    # mensajes de dominio ya aptos para la persona usuaria.
    if "versión anterior" in lowered or "version anterior" in lowered:
        return text
    if "complete la validación" in lowered or "complete la validacion" in lowered:
        return text
    if "faltan" in lowered and "imágenes" in lowered:
        return text
    if "no hay un dataset" in lowered:
        return text

    if "no está cerrado" in lowered or "no esta cerrado" in lowered:
        return "El dataset de imágenes todavía no está cerrado."
    if "se requieren al menos" in lowered:
        return text
    if "no coincide" in lowered and "hash" in lowered:
        return (
            "La verificación de imágenes falló: una imagen cambió "
            "después de cerrar el dataset."
        )
    if "no existe el manifest" in lowered or "manifest" in lowered:
        return "El dataset no tiene su manifiesto de imágenes válido."
    if "ruta no segura" in lowered or "path traversal" in lowered:
        return "El dataset contiene una ruta no permitida."
    if "otro modelo de prenda" in lowered:
        return "El dataset pertenece a otro modelo de prenda."
    if "en curso" in lowered or "duplicado" in lowered:
        return "Ya hay un entrenamiento en curso para este modelo."
    if "en proceso" in lowered:
        return (
            "Ya existe una versión de IA en proceso para este modelo."
        )
    if "silueta" in lowered:
        return (
            "No se pudo preparar una de las imágenes: la prenda no se "
            "detectó correctamente. Capture de nuevo."
        )
    if "cuda" in lowered or "out of memory" in lowered or "memory" in lowered:
        return (
            "El equipo no pudo terminar el entrenamiento por falta de "
            "recursos. Intente de nuevo."
        )
    if "download" in lowered or "url" in lowered or "network" in lowered:
        return (
            "No se pudieron descargar los pesos iniciales del modelo. "
            "Revise la conexión a internet del servidor."
        )

    return "El entrenamiento no pudo completarse."


# ============================================================
# SOLICITUD DE ENTRENAMIENTO (todas las barreras en backend/BD)
# ============================================================


def retrain_new_version_preconditions(cur, garment_model_id: int) -> dict:
    """Barreras de la acción «Reentrenar como nueva versión» (§4).

    No crea ni modifica nada: solo decide si la acción está permitida y
    aporta el contexto (versiones históricas y versión siguiente) que la
    auditoría debe registrar.
    """
    cur.execute(
        """
        SELECT id, version, status, notes
        FROM garment_ai_models
        WHERE garment_model_id = %s
        ORDER BY id ASC
        """,
        (int(garment_model_id),),
    )
    versions = [dict(row) for row in cur.fetchall()]

    available, reason = evaluate_retrain_availability(versions)

    cur.execute(
        """
        SELECT j.id
        FROM ai_jobs j
        JOIN garment_ai_models m ON m.id = j.ai_model_id
        WHERE m.garment_model_id = %s
          AND j.kind = %s
          AND j.status IN (%s, %s)
        LIMIT 1
        """,
        (
            int(garment_model_id),
            JOB_KIND_TRAINING,
            JOB_STATUS_PENDIENTE,
            JOB_STATUS_EN_CURSO,
        ),
    )

    if cur.fetchone() is not None:
        available = False
        reason = "Ya hay un entrenamiento en curso para este modelo."

    return {
        "available": bool(available),
        "reason": reason,
        "versions": versions,
        "historical": [
            {"version": item["version"], "status": item["status"]}
            for item in versions
            if is_technically_invalidated(item.get("notes"))
        ],
        "next_version": next_version_label(versions),
    }


def request_training_job(
    cur,
    *,
    garment_model_id: int,
    actor_id: int | None,
    artifacts_root=None,
    config: dict | None = None,
    as_new_version: bool = False,
) -> dict:
    """Materializa/verifica el dataset y crea el job TRAINING.

    Debe ejecutarse en una transacción abierta por el llamador.
    Lanza AIDomainError con mensaje humano si alguna barrera falla.

    Con `as_new_version=True` se pide explícitamente una NUEVA versión
    (vN+1) que reemplaza a una versión anterior no apta: nunca se
    reutiliza ni se toca la versión histórica ni su directorio de
    artefactos.
    """
    root = Path(artifacts_root) if artifacts_root else get_ai_artifacts_root()
    cfg = dict(config or get_training_config())

    _lock_garment(cur, garment_model_id)

    cur.execute(
        """
        SELECT id, code, status, active
        FROM garment_models
        WHERE id = %s
        """,
        (int(garment_model_id),),
    )
    garment = cur.fetchone()

    if garment is None:
        raise AIDomainError("El modelo de prenda no existe.")

    if garment["status"] != "APROBADO" or int(garment["active"] or 0) != 1:
        raise AIDomainError(
            "El entrenamiento solo puede iniciarse con el modelo "
            "aprobado y activo."
        )

    retrain = None

    if as_new_version:
        retrain = retrain_new_version_preconditions(cur, garment_model_id)

        if not retrain["available"]:
            raise AIDomainError(
                retrain["reason"]
                or "No es posible entrenar una versión nueva ahora."
            )

    cur.execute(
        """
        SELECT id, version, status, notes, garment_model_id,
               parent_ai_model_id, source_dataset_id,
               normal_augmentation_min_new_images
        FROM garment_ai_models
        WHERE garment_model_id = %s AND status = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(garment_model_id), AI_MODEL_STATUS_PREPARACION),
    )
    model = cur.fetchone()

    if model is not None and is_technically_invalidated(model.get("notes")):
        # Una PREPARACION invalidada es histórica: no se reutiliza.
        model = None

    if model is None:
        # Sin versión PREPARACION disponible se crea una nueva
        # (también cubre el reintento tras un FALLIDO).
        try:
            model = create_next_ai_model_version(
                cur,
                int(garment_model_id),
                actor_id,
            )
        except AIDomainError as error:
            raise AIDomainError(str(error)) from error

        record_ai_event(
            cur,
            "VERSION_PREPARED",
            actor_id=actor_id,
            ai_model_id=int(model["id"]),
            payload={"garment_model_id": int(garment_model_id)},
        )

    ai_model_id = int(model["id"])

    cur.execute(
        """
        SELECT id, status
        FROM ai_jobs
        WHERE ai_model_id = %s AND kind = %s AND status IN (%s, %s)
        LIMIT 1
        """,
        (
            ai_model_id,
            JOB_KIND_TRAINING,
            JOB_STATUS_PENDIENTE,
            JOB_STATUS_EN_CURSO,
        ),
    )
    active_row = cur.fetchone()

    if active_row is not None:
        raise AIDomainError(
            "Ya hay un entrenamiento en curso para este modelo."
        )

    if model.get("parent_ai_model_id") is not None:
        required_new = int(
            model.get("normal_augmentation_min_new_images") or 20
        )
        cur.execute(
            """SELECT COUNT(DISTINCT ti.id) AS total
               FROM ai_training_images ti
               JOIN ai_capture_sessions cs ON cs.id = ti.capture_session_id
               WHERE cs.target_ai_model_id = %s
                 AND cs.status = 'COMPLETADA'
                 AND ti.status = 'ACEPTADA'""",
            (int(model["id"]),),
        )
        new_good_count = int(cur.fetchone()["total"] or 0)
        if new_good_count < required_new:
            raise AIDomainError(
                f"La preparación v3 requiere al menos {required_new} imágenes BUENAS nuevas "
                f"(actualmente {new_good_count}); las 52 heredadas no satisfacen este requisito."
            )

    dataset = materialize_training_dataset(
        cur,
        int(garment_model_id),
        actor_id,
        artifacts_root=root,
    )

    verification = verify_training_dataset(
        cur,
        dataset_id=int(dataset["id"]),
        ai_model_id=ai_model_id,
        artifacts_root=root,
    )

    if not verification["ok"]:
        raise AIDomainError(
            "; ".join(verification["errors"])
            or "El dataset no pasó la verificación de integridad."
        )

    paths = model_artifact_paths(garment_model_id, ai_model_id)
    log_relative = relative_to_root(paths["log_path"], root)

    # Quality gate ANTES del job: staging real + validación de calidad.
    prepare_training_staging(
        cur,
        int(dataset["id"]),
        paths["staging_dir"],
        get_roi_fractions(),
        expected_count=int(dataset["image_count"]),
        contact_sheet_path=Path(paths["training_dir"]) / "contact_sheet.png",
    )

    job = create_ai_job(
        cur,
        JOB_KIND_TRAINING,
        ai_model_id=ai_model_id,
        dataset_id=int(dataset["id"]),
        created_by=actor_id,
        log_path=log_relative,
    )

    cur.execute(
        """
        UPDATE ai_jobs
        SET config_json = %s, updated_at = NOW()
        WHERE id = %s
        """,
        (
            json.dumps(cfg, ensure_ascii=False),
            int(job["id"]),
        ),
    )

    manifest_relative = (
        ensure_safe_relative_path(dataset["manifest_path"])
        if dataset.get("manifest_path")
        else None
    )

    cur.execute(
        """
        UPDATE garment_ai_models
        SET dataset_id = %s,
            dataset_name = %s,
            dataset_path = %s
        WHERE id = %s
        """,
        (
            int(dataset["id"]),
            dataset.get("version"),
            manifest_relative,
            ai_model_id,
        ),
    )

    record_ai_event(
        cur,
        "JOB_CREATED",
        actor_id=actor_id,
        ai_model_id=ai_model_id,
        dataset_id=int(dataset["id"]),
        payload={
            "job_id": int(job["id"]),
            "kind": JOB_KIND_TRAINING,
            "garment_model_id": int(garment_model_id),
            "image_count": int(dataset["image_count"]),
        },
    )

    if as_new_version:
        # Auditoría explícita del reentrenamiento como NUEVA versión:
        # versión histórica intacta, dataset reutilizado, artefactos
        # propios de la versión nueva.
        record_ai_event(
            cur,
            "NEW_VERSION_TRAINING_REQUESTED",
            actor_id=actor_id,
            ai_model_id=ai_model_id,
            dataset_id=int(dataset["id"]),
            payload={
                "job_id": int(job["id"]),
                "garment_model_id": int(garment_model_id),
                "version": model["version"],
                "superseded_versions": retrain["historical"],
                "dataset_reused": int(dataset["id"]),
                "images_reused": int(dataset["image_count"]),
                "artifacts_dir": (
                    f"garment_{int(garment_model_id)}"
                    f"/models/ai_model_{int(ai_model_id)}"
                ),
            },
        )

    return {
        "job": job,
        "dataset": dataset,
        "ai_model": {
            "id": ai_model_id,
            "garment_model_id": int(garment_model_id),
            "version": model["version"],
            "status": model["status"],
        },
        "verification": verification,
        "config": cfg,
        "retrain": retrain,
    }


def cancel_pending_training_job(
    cur,
    *,
    job_id: int,
    actor_id: int | None,
) -> dict:
    """Cancela un job aún PENDIENTE (nunca uno EN_CURSO).

    PatchCore no puede interrumpirse de forma fiable: no se simula
    una cancelación inmediata de un entrenamiento en marcha.
    """
    cur.execute(
        "SELECT id, status, ai_model_id FROM ai_jobs WHERE id = %s FOR UPDATE",
        (int(job_id),),
    )
    job = cur.fetchone()

    if job is None:
        raise AIDomainError("El trabajo de entrenamiento no existe.")

    if job["status"] == JOB_STATUS_EN_CURSO:
        raise AIDomainError(
            "El entrenamiento ya está en curso y no puede cancelarse."
        )

    if job["status"] != JOB_STATUS_PENDIENTE:
        raise AIDomainError(
            "Solo se pueden cancelar trabajos que aún no comenzaron."
        )

    transition_ai_job_status(
        cur,
        int(job_id),
        JOB_STATUS_CANCELADO,
        stage="CANCELADO",
        stage_label="Cancelado antes de iniciar",
    )

    record_ai_event(
        cur,
        "TRAINING_CANCELLED",
        actor_id=actor_id,
        ai_model_id=job["ai_model_id"],
        payload={
            "job_id": int(job_id),
            "outcome": "CANCELADO",
        },
    )

    return {"id": int(job_id), "status": JOB_STATUS_CANCELADO}


# ============================================================
# PAYLOAD DE ESTADO PARA LA UI
# ============================================================

TRAINING_UI_BLOCKED = "NO_DISPONIBLE"
TRAINING_UI_READY = "DISPONIBLE"
TRAINING_UI_RUNNING = "ENTRENANDO"
TRAINING_UI_COMPLETED = "ENTRENADO"
TRAINING_UI_FAILED = "FALLIDO"
# Versión entrenada que quedó marcada NO VALIDADA / NO APTA: es
# histórica, no entra a validación y habilita el reentrenamiento.
TRAINING_UI_HISTORICAL = "HISTORICA"

TRAINING_UI_LABELS = {
    TRAINING_UI_BLOCKED: "NO DISPONIBLE",
    TRAINING_UI_READY: "LISTO PARA ENTRENAR",
    TRAINING_UI_RUNNING: "ENTRENANDO",
    TRAINING_UI_COMPLETED: "PENDIENTE DE VALIDACIÓN",
    TRAINING_UI_FAILED: "ENTRENAMIENTO FALLIDO",
    TRAINING_UI_HISTORICAL: "VERSIÓN HISTÓRICA · NO APTA PARA VALIDACIÓN",
}

RETRAIN_UI_LABEL = "Entrenar nueva versión"


def training_status_payload(cur, garment_model_id, *, allowed=None) -> dict:
    """Estado del entrenamiento para la ficha del modelo."""
    cfg = get_ai_capture_config()
    minimum = int(cfg["min_images"])

    payload = {
        "ui_status": TRAINING_UI_BLOCKED,
        "ui_label": TRAINING_UI_LABELS[TRAINING_UI_BLOCKED],
        "can_train": False,
        "blocked_reason": None,
        "job_id": None,
        "job_status": None,
        "can_cancel": False,
        "progress": 0,
        "stage": None,
        "stage_label": None,
        "image_count": 0,
        "version": None,
        "model_id": None,
        "model_status": None,
        "dataset_status": None,
        "dataset_id": None,
        "min_count": minimum,
        "accepted_count": 0,
        "elapsed_seconds": None,
        "error_human": None,
        "started_at": None,
        "finished_at": None,
        "log_path": None,
        # Reentrenamiento como nueva versión (FASE 3A.3).
        "retrain_available": False,
        "retrain_blocked_reason": None,
        "retrain_label": RETRAIN_UI_LABEL,
        "retrain_image_count": 0,
        "next_version": None,
        "historical_version": None,
        "invalidated_versions": [],
        # Validación controlada (FASE 3B): la versión entrenada pasa a
        # validación, nunca a otra versión ni a activación.
        "validation_available": False,
        "validation_ai_model_id": None,
        "validation_version": None,
    }

    if not garment_model_id:
        payload["blocked_reason"] = "Modelo de prenda no disponible."
        return payload

    cur.execute(
        """
        SELECT COUNT(*) AS total
        FROM ai_training_images ti
        JOIN ai_capture_sessions s ON s.id = ti.capture_session_id
        WHERE ti.garment_model_id = %s
          AND ti.status = 'ACEPTADA'
          AND s.status = 'COMPLETADA'
        """,
        (int(garment_model_id),),
    )
    accepted = int(cur.fetchone()["total"] or 0)
    payload["accepted_count"] = accepted

    cur.execute(
        """
        SELECT id, version, status, image_count, manifest_path
        FROM ai_datasets
        WHERE garment_model_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(garment_model_id),),
    )
    dataset = cur.fetchone()

    if dataset:
        payload["dataset_id"] = int(dataset["id"])
        payload["dataset_status"] = dataset["status"]
        payload["image_count"] = int(dataset["image_count"] or 0)
        payload["version"] = dataset["version"]

    # Fuente única de verdad del estado de la última versión de IA.
    # Es necesaria aunque no exista job: un job pudo purgarse (o nunca
    # haberse creado) y la versión ya quedó ENTRENADO en la BD.
    cur.execute(
        """
        SELECT id, version, status, notes, normal_images_count, trained_at,
               parent_ai_model_id, source_dataset_id,
               normal_augmentation_min_new_images
        FROM garment_ai_models
        WHERE garment_model_id = %s
        ORDER BY id ASC
        """,
        (int(garment_model_id),),
    )
    versions = [dict(row) for row in cur.fetchall()]
    latest_model = versions[-1] if versions else None

    invalidated = [
        item
        for item in versions
        if is_technically_invalidated(item.get("notes"))
    ]

    if latest_model:
        payload["model_id"] = int(latest_model["id"])
        payload["model_status"] = latest_model["status"]

    payload["next_version"] = next_version_label(versions) if versions else None
    payload["invalidated_versions"] = [item["version"] for item in invalidated]

    # FASE 3B: la versión entrenada pasa a VALIDACIÓN con imágenes
    # nuevas (nunca a otra versión ni a activación). Una versión
    # invalidada técnicamente es histórica y no se valida.
    if (
        latest_model
        and latest_model["status"] in (
            AI_MODEL_STATUS_ENTRENADO,
            AI_MODEL_STATUS_VALIDACION,
            AI_MODEL_STATUS_VALIDADO,
        )
        and not is_technically_invalidated(latest_model.get("notes"))
    ):
        payload["validation_available"] = True
        payload["validation_ai_model_id"] = int(latest_model["id"])
        payload["validation_version"] = latest_model["version"]

    cur.execute(
        """
        SELECT j.id, j.status, j.progress, j.stage, j.stage_label,
               j.error_message, j.log_path, j.ai_model_id, j.dataset_id,
               j.created_at, j.started_at, j.finished_at,
               TIMESTAMPDIFF(SECOND, j.started_at, NOW()) AS elapsed,
               m.version AS model_version, m.status AS model_status
        FROM ai_jobs j
        JOIN garment_ai_models m ON m.id = j.ai_model_id
        WHERE m.garment_model_id = %s AND j.kind = %s
        ORDER BY j.id DESC
        LIMIT 1
        """,
        (int(garment_model_id), JOB_KIND_TRAINING),
    )
    job = cur.fetchone()

    active_statuses = (JOB_STATUS_PENDIENTE, JOB_STATUS_EN_CURSO)
    job_active = bool(job and job["status"] in active_statuses)

    # Disponibilidad de «Entrenar nueva versión»: reglas de dominio
    # (versión anterior no activable) + dataset sano + sin job activo.
    retrain_available, retrain_reason = evaluate_retrain_availability(versions)

    if not retrain_available:
        pass
    elif allowed is False:
        retrain_available = False
        retrain_reason = "No tiene permisos para entrenar este modelo."
    elif job_active:
        retrain_available = False
        retrain_reason = "Ya hay un entrenamiento en curso para este modelo."
    elif accepted < minimum:
        retrain_available = False
        retrain_reason = (
            f"Faltan {minimum - accepted} imágenes para alcanzar el "
            f"mínimo de {minimum}."
        )
    elif dataset is None:
        retrain_available = False
        retrain_reason = "Todavía no hay un dataset de imágenes cerrado."
    elif dataset["status"] != DATASET_STATUS_CERRADO:
        retrain_available = False
        retrain_reason = "El dataset de imágenes todavía no está cerrado."

    payload["retrain_available"] = bool(retrain_available)
    payload["retrain_blocked_reason"] = None if retrain_available else retrain_reason
    payload["retrain_image_count"] = int(
        dataset["image_count"] if dataset else 0
    )

    # Última versión entrenada pero invalidada técnicamente: histórica,
    # no apta para validación/activación, con reentrenamiento disponible.
    if (
        latest_model
        and latest_model["status"] in (
            AI_MODEL_STATUS_ENTRENADO,
            AI_MODEL_STATUS_VALIDACION,
            AI_MODEL_STATUS_VALIDADO,
        )
        and is_technically_invalidated(latest_model.get("notes"))
        and not job_active
    ):
        payload.update(
            {
                "ui_status": TRAINING_UI_HISTORICAL,
                "ui_label": TRAINING_UI_LABELS[TRAINING_UI_HISTORICAL],
                "can_train": False,
                "version": latest_model["version"],
                "model_id": int(latest_model["id"]),
                "model_status": latest_model["status"],
                "historical_version": latest_model["version"],
                "progress": 100.0,
                "stage": "HISTORICA",
                "stage_label": (
                    "Entrenamiento completado; versión no apta para "
                    "validación"
                ),
                "image_count": int(
                    dataset["image_count"]
                    if dataset
                    else (latest_model["normal_images_count"] or 0)
                ),
                "finished_at": str(latest_model["trained_at"] or ""),
            }
        )

        if job and job["status"] == JOB_STATUS_COMPLETADO:
            payload["job_id"] = int(job["id"])
            payload["log_path"] = job["log_path"]

        return payload

    if job_active:
        payload.update(
            {
                "ui_status": TRAINING_UI_RUNNING,
                "ui_label": TRAINING_UI_LABELS[TRAINING_UI_RUNNING],
                "job_id": int(job["id"]),
                "job_status": job["status"],
                "can_cancel": job["status"] == JOB_STATUS_PENDIENTE,
                "progress": float(job["progress"] or 0),
                "stage": job["stage"],
                "stage_label": job["stage_label"] or "Entrenando",
                "version": job["model_version"],
                "started_at": str(job["started_at"] or ""),
                "elapsed_seconds": (
                    int(job["elapsed"])
                    if job["started_at"] and job["elapsed"] is not None
                    else None
                ),
                "log_path": job["log_path"],
            }
        )
        return payload

    if (
        latest_model
        and latest_model.get("parent_ai_model_id") is not None
        and latest_model.get("status") == AI_MODEL_STATUS_PREPARACION
    ):
        required_new = int(
            latest_model.get("normal_augmentation_min_new_images") or 20
        )
        cur.execute(
            """SELECT COUNT(DISTINCT ti.id) AS total
               FROM ai_training_images ti
               JOIN ai_capture_sessions cs ON cs.id = ti.capture_session_id
               WHERE cs.target_ai_model_id = %s AND cs.status = 'COMPLETADA'
                 AND ti.status = 'ACEPTADA'""",
            (int(latest_model["id"]),),
        )
        new_count = int(cur.fetchone()["total"] or 0)
        cur.execute(
            """SELECT COUNT(DISTINCT ti.id) AS total
               FROM ai_training_images ti
               JOIN ai_capture_sessions cs ON cs.id = ti.capture_session_id
               WHERE cs.target_ai_model_id = %s AND cs.status = 'COMPLETADA'
                 AND ti.status = 'ACEPTADA'""",
            (int(latest_model["id"]),),
        )
        completed_new_count = int(cur.fetchone()["total"] or 0)
        payload.update({
            "version": latest_model["version"],
            "model_id": int(latest_model["id"]),
            "model_status": latest_model["status"],
            "inherited_normal_count": 52,
            "new_normal_count": new_count,
            "completed_new_normal_count": completed_new_count,
            "total_normal_available": 52 + new_count,
            "minimum_new_normal_count": required_new,
            "image_count": 52 + new_count,
        })
        if allowed is False:
            payload["blocked_reason"] = "No tiene permisos para entrenar este modelo."
            return payload
        if completed_new_count < required_new:
            payload["blocked_reason"] = (
                f"Capture y finalice al menos {required_new} imágenes BUENAS nuevas; "
                f"hay {completed_new_count} finalizadas. Las 52 heredadas no sustituyen las nuevas."
            )
            return payload
        payload.update({
            "ui_status": TRAINING_UI_READY,
            "ui_label": "LISTO PARA ENTRENAR v3 · SOLO BUENAS",
            "can_train": True,
            "blocked_reason": None,
            "accepted_count": 52 + new_count,
        })
        return payload

    if job and job["status"] == JOB_STATUS_COMPLETADO:
        payload.update(
            {
                "ui_status": TRAINING_UI_COMPLETED,
                "ui_label": TRAINING_UI_LABELS[TRAINING_UI_COMPLETED],
                "job_id": int(job["id"]),
                "progress": 100.0,
                "stage": "COMPLETADO",
                "stage_label": "Entrenamiento completado",
                "version": job["model_version"],
                "image_count": int(
                    dataset["image_count"] if dataset else payload["image_count"]
                ),
                "finished_at": str(job["finished_at"] or ""),
                "log_path": job["log_path"],
            }
        )
        return payload

    if job and job["status"] == JOB_STATUS_FALLIDO:
        payload.update(
            {
                "ui_status": TRAINING_UI_FAILED,
                "ui_label": TRAINING_UI_LABELS[TRAINING_UI_FAILED],
                "job_id": int(job["id"]),
                "job_status": job["status"],
                "progress": float(job["progress"] or 0),
                "stage": job["stage"],
                "stage_label": job["stage_label"],
                "version": job["model_version"],
                "error_human": humanize_training_error(
                    job["error_message"]
                ),
                "finished_at": str(job["finished_at"] or ""),
                "log_path": job["log_path"],
            }
        )
        return payload

    # Última versión entrenada (o en validación) sin job asociado:
    # la versión ya existe en la BD, así que no se ofrece reentrenar.
    # Mientras haya una versión pendiente de validación, la siguiente
    # acción es validar (FASE 3B), no crear otra versión.
    pending_validation_statuses = (
        AI_MODEL_STATUS_ENTRENADO,
        AI_MODEL_STATUS_VALIDACION,
        AI_MODEL_STATUS_VALIDADO,
    )

    if latest_model and latest_model["status"] in pending_validation_statuses:
        payload.update(
            {
                "ui_status": TRAINING_UI_COMPLETED,
                "ui_label": TRAINING_UI_LABELS[TRAINING_UI_COMPLETED],
                "can_train": False,
                "progress": 100.0,
                "stage": "COMPLETADO",
                "stage_label": "Entrenamiento completado",
                "version": latest_model["version"],
                "image_count": int(
                    dataset["image_count"]
                    if dataset
                    else (latest_model["normal_images_count"] or 0)
                ),
                "finished_at": str(latest_model["trained_at"] or ""),
            }
        )
        return payload

    # Sin jobs: ¿podemos entrenar?
    reason = None

    if allowed is False:
        reason = "No tiene permisos para entrenar este modelo."
    elif accepted < minimum:
        reason = (
            f"Faltan {minimum - accepted} imágenes para alcanzar el "
            f"mínimo de {minimum}."
        )
    elif dataset is None:
        # Sin dataset todavía: al entrenar se materializa y cierra.
        reason = None
    elif dataset["status"] != DATASET_STATUS_CERRADO:
        reason = "El dataset de imágenes todavía no está cerrado."

    if reason is None:
        payload.update(
            {
                "ui_status": TRAINING_UI_READY,
                "ui_label": TRAINING_UI_LABELS[TRAINING_UI_READY],
                "can_train": True,
            }
        )
    else:
        payload["blocked_reason"] = reason

    return payload


# ============================================================
# EJECUCIÓN DEL JOB (worker)
# ============================================================

STAGE_PROTOCOL = (
    (5, "VALIDANDO_DATASET", "Validando dataset"),
    (15, "PREPARANDO_ARCHIVOS", "Preparando archivos"),
    (25, "INICIALIZANDO_PATCHCORE", "Inicializando PatchCore"),
    (35, "EMBEDDINGS", "Generando representaciones"),
    (70, "MEMORY_BANK", "Construyendo modelo"),
    (85, "GUARDANDO_MODELO", "Guardando modelo"),
    (95, "VERIFICANDO_ARTEFACTOS", "Verificando artefactos"),
    (100, "COMPLETADO", "Completado"),
)


def default_connect():
    """Conexión MySQL propia del worker (sin depender de app.py)."""
    import mysql.connector

    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database=os.environ.get("MYSQL_DATABASE", "textile_quality_db"),
        connection_timeout=10,
    )


class _JobReporter:
    """Actualiza progreso/etapa en MySQL con transacciones cortas."""

    def __init__(self, connect, job_id, log_paths, on_log=None):
        self._connect = connect
        self._job_id = int(job_id)
        self._log_paths = log_paths
        self._on_log = on_log
        self._last_progress = -1.0
        self._stages = []

    @property
    def stages(self):
        return list(self._stages)

    def log(self, message):
        text = redact_secrets(str(message))
        append_training_log(self._log_paths, text)

        if self._on_log:
            self._on_log(text)

    def report(self, progress, stage, stage_label):
        value = max(0.0, min(100.0, float(progress)))

        if value < self._last_progress:
            value = self._last_progress

        self._last_progress = value

        if not self._stages or self._stages[-1]["stage"] != stage:
            self._stages.append(
                {
                    "stage": stage,
                    "stage_label": stage_label,
                    "progress": value,
                    "at": datetime.now().isoformat(timespec="seconds"),
                }
            )

        self.log(f"ETAPA {value:5.1f}%  {stage}: {stage_label}")

        try:
            conn = self._connect()
        except Exception as error:  # noqa: BLE001 - el job continúa
            self.log(
                "sin_conexion_para_progreso="
                + redact_secrets(str(error))
            )
            return value

        try:
            cur = conn.cursor(dictionary=True)
            try:
                conn.start_transaction()
                update_job_progress(
                    cur,
                    self._job_id,
                    value,
                    stage=stage,
                    stage_label=stage_label,
                )
                conn.commit()
            finally:
                cur.close()
        except Exception as error:  # noqa: BLE001 - el job continúa
            self.log(
                "no_se_pudo_reportar_progreso="
                + redact_secrets(str(error))
            )
        finally:
            conn.close()

        return value


def stage_training_images(cur, dataset_id, staging_dir, roi_fractions) -> list:
    """Genera la representación de entrenamiento (misma que producción)."""
    import cv2

    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    cur.execute(
        """
        SELECT ti.id, ti.image_path, ti.sha256, ti.garment_model_id
        FROM ai_dataset_images di
        JOIN ai_training_images ti ON ti.id = di.image_id
        WHERE di.dataset_id = %s
        ORDER BY ti.id ASC
        """,
        (int(dataset_id),),
    )
    rows = [dict(row) for row in cur.fetchall()]

    if not rows:
        raise AIDomainError("El dataset no contiene imágenes.")

    root = get_ai_artifacts_root()
    staged = []

    for row in rows:
        relative = ensure_safe_relative_path(row["image_path"])
        source = resolve_under_root(root, relative)

        if not source.is_file():
            raise AIDomainError(
                f"Falta el archivo de la imagen {row['id']} en el dataset."
            )

        frame = cv2.imread(str(source))

        if frame is None:
            raise AIDomainError(
                f"No se pudo leer la imagen {row['id']} del dataset."
            )

        roi_bounds = patchcore_preprocess.roi_bounds_from_fractions(
            frame,
            roi_fractions,
        )

        try:
            mask = patchcore_preprocess.create_garment_mask(
                frame,
                roi_bounds,
            )
        except Exception as error:
            raise AIDomainError(
                f"No se pudo preparar la imagen {row['id']}: "
                "la prenda no se detectó correctamente."
            ) from error

        patchcore_input = patchcore_preprocess.build_patchcore_input(
            frame,
            mask,
            roi_bounds,
        )

        target = staging / f"frame_{int(row['id'])}.png"

        if not cv2.imwrite(str(target), patchcore_input):
            raise AIDomainError(
                f"No se pudo preparar la imagen {row['id']} para entrenar."
            )

        staged.append(
            {
                "image_id": int(row["id"]),
                "path": target,
                "source": source,
            }
        )

    return staged


# ============================================================
# QUALITY GATE DEL STAGING (previo a crear el job TRAINING)
# ============================================================
#
# Un dataset colapsado a un único color entrena un modelo inservible y
# consume el único checkpoint disponible. El gate corre ANTES de
# create_ai_job: si falla, no existe job, no existe checkpoint y no hay
# entrenamiento que reintentar a ciegas.

STAGING_QUALITY_GATE_MESSAGE = (
    "El conjunto preparado para entrenamiento no superó la validación "
    "de calidad. No se inició el entrenamiento."
)

# Un ROI real reparte los píxeles entre muchos colores; una imagen
# colapsada (blanco total) tiene desviación ~0 y un solo color domina.
STAGING_QUALITY_MIN_STD = 2.0
STAGING_QUALITY_MAX_DOMINANT_SHARE = 0.995
STAGING_QUALITY_SAMPLE_STRIDE = 16
# Mínimo de imágenes visualmente distintas para no entrenar sobre
# un dataset repetido.
STAGING_QUALITY_MIN_DISTINCT_SHARE = 0.90


def _staged_visual_hash(image) -> str:
    """Huella visual barata (64x64 gris) para detectar duplicados."""
    import hashlib

    import cv2

    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.sha256(small.tobytes()).hexdigest()


def quality_gate_staged_images(staged, *, expected_count=None) -> dict:
    """Valida que el staging represente imágenes reales (no uniformes)."""
    import cv2
    import hashlib
    import numpy as np

    errors: list[str] = []
    shapes: dict[tuple, int] = {}
    file_hashes: list[str] = []
    visual_hashes: list[str] = []
    unique_min = None
    std_min = None
    dominant_max = None

    if expected_count is not None and len(staged) != int(expected_count):
        errors.append(
            f"Se prepararon {len(staged)} imágenes y el dataset declara "
            f"{int(expected_count)}."
        )

    if not staged:
        errors.append("No se preparó ninguna imagen para entrenar.")

    for item in staged:
        path = Path(item["path"])
        image = cv2.imread(str(path))

        if image is None:
            errors.append(
                f"No se pudo leer la imagen preparada {path.name}."
            )
            continue

        shape = tuple(int(value) for value in image.shape)
        shapes[shape] = shapes.get(shape, 0) + 1

        try:
            file_hashes.append(
                hashlib.sha256(Path(path).read_bytes()).hexdigest()
            )
        except OSError:
            errors.append(
                f"No se pudo leer el archivo preparado {path.name}."
            )
        visual_hashes.append(_staged_visual_hash(image))

        pixels = image.reshape(-1, image.shape[2])
        std = float(pixels.std())

        # Histograma 32x32x32: detecta un color dominante aplastante
        # sin recorrer los millones de píxeles uno por uno.
        histogram = cv2.calcHist(
            [image],
            [0, 1, 2],
            None,
            [32, 32, 32],
            [0, 256, 0, 256, 0, 256],
        )
        total = float(histogram.sum()) or 1.0
        dominant = float(histogram.max()) / total
        sampled = pixels[::STAGING_QUALITY_SAMPLE_STRIDE]
        colors = int(len(np.unique(sampled, axis=0)))

        unique_min = colors if unique_min is None else min(unique_min, colors)
        std_min = std if std_min is None else min(std_min, std)
        dominant_max = (
            dominant if dominant_max is None else max(dominant_max, dominant)
        )

        if (
            std < STAGING_QUALITY_MIN_STD
            or dominant >= STAGING_QUALITY_MAX_DOMINANT_SHARE
        ):
            errors.append(
                f"La imagen preparada {path.name} quedó sin variación "
                f"(desviación {std:.2f}, color dominante {dominant:.3f})."
            )

    if len(shapes) > 1:
        errors.append(
            "Las imágenes preparadas no tienen dimensiones uniformes."
        )

    distinct_files = len(set(file_hashes))
    distinct_visual = len(set(visual_hashes))
    distinct_share = (
        distinct_visual / len(visual_hashes) if visual_hashes else 0.0
    )

    if len(file_hashes) > 1 and distinct_files == 1:
        errors.append(
            "Todas las imágenes preparadas son idénticas (mismo hash de "
            "archivo)."
        )
    elif visual_hashes and (
        distinct_share < STAGING_QUALITY_MIN_DISTINCT_SHARE
    ):
        errors.append(
            f"Solo {distinct_visual} de {len(visual_hashes)} imágenes "
            f"preparadas son visualmente distintas (mínimo "
            f"{STAGING_QUALITY_MIN_DISTINCT_SHARE:.0%})."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "details": {
            "images": len(staged),
            "unique_colors_sampled_min": unique_min,
            "std_min": std_min,
            "dominant_share_max": dominant_max,
            "distinct_files": distinct_files,
            "distinct_visual": distinct_visual,
            "distinct_share": round(distinct_share, 4),
            "shapes": {str(key): value for key, value in shapes.items()},
        },
    }


def create_contact_sheet(
    image_paths,
    out_path,
    *,
    columns: int = 4,
    cell_width: int = 320,
    cell_height: int = 240,
    label_height: int = 22,
) -> Path:
    """Hoja de contactos con las imágenes preparadas (revisión humana)."""
    import cv2
    import numpy as np

    paths = [Path(item) for item in image_paths]

    if not paths:
        raise AIDomainError("No hay imágenes para la hoja de contactos.")

    columns = max(1, int(columns))
    rows = (len(paths) + columns - 1) // columns
    row_height = cell_height + label_height

    sheet = np.full(
        (rows * row_height, columns * cell_width, 3),
        32,
        dtype=np.uint8,
    )

    for index, path in enumerate(paths):
        image = cv2.imread(str(path))

        if image is None:
            continue

        height, width = image.shape[:2]
        scale = min(cell_width / float(width), cell_height / float(height))
        thumb = cv2.resize(
            image,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        x0 = (index % columns) * cell_width
        y0 = (index // columns) * row_height
        offset_x = x0 + (cell_width - thumb.shape[1]) // 2
        offset_y = y0 + (cell_height - thumb.shape[0]) // 2

        sheet[
            offset_y:offset_y + thumb.shape[0],
            offset_x:offset_x + thumb.shape[1],
        ] = thumb

        cv2.putText(
            sheet,
            f"{index + 1}. {path.name}",
            (x0 + 4, y0 + cell_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(target), sheet):
        raise AIDomainError(
            "No se pudo escribir la hoja de contactos."
        )

    return target


def _clean_staging_dir(staging_dir) -> int:
    """Elimina archivos previos del staging (nada fuera de esa carpeta)."""
    staging = Path(staging_dir)
    removed = 0

    if not staging.is_dir():
        return 0

    for path in staging.iterdir():
        if not path.is_file():
            continue

        try:
            path.unlink()
            removed += 1
        except OSError:
            # Si un archivo quedó bloqueado, imwrite lo reescribe igual.
            continue

    return removed


def prepare_training_staging(
    cur,
    dataset_id,
    staging_dir,
    roi_fractions,
    *,
    expected_count=None,
    contact_sheet_path=None,
) -> dict:
    """Prepara el staging y lo somete al quality gate.

    Se ejecuta ANTES de crear el job TRAINING: si el conjunto no
    supera el gate se lanza STAGING_QUALITY_GATE_MESSAGE y no existe
    job que pueda producir un checkpoint.
    """
    staging = Path(staging_dir)
    _clean_staging_dir(staging)

    try:
        staged = stage_training_images(
            cur,
            int(dataset_id),
            staging,
            roi_fractions,
        )
    except AIDomainError as error:
        print(f"[STAGING] No se pudo preparar el conjunto: {error}")
        raise AIDomainError(STAGING_QUALITY_GATE_MESSAGE) from error

    gate = quality_gate_staged_images(
        staged,
        expected_count=expected_count,
    )

    if contact_sheet_path is not None:
        try:
            sheet = create_contact_sheet(
                [item["path"] for item in staged],
                contact_sheet_path,
            )
            gate["details"]["contact_sheet"] = str(sheet)
        except Exception as error:  # noqa: BLE001 - el gate manda
            gate["details"]["contact_sheet_error"] = str(error)

    if not gate["ok"]:
        print(f"[STAGING] Quality gate: {'; '.join(gate['errors'])}")
        raise AIDomainError(STAGING_QUALITY_GATE_MESSAGE)

    return {"staged": staged, "gate": gate}


def _library_versions_cached():
    return library_versions()


class PatchCoreTrainer:
    """Entrenamiento PatchCore real (anomalib). Import perezoso."""

    def train(self, ctx) -> dict:
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
        from lightning.pytorch import Callback

        cfg = ctx["config"]
        report = ctx["report"]
        image_size = (int(cfg["input_size"]), int(cfg["input_size"]))

        class _Progress(Callback):
            def __init__(self):
                self.total = 1
                self.done = 0

            def on_train_start(self, _trainer, _pl_module):
                report(
                    35,
                    "EMBEDDINGS",
                    "Generando representaciones",
                )

            def on_train_batch_end(
                self, trainer, _pl_module, _outputs, _batch, _batch_idx
            ):
                self.done += 1
                estimated = getattr(
                    trainer, "estimated_stepping_batches", None
                )
                self.total = max(1, int(estimated or self.total))
                fraction = min(1.0, self.done / float(self.total))
                report(
                    35 + (70 - 35) * fraction,
                    "EMBEDDINGS",
                    "Generando representaciones",
                )

            def on_train_epoch_end(self, trainer, _pl_module):
                report(
                    70,
                    "MEMORY_BANK",
                    "Construyendo modelo",
                )

        checkpoint_dir = Path(ctx["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        datamodule = Folder(
            name=ctx["dataset_name"],
            root=str(ctx["staging_dir"]),
            normal_dir=str(ctx["staging_dir"]),
            abnormal_dir=None,
            normal_test_dir=None,
            train_batch_size=int(cfg["batch_size"]),
            eval_batch_size=1,
            num_workers=int(cfg["num_workers"]),
            test_split_mode=TestSplitMode.NONE,
            val_split_mode=ValSplitMode.NONE,
            seed=int(cfg["seed"]),
        )

        report(25, "INICIALIZANDO_PATCHCORE", "Inicializando PatchCore")

        model = Patchcore(
            backbone=str(cfg["backbone"]),
            layers=tuple(cfg["layers"]),
            pre_trained=bool(cfg["pre_trained"]),
            coreset_sampling_ratio=float(cfg["coreset_ratio"]),
            num_neighbors=int(cfg["num_neighbors"]),
            pre_processor=Patchcore.configure_pre_processor(
                image_size=image_size
            ),
        )

        engine = Engine(
            accelerator=str(cfg["accelerator"]),
            devices=int(cfg["devices"]),
            default_root_dir=str(checkpoint_dir),
            logger=False,
            deterministic=bool(cfg["deterministic"]),
            limit_val_batches=0,
            callbacks=[_Progress()],
        )

        report(35, "EMBEDDINGS", "Generando representaciones")
        engine.fit(model=model, datamodule=datamodule)

        report(85, "GUARDANDO_MODELO", "Guardando modelo")

        checkpoint = engine.best_model_path

        if not checkpoint:
            candidates = sorted(checkpoint_dir.rglob("*.ckpt"))
            checkpoint = str(candidates[0]) if candidates else ""

        if not checkpoint or not Path(checkpoint).is_file():
            raise RuntimeError(
                "PatchCore no produjo un checkpoint utilizable."
            )

        memory_bank = self._inspect_checkpoint(checkpoint)

        return {
            "checkpoint": str(checkpoint),
            "memory_bank": memory_bank,
            "image_size": list(image_size),
            "accelerator": str(cfg["accelerator"]),
        }

    @staticmethod
    def _inspect_checkpoint(checkpoint: str) -> dict:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict") or {}
        bank = state_dict.get("model.memory_bank")

        if bank is None or int(bank.numel()) == 0:
            raise RuntimeError(
                "El checkpoint no contiene un memory bank entrenado."
            )

        return {
            "shape": list(bank.shape),
            "dtype": str(bank.dtype),
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
        }


def execute_training_job(
    job_id: int,
    *,
    connect=None,
    trainer=None,
    config: dict | None = None,
) -> dict:
    """Ejecuta un job TRAINING ya reclamado (EN_CURSO).

    Devuelve {"ok": bool, ...} y nunca lanza por fallos de entrenamiento:
    el estado de job/modelo queda persistido antes de retornar.
    """
    connect = connect or default_connect
    trainer = trainer or PatchCoreTrainer()
    started = time.perf_counter()

    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            job = _load_job_context(cur, job_id)
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()

    paths = model_artifact_paths(
        job["garment_model_id"],
        job["ai_model_id"],
    )
    _ensure_dirs(paths)

    reporter = _JobReporter(connect, job_id, paths)
    cfg = dict(config or job.get("config") or get_training_config())
    dataset_id = int(job["dataset_id"])
    image_count = int(job["image_count"] or 0)

    reporter.log("=" * 70)
    reporter.log(f"INTENTO job_id={job_id}")
    reporter.log(
        f"ai_model_id={job['ai_model_id']} "
        f"garment_model_id={job['garment_model_id']} "
        f"dataset_id={dataset_id}"
    )
    reporter.log(f"image_count={image_count}")
    reporter.log(f"inicio={datetime.now().isoformat(timespec='seconds')}")
    reporter.log(f"configuracion_efectiva={json.dumps(cfg, sort_keys=True)}")

    artifacts = []
    result = None

    try:
        _transition_model(
            connect,
            job["ai_model_id"],
            AI_MODEL_STATUS_ENTRENANDO,
            f"Entrenamiento job {job_id}",
        )
        _record_training_started(connect, job)

        reporter.report(5, "VALIDANDO_DATASET", "Validando dataset")
        verification = _verify(connect, dataset_id, job["ai_model_id"])

        if not verification["ok"]:
            raise AIDomainError("; ".join(verification["errors"]))

        reporter.log(
            f"dataset_verificado={dataset_id} "
            f"imagenes={verification['details'].get('image_rows')}"
        )

        reporter.report(
            15,
            "PREPARANDO_ARCHIVOS",
            "Preparando archivos",
        )
        staged = _stage(
            connect,
            dataset_id,
            paths["staging_dir"],
            cfg,
        )
        reporter.log(f"imagenes_preparadas={len(staged)}")

        if int(job["image_count"] or 0) != len(staged):
            raise AIDomainError(
                "La cantidad de imágenes preparadas no coincide con el "
                "dataset."
            )

        _write_config(paths, cfg, job, dataset_id)

        context = {
            "config": cfg,
            "job_id": int(job_id),
            "ai_model_id": int(job["ai_model_id"]),
            "garment_model_id": int(job["garment_model_id"]),
            "dataset_id": dataset_id,
            "dataset_name": f"dataset_{dataset_id}",
            "staging_dir": Path(paths["staging_dir"]),
            "checkpoint_dir": Path(paths["checkpoint_dir"]),
            "report": reporter.report,
            "log": reporter.log,
        }

        result = trainer.train(context)

        reporter.report(
            95,
            "VERIFICANDO_ARTEFACTOS",
            "Verificando artefactos",
        )
        artifacts = _collect_artifacts(paths, result)
        reporter.log(f"artefactos={len(artifacts)}")

        for item in artifacts:
            reporter.log(
                f"artefacto {item['relative_path']} "
                f"sha256={item['sha256']} bytes={item['size']}"
            )

        _finish_success(
            connect,
            job=job,
            paths=paths,
            cfg=cfg,
            artifacts=artifacts,
            result=result,
            reporter=reporter,
            dataset_id=dataset_id,
            image_count=len(staged),
            elapsed=time.perf_counter() - started,
        )

        reporter.report(100, "COMPLETADO", "Completado")
        reporter.log("resultado=COMPLETADO")

        if not cfg.get("keep_staging", True):
            _cleanup_staging(paths)

        return {
            "ok": True,
            "job_id": int(job_id),
            "ai_model_id": int(job["ai_model_id"]),
            "dataset_id": dataset_id,
            "artifacts": artifacts,
        }

    except Exception as error:  # noqa: BLE001 - se persiste el fallo
        technical = redact_secrets(
            f"{type(error).__name__}: {error}"
        )
        human = humanize_training_error(error)

        reporter.log(f"resultado=FALLIDO detalle={technical}")
        reporter.log(f"mensaje_usuario={human}")

        try:
            _finish_failure(
                connect,
                job=job,
                human=human,
                technical=technical,
                reporter=reporter,
            )
        except Exception as persist_error:  # pragma: no cover
            reporter.log(
                "no_se_pudo_persistir_el_fallo="
                + redact_secrets(str(persist_error))
            )
            raise

        return {
            "ok": False,
            "job_id": int(job_id),
            "ai_model_id": int(job["ai_model_id"]),
            "error": human,
            "technical": technical,
        }


def _load_job_context(cur, job_id: int) -> dict:
    cur.execute(
        """
        SELECT j.id, j.status, j.ai_model_id, j.dataset_id, j.kind,
               j.config_json, j.worker_id, j.started_at, j.log_path,
               m.garment_model_id, m.version AS model_version,
               m.status AS model_status,
               d.image_count, d.status AS dataset_status,
               d.version AS dataset_version
        FROM ai_jobs j
        JOIN garment_ai_models m ON m.id = j.ai_model_id
        LEFT JOIN ai_datasets d ON d.id = j.dataset_id
        WHERE j.id = %s
        FOR UPDATE
        """,
        (int(job_id),),
    )
    row = cur.fetchone()

    if row is None:
        raise AIDomainError(f"El job {job_id} no existe.")

    if row["kind"] != JOB_KIND_TRAINING:
        raise AIDomainError("Solo se ejecutan jobs de entrenamiento.")

    if row["status"] != JOB_STATUS_EN_CURSO:
        raise AIDomainError(
            f"El job {job_id} no está EN_CURSO (estado {row['status']})."
        )

    context = dict(row)

    raw_config = context.pop("config_json", None)

    if raw_config:
        try:
            context["config"] = json.loads(raw_config)
        except (TypeError, ValueError):
            context["config"] = None
    else:
        context["config"] = None

    if context["config"] is None:
        context["config"] = get_training_config()

    if context["dataset_status"] != DATASET_STATUS_CERRADO:
        raise AIDomainError("El dataset del job no está cerrado.")

    return context


def _ensure_dirs(paths: dict) -> None:
    Path(paths["training_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)


def _transition_model(connect, ai_model_id, new_status, notes=None):
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            transition_ai_model_status(
                cur,
                int(ai_model_id),
                new_status,
                None,
                notes=notes,
            )
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def _record_training_started(connect, job) -> None:
    """Evento de auditoria del inicio real del entrenamiento."""
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            record_ai_event(
                cur,
                "TRAINING_STARTED",
                actor_id=None,
                ai_model_id=int(job["ai_model_id"]),
                dataset_id=(
                    int(job["dataset_id"]) if job.get("dataset_id") else None
                ),
                payload={
                    "job_id": int(job["id"]),
                    "worker_id": job.get("worker_id"),
                    "image_count": int(job.get("image_count") or 0),
                },
            )
            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def _verify(connect, dataset_id, ai_model_id) -> dict:
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            verification = verify_training_dataset(
                cur,
                dataset_id=int(dataset_id),
                ai_model_id=int(ai_model_id),
            )
            conn.commit()
            return verification
        finally:
            cur.close()
    finally:
        conn.close()


def _stage(connect, dataset_id, staging_dir, cfg):
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            # Mismo gate que corre antes de crear el job: si el
            # conjunto quedara inválido aquí, el job termina FALLIDO
            # sin llegar a entrenar ni a escribir un checkpoint.
            staged = prepare_training_staging(
                cur,
                int(dataset_id),
                staging_dir,
                get_roi_fractions(),
                contact_sheet_path=Path(staging_dir).parent
                / "contact_sheet.png",
            )["staged"]
            conn.commit()
            return staged
        finally:
            cur.close()
    finally:
        conn.close()


def _write_config(paths, cfg, job, dataset_id) -> None:
    payload = {
        "job_id": int(job["id"]),
        "ai_model_id": int(job["ai_model_id"]),
        "garment_model_id": int(job["garment_model_id"]),
        "dataset_id": int(dataset_id),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "library_versions": _library_versions_cached(),
    }

    target = Path(paths["config_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _collect_artifacts(paths, result) -> list:
    """Verifica y describe todos los artefactos necesarios para cargar."""
    root = Path(paths["root"])
    candidates = []

    checkpoint = (result or {}).get("checkpoint")

    if checkpoint:
        candidates.append(Path(checkpoint))

    for extra in (result or {}).get("extra_artifacts", []) or []:
        candidates.append(Path(extra))

    if not candidates:
        raise RuntimeError(
            "El entrenamiento no produjo artefactos verificables."
        )

    artifacts = []

    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()

        if not resolved.is_file():
            raise RuntimeError(
                f"No existe el artefacto esperado: {resolved.name}"
            )

        size = resolved.stat().st_size

        if size <= 0:
            raise RuntimeError(
                f"El artefacto {resolved.name} está vacío."
            )

        artifacts.append(
            {
                "relative_path": relative_to_root(resolved, root),
                "sha256": sha256_file(resolved),
                "size": size,
                "modified_at": datetime.fromtimestamp(
                    resolved.stat().st_mtime
                ).isoformat(timespec="seconds"),
            }
        )

    return artifacts


def _write_metadata(paths, payload) -> None:
    target = Path(paths["metadata_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cleanup_staging(paths) -> None:
    import shutil

    staging = Path(paths["staging_dir"])

    if staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)


def _finish_success(
    connect,
    *,
    job,
    paths,
    cfg,
    artifacts,
    result,
    reporter,
    dataset_id,
    image_count,
    elapsed,
):
    primary = artifacts[0]
    finished = datetime.now().isoformat(timespec="seconds")

    metadata = {
        "job_id": int(job["id"]),
        "ai_model_id": int(job["ai_model_id"]),
        "garment_model_id": int(job["garment_model_id"]),
        "dataset_id": int(dataset_id),
        "dataset_version": job.get("dataset_version"),
        "image_count": int(image_count),
        "started_at": str(job.get("started_at") or ""),
        "finished_at": finished,
        "duration_seconds": round(float(elapsed), 2),
        "config": cfg,
        "stages": reporter.stages,
        "artifacts": artifacts,
        "result": "COMPLETADO",
        "library_versions": _library_versions_cached(),
        "checkpoint_detail": (result or {}).get("memory_bank"),
        "image_size": (result or {}).get("image_size"),
        "accelerator": (result or {}).get("accelerator"),
    }
    _write_metadata(paths, metadata)

    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()

            transition_ai_job_status(
                cur,
                int(job["id"]),
                JOB_STATUS_COMPLETADO,
                progress=100,
                stage="COMPLETADO",
                stage_label="Entrenamiento completado",
            )

            cur.execute(
                """
                UPDATE ai_jobs
                SET artifacts_json = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (
                    json.dumps(artifacts, ensure_ascii=False),
                    int(job["id"]),
                ),
            )

            transition_ai_model_status(
                cur,
                int(job["ai_model_id"]),
                AI_MODEL_STATUS_ENTRENADO,
                None,
            )

            cur.execute(
                """
                UPDATE garment_ai_models
                SET checkpoint_path = %s,
                    checkpoint_hash = %s,
                    input_size = %s,
                    dataset_id = %s,
                    normal_images_count = %s,
                    metrics_json = %s
                WHERE id = %s
                """,
                (
                    primary["relative_path"],
                    primary["sha256"],
                    str(cfg.get("input_size")),
                    int(dataset_id),
                    int(image_count),
                    json.dumps(
                        {
                            "job_id": int(job["id"]),
                            "duration_seconds": round(float(elapsed), 2),
                            "memory_bank": (result or {}).get("memory_bank"),
                            "artifacts": len(artifacts),
                        },
                        ensure_ascii=False,
                    ),
                    int(job["ai_model_id"]),
                ),
            )

            record_ai_event(
                cur,
                "TRAINING_COMPLETED",
                actor_id=None,
                ai_model_id=int(job["ai_model_id"]),
                dataset_id=int(dataset_id),
                payload={
                    "job_id": int(job["id"]),
                    "image_count": int(image_count),
                    "checkpoint_path": primary["relative_path"],
                    "checkpoint_hash": primary["sha256"],
                },
            )

            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def _finish_failure(connect, *, job, human, technical, reporter):
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()

            cur.execute(
                "SELECT status FROM ai_jobs WHERE id = %s FOR UPDATE",
                (int(job["id"]),),
            )
            current = cur.fetchone()

            if current and current["status"] == JOB_STATUS_EN_CURSO:
                transition_ai_job_status(
                    cur,
                    int(job["id"]),
                    JOB_STATUS_FALLIDO,
                    error_message=technical[:500],
                    stage="FALLIDO",
                    stage_label="Entrenamiento no completado",
                )

                cur.execute(
                    "SELECT status FROM garment_ai_models WHERE id = %s "
                    "FOR UPDATE",
                    (int(job["ai_model_id"]),),
                )
                model = cur.fetchone()

                if model and model["status"] == AI_MODEL_STATUS_ENTRENANDO:
                    transition_ai_model_status(
                        cur,
                        int(job["ai_model_id"]),
                        AI_MODEL_STATUS_FALLIDO,
                        None,
                        notes=human[:500],
                    )

                record_ai_event(
                    cur,
                    "TRAINING_FAILED",
                    actor_id=None,
                    ai_model_id=int(job["ai_model_id"]),
                    dataset_id=(
                        int(job["dataset_id"]) if job.get("dataset_id") else None
                    ),
                    payload={
                        "job_id": int(job["id"]),
                        "error": human,
                        "detail": technical[:400],
                    },
                )

            conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()
