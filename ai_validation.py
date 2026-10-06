"""FASE 3B — Validación controlada de modelos PatchCore.

Separa formalmente entrenamiento y validación:

- las imágenes de validación viven en AI_ARTIFACTS_ROOT/validations
  y jamás se registran en ai_training_images ni ai_dataset_images;
- la inferencia se ejecuta SIEMPRE sobre los artefactos de la versión
  que se valida: predict(ai_model_id=..., image) resuelve
  ai_model -> artifact path -> config -> checkpoint/memory bank ->
  preprocessing. El checkpoint productivo global (PATCHCORE_CKPT)
  no se lee ni se modifica;
- el score bruto se persiste sin elegir umbral final: FASE 3B deja
  la calibración para cuando existan muestras reales.

Este módulo no importa Flask ni app.py.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ai_domain import (
    AIDomainError,
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_VALIDADO,
    AI_MODEL_STATUS_VALIDACION,
    VALIDATION_CATEGORIES,
    VALIDATION_CATEGORY_LABELS,
    VALIDATION_CATEGORY_NORMAL,
    VALIDATION_CLASSIFICATION_LABELS,
    VALIDATION_BINARY_NOTICE,
    VALIDATION_DEFECT_TYPE_LABELS,
    VALIDATION_DEFECT_TYPES,
    VALIDATION_ESTADO_BUENA,
    VALIDATION_ESTADO_DEFECTUOSA,
    VALIDATION_ESTADO_LABELS,
    VALIDATION_ESTADOS,
    VALIDATION_IMAGE_NOTICE,
    VALIDATION_MODEL_STATUSES,
    VALIDATION_RESULT_PENDING,
    VALIDATION_SESSION_STATUS_ABIERTA,
    VALIDATION_SESSION_STATUS_CERRADA,
    VALIDATION_SESSION_STATUS_EVALUADA,
    atomic_write_bytes,
    defect_type_from_category,
    estado_real_from_category,
    get_ai_artifacts_root,
    is_technically_invalidated,
    normalize_validation_category,
    record_ai_event,
    resolve_under_root,
    resolve_validation_category,
    sha256_bytes,
    transition_ai_model_status,
    validate_validation_case_result,
    validation_case_relative_dir,
    validation_case_relative_path,
    validation_classification,
    sha256_file,
    compute_manifest_hash,
)
from ai_training import default_connect, get_roi_fractions


# ============================================================
# SCORE
# ============================================================

# Estados desde los cuales un caso todavía puede corregirse o
# eliminarse (la validación abierta; nunca una sesión CERRADA).
CASE_EDITABLE_MODEL_STATUSES = (
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_VALIDACION,
)

def normalize_score_percent(raw) -> float | None:
    """Score en escala porcentual 0..100 (misma regla que producción).

    PatchCore devuelve 0..1 en calibraciones y 0..100 en otras; la
    escala mostrada/guardada en validación es siempre porcentual.
    """
    if raw is None:
        return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    if value != value:  # NaN
        return None

    if value <= 1.0:
        value *= 100.0

    return round(max(0.0, min(value, 100.0)), 4)


# ============================================================
# BANCO DE IMÁGENES DE VALIDACIÓN (independiente del training)
# ============================================================

def validation_case_abs_dir(garment_model_id, ai_model_id, category, case_id) -> Path:
    """Directorio absoluto del caso bajo AI_ARTIFACTS_ROOT/validations."""
    relative = validation_case_relative_dir(
        garment_model_id,
        ai_model_id,
        category,
        case_id,
    )
    return resolve_under_root(get_ai_artifacts_root(), relative)


# ============================================================
# CARGA AISLADA DE LA VERSIÓN (predict por ai_model_id)
# ============================================================

_INSPECTORS: dict = {}
_INSPECTORS_LOCK = threading.Lock()

# Lightning/anomalib mantienen estado durante predict(): toda
# inferencia de validación pasa por este candado.
VALIDATION_INFERENCE_LOCK = threading.Lock()


def clear_model_inspectors() -> None:
    """Descarta los inspectores cacheados (uso en tests)."""
    with _INSPECTORS_LOCK:
        _INSPECTORS.clear()


def _training_config_path(garment_model_id, ai_model_id) -> Path:
    root = get_ai_artifacts_root()
    return (
        root
        / f"garment_{int(garment_model_id)}"
        / "models"
        / f"ai_model_{int(ai_model_id)}"
        / "training"
        / "config.json"
    )


def load_model_bundle(cur, ai_model_id) -> dict:
    """Resuelve la versión -> artefactos -> config -> input_size.

    Es el núcleo de "predict(ai_model_id=...)": nunca mira
    PATCHCORE_CKPT ni ningún otro artefacto global.
    """
    try:
        model_id = int(ai_model_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError("ai_model_id inválido.") from error

    cur.execute(
        """
        SELECT id, garment_model_id, version, status, notes,
               checkpoint_path, checkpoint_hash, input_size,
               dataset_id, active, threshold_final, threshold_frozen_at,
               threshold_frozen_by, threshold_provenance
        FROM garment_ai_models
        WHERE id = %s
        """,
        (model_id,),
    )
    row = cur.fetchone()

    if row is None:
        raise AIDomainError("La versión de IA solicitada no existe.")

    if is_technically_invalidated(row.get("notes")):
        raise AIDomainError(
            "Esta versión está marcada NO VALIDADA / NO APTO PARA "
            "ACTIVACIÓN: no puede validarse."
        )

    status = str(row.get("status") or "").strip().upper()

    if status not in VALIDATION_MODEL_STATUSES:
        raise AIDomainError(
            "La validación solo admite versiones ENTRENADO, "
            f"VALIDACION o VALIDADO (estado actual: {status or 'SIN ESTADO'})."
        )

    checkpoint_relative = str(row.get("checkpoint_path") or "").strip()

    if not checkpoint_relative:
        raise AIDomainError(
            "La versión no tiene un checkpoint asociado; no puede validarse."
        )

    checkpoint_abs = resolve_under_root(
        get_ai_artifacts_root(),
        checkpoint_relative,
    )

    if not checkpoint_abs.exists():
        raise AIDomainError(
            "No se encontró el artefacto de la versión solicitada."
        )

    config = {}
    config_path = _training_config_path(
        row["garment_model_id"],
        row["id"],
    )

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}

    raw_size = (
        config.get("input_size")
        or row.get("input_size")
        or 256
    )

    try:
        input_size = int(raw_size)
    except (TypeError, ValueError):
        input_size = 256

    if input_size <= 0:
        input_size = 256

    training_config = config.get("config") if isinstance(config, dict) else None
    preprocessing_profile = (
        training_config.get("validation_preprocessing_profile")
        if isinstance(training_config, dict) else None
    )

    return {
        "ai_model_id": int(row["id"]),
        "garment_model_id": int(row["garment_model_id"]),
        "version": row.get("version"),
        "status": status,
        "dataset_id": row.get("dataset_id"),
        "checkpoint_path": checkpoint_relative,
        "checkpoint_abs": str(checkpoint_abs),
        "checkpoint_hash": row.get("checkpoint_hash"),
        "input_size": input_size,
        "config": config,
        "preprocessing_profile": str(preprocessing_profile or "PER_IMAGE").upper(),
        "active": int(row.get("active") or 0),
        "threshold_final": (
            float(row["threshold_final"])
            if row.get("threshold_final") is not None else None
        ),
        "threshold_frozen_at": row.get("threshold_frozen_at"),
        "threshold_frozen_by": row.get("threshold_frozen_by"),
        "threshold_provenance": row.get("threshold_provenance"),
    }


def get_model_inspector(bundle, inspector_factory=None) -> object:
    """Inspector cacheado por versión (uno por checkpoint/input_size)."""
    key = (
        int(bundle["ai_model_id"]),
        str(bundle["checkpoint_path"]),
        int(bundle["input_size"]),
    )

    with _INSPECTORS_LOCK:
        inspector = _INSPECTORS.get(key)

        if inspector is not None:
            return inspector

        factory = inspector_factory

        if factory is None:
            from patchcore_inference import PatchCoreInspector

            factory = PatchCoreInspector

        inspector = factory(
            bundle["checkpoint_abs"],
            image_size=int(bundle["input_size"]),
        )
        _INSPECTORS[key] = inspector

        return inspector


def _roi_bounds(frame):
    """Mismas fracciones ROI que entrenamiento e inferencia productiva."""
    x1f, y1f, x2f, y2f = get_roi_fractions()
    height, width = frame.shape[:2]

    x1 = int(max(0.0, min(x1f, 0.99)) * width)
    y1 = int(max(0.0, min(y1f, 0.99)) * height)
    x2 = int(max(0.01, min(x2f, 1.0)) * width)
    y2 = int(max(0.01, min(y2f, 1.0)) * height)

    if x2 <= x1 or y2 <= y1:
        raise AIDomainError("La región ROI configurada no es válida.")

    return x1, y1, x2, y2


def build_validation_input(image_path, *, preprocessing_profile="PER_IMAGE") -> Path:
    """Preprocesamiento idéntico a producción (máscara -> ROI -> blanco).

    Devuelve un PNG temporal; el llamador lo elimina.
    """
    try:
        import cv2
        import numpy as np

        import patchcore_preprocess
    except Exception as error:  # pragma: no cover - entorno sin cv2
        raise AIDomainError(
            "No se pudo preparar la imagen de validación."
        ) from error

    image = cv2.imread(str(image_path))

    if image is None:
        raise AIDomainError(
            f"No se pudo abrir la imagen de validación: {image_path}"
        )

    bounds = _roi_bounds(image)

    profile = str(preprocessing_profile or "PER_IMAGE").strip().upper()
    if profile == "FULL_ROI":
        # Política versionada a partir del preprocesamiento que alimentó
        # el checkpoint. Evita que cambios de cobertura de la máscara entre
        # cohortes cambien de forma abrupta el fondo visto por PatchCore.
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        print("[VALIDACION] preprocessing_profile=FULL_ROI; ROI rectangular.")
    else:
        try:
            mask = patchcore_preprocess.create_garment_mask(image, bounds)
        except Exception:
            # Mismo fallback geométrico compartido con training.
            print("[VALIDACION] Sin silueta utilizable: se usará el ROI completo.")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

    _, _, patchcore_input = patchcore_preprocess.build_patchcore_regions(
        image,
        mask,
        bounds,
    )

    handle, temp_name = tempfile.mkstemp(
        prefix="astrid_validation_",
        suffix=".png",
    )
    os.close(handle)
    temp_path = Path(temp_name)

    if not cv2.imwrite(str(temp_path), patchcore_input):
        temp_path.unlink(missing_ok=True)
        raise AIDomainError(
            "No se pudo preparar la imagen de validación."
        )

    return temp_path


def predict_with_ai_model(
    ai_model_id,
    image_path,
    *,
    cur=None,
    inspector_factory=None,
    preprocess=True,
    preprocessing_profile=None,
) -> dict:
    """predict(ai_model_id=vN, image) sobre los artefactos de esa versión.

    No lee PATCHCORE_CKPT ni altera el modelo productivo.
    """
    own_connection = None

    if cur is None:
        own_connection = default_connect()
        cursor = own_connection.cursor(dictionary=True)
    else:
        cursor = cur

    temp_input = None

    try:
        bundle = load_model_bundle(cursor, ai_model_id)
        target = str(image_path)

        if preprocess:
            temp_input = build_validation_input(
                target,
                preprocessing_profile=(
                    preprocessing_profile
                    or bundle.get("preprocessing_profile")
                ),
            )
            target = str(temp_input)

        inspector = get_model_inspector(
            bundle,
            inspector_factory=inspector_factory,
        )

        with VALIDATION_INFERENCE_LOCK:
            prediction = inspector.inspect(target)

        anomaly_map = prediction.get("anomaly_map")
        score_percent = normalize_score_percent(prediction.get("score"))

        return {
            "ai_model_id": bundle["ai_model_id"],
            "garment_model_id": bundle["garment_model_id"],
            "version": bundle["version"],
            "checkpoint_path": bundle["checkpoint_path"],
            "checkpoint_hash": bundle["checkpoint_hash"],
            "input_size": bundle["input_size"],
            "status": bundle["status"],
            "score": prediction.get("score"),
            "score_percent": score_percent,
            "is_anomaly": bool(prediction.get("is_anomaly")),
            "anomaly_map": anomaly_map,
            "source": "ai_model_artifacts",
            "preprocessing_profile": (
                str(preprocessing_profile or bundle.get("preprocessing_profile") or "PER_IMAGE").upper()
            ),
        }
    finally:
        if temp_input is not None:
            try:
                temp_input.unlink(missing_ok=True)
            except OSError:
                pass

        if own_connection is not None:
            try:
                cursor.close()
            finally:
                own_connection.close()


# ============================================================
# HEATMAP / COMPARACIÓN DIAGNÓSTICA
# ============================================================

def _encode_png(image) -> bytes | None:
    try:
        import cv2
    except Exception:  # pragma: no cover
        return None

    if image is None:
        return None

    try:
        ok, buffer = cv2.imencode(".png", image)
    except Exception:  # pragma: no cover
        return None

    if not ok:
        return None

    return buffer.tobytes()


def render_heatmap(anomaly_map, reference_shape=None):
    """Mapa de anomalía coloreado (JET) o None si no es utilizable."""
    if anomaly_map is None:
        return None

    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover
        return None

    values = np.asarray(anomaly_map, dtype=float).squeeze()

    if values.ndim != 2 or values.size == 0:
        return None

    low = float(np.nanmin(values))
    high = float(np.nanmax(values))

    if not (high > low):
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        scaled = (
            (values - low) / (high - low) * 255.0
        ).astype(np.uint8)

    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    if reference_shape is not None:
        target_height, target_width = reference_shape[:2]

        if heatmap.shape[0] != target_height or heatmap.shape[1] != target_width:
            heatmap = cv2.resize(
                heatmap,
                (int(target_width), int(target_height)),
                interpolation=cv2.INTER_LINEAR,
            )

    return heatmap


def render_comparison(original, heatmap):
    """ORIGINAL | HEATMAP lado a lado (diagnóstico visual)."""
    if original is None or heatmap is None:
        return None

    try:
        import cv2
    except Exception:  # pragma: no cover
        return None

    if original.shape[0] != heatmap.shape[0] or original.shape[1] != heatmap.shape[1]:
        heatmap = cv2.resize(
            heatmap,
            (original.shape[1], original.shape[0]),
        )

    return cv2.hconcat([original, heatmap])


def _original_extension(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"

    if image_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"

    return ".png"


# ============================================================
# SESIONES DE VALIDACIÓN
# ============================================================

def _fetch_validation_model(cur, ai_model_id) -> dict:
    cur.execute(
        """
        SELECT id, garment_model_id, version, status, notes, active, dataset_id,
               threshold_final, threshold_frozen_at, threshold_frozen_by,
               threshold_provenance
        FROM garment_ai_models
        WHERE id = %s
        """,
        (int(ai_model_id),),
    )
    row = cur.fetchone()

    if row is None:
        raise AIDomainError("La versión de IA solicitada no existe.")

    return dict(row)


def open_validation_session(cur, ai_model_id) -> dict | None:
    cur.execute(
        """
        SELECT id, garment_model_id, ai_model_id, status,
               threshold_candidate, metrics_json, created_by,
               created_at, evaluated_at, closed_at
        FROM ai_validation_sessions
        WHERE ai_model_id = %s AND status = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(ai_model_id), VALIDATION_SESSION_STATUS_ABIERTA),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def start_validation_session(cur, *, ai_model_id, actor_id=None) -> dict:
    """Abre (o reutiliza) la sesión de validación de una versión.

    Separa formalmente entrenamiento de validación: la versión pasa
    de ENTRENADO a VALIDACION en la misma transacción.
    """
    model = _fetch_validation_model(cur, ai_model_id)

    if is_technically_invalidated(model.get("notes")):
        raise AIDomainError(
            "Esta versión está marcada NO VALIDADA / NO APTO PARA "
            "ACTIVACIÓN: no puede validarse."
        )

    status = str(model["status"] or "").strip().upper()

    if status == AI_MODEL_STATUS_VALIDADO:
        raise AIDomainError(
            "La versión ya está VALIDADA: su sesión de validación "
            "quedó cerrada."
        )

    if status not in VALIDATION_MODEL_STATUSES:
        raise AIDomainError(
            "La validación solo admite versiones entrenadas "
            f"(estado actual: {status or 'SIN ESTADO'})."
        )

    session = open_validation_session(cur, ai_model_id)

    if session is not None:
        return session

    if status == AI_MODEL_STATUS_ENTRENADO:
        transition_ai_model_status(
            cur,
            int(ai_model_id),
            AI_MODEL_STATUS_VALIDACION,
            actor_id,
        )
        record_ai_event(
            cur,
            "VALIDATION_STARTED",
            actor_id=actor_id,
            ai_model_id=int(ai_model_id),
            payload={"version": model.get("version")},
        )

    cur.execute(
        """
        INSERT INTO ai_validation_sessions (
            garment_model_id, ai_model_id, status, created_by
        )
        VALUES (%s, %s, %s, %s)
        """,
        (
            int(model["garment_model_id"]),
            int(ai_model_id),
            VALIDATION_SESSION_STATUS_ABIERTA,
            actor_id,
        ),
    )

    session_id = int(cur.lastrowid)

    cur.execute(
        """
        SELECT id, garment_model_id, ai_model_id, status,
               threshold_candidate, metrics_json, created_by,
               created_at, evaluated_at, closed_at
        FROM ai_validation_sessions
        WHERE id = %s
        """,
        (session_id,),
    )

    return dict(cur.fetchone())


# ============================================================
# CASOS DE VALIDACIÓN
# ============================================================

def training_sha256_match(cur, sha256: str) -> dict | None:
    """True si la imagen ya pertenece al entrenamiento (cualquier sesión)."""
    cur.execute(
        """
        SELECT id, garment_model_id, image_path
        FROM ai_training_images
        WHERE sha256 = %s
        LIMIT 1
        """,
        (str(sha256).lower(),),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _classification_fields(category_key) -> dict:
    """Campos binarios (estado/tipo) de una categoría persistida."""
    return validation_classification(category_key)


def register_validation_case(
    cur,
    *,
    ai_model_id,
    category=None,
    image_bytes,
    observation=None,
    actor_id=None,
    predict_fn=None,
    preprocess=False,
    estado_real=None,
    tipo_defecto=None,
) -> dict:
    """Registra un caso de validación con su score y artefactos.

    - exige imagen nueva (SHA-256 distinto de todo el training);
    - usa los artefactos de la versión (o ``predict_fn`` en tests);
    - guarda el score bruto con resultado pendiente de calibración;
    - acepta el estado binario (BUENA/DEFECTUOSA + tipo de defecto)
      o la categoría histórica (NORMAL/MANCHA/AGUJERO).
    """
    if not image_bytes:
        raise AIDomainError(
            "No se recibió ninguna imagen para validar."
        )

    if not isinstance(image_bytes, (bytes, bytearray)):
        raise AIDomainError("La imagen de validación es inválida.")

    image_bytes = bytes(image_bytes)
    category_key = resolve_validation_category(
        category=category,
        estado_real=estado_real,
        tipo_defecto=tipo_defecto,
    )

    bundle = load_model_bundle(cur, ai_model_id)
    sha256 = sha256_bytes(image_bytes)

    match = training_sha256_match(cur, sha256)

    if match is not None:
        raise AIDomainError(
            "Esta imagen ya forma parte del dataset de entrenamiento "
            f"(SHA-256 {sha256[:12]}…). Las imágenes de validación "
            "deben ser nuevas."
        )

    session = start_validation_session(
        cur,
        ai_model_id=ai_model_id,
        actor_id=actor_id,
    )

    handle, temp_name = tempfile.mkstemp(
        prefix="astrid_validation_case_",
        suffix=_original_extension(image_bytes),
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(image_bytes)

        if predict_fn is not None:
            raw = predict_fn(str(temp_path))
            score_percent = normalize_score_percent(
                raw.get("score_percent", raw.get("score"))
                if isinstance(raw, dict)
                else raw
            )
            is_anomaly = bool(
                raw.get("is_anomaly")
                if isinstance(raw, dict)
                else False
            )
            anomaly_map = (
                raw.get("anomaly_map") if isinstance(raw, dict) else None
            )
            inference_source = (
                raw.get("source", "predict_fn")
                if isinstance(raw, dict)
                else "predict_fn"
            )
        else:
            prediction = predict_with_ai_model(
                bundle["ai_model_id"],
                str(temp_path),
                cur=cur,
                preprocess=preprocess,
            )
            score_percent = prediction["score_percent"]
            is_anomaly = bool(prediction["is_anomaly"])
            anomaly_map = prediction["anomaly_map"]
            inference_source = prediction["source"]
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    threshold_final = bundle.get("threshold_final")
    if threshold_final is not None and score_percent is not None:
        is_anomaly = float(score_percent) >= float(threshold_final)
    prediction_label = "ANOMALIA" if is_anomaly else "NORMAL"
    cohort = "POST_FREEZE_FINAL_TEST" if threshold_final is not None else None
    result = validate_validation_case_result(
        category_key,
        score_percent,
        threshold_final,
    ) or VALIDATION_RESULT_PENDING

    cur.execute(
        """
        INSERT INTO ai_validation_cases (
            validation_session_id,
            ai_model_id,
            garment_model_id,
            category,
            image_path,
            image_sha256,
            anomaly_score,
            threshold_used,
            prediction,
            result,
            validation_cohort,
            observation,
            created_by
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            int(session["id"]),
            int(bundle["ai_model_id"]),
            int(bundle["garment_model_id"]),
            category_key,
            "PENDIENTE",
            sha256,
            score_percent,
            threshold_final,
            prediction_label,
            result,
            cohort,
            str(observation).strip() if observation else None,
            actor_id,
        ),
    )

    case_id = int(cur.lastrowid)
    case_dir = validation_case_relative_dir(
        bundle["garment_model_id"],
        bundle["ai_model_id"],
        category_key,
        case_id,
    )

    original_relative = validation_case_relative_path(
        bundle["garment_model_id"],
        bundle["ai_model_id"],
        category_key,
        case_id,
        f"original{_original_extension(image_bytes)}",
    )

    atomic_write_bytes(
        resolve_under_root(get_ai_artifacts_root(), original_relative),
        image_bytes,
    )

    heatmap_relative = None
    comparison_relative = None

    if anomaly_map is not None:
        try:
            import cv2
            import numpy as np

            original_bgr = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            heatmap_bgr = render_heatmap(
                anomaly_map,
                reference_shape=(
                    original_bgr.shape if original_bgr is not None else None
                ),
            )
            heatmap_png = _encode_png(heatmap_bgr)

            if heatmap_png:
                heatmap_relative = validation_case_relative_path(
                    bundle["garment_model_id"],
                    bundle["ai_model_id"],
                    category_key,
                    case_id,
                    "heatmap.png",
                )
                atomic_write_bytes(
                    resolve_under_root(
                        get_ai_artifacts_root(),
                        heatmap_relative,
                    ),
                    heatmap_png,
                )

            if original_bgr is not None and heatmap_bgr is not None:
                comparison_png = _encode_png(
                    render_comparison(original_bgr, heatmap_bgr)
                )

                if comparison_png:
                    comparison_relative = validation_case_relative_path(
                        bundle["garment_model_id"],
                        bundle["ai_model_id"],
                        category_key,
                        case_id,
                        "comparison.png",
                    )
                    atomic_write_bytes(
                        resolve_under_root(
                            get_ai_artifacts_root(),
                            comparison_relative,
                        ),
                        comparison_png,
                    )
        except Exception:
            # El heatmap es diagnóstico: su ausencia no anula el caso.
            print("[VALIDACION] No se pudieron generar el heatmap/diagnóstico.")
            heatmap_relative = None
            comparison_relative = None

    cur.execute(
        """
        UPDATE ai_validation_cases
        SET image_path = %s,
            heatmap_path = %s,
            comparison_path = %s
        WHERE id = %s
        """,
        (
            original_relative,
            heatmap_relative,
            comparison_relative,
            case_id,
        ),
    )

    record_ai_event(
        cur,
        "VALIDATION_CASE_REGISTERED",
        actor_id=actor_id,
        ai_model_id=int(bundle["ai_model_id"]),
        payload={
            "validation_session_id": int(session["id"]),
            "validation_case_id": case_id,
            "category": category_key,
            "estado_real": estado_real_from_category(category_key),
            "tipo_defecto": defect_type_from_category(category_key),
            "anomaly_score": score_percent,
            "prediction": prediction_label,
            "result": result,
            "threshold_used": threshold_final,
            "validation_cohort": cohort,
            "image_sha256": sha256,
            "source": inference_source,
            "heatmap": bool(heatmap_relative),
        },
    )

    return {
        "id": case_id,
        "validation_session_id": int(session["id"]),
        "ai_model_id": int(bundle["ai_model_id"]),
        "garment_model_id": int(bundle["garment_model_id"]),
        "version": bundle["version"],
        "category": category_key,
        "category_label": VALIDATION_CATEGORY_LABELS[category_key],
        **_classification_fields(category_key),
        "image_path": original_relative,
        "image_sha256": sha256,
        "anomaly_score": score_percent,
        "threshold_used": threshold_final,
        "validation_cohort": cohort,
        "prediction": prediction_label,
        "result": result,
        "result_label": (
            "Pendiente de calibración"
            if result == VALIDATION_RESULT_PENDING else result
        ),
        "observation": str(observation).strip() if observation else None,
        "heatmap_path": heatmap_relative,
        "comparison_path": comparison_relative,
        "case_dir": case_dir,
        "notice": VALIDATION_IMAGE_NOTICE,
        "source": inference_source,
    }


def list_validation_cases(cur, ai_model_id) -> list[dict]:
    cur.execute(
        """
        SELECT c.id, c.validation_session_id, c.ai_model_id,
               c.garment_model_id, c.category, c.image_path,
               c.image_sha256, c.anomaly_score, c.threshold_used,
               c.prediction, c.result, c.validation_cohort, c.observation,
               c.heatmap_path, c.comparison_path, c.created_at,
               c.created_by
        FROM ai_validation_cases c
        JOIN ai_validation_sessions s ON s.id = c.validation_session_id
        WHERE c.ai_model_id = %s
        ORDER BY c.id ASC
        """,
        (int(ai_model_id),),
    )
    return [dict(row) for row in cur.fetchall()]


def count_cases_by_category(cases) -> dict:
    counts = {key: 0 for key in VALIDATION_CATEGORIES}

    for case in cases or []:
        key = str(case.get("category") or "").strip().upper()
        if key in counts:
            counts[key] += 1

    return counts


def count_cases_by_estado(cases) -> dict:
    """Conteo binario BUENA/DEFECTUOSA derivado de ``category``."""
    counts = {key: 0 for key in VALIDATION_ESTADOS}

    for case in cases or []:
        key = str(case.get("category") or "").strip().upper()
        if key not in VALIDATION_CATEGORIES:
            continue

        counts[estado_real_from_category(key)] += 1

    return counts


def count_defect_cases(cases) -> dict:
    """Desglose secundario: cuántos Mancha y cuántos Agujero."""
    counts = {key: 0 for key in VALIDATION_DEFECT_TYPES}

    for case in cases or []:
        key = str(case.get("category") or "").strip().upper()
        if key not in VALIDATION_CATEGORIES:
            continue

        tipo = defect_type_from_category(key)
        if tipo in counts:
            counts[tipo] += 1

    return counts


def export_validation_bank(cur, *, ai_model_id, artifacts_root=None) -> dict:
    """Exporta todos los casos vigentes de una versión a un banco derivado.

    La BD sigue siendo la fuente de verdad. Se verifica cada fuente y copia,
    y cualquier colisión exacta con training impide completar la exportación.
    No ejecuta inferencia ni modifica sesiones, scores o umbrales.
    """
    cur.execute(
        """SELECT id, garment_model_id, version, dataset_id, active, status
           FROM garment_ai_models WHERE id = %s""",
        (int(ai_model_id),),
    )
    model = cur.fetchone()
    if not model:
        raise AIDomainError("La versión de IA solicitada no existe.")
    model = dict(model)
    root = Path(artifacts_root) if artifacts_root is not None else get_ai_artifacts_root()
    root = root.expanduser().resolve()
    cur.execute(
        """SELECT c.id, c.validation_session_id, c.ai_model_id,
                  c.garment_model_id, c.category, c.image_path,
                  c.image_sha256, c.anomaly_score, c.heatmap_path,
                  c.comparison_path, c.created_at
           FROM ai_validation_cases c
           JOIN ai_validation_sessions s ON s.id = c.validation_session_id
           WHERE c.ai_model_id = %s ORDER BY c.id""",
        (int(ai_model_id),),
    )
    cases = [dict(row) for row in cur.fetchall()]
    categories = {"NORMAL": "buenas", "MANCHA": "manchas", "AGUJERO": "agujeros"}
    counts = {name: 0 for name in categories.values()}
    seen_case_ids = set()
    by_hash = defaultdict(list)
    resolved = []
    for case in cases:
        case_id = int(case["id"])
        if case_id in seen_case_ids:
            raise AIDomainError(f"case_id duplicado en BD: {case_id}.")
        seen_case_ids.add(case_id)
        category = str(case.get("category") or "").strip().upper()
        if category not in categories:
            raise AIDomainError(f"Clasificación inválida en case_id={case_id}.")
        case_dir = categories[category]
        counts[case_dir] += 1
        source = resolve_under_root(root, case["image_path"])
        if not source.is_file():
            raise AIDomainError(f"No existe el original del case_id={case_id}.")
        digest = sha256_file(source)
        if digest.lower() != str(case.get("image_sha256") or "").lower():
            raise AIDomainError(f"SHA-256 del original no coincide, case_id={case_id}.")
        by_hash[digest].append(case_id)
        case["_source"] = source
        case["_digest"] = digest
        case["_category_dir"] = case_dir
        resolved.append(case)
    duplicates = {digest: ids for digest, ids in by_hash.items() if len(ids) > 1}
    if duplicates:
        detail = ", ".join(f"{digest}: case_ids={ids}" for digest, ids in duplicates.items())
        raise AIDomainError("Hashes duplicados dentro de VALIDATION: " + detail)

    dataset_id = model.get("dataset_id")
    if dataset_id is None:
        raise AIDomainError("La versión no referencia su dataset de training.")
    cur.execute(
        """SELECT ti.id, ti.sha256, ti.image_path
           FROM ai_dataset_images di JOIN ai_training_images ti ON ti.id = di.image_id
           WHERE di.dataset_id = %s ORDER BY ti.id""",
        (int(dataset_id),),
    )
    training_rows = [dict(row) for row in cur.fetchall()]
    training_hashes = {str(row["sha256"]).lower() for row in training_rows}
    contaminated = [(c["id"], c["_digest"]) for c in resolved if c["_digest"].lower() in training_hashes]
    if contaminated:
        raise AIDomainError(f"Contaminación TRAIN/VALIDATION: {contaminated}.")
    # Also validate every physical training image and the dataset manifest itself.
    for row in training_rows:
        training_path = resolve_under_root(root, row["image_path"])
        if not training_path.is_file() or sha256_file(training_path).lower() != str(row["sha256"]).lower():
            raise AIDomainError(f"Training source/hash inválido, image_id={row['id']}.")
    cur.execute("SELECT manifest_path, manifest_hash, image_count FROM ai_datasets WHERE id = %s", (int(dataset_id),))
    dataset = cur.fetchone()
    if not dataset or int(dataset["image_count"] or 0) != len(training_rows):
        raise AIDomainError("Conteo del dataset de training no coincide.")
    manifest_source = resolve_under_root(root, dataset["manifest_path"])
    if not manifest_source.is_file():
        raise AIDomainError("Manifest del dataset de training inexistente.")
    try:
        training_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AIDomainError("Manifest del dataset de training ilegible.") from error
    content_hash = compute_manifest_hash([
            {"id": item.get("image_id"), "sha256": item.get("sha256")}
            for item in training_manifest.get("images", [])
        ])
    if training_manifest.get("content_hash") is not None:
        manifest_valid = (
            hashlib.sha256(manifest_source.read_bytes()).hexdigest()
            == str(dataset["manifest_hash"]).lower()
            and content_hash == str(training_manifest.get("content_hash")).lower()
        )
    else:
        manifest_valid = (
            str(training_manifest.get("manifest_hash") or "").lower()
            == str(dataset["manifest_hash"]).lower()
            and content_hash == str(dataset["manifest_hash"]).lower()
        )
    if int(training_manifest.get("image_count") or 0) != len(training_rows) or not manifest_valid:
        raise AIDomainError("El contenido del manifest de training no coincide con su snapshot registrado.")

    bank_rel = f"garment_{int(model['garment_model_id'])}/validation_datasets/ai_model_{int(model['id'])}_{str(model['version']).lower()}"
    bank = resolve_under_root(root, bank_rel)
    for folder in (*counts.keys(), "artifacts/heatmaps", "artifacts/comparisons"):
        (bank / folder).mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    desired_files = set()
    for case in resolved:
        case_id = int(case["id"])
        digest = case["_digest"]
        suffix = case["_source"].suffix.lower() or ".img"
        filename = f"case_{case_id}_{digest[:12]}{suffix}"
        original_rel = f"{case['_category_dir']}/{filename}"
        destination = resolve_under_root(bank, original_rel)
        payload = case["_source"].read_bytes()
        if destination.exists():
            if sha256_file(destination) != digest:
                raise AIDomainError(f"Destino existente con contenido distinto: {original_rel}.")
        else:
            from ai_domain import atomic_write_bytes
            atomic_write_bytes(destination, payload)
        if sha256_file(destination) != digest:
            raise AIDomainError(f"La copia no verifica SHA-256: {original_rel}.")
        desired_files.add(original_rel)
        output = {
            "case_id": case_id,
            "validation_session_id": int(case["validation_session_id"]),
            "ai_model_id": int(case["ai_model_id"]),
            "estado_real": estado_real_from_category(case["category"]),
            "tipo_defecto": defect_type_from_category(case["category"]),
            "classification": validation_classification(case["category"])["classification"],
            "category_current": str(case["category"]).upper(),
            "anomaly_score": float(case["anomaly_score"]) if case.get("anomaly_score") is not None else None,
            "captured_at": case["created_at"].isoformat(sep=" ") if hasattr(case["created_at"], "isoformat") else str(case.get("created_at")),
            "original_relative_path": original_rel,
            "original_sha256": digest,
        }
        for db_key, artifact_folder, out_path, out_hash in (
            ("heatmap_path", "artifacts/heatmaps", "heatmap_relative_path", "heatmap_sha256"),
            ("comparison_path", "artifacts/comparisons", "comparison_relative_path", "comparison_sha256"),
        ):
            rel = case.get(db_key)
            if rel:
                artifact = resolve_under_root(root, rel)
                if artifact.is_file():
                    artifact_hash = sha256_file(artifact)
                    artifact_dest_rel = f"{artifact_folder}/case_{case_id}_{artifact_hash[:12]}{artifact.suffix.lower()}"
                    artifact_dest = resolve_under_root(bank, artifact_dest_rel)
                    if artifact_dest.exists() and sha256_file(artifact_dest) != artifact_hash:
                        raise AIDomainError(f"Artefacto destino distinto: {artifact_dest_rel}.")
                    if not artifact_dest.exists():
                        from ai_domain import atomic_write_bytes
                        atomic_write_bytes(artifact_dest, artifact.read_bytes())
                    if sha256_file(artifact_dest) != artifact_hash:
                        raise AIDomainError(f"No verifica el artefacto: {artifact_dest_rel}.")
                    output[out_path] = artifact_dest_rel
                    output[out_hash] = artifact_hash
                    desired_files.add(artifact_dest_rel)
        manifest_cases.append(output)

    for category in counts:
        if sum(1 for item in manifest_cases if item["original_relative_path"].startswith(category + "/")) != counts[category]:
            raise AIDomainError("Conteos del banco no coinciden con casos exportados.")
    # Elimina únicamente copias derivadas no referenciadas en este banco (nunca originales).
    for candidate in bank.rglob("*"):
        if candidate.is_file() and candidate.name != "manifest.json":
            relative = candidate.relative_to(bank).as_posix()
            if relative not in desired_files:
                candidate.unlink()
    sessions = sorted({int(case["validation_session_id"]) for case in resolved})
    logical_manifest = {
        "dataset_type": "VALIDATION",
        "garment_model_id": int(model["garment_model_id"]),
        "ai_model_id": int(model["id"]),
        "ai_model_version": model["version"],
        "created_at": min((c["created_at"].isoformat(sep=" ") for c in resolved if hasattr(c["created_at"], "isoformat")), default=datetime.utcnow().isoformat(sep=" ")),
        "source_validation_sessions": sessions,
        "training_dataset_reference": {"dataset_id": int(dataset_id), "manifest_path": dataset["manifest_path"], "manifest_sha256": dataset["manifest_hash"], "image_count": len(training_rows)},
        "counts": {**counts, "total": len(manifest_cases)},
        "validation_unique_hashes": len(by_hash),
        "validation_duplicate_hashes": len(duplicates),
        "cases": manifest_cases,
    }
    snapshot_payload = json.dumps(logical_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot_hash = hashlib.sha256(snapshot_payload).hexdigest()
    previous_exported_at = None
    existing_manifest = bank / "manifest.json"
    if existing_manifest.is_file():
        try:
            old_manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if old_manifest.get("snapshot_sha256") == snapshot_hash:
                previous_exported_at = old_manifest.get("exported_at")
        except (OSError, ValueError):
            pass
    manifest = {
        **logical_manifest,
        "exported_at": previous_exported_at or datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "snapshot_sha256": snapshot_hash,
    }
    from ai_domain import atomic_write_bytes
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(bank / "manifest.json", manifest_bytes)
    return {"path": str(bank), "manifest_path": str(bank / "manifest.json"), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "counts": manifest["counts"], "source_validation_sessions": sessions, "training_images": len(training_rows), "validation_unique_hashes": len(by_hash), "validation_duplicate_hashes": len(duplicates), "cases": len(manifest_cases)}


def verify_validation_bank_snapshot(cur, *, ai_model_id, cases=None, session=None, metrics=None) -> dict:
    """Read-only integrity check for the evaluated calibration bank."""
    model = _fetch_validation_model(cur, ai_model_id)
    rows = [dict(row) for row in (cases if cases is not None else list_validation_cases(cur, ai_model_id))]
    evaluated_session = dict(session or {})
    evaluated_metrics = dict(metrics or {})
    if not rows:
        raise AIDomainError("El banco de validación no contiene casos.")
    if any(row.get("anomaly_score") is None for row in rows):
        raise AIDomainError("El banco de validación contiene casos sin score.")
    hashes = [str(row.get("image_sha256") or "").lower() for row in rows]
    if any(len(digest) != 64 for digest in hashes) or len(set(hashes)) != len(hashes):
        raise AIDomainError("El banco de validación tiene hashes vacíos o duplicados.")

    candidate = evaluated_session.get("threshold_candidate")
    if candidate is None or evaluated_metrics.get("best_threshold") is None:
        raise AIDomainError("La sesión EVALUADA no tiene threshold candidato verificable.")
    if round(float(candidate), 2) != round(float(evaluated_metrics["best_threshold"]), 2):
        raise AIDomainError("El candidato guardado no coincide con las métricas EVALUADAS.")
    candidate_metrics = evaluated_metrics.get("threshold_candidate_metrics") or {}
    if round(float(candidate_metrics.get("threshold") or -1), 2) != round(float(candidate), 2):
        raise AIDomainError("Las métricas del candidato no corresponden al threshold de la sesión.")
    expected_count = int(evaluated_metrics.get("calibration_case_count") or evaluated_metrics.get("total") or -1)
    if expected_count != len(rows) or int(evaluated_metrics.get("scored") or -1) != len(rows):
        raise AIDomainError("El banco actual no coincide con el conjunto EVALUADO.")

    dataset_id = model.get("dataset_id")
    if dataset_id is None:
        raise AIDomainError("La versión no tiene dataset de training para validar la separación del banco.")
    cur.execute("SELECT sha256 FROM ai_dataset_images di JOIN ai_training_images ti ON ti.id=di.image_id WHERE di.dataset_id=%s", (int(dataset_id),))
    training_hashes = {str(row["sha256"]).lower() for row in cur.fetchall()}
    if training_hashes.intersection(hashes):
        raise AIDomainError("El banco de validación comparte hashes con training.")

    root = get_ai_artifacts_root().expanduser().resolve()
    for row in rows:
        source = resolve_under_root(root, row["image_path"])
        if not source.is_file() or sha256_file(source).lower() != str(row["image_sha256"]).lower():
            raise AIDomainError(f"El original de validación del caso {row['id']} falta o no verifica su hash.")

    bank_rel = f"garment_{int(model['garment_model_id'])}/validation_datasets/ai_model_{int(model['id'])}_{str(model['version']).lower()}"
    bank = resolve_under_root(root, bank_rel)
    manifest_path = bank / "manifest.json"
    snapshot_hash = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AIDomainError("El manifest del banco de validación no se puede leer.") from error
        manifest_cases = manifest.get("cases") or []
        by_id = {int(item.get("case_id") or 0): item for item in manifest_cases}
        if (
            manifest.get("dataset_type") != "VALIDATION"
            or int(manifest.get("ai_model_id") or 0) != int(ai_model_id)
            or len(by_id) != len(rows)
            or set(by_id) != {int(row["id"]) for row in rows}
            or int((manifest.get("counts") or {}).get("total") or 0) != len(rows)
        ):
            raise AIDomainError("El manifest exportado no coincide con los casos actuales del banco.")
        for row in rows:
            item = by_id[int(row["id"])]
            if (
                str(item.get("original_sha256") or "").lower() != str(row["image_sha256"]).lower()
                or str(item.get("category_current") or "").upper() != str(row.get("category") or "").upper()
                or round(float(item.get("anomaly_score") or -1), 4) != round(float(row.get("anomaly_score") or -1), 4)
            ):
                raise AIDomainError("El manifest del banco contiene etiqueta, score o hash distinto al caso vigente.")
            copied = resolve_under_root(bank, item.get("original_relative_path") or "")
            if not copied.is_file() or sha256_file(copied).lower() != str(row["image_sha256"]).lower():
                raise AIDomainError("La copia original del banco no verifica su hash.")
        logical = dict(manifest)
        logical.pop("exported_at", None)
        snapshot_hash = logical.pop("snapshot_sha256", None)
        digest = hashlib.sha256(json.dumps(logical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if snapshot_hash != digest:
            raise AIDomainError("El snapshot del banco no verifica integridad.")

    return {
        "valid": True,
        "case_count": len(rows),
        "threshold_candidate": round(float(candidate), 2),
        "manifest_verified": manifest_path.is_file(),
        "snapshot_sha256": snapshot_hash,
    }


def get_validation_state(cur, ai_model_id) -> dict:
    """Estado completo de la validación para la UI."""
    model = _fetch_validation_model(cur, ai_model_id)
    session = None

    cur.execute(
        """
        SELECT id, status, threshold_candidate, metrics_json,
               created_at, evaluated_at, closed_at
        FROM ai_validation_sessions
        WHERE ai_model_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(ai_model_id),),
    )
    row = cur.fetchone()

    if row:
        session = dict(row)

    cases = list_validation_cases(cur, ai_model_id)
    counts = count_cases_by_category(cases)
    invalidated = is_technically_invalidated(model.get("notes"))
    status = str(model["status"] or "").strip().upper()
    final_test = None
    if (
        int(ai_model_id) == FINAL_TEST_AI_MODEL_ID
        or (
            status == AI_MODEL_STATUS_VALIDADO
            and not int(model.get("active") or 0)
            and model.get("threshold_final") is not None
        )
    ):
        final_test = get_final_test_state(cur, ai_model_id=ai_model_id)

    metrics = None
    if session and session.get("metrics_json"):
        try:
            metrics = json.loads(session["metrics_json"])
        except (TypeError, ValueError):
            metrics = None

    validation_bank = {
        "valid": False,
        "case_count": 0,
        "threshold_candidate": None,
        "manifest_verified": False,
        "error": None,
    }
    if (
        session
        and str(session.get("status") or "").upper() in (
            VALIDATION_SESSION_STATUS_EVALUADA,
            VALIDATION_SESSION_STATUS_CERRADA,
        )
        and session.get("threshold_candidate") is not None
        and metrics is not None
    ):
        try:
            validation_bank.update(verify_validation_bank_snapshot(
                cur,
                ai_model_id=ai_model_id,
                cases=cases,
                session=session,
                metrics=metrics,
            ))
        except AIDomainError as error:
            validation_bank["error"] = str(error)

    # Fuente única de verdad: cada caso serializado con su estado
    # binario derivado (tabla, miniaturas y edición usan esto).
    cases_ui = [_case_ui_payload(case) for case in cases]

    return {
        "ai_model_id": int(model["id"]),
        "garment_model_id": int(model["garment_model_id"]),
        "version": model.get("version"),
        "model_status": status,
        "ui_status": status,
        "technically_invalidated": bool(invalidated),
        "active": int(model.get("active") or 0),
        "session_id": int(session["id"]) if session else None,
        "session_status": session["status"] if session else None,
        "threshold_candidate": (
            float(session["threshold_candidate"])
            if session and session.get("threshold_candidate") is not None
            else None
        ),
        "threshold_final": (
            float(model["threshold_final"])
            if model.get("threshold_final") is not None else None
        ),
        "threshold_frozen_at": model.get("threshold_frozen_at"),
        "threshold_frozen_by": model.get("threshold_frozen_by"),
        "threshold_provenance": model.get("threshold_provenance"),
        "validation_bank": validation_bank,
        "can_freeze_threshold": bool(
            not invalidated
            and status == AI_MODEL_STATUS_VALIDACION
            and not int(model.get("active") or 0)
            and model.get("threshold_final") is None
            and session is not None
            and str(session.get("status") or "").upper() == VALIDATION_SESSION_STATUS_EVALUADA
            and session.get("threshold_candidate") is not None
            and validation_bank["valid"]
        ),
        "counts": counts,
        "counts_estado": count_cases_by_estado(cases),
        "counts_defects": count_defect_cases(cases),
        "total_cases": len(cases),
        "source_session_count": len({int(case["validation_session_id"]) for case in cases}),
        "categories": list(VALIDATION_CATEGORIES),
        "category_labels": dict(VALIDATION_CATEGORY_LABELS),
        "estados": list(VALIDATION_ESTADOS),
        "estado_labels": dict(VALIDATION_ESTADO_LABELS),
        "defect_types": list(VALIDATION_DEFECT_TYPES),
        "defect_type_labels": dict(VALIDATION_DEFECT_TYPE_LABELS),
        "classification_labels": dict(VALIDATION_CLASSIFICATION_LABELS),
        "binary_notice": VALIDATION_BINARY_NOTICE,
        "cases": cases_ui,
        "last_case": cases_ui[-1] if cases_ui else None,
        "metrics": metrics,
        "can_capture": (not invalidated)
        and status in (AI_MODEL_STATUS_ENTRENADO, AI_MODEL_STATUS_VALIDACION),
        "can_evaluate": bool(cases)
        and (not invalidated)
        and status in (AI_MODEL_STATUS_ENTRENADO, AI_MODEL_STATUS_VALIDACION),
        "can_complete": bool(metrics)
        and status == AI_MODEL_STATUS_VALIDACION,
        "can_edit_cases": bool(cases)
        and (not invalidated)
        and status in CASE_EDITABLE_MODEL_STATUSES
        and (
            session is None
            or str(session.get("status") or "").strip().upper()
            != VALIDATION_SESSION_STATUS_CERRADA
        ),
        "notice": VALIDATION_IMAGE_NOTICE,
        "pending_calibration": model.get("threshold_final") is None,
        "final_test": final_test,
    }


# ============================================================
# MÉTRICAS Y CALIBRACIÓN (sin elegir umbral definitivo)
# ============================================================

DEFAULT_THRESHOLD_GRID = tuple(range(5, 100, 5))

# FASE 3C.5 — these values are intentionally not caller-configurable.
FINAL_TEST_GARMENT_MODEL_ID = 1082
FINAL_TEST_AI_MODEL_ID = 1649
FINAL_TEST_THRESHOLD = 47.32
FINAL_TEST_PREPROCESSING = "FULL_ROI"
FINAL_TEST_TARGET_PER_CLASS = 10


def _numeric_score(value):
    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if score != score:
        return None

    return score


def _distribution(values) -> dict | None:
    if not values:
        return None

    ordered = sorted(values)
    total = len(ordered)
    middle = total // 2

    if total % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0

    return {
        "count": total,
        "min": round(min(ordered), 4),
        "max": round(max(ordered), 4),
        "mean": round(sum(ordered) / total, 4),
        "median": round(float(median), 4),
    }


def default_threshold_candidates(cases=None) -> list[float]:
    """Candidatos: rejilla fija + puntos medios entre scores observados."""
    scores = []

    for case in cases or []:
        score = _numeric_score(case.get("anomaly_score"))
        if score is not None:
            scores.append(round(score, 2))

    distinct = sorted(set(scores))
    midpoints = [
        round((a + b) / 2.0, 2)
        for a, b in zip(distinct, distinct[1:])
    ]

    return sorted(set(DEFAULT_THRESHOLD_GRID) | set(midpoints))


def binary_metrics(cases, threshold) -> dict:
    """BUENA (negativa) vs DEFECTUOSA (positiva) para un umbral dado."""
    limit = float(threshold)
    tp = fp = fn = tn = 0
    false_positives = false_negatives = 0
    evaluated = 0

    for case in cases or []:
        score = _numeric_score(case.get("anomaly_score"))

        if score is None:
            continue

        evaluated += 1
        estado = _case_estado(case)
        real_anomaly = estado == VALIDATION_ESTADO_DEFECTUOSA
        predicted_anomaly = score >= limit

        if real_anomaly and predicted_anomaly:
            tp += 1
        elif real_anomaly and not predicted_anomaly:
            fn += 1
            false_negatives += 1
        elif not real_anomaly and predicted_anomaly:
            fp += 1
            false_positives += 1
        else:
            tn += 1

    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    specificity = round(tn / (tn + fp), 4) if (tn + fp) else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    accuracy = (
        round((tp + tn) / evaluated, 4) if evaluated else None
    )

    return {
        "threshold": round(float(threshold), 4),
        "evaluated": evaluated,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_threshold_candidates(cases, candidates) -> list[dict]:
    """Evalúa varios thresholds candidatos sobre los datos registrados."""
    results = [
        binary_metrics(cases, candidate)
        for candidate in (candidates or [])
    ]

    results.sort(
        key=lambda item: (
            item["f1"] if item["f1"] is not None else -1.0,
            -item["threshold"],
        ),
        reverse=True,
    )
    return results


def _case_estado(case) -> str | None:
    """Estado binario de un caso; None si la categoría es desconocida."""
    key = str(case.get("category") or "").strip().upper()

    if key not in VALIDATION_CATEGORIES:
        return None

    return estado_real_from_category(key)


def compute_defect_breakdown(cases, threshold=None) -> dict:
    """Desglose secundario Mancha/Agujero (detalle humano, no IA).

    «Detectada» = score >= umbral de referencia. La decisión del
    modelo sigue siendo binaria (BUENA vs DEFECTUOSA): este desglose
    permite demostrar los dos defectos de la tesis sin afirmar que
    PatchCore clasifica semánticamente el tipo.
    """
    limit = None if threshold is None else float(threshold)

    breakdown = {
        "threshold": (
            round(float(threshold), 4) if threshold is not None else None
        ),
        "labels": dict(VALIDATION_DEFECT_TYPE_LABELS),
    }

    for tipo in VALIDATION_DEFECT_TYPES:
        evaluated = 0
        detected = 0

        for case in cases or []:
            if str(case.get("category") or "").strip().upper() != tipo:
                continue

            score = _numeric_score(case.get("anomaly_score"))

            if score is None:
                continue

            evaluated += 1

            if limit is not None and score >= limit:
                detected += 1

        breakdown[tipo] = {
            "evaluated": evaluated,
            "detected": detected if limit is not None else None,
            "missed": (
                (evaluated - detected) if limit is not None else None
            ),
        }

    return breakdown


def compute_validation_metrics(
    cases,
    *,
    threshold=None,
    candidates=None,
) -> dict:
    """Métricas binarias (BUENA/DEFECTUOSA) + candidatos de umbral."""
    rows = list(cases or [])

    by_category = {}
    scored_by_category = {}
    by_estado = {}

    for key in VALIDATION_CATEGORIES:
        scores = [
            _numeric_score(case.get("anomaly_score"))
            for case in rows
            if str(case.get("category") or "").strip().upper() == key
        ]
        numeric = [score for score in scores if score is not None]

        by_category[key] = {
            "count": len(scores),
            "evaluated": len(numeric),
            "distribution": _distribution(numeric),
        }
        scored_by_category[key] = numeric

    # Desglose binario principal: BUENA (clase negativa) frente a
    # DEFECTUOSA (clase positiva).
    for estado in VALIDATION_ESTADOS:
        scores = [
            _numeric_score(case.get("anomaly_score"))
            for case in rows
            if _case_estado(case) == estado
        ]
        numeric = [score for score in scores if score is not None]

        by_estado[estado] = {
            "count": len(scores),
            "evaluated": len(numeric),
            "distribution": _distribution(numeric),
        }

    all_scores = [
        _numeric_score(case.get("anomaly_score")) for case in rows
    ]
    numeric_scores = [score for score in all_scores if score is not None]
    missing = len(rows) - len(numeric_scores)

    metrics = {
        "total": len(rows),
        "by_category": by_category,
        "by_estado": by_estado,
        "estado_counts": count_cases_by_estado(rows),
        "defect_counts": count_defect_cases(rows),
        "estado_labels": dict(VALIDATION_ESTADO_LABELS),
        "defect_labels": dict(VALIDATION_DEFECT_TYPE_LABELS),
        "classification_labels": dict(VALIDATION_CLASSIFICATION_LABELS),
        "scored": len(numeric_scores),
        "pending_scores": missing,
        "pending_calibration": threshold is None,
        "threshold": (
            round(float(threshold), 4) if threshold is not None else None
        ),
        "threshold_reference": None,
        "threshold_source": None,
        "distribution": _distribution(numeric_scores),
        "confusion": None,
        "false_positives": None,
        "false_negatives": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1": None,
        "accuracy": None,
        "candidates": [],
        "best_threshold": None,
        "threshold_candidate_metrics": None,
        "defects": compute_defect_breakdown(rows, None),
        "status": (
            "PENDIENTE_CALIBRACION"
            if threshold is None
            else "EVALUADO"
        ),
    }

    if threshold is not None and numeric_scores:
        metrics.update(binary_metrics(rows, threshold))

    if candidates:
        evaluated = evaluate_threshold_candidates(rows, candidates)
        metrics["candidates"] = evaluated

        if evaluated and evaluated[0]["f1"] is not None:
            metrics["best_threshold"] = evaluated[0]["threshold"]

    # Umbral de referencia para el desglose y las métricas binarias
    # NUNCA sustituye el umbral aplicado: es sólo el candidato.
    reference = (
        threshold if threshold is not None else metrics["best_threshold"]
    )

    if reference is not None:
        metrics["threshold_reference"] = round(float(reference), 4)
        metrics["threshold_source"] = (
            "applied" if threshold is not None else "candidate"
        )

        if threshold is None:
            metrics["threshold_candidate_metrics"] = binary_metrics(
                rows, reference
            )

    metrics["defects"] = compute_defect_breakdown(rows, reference)

    return metrics


def _final_test_contract(cur, *, ai_model_id, verify_bank=False) -> dict:
    """Resolve the immutable FINAL_TEST contract for one AI version.

    New versions must be VALIDADO, inactive and have a frozen definitive
    threshold. BLUSA-762 v1 keeps its historical 47.32/FULL_ROI contract so
    existing thesis evidence remains readable and reproducible.
    """
    model = _fetch_validation_model(cur, ai_model_id)
    status = str(model.get("status") or "").strip().upper()
    invalidated = is_technically_invalidated(model.get("notes"))
    legacy = (
        int(model.get("id") or 0) == FINAL_TEST_AI_MODEL_ID
        and int(model.get("garment_model_id") or 0) == FINAL_TEST_GARMENT_MODEL_ID
    )

    if legacy and model.get("threshold_final") is None:
        threshold_fixed = FINAL_TEST_THRESHOLD
        preprocessing_profile = FINAL_TEST_PREPROCESSING
        can_start = (
            not int(model.get("active") or 0)
            and not invalidated
            and status in (
                AI_MODEL_STATUS_ENTRENADO,
                AI_MODEL_STATUS_VALIDACION,
                AI_MODEL_STATUS_VALIDADO,
            )
        )
        reason = None if can_start else (
            "BLUSA-762 v1 debe permanecer inactiva y apta para evaluación final."
        )
        return {
            "model": model,
            "legacy": True,
            "threshold_fixed": float(threshold_fixed),
            "preprocessing_profile": preprocessing_profile,
            "can_start": bool(can_start),
            "start_block_reason": reason,
        }

    threshold = model.get("threshold_final")
    threshold_fixed = float(threshold) if threshold is not None else None
    preprocessing_profile = None
    reason = None

    try:
        bundle = load_model_bundle(cur, ai_model_id)
        preprocessing_profile = str(
            bundle.get("preprocessing_profile") or ""
        ).strip().upper() or None
    except AIDomainError as error:
        reason = str(error)

    if reason is None and int(model.get("active") or 0):
        reason = "FINAL_TEST debe ejecutarse antes de activar la versión."
    elif reason is None and invalidated:
        reason = "La versión está invalidada técnicamente y no puede ejecutar FINAL_TEST."
    elif reason is None and status != AI_MODEL_STATUS_VALIDADO:
        reason = "La versión debe estar VALIDADA antes de iniciar FINAL_TEST."
    elif reason is None and threshold_fixed is None:
        reason = "La versión necesita un threshold definitivo antes de FINAL_TEST."
    elif reason is None and model.get("threshold_frozen_at") is None:
        reason = "El threshold definitivo debe estar CONGELADO antes de FINAL_TEST."
    elif reason is None and not preprocessing_profile:
        reason = "El checkpoint no declara un perfil de preprocesamiento verificable."

    if reason is None and verify_bank:
        cur.execute(
            """SELECT id, status, threshold_candidate, metrics_json,
                      created_at, evaluated_at, closed_at
               FROM ai_validation_sessions
               WHERE ai_model_id = %s
               ORDER BY id DESC LIMIT 1""",
            (int(ai_model_id),),
        )
        session = cur.fetchone()
        session = dict(session) if session else None
        if not session or str(session.get("status") or "").upper() != VALIDATION_SESSION_STATUS_CERRADA:
            reason = "La sesión de validación debe estar CERRADA antes de FINAL_TEST."
        elif session.get("threshold_candidate") is None or not session.get("metrics_json"):
            reason = "La validación cerrada no conserva candidato y métricas verificables."
        elif round(float(session["threshold_candidate"]), 2) != round(float(threshold_fixed), 2):
            reason = "El threshold congelado no coincide con el candidato de la validación cerrada."
        else:
            try:
                metrics = json.loads(session.get("metrics_json") or "{}")
                verify_validation_bank_snapshot(
                    cur,
                    ai_model_id=ai_model_id,
                    cases=list_validation_cases(cur, ai_model_id),
                    session=session,
                    metrics=metrics,
                )
            except (AIDomainError, TypeError, ValueError) as error:
                reason = f"El banco de validación no supera la verificación previa a FINAL_TEST: {error}"

    return {
        "model": model,
        "legacy": False,
        "threshold_fixed": threshold_fixed,
        "preprocessing_profile": preprocessing_profile,
        "can_start": reason is None,
        "start_block_reason": reason,
    }


def start_final_test(cur, *, ai_model_id, actor_id=None) -> dict:
    """Open an isolated, initially empty FINAL_TEST cohort for one version."""
    cur.execute(
        "SELECT id FROM ai_final_test_sessions WHERE ai_model_id = %s LIMIT 1",
        (int(ai_model_id),),
    )
    if cur.fetchone():
        return get_final_test_state(cur, ai_model_id=ai_model_id)

    contract = _final_test_contract(cur, ai_model_id=ai_model_id, verify_bank=True)
    if not contract["can_start"]:
        raise AIDomainError(contract["start_block_reason"] or "FINAL_TEST no disponible.")

    model = contract["model"]
    threshold_fixed = float(contract["threshold_fixed"])
    preprocessing_profile = str(contract["preprocessing_profile"])
    garment_model_id = int(model["garment_model_id"])

    cur.execute(
        """INSERT INTO ai_final_test_sessions
           (garment_model_id, ai_model_id, cohort, threshold_fixed,
            preprocessing_profile, status, created_by)
           VALUES (%s, %s, 'FINAL_TEST', %s, %s, 'ABIERTA', %s)""",
        (
            garment_model_id,
            int(ai_model_id),
            threshold_fixed,
            preprocessing_profile,
            actor_id,
        ),
    )
    session_id = int(cur.lastrowid)
    record_ai_event(
        cur,
        "FINAL_TEST_STARTED",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={
            "final_test_session_id": session_id,
            "garment_model_id": garment_model_id,
            "threshold_fixed": threshold_fixed,
            "preprocessing_profile": preprocessing_profile,
            "cohort": "FINAL_TEST",
            "initial_case_count": 0,
            "legacy_contract": bool(contract.get("legacy")),
        },
    )
    return get_final_test_state(cur, ai_model_id=ai_model_id)


def get_final_test_state(cur, *, ai_model_id=FINAL_TEST_AI_MODEL_ID) -> dict:
    cur.execute(
        """SELECT id, garment_model_id, ai_model_id, cohort,
                  threshold_fixed, preprocessing_profile, status,
                  metrics_json, created_by, created_at, evaluated_at
           FROM ai_final_test_sessions WHERE ai_model_id = %s LIMIT 1""",
        (int(ai_model_id),),
    )
    session_row = cur.fetchone()

    # Preserve the historical BLUSA-762 contract even in lightweight unit
    # cursors that do not expose garment_ai_models.
    if not session_row and int(ai_model_id) == FINAL_TEST_AI_MODEL_ID:
        return {
            "cohort": "FINAL_TEST",
            "started": False,
            "status": "INACTIVA",
            "status_label": "INACTIVA",
            "cases": [],
            "counts": {"BUENA": 0, "MANCHA": 0, "total": 0},
            "threshold_fixed": FINAL_TEST_THRESHOLD,
            "preprocessing_profile": FINAL_TEST_PREPROCESSING,
            "metrics": None,
            "ready_to_evaluate": False,
            "can_start": False,
            "start_block_reason": None,
        }

    if int(ai_model_id) == FINAL_TEST_AI_MODEL_ID:
        contract = {
            "model": {
                "id": FINAL_TEST_AI_MODEL_ID,
                "garment_model_id": FINAL_TEST_GARMENT_MODEL_ID,
            },
            "threshold_fixed": FINAL_TEST_THRESHOLD,
            "preprocessing_profile": FINAL_TEST_PREPROCESSING,
            "can_start": False,
            "start_block_reason": None,
        }
    else:
        contract = _final_test_contract(cur, ai_model_id=ai_model_id, verify_bank=False)
    model = contract["model"]
    expected_threshold = contract.get("threshold_fixed")
    expected_profile = contract.get("preprocessing_profile")

    if not session_row:
        return {
            "cohort": "FINAL_TEST",
            "started": False,
            "status": "INACTIVA",
            "status_label": "PENDIENTE DE PRUEBA FINAL" if contract["can_start"] else "INACTIVA",
            "cases": [],
            "counts": {"BUENA": 0, "MANCHA": 0, "total": 0},
            "threshold_fixed": expected_threshold,
            "preprocessing_profile": expected_profile,
            "metrics": None,
            "ready_to_evaluate": False,
            "can_start": bool(contract["can_start"]),
            "start_block_reason": contract.get("start_block_reason"),
            "garment_model_id": int(model["garment_model_id"]),
            "ai_model_id": int(model["id"]),
        }

    session_row = dict(session_row)
    integrity_ok = (
        expected_threshold is not None
        and round(float(session_row.get("threshold_fixed") or 0), 2)
        == round(float(expected_threshold), 2)
        and bool(expected_profile)
        and str(session_row.get("preprocessing_profile") or "").upper()
        == str(expected_profile).upper()
        and str(session_row.get("cohort") or "").upper() == "FINAL_TEST"
        and int(session_row.get("ai_model_id") or 0) == int(ai_model_id)
        and int(session_row.get("garment_model_id") or 0) == int(model["garment_model_id"])
    )
    cur.execute(
        """SELECT id, category, image_path, image_sha256, anomaly_score,
                  threshold_used, prediction, correct, heatmap_path,
                  comparison_path, created_at
           FROM ai_final_test_cases WHERE final_test_session_id = %s
           ORDER BY id ASC""",
        (int(session_row["id"]),),
    )
    cases = [dict(row) for row in cur.fetchall()]
    # MySQL DECIMAL puede llegar como str/Decimal según el cursor. Normalizamos
    # aquí para que la UI y el JSON reciban siempre números reales.
    for case in cases:
        if case.get("anomaly_score") is not None:
            case["anomaly_score"] = float(case["anomaly_score"])
        if case.get("threshold_used") is not None:
            case["threshold_used"] = float(case["threshold_used"])
        case["correct"] = bool(case.get("correct"))
    good = sum(str(row["category"]).upper() == "NORMAL" for row in cases)
    stains = sum(str(row["category"]).upper() == "MANCHA" for row in cases)
    metrics = None
    if session_row.get("metrics_json"):
        try:
            metrics = json.loads(session_row["metrics_json"])
        except (TypeError, ValueError):
            metrics = None
    return {
        **session_row,
        "started": True,
        "integrity_ok": integrity_ok,
        "integrity_error": None if integrity_ok else "Configuración FINAL_TEST alterada; captura y evaluación bloqueadas.",
        "status_label": {
            "ABIERTA": "EN CURSO",
            "EVALUADA": "EVALUADA",
        }.get(str(session_row.get("status") or "").upper(), "INACTIVA"),
        "cases": cases,
        "counts": {"BUENA": good, "MANCHA": stains, "total": len(cases)},
        "threshold_fixed": float(session_row["threshold_fixed"]),
        "preprocessing_profile": session_row["preprocessing_profile"],
        "metrics": metrics,
        "ready_to_evaluate": good == FINAL_TEST_TARGET_PER_CLASS
        and stains == FINAL_TEST_TARGET_PER_CLASS,
        "can_start": False,
        "start_block_reason": None,
    }


def register_final_test_case(
    cur, *, ai_model_id=FINAL_TEST_AI_MODEL_ID, image_bytes, category,
    actor_id=None, inspector_factory=None
) -> dict:
    """Infer and persist one camera-captured item in an isolated FINAL_TEST."""
    category_key = str(category or "").strip().upper()
    if category_key not in ("NORMAL", "MANCHA"):
        raise AIDomainError("La prueba final solo admite BUENA o DEFECTUOSA → MANCHA.")
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise AIDomainError("No se recibió un frame de cámara válido.")

    state = get_final_test_state(cur, ai_model_id=ai_model_id)
    if not state.get("started") or state.get("status") != "ABIERTA":
        raise AIDomainError("Inicie una cohorte FINAL_TEST abierta antes de capturar.")
    if not state.get("integrity_ok"):
        raise AIDomainError(state.get("integrity_error") or "Configuración FINAL_TEST inválida.")
    if state["counts"]["total"] >= 2 * FINAL_TEST_TARGET_PER_CLASS:
        raise AIDomainError("La cohorte FINAL_TEST ya alcanzó 20 imágenes.")
    if state["counts"]["BUENA"] >= FINAL_TEST_TARGET_PER_CLASS and category_key == "NORMAL":
        raise AIDomainError("La cuota de 10 BUENAS de FINAL_TEST ya está completa.")
    if state["counts"]["MANCHA"] >= FINAL_TEST_TARGET_PER_CLASS and category_key == "MANCHA":
        raise AIDomainError("La cuota de 10 MANCHAS de FINAL_TEST ya está completa.")

    digest = sha256_bytes(bytes(image_bytes))
    cur.execute("SELECT id FROM ai_training_images WHERE LOWER(sha256) = %s LIMIT 1", (digest,))
    if cur.fetchone():
        raise AIDomainError("Imagen rechazada: el hash ya existe en TRAINING.")
    cur.execute("SELECT id FROM ai_validation_cases WHERE LOWER(image_sha256) = %s LIMIT 1", (digest,))
    if cur.fetchone():
        raise AIDomainError("Imagen rechazada: el hash ya existe en VALIDATION/calibración.")
    cur.execute("SELECT id FROM ai_final_test_cases WHERE image_sha256 = %s LIMIT 1", (digest,))
    if cur.fetchone():
        raise AIDomainError("Imagen duplicada dentro de FINAL_TEST.")

    threshold_fixed = float(state["threshold_fixed"])
    preprocessing_profile = str(state["preprocessing_profile"])
    garment_model_id = int(state["garment_model_id"])
    session_id = int(state["id"])

    handle, temp_name = tempfile.mkstemp(prefix="astrid_final_test_", suffix=_original_extension(bytes(image_bytes)))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(bytes(image_bytes))
        prediction = predict_with_ai_model(
            int(ai_model_id),
            str(temp_path),
            cur=cur,
            inspector_factory=inspector_factory,
            preprocess=True,
            preprocessing_profile=preprocessing_profile,
        )
    finally:
        temp_path.unlink(missing_ok=True)
    score = prediction.get("score_percent")
    if score is None:
        raise AIDomainError("La inferencia FINAL_TEST no produjo score.")
    anomaly = float(score) >= threshold_fixed
    predicted_category = "MANCHA" if anomaly else "NORMAL"
    correct = predicted_category == category_key
    case_dir = (
        f"garment_{garment_model_id}/final_test/"
        f"ai_model_{int(ai_model_id)}/session_{session_id}/case_{digest[:16]}"
    )
    original_rel = f"{case_dir}/original{_original_extension(bytes(image_bytes))}"
    heatmap_rel = f"{case_dir}/heatmap.png" if prediction.get("anomaly_map") is not None else None
    comparison_rel = f"{case_dir}/comparison.png" if prediction.get("anomaly_map") is not None else None
    cur.execute(
        """INSERT INTO ai_final_test_cases
           (final_test_session_id, ai_model_id, garment_model_id, category,
            image_path, image_sha256, anomaly_score, threshold_used,
            prediction, correct, heatmap_path, comparison_path, created_by)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            session_id, int(ai_model_id), garment_model_id,
            category_key, original_rel, digest, float(score),
            threshold_fixed, predicted_category, int(correct),
            heatmap_rel, comparison_rel, actor_id,
        ),
    )
    case_id = int(cur.lastrowid)
    root = get_ai_artifacts_root()
    atomic_write_bytes(resolve_under_root(root, original_rel), bytes(image_bytes))
    if heatmap_rel:
        try:
            import cv2
            import numpy as np
            original = cv2.imdecode(np.frombuffer(bytes(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            heatmap = render_heatmap(prediction.get("anomaly_map"), reference_shape=original.shape if original is not None else None)
            heatmap_png = _encode_png(heatmap)
            comparison_png = _encode_png(render_comparison(original, heatmap)) if original is not None else None
            if heatmap_png:
                atomic_write_bytes(resolve_under_root(root, heatmap_rel), heatmap_png)
            if comparison_png:
                atomic_write_bytes(resolve_under_root(root, comparison_rel), comparison_png)
        except Exception:
            print("[FINAL_TEST] No se pudo generar heatmap/comparación.")
            heatmap_rel = comparison_rel = None
            cur.execute("UPDATE ai_final_test_cases SET heatmap_path=NULL, comparison_path=NULL WHERE id=%s", (case_id,))
    record_ai_event(
        cur,
        "FINAL_TEST_CASE_REGISTERED",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={
            "final_test_session_id": session_id,
            "final_test_case_id": case_id,
            "category": category_key,
            "score": float(score),
            "prediction": predicted_category,
            "correct": bool(correct),
            "threshold_fixed": threshold_fixed,
            "preprocessing_profile": preprocessing_profile,
            "image_sha256": digest,
        },
    )
    return {
        "id": case_id,
        "category": category_key,
        "anomaly_score": float(score),
        "threshold_used": threshold_fixed,
        "prediction": predicted_category,
        "correct": bool(correct),
        "image_sha256": digest,
        "image_path": original_rel,
        "heatmap_path": heatmap_rel,
        "comparison_path": comparison_rel,
        "created_at": None,
    }


def evaluate_final_test(cur, *, ai_model_id=FINAL_TEST_AI_MODEL_ID, actor_id=None) -> dict:
    """Calculate metrics only over this final cohort; never search thresholds."""
    state = get_final_test_state(cur, ai_model_id=ai_model_id)
    if not state.get("integrity_ok"):
        raise AIDomainError(state.get("integrity_error") or "Configuración FINAL_TEST inválida.")
    if not state.get("ready_to_evaluate"):
        raise AIDomainError("La evaluación requiere exactamente 10 BUENAS y 10 MANCHAS nuevas.")
    threshold_fixed = float(state["threshold_fixed"])
    cases = state["cases"]
    metrics = binary_metrics(cases, threshold_fixed)
    metrics.update({
        "cohort": "FINAL_TEST",
        "threshold_fixed": threshold_fixed,
        "candidate_thresholds": [],
        "recalibration_performed": False,
        "confusion_matrix": [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        "good_cases": state["counts"]["BUENA"],
        "stain_cases": state["counts"]["MANCHA"],
    })
    cur.execute(
        "UPDATE ai_final_test_sessions SET status='EVALUADA', metrics_json=%s, evaluated_at=NOW() WHERE id=%s AND status='ABIERTA'",
        (json.dumps(metrics, ensure_ascii=False), int(state["id"])),
    )
    record_ai_event(
        cur,
        "FINAL_TEST_EVALUATED",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={"final_test_session_id": int(state["id"]), "metrics": metrics, "activated": False},
    )
    return metrics


def evaluate_validation(
    cur,
    *,
    ai_model_id,
    actor_id=None,
    thresholds=None,
) -> dict:
    """Acción administrativa «Evaluar validación».

    Calcula métricas y candidatos de umbral, y los persiste en la
    sesión. NO cambia el estado del modelo (eso es FASE 3C).
    """
    model = _fetch_validation_model(cur, ai_model_id)

    if is_technically_invalidated(model.get("notes")):
        raise AIDomainError(
            "Esta versión está marcada NO VALIDADA / NO APTO PARA "
            "ACTIVACIÓN: no puede evaluarse."
        )

    cur.execute(
        """
        SELECT id, status, metrics_json
        FROM ai_validation_sessions
        WHERE ai_model_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(ai_model_id),),
    )
    session = cur.fetchone()

    if session is None:
        raise AIDomainError(
            "Todavía no existe una sesión de validación para esta versión."
        )

    session = dict(session)

    if session["status"] == VALIDATION_SESSION_STATUS_CERRADA:
        raise AIDomainError(
            "La validación de esta versión ya fue cerrada: no se "
            "puede reevaluar en FASE 3B."
        )

    cases = list_validation_cases(cur, ai_model_id)

    if not cases:
        raise AIDomainError(
            "Todavía no hay casos de validación registrados."
        )

    post_freeze_cases = [
        case for case in cases
        if case.get("validation_cohort") == "POST_FREEZE_FINAL_TEST"
    ]
    calibration_cases = [
        case for case in cases
        if case.get("validation_cohort") != "POST_FREEZE_FINAL_TEST"
    ]
    candidates = None

    if thresholds:
        candidates = []
        for raw in thresholds:
            value = _numeric_score(raw)
            if value is None:
                raise AIDomainError(
                    "Los thresholds candidatos deben ser numéricos."
                )
            candidates.append(round(value, 4))

    if not candidates:
        candidates = default_threshold_candidates(calibration_cases)

    metrics = compute_validation_metrics(
        calibration_cases,
        candidates=candidates,
    )
    metrics["calibration_case_count"] = len(calibration_cases)
    metrics["post_freeze_final_test"] = {
        "case_count": len(post_freeze_cases),
        "case_ids": [int(case["id"]) for case in post_freeze_cases],
        "threshold_final": (
            float(model["threshold_final"])
            if model.get("threshold_final") is not None else None
        ),
        "metrics_at_threshold_final": (
            binary_metrics(post_freeze_cases, model["threshold_final"])
            if post_freeze_cases and model.get("threshold_final") is not None
            else None
        ),
    }
    metrics["by_category_labels"] = dict(VALIDATION_CATEGORY_LABELS)
    metrics["counts"] = count_cases_by_category(calibration_cases)
    metrics["counts_estado"] = count_cases_by_estado(calibration_cases)
    metrics["counts_defects"] = count_defect_cases(calibration_cases)
    metrics["binary_notice"] = VALIDATION_BINARY_NOTICE

    best = metrics.get("best_threshold")

    cur.execute(
        """
        UPDATE ai_validation_sessions
        SET status = %s,
            metrics_json = %s,
            threshold_candidate = %s,
            evaluated_at = NOW()
        WHERE id = %s
        """,
        (
            VALIDATION_SESSION_STATUS_EVALUADA,
            json.dumps(metrics, ensure_ascii=False),
            best,
            int(session["id"]),
        ),
    )

    record_ai_event(
        cur,
        "VALIDATION_EVALUATED",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={
            "validation_session_id": int(session["id"]),
            "cases": len(calibration_cases),
            "post_freeze_final_test_cases": len(post_freeze_cases),
            "best_threshold": best,
            "counts_estado": count_cases_by_estado(calibration_cases),
            "counts_defects": count_defect_cases(calibration_cases),
            "model_status": model.get("status"),
        },
    )

    return {
        "ai_model_id": int(ai_model_id),
        "garment_model_id": int(model["garment_model_id"]),
        "version": model.get("version"),
        "model_status": model.get("status"),
        "validation_session_id": int(session["id"]),
        "metrics": metrics,
        "notice": VALIDATION_IMAGE_NOTICE,
    }


def freeze_validation_threshold(
    cur, *, ai_model_id, threshold, actor_id, provenance
) -> dict:
    """Fija una sola vez el threshold final; nunca activa la versión."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise AIDomainError("El threshold definitivo debe ser numérico.") from error
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise AIDomainError("El threshold definitivo debe estar entre 0 y 100.")
    value = round(value, 2)
    note = str(provenance or "").strip()
    if not note or len(note) > 500:
        raise AIDomainError("Indique una procedencia de hasta 500 caracteres.")
    cur.execute(
        """SELECT id, garment_model_id, version, status, active,
                  threshold_final, threshold_frozen_at, threshold_frozen_by,
                  threshold_provenance
           FROM garment_ai_models WHERE id = %s FOR UPDATE""",
        (int(ai_model_id),),
    )
    row = cur.fetchone()
    if row is None:
        raise AIDomainError("La versión de IA solicitada no existe.")
    model = dict(row)
    if is_technically_invalidated(model.get("notes")):
        raise AIDomainError("La versión invalidada técnicamente no puede congelarse.")
    if str(model.get("status") or "").upper() != AI_MODEL_STATUS_VALIDACION:
        raise AIDomainError("Solo se puede congelar el threshold durante VALIDACION.")
    if int(model.get("active") or 0):
        raise AIDomainError("No se puede congelar un threshold de una versión ACTIVA.")
    if model.get("threshold_final") is not None:
        if round(float(model["threshold_final"]), 2) != value:
            raise AIDomainError("El threshold final ya está congelado y no puede cambiarse.")
        return {
            "ai_model_id": int(ai_model_id),
            "threshold_final": round(float(model["threshold_final"]), 2),
            "frozen_at": model.get("threshold_frozen_at"),
            "frozen_by": model.get("threshold_frozen_by"),
            "provenance": model.get("threshold_provenance"),
            "already_frozen": True,
        }
    cur.execute(
        """SELECT id, status, threshold_candidate, metrics_json FROM ai_validation_sessions
           WHERE ai_model_id = %s ORDER BY id DESC LIMIT 1""",
        (int(ai_model_id),),
    )
    session = cur.fetchone()
    if (
        session is None
        or str(session.get("status") or "").upper()
        != VALIDATION_SESSION_STATUS_EVALUADA
    ):
        raise AIDomainError(
            "Evalúe la sesión vigente antes de congelar el threshold definitivo."
        )
    if session.get("threshold_candidate") is None:
        raise AIDomainError("No existe un threshold candidato para congelar.")
    candidate = round(float(session["threshold_candidate"]), 2)
    if value != candidate:
        raise AIDomainError(
            f"Solo puede congelarse el threshold candidato EVALUADO ({candidate:.2f}); no se permite editarlo."
        )
    try:
        evaluated_metrics = json.loads(session.get("metrics_json") or "{}")
    except (TypeError, ValueError) as error:
        raise AIDomainError("Las métricas de la sesión EVALUADA no se pueden verificar.") from error
    cases = list_validation_cases(cur, ai_model_id)
    bank_check = verify_validation_bank_snapshot(
        cur,
        ai_model_id=ai_model_id,
        cases=cases,
        session=session,
        metrics=evaluated_metrics,
    )
    cur.execute(
        """UPDATE garment_ai_models
           SET threshold_final = %s, threshold_frozen_at = NOW(),
               threshold_frozen_by = %s, threshold_provenance = %s
           WHERE id = %s AND threshold_final IS NULL""",
        (value, actor_id, note, int(ai_model_id)),
    )
    record_ai_event(
        cur,
        "THRESHOLD_FROZEN",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={
            "threshold_final": value,
            "threshold_candidate_at_freeze": (
                float(candidate) if candidate is not None else None
            ),
            "source": "ADMIN_CONFIRMED",
            "provenance": note,
            "bank_case_count": bank_check["case_count"],
            "bank_snapshot_sha256": bank_check.get("snapshot_sha256"),
        },
    )
    return {
        "ai_model_id": int(ai_model_id),
        "garment_model_id": int(model["garment_model_id"]),
        "version": model.get("version"),
        "threshold_final": value,
        "threshold_candidate_at_freeze": (
            float(candidate) if candidate is not None else None
        ),
        "frozen_by": actor_id,
        "provenance": note,
        "already_frozen": False,
        "active": 0,
    }


def complete_validation(cur, *, ai_model_id, actor_id=None) -> dict:
    """Cierra la validación marcando VALIDADO (nunca ACTIVO)."""
    model = _fetch_validation_model(cur, ai_model_id)

    if is_technically_invalidated(model.get("notes")):
        raise AIDomainError(
            "Esta versión está marcada NO VALIDADA / NO APTO PARA "
            "ACTIVACIÓN: no puede validarse."
        )

    status = str(model["status"] or "").strip().upper()

    cur.execute(
        """
        SELECT id, status, metrics_json, threshold_candidate
        FROM ai_validation_sessions
        WHERE ai_model_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (int(ai_model_id),),
    )
    session = cur.fetchone()

    if session is None:
        raise AIDomainError(
            "Todavía no existe una sesión de validación para esta versión."
        )

    session = dict(session)

    # Idempotente: una versión ya VALIDADA no vuelve a transicionar y
    # solo asegura que su sesión quede cerrada.
    if status == AI_MODEL_STATUS_VALIDADO:
        if session["status"] != VALIDATION_SESSION_STATUS_CERRADA:
            cur.execute(
                "UPDATE ai_validation_sessions SET status = %s, closed_at = NOW() "
                "WHERE id = %s",
                (
                    VALIDATION_SESSION_STATUS_CERRADA,
                    int(session["id"]),
                ),
            )
        return {
            "ai_model_id": int(ai_model_id),
            "garment_model_id": int(model["garment_model_id"]),
            "version": model.get("version"),
            "status": AI_MODEL_STATUS_VALIDADO,
            "already_validated": True,
        }

    if session["status"] != VALIDATION_SESSION_STATUS_EVALUADA:
        raise AIDomainError(
            "Evalúe la validación antes de marcarla como completada."
        )

    if model.get("threshold_final") is None:
        raise AIDomainError(
            "Confirme y congele el threshold definitivo antes de marcar la versión como VALIDADA."
        )

    if status == AI_MODEL_STATUS_ENTRENADO:
        raise AIDomainError(
            "Primero debe iniciar la validación de la versión."
        )

    transition_ai_model_status(
        cur,
        int(ai_model_id),
        AI_MODEL_STATUS_VALIDADO,
        actor_id,
    )

    cur.execute(
        "UPDATE ai_validation_sessions SET status = %s, closed_at = NOW() "
        "WHERE id = %s",
        (VALIDATION_SESSION_STATUS_CERRADA, int(session["id"])),
    )

    record_ai_event(
        cur,
        "VALIDATED",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={
            "validation_session_id": int(session["id"]),
            "threshold_candidate": (
                float(session["threshold_candidate"])
                if session.get("threshold_candidate") is not None
                else None
            ),
            "threshold_final": float(model["threshold_final"]),
        },
    )

    return {
        "ai_model_id": int(ai_model_id),
        "garment_model_id": int(model["garment_model_id"]),
        "version": model.get("version"),
        "status": AI_MODEL_STATUS_VALIDADO,
        "already_validated": False,
    }


# ============================================================
# FASE 3B.1 — CORRECCIÓN DEL GROUND TRUTH Y ELIMINACIÓN DE CASOS
#
# Permite corregir la etiqueta humana de un caso (o eliminarlo si
# fue un error operativo) SIN repetir la inferencia: la imagen, su
# SHA-256, el score, los artefactos y la fecha quedan intactos.
# Toda edición invalida las métricas de la sesión.
# ============================================================

_VALIDATION_CASE_COLUMNS = """
    c.id, c.validation_session_id, c.ai_model_id,
    c.garment_model_id, c.category, c.image_path,
    c.image_sha256, c.anomaly_score, c.threshold_used,
    c.prediction, c.result, c.validation_cohort, c.observation,
    c.heatmap_path, c.comparison_path, c.created_at,
    c.created_by
"""


def _case_ui_payload(case: dict) -> dict:
    """Caso serializado para la UI/API (score y umbral como float)."""
    score = case.get("anomaly_score")
    threshold = case.get("threshold_used")
    category = str(case.get("category") or "").strip().upper()

    try:
        derived = validation_classification(category)
    except AIDomainError:
        # Fila con categoría fuera de catálogo: la UI nunca debe romperse.
        derived = {
            "estado_real": None,
            "estado_label": None,
            "tipo_defecto": None,
            "tipo_defecto_label": None,
            "classification": category,
        }

    return {
        **case,
        "id": int(case["id"]),
        "anomaly_score": float(score) if score is not None else None,
        "threshold_used": (
            float(threshold) if threshold is not None else None
        ),
        "category": category,
        "category_label": VALIDATION_CATEGORY_LABELS.get(category, category),
        **derived,
        "result_label": (
            "Pendiente de calibración"
            if str(case.get("result") or "") == VALIDATION_RESULT_PENDING
            else case.get("result")
        ),
    }


def _fetch_validation_case(
    cur, case_id, ai_model_id=None
) -> dict:
    """Trae un caso por id (y opcionalmente por versión)."""
    try:
        case_key = int(case_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError("case_id inválido.") from error

    sql = (
        f"SELECT {_VALIDATION_CASE_COLUMNS} "
        "FROM ai_validation_cases c "
        "WHERE c.id = %s"
    )
    params: list = [case_key]

    if ai_model_id is not None:
        sql += " AND c.ai_model_id = %s"
        params.append(int(ai_model_id))

    cur.execute(sql, tuple(params))
    row = cur.fetchone()

    if row is None:
        raise AIDomainError(
            "El caso de validación solicitado no existe en esta versión."
        )

    return dict(row)


def _fetch_validation_session_by_id(cur, session_id) -> dict:
    try:
        session_key = int(session_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError("validation_session_id inválido.") from error

    cur.execute(
        """
        SELECT id, ai_model_id, status, metrics_json,
               threshold_candidate, evaluated_at, closed_at
        FROM ai_validation_sessions
        WHERE id = %s
        """,
        (session_key,),
    )
    row = cur.fetchone()

    if row is None:
        raise AIDomainError("La sesión de validación del caso no existe.")

    return dict(row)


def _ensure_case_editable(model: dict, session: dict | None) -> None:
    """Guardas comunes: versión en validación y sesión abierta."""
    if is_technically_invalidated(model.get("notes")):
        raise AIDomainError(
            "Esta versión está marcada NO VALIDADA / NO APTO PARA "
            "ACTIVACIÓN: no se pueden editar sus casos."
        )

    if session is None:
        raise AIDomainError(
            "No existe una sesión de validación para este caso."
        )

    session_status = str(session.get("status") or "").strip().upper()

    if session_status == VALIDATION_SESSION_STATUS_CERRADA:
        raise AIDomainError(
            "La validación de esta versión ya fue cerrada: no se "
            "pueden editar ni eliminar sus casos."
        )

    status = str(model.get("status") or "").strip().upper()

    if status not in CASE_EDITABLE_MODEL_STATUSES:
        raise AIDomainError(
            "Los casos solo pueden editarse mientras la versión está "
            "ENTRENADO o EN VALIDACIÓN (estado actual: "
            f"{status or 'SIN ESTADO'})."
        )


def _invalidate_session_metrics(cur, session: dict) -> dict:
    """Deja la sesión ABIERTA y sin métricas tras tocar sus casos.

    Las métricas derivadas del ground truth quedan obsoletas: vuelve
    a exigirse la acción «Evaluar validación».
    """
    session_id = int(session["id"])
    previous_status = str(session.get("status") or "").strip().upper()
    had_metrics = bool(session.get("metrics_json"))

    cur.execute(
        """
        UPDATE ai_validation_sessions
        SET status = %s,
            metrics_json = NULL,
            threshold_candidate = NULL,
            evaluated_at = NULL
        WHERE id = %s
        """,
        (VALIDATION_SESSION_STATUS_ABIERTA, session_id),
    )

    return {
        "validation_session_id": session_id,
        "previous_status": previous_status,
        "status": VALIDATION_SESSION_STATUS_ABIERTA,
        "had_metrics": had_metrics,
        "invalidated": had_metrics
        or previous_status == VALIDATION_SESSION_STATUS_EVALUADA,
    }


def _remove_case_artifacts(case: dict) -> list[str]:
    """Borra los archivos del caso; devuelve las rutas eliminadas."""
    root = get_ai_artifacts_root()
    removed: list[str] = []

    for key in ("image_path", "heatmap_path", "comparison_path"):
        relative = str(case.get(key) or "").strip()

        if not relative:
            continue

        try:
            target = resolve_under_root(root, relative)
        except AIDomainError:
            continue

        try:
            if target.is_file():
                target.unlink()
                removed.append(relative)
        except OSError:
            continue

    try:
        case_dir = resolve_under_root(
            root,
            validation_case_relative_dir(
                case.get("garment_model_id"),
                case.get("ai_model_id"),
                case.get("category"),
                case.get("id"),
            ),
        )

        if case_dir.is_dir() and not any(case_dir.iterdir()):
            case_dir.rmdir()
    except (AIDomainError, OSError):
        pass

    return removed


def update_validation_case_category(
    cur,
    *,
    case_id,
    ai_model_id,
    category=None,
    actor_id=None,
    estado_real=None,
    tipo_defecto=None,
) -> dict:
    """Corrige la clasificación real (ground truth) de un caso.

    Acepta el estado binario (BUENA/DEFECTUOSA + tipo de defecto) o la
    categoría histórica. Solo cambia la columna ``category``: imagen,
    SHA-256, score de anomalía, artefactos, versión y fecha permanecen
    intactos, y NO se vuelve a ejecutar la inferencia. La edición
    invalida las métricas de la sesión.
    """
    model = _fetch_validation_model(cur, ai_model_id)
    case = _fetch_validation_case(cur, case_id, ai_model_id)
    session = _fetch_validation_session_by_id(
        cur, case["validation_session_id"]
    )
    _ensure_case_editable(model, session)

    new_category = resolve_validation_category(
        category=category,
        estado_real=estado_real,
        tipo_defecto=tipo_defecto,
    )
    previous_category = normalize_validation_category(case.get("category"))
    new_label = VALIDATION_CLASSIFICATION_LABELS[new_category]
    previous_label = VALIDATION_CLASSIFICATION_LABELS[previous_category]

    if new_category == previous_category:
        return {
            "changed": False,
            "message": (
                f"La clasificación real ya era {previous_label}."
            ),
            "ai_model_id": int(model["id"]),
            "validation_session_id": int(session["id"]),
            "case": _case_ui_payload(case),
            "metrics_invalidated": False,
            "notice": VALIDATION_IMAGE_NOTICE,
        }

    cur.execute(
        "UPDATE ai_validation_cases SET category = %s WHERE id = %s",
        (new_category, int(case["id"])),
    )

    invalidation = _invalidate_session_metrics(cur, session)
    updated = _fetch_validation_case(cur, case["id"])

    record_ai_event(
        cur,
        "VALIDATION_CASE_CATEGORY_CHANGED",
        actor_id=actor_id,
        ai_model_id=int(model["id"]),
        payload={
            "validation_session_id": int(session["id"]),
            "validation_case_id": int(case["id"]),
            "before_category": previous_category,
            "after_category": new_category,
            "before_estado_real": estado_real_from_category(
                previous_category
            ),
            "after_estado_real": estado_real_from_category(new_category),
            "before_tipo_defecto": defect_type_from_category(
                previous_category
            ),
            "after_tipo_defecto": defect_type_from_category(new_category),
            "image_sha256": case.get("image_sha256"),
            "image_path": case.get("image_path"),
            "anomaly_score": (
                float(case["anomaly_score"])
                if case.get("anomaly_score") is not None
                else None
            ),
            "session_status_before": invalidation["previous_status"],
            "session_status_after": invalidation["status"],
            "metrics_invalidated": invalidation["invalidated"],
        },
    )

    return {
        "changed": True,
        "message": (
            f"Clasificación real actualizada a {new_label}. El score y "
            "los artefactos no se modificaron; vuelva a «Evaluar "
            "validación» para recalcular las métricas."
        ),
        "ai_model_id": int(model["id"]),
        "validation_session_id": int(session["id"]),
        "case": _case_ui_payload(updated),
        "metrics_invalidated": invalidation["invalidated"],
        "notice": VALIDATION_IMAGE_NOTICE,
    }


def delete_validation_case(
    cur,
    *,
    case_id,
    ai_model_id,
    actor_id=None,
) -> dict:
    """Elimina un caso registrado por error operativo.

    No toca el entrenamiento, la versión ni los demás casos; solo
    borra la fila del caso y sus artefactos, e invalida las métricas
    de la sesión. Queda registrado en el historial de auditoría.
    """
    model = _fetch_validation_model(cur, ai_model_id)
    case = _fetch_validation_case(cur, case_id, ai_model_id)
    session = _fetch_validation_session_by_id(
        cur, case["validation_session_id"]
    )
    _ensure_case_editable(model, session)

    removed_case = _case_ui_payload(case)

    cur.execute(
        "DELETE FROM ai_validation_cases WHERE id = %s",
        (int(case["id"]),),
    )

    if cur.rowcount == 0:
        raise AIDomainError("El caso de validación ya no existe.")

    removed_files = _remove_case_artifacts(case)
    invalidation = _invalidate_session_metrics(cur, session)

    record_ai_event(
        cur,
        "VALIDATION_CASE_DELETED",
        actor_id=actor_id,
        ai_model_id=int(model["id"]),
        payload={
            "validation_session_id": int(session["id"]),
            "validation_case_id": int(case["id"]),
            "category": removed_case["category"],
            "estado_real": removed_case.get("estado_real"),
            "tipo_defecto": removed_case.get("tipo_defecto"),
            "image_sha256": case.get("image_sha256"),
            "image_path": case.get("image_path"),
            "anomaly_score": (
                float(case["anomaly_score"])
                if case.get("anomaly_score") is not None
                else None
            ),
            "files_removed": removed_files,
            "session_status_before": invalidation["previous_status"],
            "session_status_after": invalidation["status"],
            "metrics_invalidated": invalidation["invalidated"],
        },
    )

    return {
        "deleted": True,
        "message": (
            "Caso de validación eliminado. Las métricas de la sesión "
            "quedaron invalidadas; vuelva a «Evaluar validación» antes "
            "de cerrar la validación."
        ),
        "ai_model_id": int(model["id"]),
        "validation_session_id": int(session["id"]),
        "case": removed_case,
        "files_removed": removed_files,
        "metrics_invalidated": invalidation["invalidated"],
        "notice": VALIDATION_IMAGE_NOTICE,
    }
