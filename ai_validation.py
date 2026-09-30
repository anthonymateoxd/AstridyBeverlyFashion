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
import os
import tempfile
import threading
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
               dataset_id, active
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
        "active": int(row.get("active") or 0),
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


def build_validation_input(image_path) -> Path:
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

    try:
        mask = patchcore_preprocess.create_garment_mask(image, bounds)
    except Exception:
        # Misma tolerancia que producción: si la silueta no se puede
        # describir, build_patchcore_regions cae al ROI completo.
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
            temp_input = build_validation_input(target)
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
        SELECT id, garment_model_id, version, status, notes, active
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

    prediction_label = "ANOMALIA" if is_anomaly else "NORMAL"
    result = validate_validation_case_result(
        category_key,
        score_percent,
        None,
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
            observation,
            created_by
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            int(session["id"]),
            int(bundle["ai_model_id"]),
            int(bundle["garment_model_id"]),
            category_key,
            "PENDIENTE",
            sha256,
            score_percent,
            None,
            prediction_label,
            result,
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
        "threshold_used": None,
        "prediction": prediction_label,
        "result": result,
        "result_label": "Pendiente de calibración",
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
               c.prediction, c.result, c.observation,
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

    metrics = None
    if session and session.get("metrics_json"):
        try:
            metrics = json.loads(session["metrics_json"])
        except (TypeError, ValueError):
            metrics = None

    invalidated = is_technically_invalidated(model.get("notes"))
    status = str(model["status"] or "").strip().upper()

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
        "counts": counts,
        "counts_estado": count_cases_by_estado(cases),
        "counts_defects": count_defect_cases(cases),
        "total_cases": len(cases),
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
        "pending_calibration": True,
    }


# ============================================================
# MÉTRICAS Y CALIBRACIÓN (sin elegir umbral definitivo)
# ============================================================

DEFAULT_THRESHOLD_GRID = tuple(range(5, 100, 5))


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
        candidates = default_threshold_candidates(cases)

    metrics = compute_validation_metrics(
        cases,
        candidates=candidates,
    )
    metrics["by_category_labels"] = dict(VALIDATION_CATEGORY_LABELS)
    metrics["counts"] = count_cases_by_category(cases)
    metrics["counts_estado"] = count_cases_by_estado(cases)
    metrics["counts_defects"] = count_defect_cases(cases)
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
            "cases": len(cases),
            "best_threshold": best,
            "counts_estado": count_cases_by_estado(cases),
            "counts_defects": count_defect_cases(cases),
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
    c.prediction, c.result, c.observation,
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
