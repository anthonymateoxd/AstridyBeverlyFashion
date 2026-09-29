"""Dominio de preparación y versionado de IA (Fase 1).

Contiene la máquina de estados, validación de rutas/hashes y helpers de
persistencia que reciben un cursor abierto. No importa Flask ni app.py
para poder probarse de forma aislada.

No implementa cámara, entrenamiento PatchCore ni cambio de checkpoint
de producción.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path, PurePosixPath


# ============================================================
# RAÍZ PERSISTENTE DE ARTEFACTOS IA
# ============================================================

AI_ARTIFACT_SUBDIRS = ("datasets", "checkpoints", "validations")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_FORBIDDEN_PAYLOAD_KEYS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "private_key",
    "secret_key",
)


class AIDomainError(ValueError):
    """Error de negocio del dominio de IA."""


def _scalar(row):
    """Extrae el primer valor de una fila (tuple o dict)."""
    if row is None:
        return None

    if isinstance(row, dict):
        values = list(row.values())
        return values[0] if values else None

    return row[0]


def get_ai_artifacts_root() -> Path:
    """Raíz configurable para artefactos IA (env AI_ARTIFACTS_ROOT)."""
    raw = str(os.getenv("AI_ARTIFACTS_ROOT", "") or "").strip()

    if raw:
        return Path(raw).expanduser()

    return Path(__file__).resolve().parent / "ai_artifacts"


def build_garment_artifact_dir(
    garment_model_id: int,
    root: Path | None = None,
) -> Path:
    """Devuelve <root>/garment_<id> validando que el id sea un entero > 0."""
    try:
        model_id = int(garment_model_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "garment_model_id debe ser un entero positivo."
        ) from error

    if model_id <= 0:
        raise AIDomainError(
            "garment_model_id debe ser un entero positivo."
        )

    base = Path(root) if root is not None else get_ai_artifacts_root()
    return base / f"garment_{model_id}"


def ensure_ai_artifact_dirs(
    garment_model_id: int,
    root: Path | None = None,
) -> Path:
    """Crea garment_<id>/{datasets,checkpoints,validations} de forma segura."""
    garment_dir = build_garment_artifact_dir(garment_model_id, root=root)

    for name in AI_ARTIFACT_SUBDIRS:
        (garment_dir / name).mkdir(parents=True, exist_ok=True)

    return garment_dir


# ============================================================
# FASE 2A — CONFIGURACIÓN Y ALMACENAMIENTO DE CAPTURA IA
# ============================================================

# Defaults documentados (override por env). No son mágicos:
# - min_coverage alinea con AUTO_GARMENT_CAPTURE_COVERAGE de producción.
# - min_sharpness: varianza Laplaciana sobre el ROI (más bajo = más borroso).
# - target_images: objetivo recomendado, no mínimo obligatorio.
# - min_images: mínimo configurable de imágenes válidas requerido
#   tanto para finalizar la preparación como para habilitar entrenamiento.
# - duplicate_max_distance: distancia Hamming máxima (0–64) del dHash 64-bit
#   para considerar dos imágenes duplicadas perceptualmente.
# - keep_rejected: si es false solo se persiste metadata de rechazos.

AI_CAPTURE_DEFAULTS = {
    "min_coverage": 0.48,
    "min_sharpness": 40.0,
    "target_images": 40,
    "min_images": 20,
    "duplicate_max_distance": 6,
    "keep_rejected": False,
    "jpeg_quality": 92,
}


def _env_bool(raw: str | None, default: bool) -> bool:
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def get_ai_capture_config() -> dict:
    """Configuración de captura IA desde env con defaults documentados."""
    env_map = {
        "AI_CAPTURE_MIN_COVERAGE": ("min_coverage", float),
        "AI_CAPTURE_MIN_SHARPNESS": ("min_sharpness", float),
        "AI_CAPTURE_TARGET_IMAGES": ("target_images", int),
        "AI_CAPTURE_MIN_IMAGES": ("min_images", int),
        "AI_CAPTURE_DUPLICATE_THRESHOLD": ("duplicate_max_distance", int),
        "AI_CAPTURE_JPEG_QUALITY": ("jpeg_quality", int),
    }

    values = dict(AI_CAPTURE_DEFAULTS)

    for env_name, (key, caster) in env_map.items():
        raw = os.getenv(env_name, "")
        if str(raw).strip() == "":
            continue
        try:
            values[key] = caster(raw)
        except ValueError as error:
            raise AIDomainError(
                f"{env_name} inválido: {raw!r}"
            ) from error

    values["keep_rejected"] = _env_bool(
        os.getenv("AI_CAPTURE_KEEP_REJECTED"),
        bool(AI_CAPTURE_DEFAULTS["keep_rejected"]),
    )

    if not 0.0 <= float(values["min_coverage"]) <= 1.0:
        raise AIDomainError(
            "AI_CAPTURE_MIN_COVERAGE debe estar entre 0 y 1."
        )
    if float(values["min_sharpness"]) < 0:
        raise AIDomainError(
            "AI_CAPTURE_MIN_SHARPNESS no puede ser negativa."
        )
    if int(values["target_images"]) < 1:
        raise AIDomainError(
            "AI_CAPTURE_TARGET_IMAGES debe ser >= 1."
        )
    if int(values["min_images"]) < 1:
        raise AIDomainError(
            "AI_CAPTURE_MIN_IMAGES debe ser >= 1."
        )
    if not 0 <= int(values["duplicate_max_distance"]) <= 64:
        raise AIDomainError(
            "AI_CAPTURE_DUPLICATE_THRESHOLD debe estar entre 0 y 64."
        )
    if not 1 <= int(values["jpeg_quality"]) <= 100:
        raise AIDomainError(
            "AI_CAPTURE_JPEG_QUALITY debe estar entre 1 y 100."
        )

    return {
        "min_coverage": float(values["min_coverage"]),
        "min_sharpness": float(values["min_sharpness"]),
        "target_images": int(values["target_images"]),
        "min_images": int(values["min_images"]),
        "duplicate_max_distance": int(values["duplicate_max_distance"]),
        "keep_rejected": bool(values["keep_rejected"]),
        "jpeg_quality": int(values["jpeg_quality"]),
    }


def capture_progress_gate(
    accepted_count,
    *,
    min_images,
    target_count=None,
) -> dict:
    """
    Compuertas de la preparación de IA (sin efectos secundarios).

    - can_finalize: exige alcanzar el mínimo configurable.
    - can_train: recién se habilita cuando se alcanza el mínimo
      configurable (AI_CAPTURE_MIN_IMAGES).
    - target_count: objetivo recomendado (no bloquea nada).
    """
    accepted = int(accepted_count or 0)
    minimum = max(0, int(min_images or 0))

    return {
        "accepted_count": accepted,
        "min_count": minimum,
        "target_count": (
            None if target_count is None else int(target_count)
        ),
        "can_finalize": minimum > 0 and accepted >= minimum,
        "can_train": minimum > 0 and accepted >= minimum,
        "missing_to_train": max(0, minimum - accepted),
    }


def build_capture_session_dir(
    garment_model_id: int,
    capture_session_id: int,
    root: Path | None = None,
) -> Path:
    """garment_<id>/capture_sessions/session_<id> (sin crear)."""
    garment_dir = build_garment_artifact_dir(garment_model_id, root=root)

    try:
        session_id = int(capture_session_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "capture_session_id debe ser un entero positivo."
        ) from error

    if session_id <= 0:
        raise AIDomainError(
            "capture_session_id debe ser un entero positivo."
        )

    return garment_dir / "capture_sessions" / f"session_{session_id}"


def ensure_capture_session_dirs(
    garment_model_id: int,
    capture_session_id: int,
    root: Path | None = None,
    *,
    keep_rejected: bool = False,
) -> Path:
    """Crea accepted/ (y rejected/ solo si keep_rejected)."""
    session_dir = build_capture_session_dir(
        garment_model_id,
        capture_session_id,
        root=root,
    )
    (session_dir / "accepted").mkdir(parents=True, exist_ok=True)
    if keep_rejected:
        (session_dir / "rejected").mkdir(parents=True, exist_ok=True)
    return session_dir


def capture_image_relative_path(
    garment_model_id: int,
    capture_session_id: int,
    filename: str,
    *,
    rejected: bool = False,
) -> str:
    """Ruta relativa controlada bajo AI_ARTIFACTS_ROOT."""
    safe_name = ensure_safe_relative_path(str(filename).strip())
    if "/" in safe_name or "\\" in safe_name:
        raise AIDomainError("filename no puede contener separadores.")
    folder = "rejected" if rejected else "accepted"
    try:
        model_id = int(garment_model_id)
        session_id = int(capture_session_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "garment_model_id y capture_session_id deben ser enteros."
        ) from error
    if model_id <= 0 or session_id <= 0:
        raise AIDomainError(
            "garment_model_id y capture_session_id deben ser positivos."
        )
    return (
        f"garment_{model_id}/capture_sessions/"
        f"session_{session_id}/{folder}/{safe_name}"
    )


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Escritura atómica: temporal en el mismo directorio → fsync → replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return target


def hamming_distance(a: int, b: int) -> int:
    """Distancia de Hamming entre dos enteros (dHash 64-bit)."""
    return int((a ^ b).bit_count())


def is_perceptual_duplicate(
    hash_a: int,
    hash_b: int,
    max_distance: int,
) -> bool:
    """True si la distancia Hamming no supera el umbral configurado."""
    try:
        max_d = int(max_distance)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "duplicate_max_distance debe ser un entero."
        ) from error
    if not 0 <= max_d <= 64:
        raise AIDomainError(
            "duplicate_max_distance debe estar entre 0 y 64."
        )
    return hamming_distance(int(hash_a), int(hash_b)) <= max_d


def compute_quality_score(
    coverage: float,
    sharpness: float,
    *,
    min_coverage: float,
    min_sharpness: float,
) -> float:
    """Score 0–1 combinando cobertura y nitidez (documentado, no mágico)."""
    cov = max(0.0, float(coverage))
    sharp = max(0.0, float(sharpness))
    base_cov = max(float(min_coverage), 1e-6)
    base_sharp = max(float(min_sharpness), 1e-6)
    cov_norm = min(1.0, cov / base_cov)
    # sharpness ~2x el mínimo ya satura el peso de nitidez
    sharp_norm = min(1.0, sharp / (base_sharp * 2.0))
    score = 0.6 * cov_norm + 0.4 * sharp_norm
    return round(max(0.0, min(1.0, score)), 3)


def evaluate_capture_metrics(
    *,
    coverage: float | None,
    sharpness: float | None,
    roi_valid: bool,
    config: dict | None = None,
) -> dict:
    """
    Decide ACEPTADA/RECHAZADA a partir de métricas ya calculadas.

    No escribe archivos ni toca BD.
    """
    cfg = dict(config or get_ai_capture_config())
    min_coverage = float(cfg["min_coverage"])
    min_sharpness = float(cfg["min_sharpness"])

    if not roi_valid:
        return {
            "accepted": False,
            "reject_reason": "INVALID_ROI",
            "quality_score": 0.0,
        }

    if coverage is None:
        return {
            "accepted": False,
            "reject_reason": "INVALID_COVERAGE",
            "quality_score": 0.0,
        }

    try:
        cov = float(coverage)
    except (TypeError, ValueError):
        return {
            "accepted": False,
            "reject_reason": "INVALID_COVERAGE",
            "quality_score": 0.0,
        }

    if sharpness is None:
        return {
            "accepted": False,
            "reject_reason": "INVALID_SHARPNESS",
            "quality_score": 0.0,
        }

    try:
        sharp = float(sharpness)
    except (TypeError, ValueError):
        return {
            "accepted": False,
            "reject_reason": "INVALID_SHARPNESS",
            "quality_score": 0.0,
        }

    score = compute_quality_score(
        cov,
        sharp,
        min_coverage=min_coverage,
        min_sharpness=min_sharpness,
    )

    if cov < min_coverage:
        return {
            "accepted": False,
            "reject_reason": "LOW_COVERAGE",
            "quality_score": score,
        }

    if sharp < min_sharpness:
        return {
            "accepted": False,
            "reject_reason": "BLUR",
            "quality_score": score,
        }

    return {
        "accepted": True,
        "reject_reason": None,
        "quality_score": score,
    }


def ensure_safe_relative_path(relative_path: str) -> str:
    """Valida una ruta relativa controlada (sin traversal ni absoluta)."""
    raw = str(relative_path or "").strip()

    if not raw:
        raise AIDomainError("La ruta no puede estar vacía.")

    if "\x00" in raw:
        raise AIDomainError("La ruta contiene caracteres no permitidos.")

    if any(ord(char) < 32 for char in raw):
        raise AIDomainError("La ruta contiene caracteres no permitidos.")

    posix = PurePosixPath(raw.replace("\\", "/"))

    if posix.is_absolute() or raw.startswith("/"):
        raise AIDomainError("Solo se permiten rutas relativas.")

    if posix.parts and posix.parts[0].endswith(":"):
        raise AIDomainError("Solo se permiten rutas relativas.")

    if ".." in posix.parts:
        raise AIDomainError("Ruta inválida (path traversal).")

    if not posix.parts or posix.parts in ((".",),):
        raise AIDomainError("La ruta no puede estar vacía.")

    return posix.as_posix()


def resolve_under_root(root: Path, relative_path: str) -> Path:
    """Resuelve una ruta relativa garantizando que quede bajo root."""
    safe = ensure_safe_relative_path(relative_path)
    root_resolved = Path(root).expanduser().resolve()
    target = (root_resolved / safe).resolve()

    if target != root_resolved and root_resolved not in target.parents:
        raise AIDomainError("Ruta fuera de la raíz de artefactos IA.")

    return target


def is_valid_sha256(value) -> bool:
    """True si value es un hash SHA-256 hexadecimal de 64 caracteres."""
    if not isinstance(value, str):
        return False

    raw = value.strip()

    if not _SHA256_RE.fullmatch(raw):
        return False

    return raw == raw.lower() or raw == raw.upper()


def normalize_sha256(value: str) -> str:
    """Valida y normaliza un SHA-256 a minúsculas."""
    if not isinstance(value, str):
        raise AIDomainError("sha256 debe ser una cadena hexadecimal.")

    raw = value.strip()

    if not _SHA256_RE.fullmatch(raw):
        raise AIDomainError(
            "sha256 debe ser un hash hexadecimal de 64 caracteres."
        )

    return raw.lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sanitize_event_payload(payload) -> dict | None:
    """Normaliza el payload de auditoría y bloquea secretos evidentes."""
    if payload is None:
        return None

    if not isinstance(payload, dict):
        raise AIDomainError(
            "payload_json debe ser un objeto JSON."
        )

    def _walk(value, key_hint=""):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                lowered = str(key).strip().lower()
                if any(marker in lowered for marker in _FORBIDDEN_PAYLOAD_KEYS):
                    raise AIDomainError(
                        "payload_json no puede contener secretos."
                    )
                cleaned[key] = _walk(item, key_hint=lowered)
            return cleaned

        if isinstance(value, (list, tuple)):
            return [_walk(item, key_hint=key_hint) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and key_hint:
                if any(
                    marker in key_hint
                    for marker in _FORBIDDEN_PAYLOAD_KEYS
                ):
                    raise AIDomainError(
                        "payload_json no puede contener secretos."
                    )
            return value

        raise AIDomainError(
            "payload_json contiene un tipo no serializable."
        )

    cleaned = _walk(payload)

    try:
        json.dumps(cleaned, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "payload_json debe ser JSON serializable."
        ) from error

    return cleaned


# ============================================================
# MÁQUINA DE ESTADOS — MODELOS DE IA
# ============================================================

AI_MODEL_STATUS_PREPARACION = "PREPARACION"
AI_MODEL_STATUS_ENTRENANDO = "ENTRENANDO"
AI_MODEL_STATUS_ENTRENADO = "ENTRENADO"
AI_MODEL_STATUS_VALIDACION = "VALIDACION"
AI_MODEL_STATUS_VALIDADO = "VALIDADO"
AI_MODEL_STATUS_ACTIVO = "ACTIVO"
AI_MODEL_STATUS_RETIRADO = "RETIRADO"
AI_MODEL_STATUS_RECHAZADO = "RECHAZADO"
AI_MODEL_STATUS_FALLIDO = "FALLIDO"

AI_MODEL_STATUSES = {
    AI_MODEL_STATUS_PREPARACION,
    AI_MODEL_STATUS_ENTRENANDO,
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_VALIDACION,
    AI_MODEL_STATUS_VALIDADO,
    AI_MODEL_STATUS_ACTIVO,
    AI_MODEL_STATUS_RETIRADO,
    AI_MODEL_STATUS_RECHAZADO,
    AI_MODEL_STATUS_FALLIDO,
}

# Invalidación técnica: se escribe en garment_ai_models.notes y bloquea
# cualquier paso a ACTIVO mientras siga presente. No es un estado nuevo
# (la máquina de estados no tiene transición de invalidación técnica).
AI_MODEL_TECHNICAL_INVALIDATION_PREFIX = (
    "NO VALIDADO / NO APTO PARA ACTIVACIÓN"
)


def is_technically_invalidated(notes) -> bool:
    """True si las notas marcan la versión como no apta para activar."""
    return AI_MODEL_TECHNICAL_INVALIDATION_PREFIX in str(
        notes or ""
    ).upper()

# Estados que cuentan como "versión en proceso" a la hora de decidir si
# ya existe un trabajo pendiente para el modelo de prenda.
AI_MODEL_PENDING_STATUSES = (
    AI_MODEL_STATUS_PREPARACION,
    AI_MODEL_STATUS_ENTRENANDO,
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_VALIDACION,
    AI_MODEL_STATUS_VALIDADO,
)

# Estados desde los cuales una versión todavía podría llegar a ACTIVO
# (si nadie la invalidó técnicamente).
AI_MODEL_ACTIVATABLE_STATUSES = (
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_VALIDACION,
    AI_MODEL_STATUS_VALIDADO,
)


def evaluate_retrain_availability(versions) -> tuple:
    """¿Corresponde "Reentrenar como nueva versión"?

    Reglas de dominio (§4 de FASE 3A.3):

    - debe existir al menos una versión anterior que reemplazar;
    - ninguna versión en proceso sin invalidar bloquea el reentrenamiento
      (si la última está ENTRENADO/VALIDACION/VALIDADO y es válida, la
      acción correcta es validar, no crear otra versión);
    - una versión invalidada técnicamente (o FALLIDO/RECHAZADO) nunca
      bloquea: es histórica y no activable.

    Devuelve (disponible, motivo_bloqueo). No toca la base de datos.
    """
    rows = []

    for item in versions or []:
        if isinstance(item, dict):
            row = {
                "version": str(item.get("version") or "").strip(),
                "status": str(item.get("status") or "").strip().upper(),
                "notes": item.get("notes"),
            }
        else:
            row = {"version": str(item or "").strip(), "status": "", "notes": None}

        if row["version"]:
            rows.append(row)

    if not rows:
        return False, "No existe una versión anterior que reemplazar."

    blocking = [
        row
        for row in rows
        if row["status"] in AI_MODEL_PENDING_STATUSES
        and not is_technically_invalidated(row["notes"])
    ]

    if blocking:
        last = blocking[-1]

        if last["status"] in AI_MODEL_ACTIVATABLE_STATUSES:
            return (
                False,
                f"La versión {last['version']} está pendiente de "
                "validación; complete la validación antes de reentrenar.",
            )

        return (
            False,
            f"Ya existe la versión {last['version']} en proceso "
            f"({last['status']}).",
        )

    return True, None


# Rollback futuro: RETIRADO -> ACTIVO (reactivación controlada).
AI_MODEL_TRANSITIONS = {
    AI_MODEL_STATUS_PREPARACION: {AI_MODEL_STATUS_ENTRENANDO},
    AI_MODEL_STATUS_ENTRENANDO: {
        AI_MODEL_STATUS_ENTRENADO,
        AI_MODEL_STATUS_FALLIDO,
    },
    AI_MODEL_STATUS_ENTRENADO: {AI_MODEL_STATUS_VALIDACION},
    AI_MODEL_STATUS_VALIDACION: {
        AI_MODEL_STATUS_VALIDADO,
        AI_MODEL_STATUS_RECHAZADO,
        AI_MODEL_STATUS_FALLIDO,
    },
    AI_MODEL_STATUS_VALIDADO: {AI_MODEL_STATUS_ACTIVO},
    AI_MODEL_STATUS_ACTIVO: {AI_MODEL_STATUS_RETIRADO},
    AI_MODEL_STATUS_RETIRADO: {AI_MODEL_STATUS_ACTIVO},
    AI_MODEL_STATUS_RECHAZADO: set(),
    AI_MODEL_STATUS_FALLIDO: set(),
}


def validate_ai_status_transition(current: str, new_status: str) -> None:
    """Valida la transición de estado de una versión de IA."""
    current_raw = str(current or "").strip().upper()
    new_raw = str(new_status or "").strip().upper()

    if current_raw not in AI_MODEL_STATUSES:
        raise AIDomainError(f"Estado actual desconocido: {current!r}")

    if new_raw not in AI_MODEL_STATUSES:
        raise AIDomainError(f"Estado destino desconocido: {new_status!r}")

    if current_raw == new_raw:
        raise AIDomainError(
            f"La versión ya se encuentra en {current_raw}."
        )

    allowed = AI_MODEL_TRANSITIONS.get(current_raw, set())

    if new_raw not in allowed:
        raise AIDomainError(
            f"Transición no permitida: {current_raw} -> {new_raw}."
        )

    if (
        new_raw == AI_MODEL_STATUS_ACTIVO
        and current_raw != AI_MODEL_STATUS_VALIDADO
        and current_raw != AI_MODEL_STATUS_RETIRADO
    ):
        raise AIDomainError(
            "Solo una versión VALIDADA o RETIRADA puede pasar a ACTIVO."
        )


def next_version_label(existing_versions) -> str:
    """Calcula la siguiente versión vN para un modelo de prenda."""
    numbers = []

    for item in existing_versions or []:
        if isinstance(item, dict):
            value = str(item.get("version") or "").strip()
        else:
            value = str(item or "").strip()

        lowered = value.lower()

        if lowered.startswith("v") and lowered[1:].isdigit():
            numbers.append(int(lowered[1:]))

    return f"v{max(numbers, default=0) + 1}"


def next_dataset_version_label(existing_versions) -> str:
    """Calcula la siguiente versión de dataset dN para un modelo."""
    numbers = []

    for item in existing_versions or []:
        if isinstance(item, dict):
            value = str(item.get("version") or "").strip()
        else:
            value = str(item or "").strip()

        lowered = value.lower()

        if lowered.startswith("d") and lowered[1:].isdigit():
            numbers.append(int(lowered[1:]))

    return f"d{max(numbers, default=0) + 1}"


# ============================================================
# FASE 3B — VALIDACIÓN CONTROLADA DE MODELOS
#
# Sesiones/casos de validación con imágenes NUEVAS, separadas por
# diseño del dataset de entrenamiento. La validación nunca modifica
# el dataset ni los artefactos de entrenamiento.
# ============================================================

VALIDATION_CATEGORY_NORMAL = "NORMAL"
VALIDATION_CATEGORY_MANCHA = "MANCHA"
VALIDATION_CATEGORY_AGUJERO = "AGUJERO"

# Orden estable para la UI y los tests.
VALIDATION_CATEGORIES = (
    VALIDATION_CATEGORY_NORMAL,
    VALIDATION_CATEGORY_MANCHA,
    VALIDATION_CATEGORY_AGUJERO,
)

# Directorios del banco de validación (AI_ARTIFACTS_ROOT/validations).
# La separación por prenda/versión permite servir a futuros modelos.
VALIDATION_CATEGORY_DIRS = {
    VALIDATION_CATEGORY_NORMAL: "normal",
    VALIDATION_CATEGORY_MANCHA: "manchas",
    VALIDATION_CATEGORY_AGUJERO: "aguajeros",
}

VALIDATION_CATEGORY_LABELS = {
    VALIDATION_CATEGORY_NORMAL: "Normal",
    VALIDATION_CATEGORY_MANCHA: "Mancha",
    VALIDATION_CATEGORY_AGUJERO: "Agujero",
}

VALIDATION_SESSION_STATUS_ABIERTA = "ABIERTA"
VALIDATION_SESSION_STATUS_EVALUADA = "EVALUADA"
VALIDATION_SESSION_STATUS_CERRADA = "CERRADA"

VALIDATION_SESSION_STATUSES = (
    VALIDATION_SESSION_STATUS_ABIERTA,
    VALIDATION_SESSION_STATUS_EVALUADA,
    VALIDATION_SESSION_STATUS_CERRADA,
)

# Predicción sin umbral definido: FASE 3B registra el score bruto y
# deja la calibración para cuando existan muestras reales.
VALIDATION_PREDICTION_PENDING = "PENDIENTE_CALIBRACION"
VALIDATION_RESULT_PENDING = "PENDIENTE_CALIBRACION"

VALIDATION_RESULT_CORRECT = "CORRECTO"
VALIDATION_RESULT_INCORRECT = "INCORRECTO"

# Estados de garment_ai_models desde los cuales se admiten casos.
VALIDATION_MODEL_STATUSES = (
    AI_MODEL_STATUS_ENTRENADO,
    AI_MODEL_STATUS_VALIDACION,
    AI_MODEL_STATUS_VALIDADO,
)

VALIDATION_BANK_SUBDIR = "validations"

# Aviso obligatorio en toda captura de validación.
VALIDATION_IMAGE_NOTICE = (
    "Imagen de validación — no se utilizará para entrenamiento."
)


def normalize_validation_category(value) -> str:
    """Normaliza y valida la categoría real de un caso de validación."""
    raw = str(value or "").strip().upper()

    aliases = {
        "HUECO": VALIDATION_CATEGORY_AGUJERO,
        "AGUJEROS": VALIDATION_CATEGORY_AGUJERO,
        "MANCHAS": VALIDATION_CATEGORY_MANCHA,
        "SIN_DEFECTO": VALIDATION_CATEGORY_NORMAL,
        "OK": VALIDATION_CATEGORY_NORMAL,
    }

    raw = aliases.get(raw, raw)

    if raw not in VALIDATION_CATEGORIES:
        raise AIDomainError(
            "Categoría inválida: use NORMAL, MANCHA o AGUJERO."
        )

    return raw


def validate_validation_case_result(category, score, threshold=None) -> str | None:
    """Resultado CORRECTO/INCORRECTO frente a la etiqueta humana.

    Devuelve None mientras no exista umbral definido (pendiente de
    calibración): la categoría real la pone la persona, no el modelo.
    """
    if threshold is None or score is None:
        return None

    predicted_anomaly = float(score) >= float(threshold)
    real_anomaly = str(category) != VALIDATION_CATEGORY_NORMAL

    return (
        VALIDATION_RESULT_CORRECT
        if predicted_anomaly == real_anomaly
        else VALIDATION_RESULT_INCORRECT
    )


def validation_bank_root(root: Path | None = None) -> Path:
    """AI_ARTIFACTS_ROOT/validations (banco independiente del training)."""
    base = Path(root) if root is not None else get_ai_artifacts_root()
    return Path(base) / VALIDATION_BANK_SUBDIR


def validation_case_relative_dir(
    garment_model_id: int,
    ai_model_id: int,
    category: str,
    case_id: int,
) -> str:
    """Ruta relativa del caso: validations/garment_X/model_Y/<cat>/case_Z."""
    try:
        garment_id = int(garment_model_id)
        model_id = int(ai_model_id)
        case = int(case_id)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "garment_model_id, ai_model_id y case_id deben ser enteros."
        ) from error

    if garment_id <= 0 or model_id <= 0 or case <= 0:
        raise AIDomainError(
            "garment_model_id, ai_model_id y case_id deben ser positivos."
        )

    category_key = normalize_validation_category(category)
    folder = VALIDATION_CATEGORY_DIRS[category_key]

    return (
        f"{VALIDATION_BANK_SUBDIR}/garment_{garment_id}/"
        f"model_{model_id}/{folder}/case_{case}"
    )


def validation_case_relative_path(
    garment_model_id: int,
    ai_model_id: int,
    category: str,
    case_id: int,
    filename: str,
) -> str:
    """Ruta relativa de un artefacto del caso (original/heatmap/comparación)."""
    safe_name = ensure_safe_relative_path(str(filename).strip())

    if "/" in safe_name or "\\" in safe_name:
        raise AIDomainError("filename no puede contener separadores.")

    return (
        f"{validation_case_relative_dir(garment_model_id, ai_model_id, category, case_id)}"
        f"/{safe_name}"
    )


# ============================================================
# CÓDIGOS DE MODELO DE PRENDA (alta automática, Fase 2B)
# ============================================================

GARMENT_CODE_PREFIX = "BLUSA"
GARMENT_CODE_WIDTH = 3
GARMENT_CODE_LOCK = "astrid_garment_model_code"
GARMENT_CODE_STATE_TABLE = "garment_model_code_state"

# Solo BLUSA-<n> compite por la serie automática. Códigos históricos con
# otro formato (TEST-AI-001, BLUSA-20240101-101010, ...) quedan intactos.
_GARMENT_CODE_RE_TEMPLATE = r"^{prefix}-([0-9]+)$"


def format_garment_model_code(
    number,
    *,
    prefix: str = GARMENT_CODE_PREFIX,
    width: int = GARMENT_CODE_WIDTH,
) -> str:
    """Formatea BLUSA-001 (relleno a `width` dígitos, sin tope artificial)."""
    try:
        value = int(number)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "El número de código de modelo debe ser entero."
        ) from error

    if value < 1:
        raise AIDomainError(
            "El número de código de modelo debe ser mayor que cero."
        )

    digits = str(value).zfill(int(width)) if value < 10 ** int(width) else str(value)
    return f"{prefix}-{digits}"


def _garment_code_pattern(prefix: str = GARMENT_CODE_PREFIX) -> "re.Pattern":
    safe_prefix = re.escape(str(prefix))
    return re.compile(
        _GARMENT_CODE_RE_TEMPLATE.format(prefix=safe_prefix),
        re.IGNORECASE,
    )


# La preparación de la tabla ejecuta DDL (commit implícito) y toma
# bloqueos: se hace UNA sola vez por proceso, nunca en cada llamada.
_GARMENT_CODE_STATE_READY = False
_GARMENT_CODE_STATE_LOCK = threading.Lock()


def ensure_garment_code_state(cur) -> None:
    """Crea la tabla que recuerda el último código emitido (si no existe)."""
    global _GARMENT_CODE_STATE_READY

    if _GARMENT_CODE_STATE_READY:
        return

    with _GARMENT_CODE_STATE_LOCK:
        if _GARMENT_CODE_STATE_READY:
            return

        try:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {GARMENT_CODE_STATE_TABLE} (
                    singleton TINYINT NOT NULL PRIMARY KEY,
                    last_number INT NOT NULL DEFAULT 0
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                f"""
                INSERT IGNORE INTO {GARMENT_CODE_STATE_TABLE}
                    (singleton, last_number)
                VALUES (1, 0)
                """
            )
            _GARMENT_CODE_STATE_READY = True
        except Exception:
            # Otra sesión pudo crearla al mismo tiempo. Si la tabla
            # realmente no existe, el SELECT posterior falla de forma
            # explícita y se vuelve a intentar en la siguiente llamada.
            pass


def _max_existing_garment_code_number(
    cur,
    prefix: str = GARMENT_CODE_PREFIX,
) -> int:
    """Mayor número de la serie `prefix` presente en garment_models."""
    pattern_re = _garment_code_pattern(prefix)
    cur.execute("SELECT code FROM garment_models")

    maximum = 0
    for row in cur.fetchall():
        match = pattern_re.match(str(_scalar(row) or ""))
        if match:
            maximum = max(maximum, int(match.group(1)))

    return maximum


def next_garment_model_code(
    cur,
    *,
    prefix: str = GARMENT_CODE_PREFIX,
    width: int = GARMENT_CODE_WIDTH,
) -> str:
    """
    Genera el siguiente código de modelo de prenda (BLUSA-001, BLUSA-002...).

    Reglas:
    - Se emite bajo un lock de servidor: nunca dos procesos obtienen el
      mismo código aunque corran en paralelo.
    - Respeta los códigos históricos existentes (de cualquier formato).
    - No reutiliza códigos ya emitidos aunque la fila haya sido eliminada
      (el contador solo avanza).
    - No depende del frontend para garantizar unicidad.
    """
    ensure_garment_code_state(cur)

    cur.execute(
        "SELECT GET_LOCK(%s, 5) AS got",
        (GARMENT_CODE_LOCK,),
    )
    lock_row = cur.fetchone()

    if int(_scalar(lock_row) or 0) != 1:
        raise AIDomainError(
            "No se pudo asignar el código del modelo. "
            "Intente guardar de nuevo."
        )

    try:
        # Lectura bloqueante: si el proceso anterior soltó el lock de
        # servidor pero todavía no confirmó, esperamos su commit para no
        # repetir el mismo número.
        cur.execute(
            f"SELECT last_number FROM {GARMENT_CODE_STATE_TABLE} "
            "WHERE singleton = 1 FOR UPDATE"
        )
        state_row = cur.fetchone()
        last_number = int(_scalar(state_row) or 0)

        candidate = max(
            last_number,
            _max_existing_garment_code_number(cur, prefix=prefix),
        )

        # El lock ya serializa, pero el UNIQUE de garment_models es el
        # respaldo definitivo contra duplicados.
        while True:
            candidate += 1
            code = format_garment_model_code(
                candidate,
                prefix=prefix,
                width=width,
            )
            cur.execute(
                "SELECT id FROM garment_models WHERE code = %s",
                (code,),
            )
            if cur.fetchone() is None:
                break

        cur.execute(
            f"""
            UPDATE {GARMENT_CODE_STATE_TABLE}
            SET last_number = %s
            WHERE singleton = 1
            """,
            (candidate,),
        )
        return code
    finally:
        cur.execute(
            "SELECT RELEASE_LOCK(%s) AS released",
            (GARMENT_CODE_LOCK,),
        )
        cur.fetchone()


# ============================================================
# MÁQUINAS DE ESTADOS — CAPTURA, DATASETS, JOBS
# ============================================================

CAPTURE_STATUS_ABIERTA = "ABIERTA"
CAPTURE_STATUS_COMPLETADA = "COMPLETADA"
CAPTURE_STATUS_CANCELADA = "CANCELADA"

CAPTURE_SESSION_STATUSES = {
    CAPTURE_STATUS_ABIERTA,
    CAPTURE_STATUS_COMPLETADA,
    CAPTURE_STATUS_CANCELADA,
}

CAPTURE_SESSION_TRANSITIONS = {
    CAPTURE_STATUS_ABIERTA: {
        CAPTURE_STATUS_COMPLETADA,
        CAPTURE_STATUS_CANCELADA,
    },
    CAPTURE_STATUS_COMPLETADA: set(),
    CAPTURE_STATUS_CANCELADA: set(),
}

TRAINING_IMAGE_STATUS_ACEPTADA = "ACEPTADA"
TRAINING_IMAGE_STATUS_RECHAZADA = "RECHAZADA"

TRAINING_IMAGE_STATUSES = {
    TRAINING_IMAGE_STATUS_ACEPTADA,
    TRAINING_IMAGE_STATUS_RECHAZADA,
}

DATASET_STATUS_ABIERTO = "ABIERTO"
DATASET_STATUS_CERRADO = "CERRADO"
DATASET_STATUS_ARCHIVADO = "ARCHIVADO"

DATASET_STATUSES = {
    DATASET_STATUS_ABIERTO,
    DATASET_STATUS_CERRADO,
    DATASET_STATUS_ARCHIVADO,
}

DATASET_TRANSITIONS = {
    DATASET_STATUS_ABIERTO: {DATASET_STATUS_CERRADO},
    DATASET_STATUS_CERRADO: {DATASET_STATUS_ARCHIVADO},
    DATASET_STATUS_ARCHIVADO: set(),
}

JOB_KIND_TRAINING = "TRAINING"
JOB_KIND_VALIDATION = "VALIDATION"

JOB_KINDS = {JOB_KIND_TRAINING, JOB_KIND_VALIDATION}

JOB_STATUS_PENDIENTE = "PENDIENTE"
JOB_STATUS_EN_CURSO = "EN_CURSO"
JOB_STATUS_COMPLETADO = "COMPLETADO"
JOB_STATUS_FALLIDO = "FALLIDO"
JOB_STATUS_CANCELADO = "CANCELADO"

JOB_STATUSES = {
    JOB_STATUS_PENDIENTE,
    JOB_STATUS_EN_CURSO,
    JOB_STATUS_COMPLETADO,
    JOB_STATUS_FALLIDO,
    JOB_STATUS_CANCELADO,
}

JOB_TRANSITIONS = {
    JOB_STATUS_PENDIENTE: {JOB_STATUS_EN_CURSO, JOB_STATUS_CANCELADO},
    JOB_STATUS_EN_CURSO: {
        JOB_STATUS_COMPLETADO,
        JOB_STATUS_FALLIDO,
        JOB_STATUS_CANCELADO,
    },
    JOB_STATUS_COMPLETADO: set(),
    JOB_STATUS_FALLIDO: set(),
    JOB_STATUS_CANCELADO: set(),
}

AI_EVENT_TYPES = {
    "CAPTURE_STARTED",
    "CAPTURE_COMPLETED",
    "CAPTURE_CANCELLED",
    "CAPTURE_IMAGE_REVIEWED",
    "DATASET_CLOSED",
    "DATASET_ARCHIVED",
    "TRAINING_STARTED",
    "TRAINING_COMPLETED",
    "TRAINING_FAILED",
    "TRAINING_CANCELLED",
    "VALIDATION_STARTED",
    "VALIDATION_CASE_REGISTERED",
    "VALIDATION_EVALUATED",
    "VALIDATED",
    "REJECTED",
    "ACTIVATED",
    "RETIRED",
    "ROLLBACK",
    "VERSION_PREPARED",
    "JOB_CREATED",
    "NEW_VERSION_TRAINING_REQUESTED",
    "TECHNICAL_INVALIDATION",
}


def validate_capture_status_transition(current: str, new_status: str) -> None:
    current_raw = str(current or "").strip().upper()
    new_raw = str(new_status or "").strip().upper()

    if current_raw not in CAPTURE_SESSION_STATUSES:
        raise AIDomainError(f"Estado de sesión desconocido: {current!r}")

    if new_raw not in CAPTURE_SESSION_STATUSES:
        raise AIDomainError(
            f"Estado de sesión destino desconocido: {new_status!r}"
        )

    if new_raw not in CAPTURE_SESSION_TRANSITIONS.get(current_raw, set()):
        raise AIDomainError(
            f"Transición de sesión no permitida: "
            f"{current_raw} -> {new_raw}."
        )


def validate_dataset_status_transition(current: str, new_status: str) -> None:
    current_raw = str(current or "").strip().upper()
    new_raw = str(new_status or "").strip().upper()

    if current_raw not in DATASET_STATUSES:
        raise AIDomainError(f"Estado de dataset desconocido: {current!r}")

    if new_raw not in DATASET_STATUSES:
        raise AIDomainError(
            f"Estado de dataset destino desconocido: {new_status!r}"
        )

    if new_raw not in DATASET_TRANSITIONS.get(current_raw, set()):
        raise AIDomainError(
            f"Transición de dataset no permitida: "
            f"{current_raw} -> {new_raw}."
        )


def validate_job_status_transition(current: str, new_status: str) -> None:
    current_raw = str(current or "").strip().upper()
    new_raw = str(new_status or "").strip().upper()

    if current_raw not in JOB_STATUSES:
        raise AIDomainError(f"Estado de job desconocido: {current!r}")

    if new_raw not in JOB_STATUSES:
        raise AIDomainError(f"Estado de job destino desconocido: {new_status!r}")

    if new_raw not in JOB_TRANSITIONS.get(current_raw, set()):
        raise AIDomainError(
            f"Transición de job no permitida: "
            f"{current_raw} -> {new_raw}."
        )


def validate_job_kind(kind: str) -> str:
    raw = str(kind or "").strip().upper()

    if raw not in JOB_KINDS:
        raise AIDomainError(f"Tipo de job no soportado: {kind!r}")

    return raw


def validate_event_type(event_type: str) -> str:
    raw = str(event_type or "").strip().upper()

    if raw not in AI_EVENT_TYPES:
        raise AIDomainError(f"Tipo de evento desconocido: {event_type!r}")

    return raw


def compute_manifest_hash(image_rows) -> str:
    """Hash SHA-256 determinista del contenido de un dataset.

    image_rows: iterable de (image_id, sha256) o dicts equivalentes.
    """
    normalized = []

    for row in image_rows or []:
        if isinstance(row, dict):
            image_id = row.get("id", row.get("image_id"))
            digest = row.get("sha256")
        else:
            image_id, digest = row

        if image_id is None:
            raise AIDomainError(
                "Fila de manifest sin image_id."
            )

        normalized.append(
            f"{int(image_id)}:{normalize_sha256(str(digest))}"
        )

    normalized.sort(key=lambda line: int(line.split(":", 1)[0]))
    payload = "\n".join(normalized)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_manifest_path(
    garment_model_id: int,
    dataset_id: int,
) -> str:
    """Ruta relativa (bajo la raíz IA) del manifest de un dataset."""
    model_id = int(garment_model_id)
    ds_id = int(dataset_id)

    if model_id <= 0 or ds_id <= 0:
        raise AIDomainError(
            "garment_model_id y dataset_id deben ser positivos."
        )

    return (
        f"garment_{model_id}/datasets/dataset_{ds_id}/manifest.json"
    )


# ============================================================
# PERSISTENCIA (cursor abierto; la transacción la maneja el llamador)
# ============================================================

def _ensure_column_fallback(cur, db_name, table_name, column_name, definition):
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (db_name, table_name, column_name),
    )

    if int(_scalar(cur.fetchone()) or 0) == 0:
        cur.execute(
            f"ALTER TABLE `{table_name}` ADD COLUMN {definition}"
        )


def _ensure_index(
    cur,
    db_name,
    table_name,
    index_name,
    definition,
    unique=False,
):
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND INDEX_NAME = %s
        """,
        (db_name, table_name, index_name),
    )

    if int(_scalar(cur.fetchone()) or 0) == 0:
        keyword = "UNIQUE INDEX" if unique else "INDEX"
        cur.execute(
            f"ALTER TABLE `{table_name}` "
            f"ADD {keyword} `{index_name}` {definition}"
        )


def _ensure_foreign_key(
    cur,
    db_name,
    table_name,
    constraint_name,
    definition,
):
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.REFERENTIAL_CONSTRAINTS
        WHERE CONSTRAINT_SCHEMA = %s
          AND TABLE_NAME = %s
          AND CONSTRAINT_NAME = %s
        """,
        (db_name, table_name, constraint_name),
    )

    if int(_scalar(cur.fetchone()) or 0) == 0:
        cur.execute(
            f"ALTER TABLE `{table_name}` "
            f"ADD CONSTRAINT `{constraint_name}` {definition}"
        )


def _ensure_trigger(cur, db_name, trigger_name, definition):
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.TRIGGERS
        WHERE TRIGGER_SCHEMA = %s
          AND TRIGGER_NAME = %s
        """,
        (db_name, trigger_name),
    )

    if int(_scalar(cur.fetchone()) or 0) > 0:
        return

    # No se ejecuta SET GLOBAL: log_bin_trust_function_creators se
    # fija solo en infraestructura (docker-compose de desarrollo).
    # El usuario de la aplicación no necesita privilegios admin.
    try:
        cur.execute(definition)
    except Exception as error:
        errno = getattr(error, "errno", None)

        if errno == 1419:
            raise AIDomainError(
                "MySQL no permite crear triggers (binary logging "
                "sin log_bin_trust_function_creators). Configure "
                "ese parámetro en el servidor MySQL de desarrollo "
                "o infraestructura equivalente."
            ) from error

        raise


def ensure_ai_schema(cur, db_name, ensure_column=None):
    """Crea/extendiendo el esquema de Fase 1 de forma idempotente.

    No toca imágenes de catálogo (garment_model_images) ni datos
    productivos existentes.
    """
    add_column = ensure_column or (
        lambda table_name, column_name, definition: _ensure_column_fallback(
            cur, db_name, table_name, column_name, definition
        )
    )

    # --------------------------------------------------------
    # SESIONES DE CAPTURA (preparación; sin cámara en Fase 1)
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_capture_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            garment_model_id INT NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'ABIERTA',
            started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at DATETIME NULL,
            created_by INT NULL,
            notes TEXT NULL,

            KEY idx_ai_capture_garment (garment_model_id),
            KEY idx_ai_capture_status (status),

            CONSTRAINT fk_ai_capture_garment
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_capture_user
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # --------------------------------------------------------
    # IMÁGENES DE ENTRENAMIENTO (independientes del catálogo)
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_training_images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            capture_session_id INT NULL,
            garment_model_id INT NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            sha256 CHAR(64) NOT NULL,
            captured_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            frame_sequence INT NULL,
            coverage DECIMAL(6,4) NULL,
            quality_score DECIMAL(8,3) NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'ACEPTADA',
            reject_reason VARCHAR(255) NULL,

            KEY idx_ai_train_img_garment (garment_model_id),
            KEY idx_ai_train_img_session (capture_session_id),
            KEY idx_ai_train_img_status (status),
            KEY idx_ai_train_img_sha256 (sha256),

            CONSTRAINT fk_ai_train_img_garment
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_train_img_session
                FOREIGN KEY (capture_session_id)
                REFERENCES ai_capture_sessions(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # --------------------------------------------------------
    # DATASETS INMUTABLES
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_datasets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            garment_model_id INT NOT NULL,
            version VARCHAR(40) NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'ABIERTO',
            image_count INT NOT NULL DEFAULT 0,
            manifest_path VARCHAR(500) NULL,
            manifest_hash CHAR(64) NULL,
            created_by INT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            closed_at DATETIME NULL,

            UNIQUE KEY uq_ai_dataset_version (
                garment_model_id,
                version
            ),
            KEY idx_ai_dataset_garment (garment_model_id),
            KEY idx_ai_dataset_status (status),

            CONSTRAINT fk_ai_dataset_garment
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_dataset_user
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # --------------------------------------------------------
    # RELACIÓN DATASET <-> IMÁGENES (opción B: tabla intermedia)
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_dataset_images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            dataset_id INT NOT NULL,
            image_id INT NOT NULL,
            added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            UNIQUE KEY uq_ai_dataset_image (dataset_id, image_id),
            KEY idx_ai_dimg_image (image_id),

            CONSTRAINT fk_ai_dimg_dataset
                FOREIGN KEY (dataset_id)
                REFERENCES ai_datasets(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_dimg_image
                FOREIGN KEY (image_id)
                REFERENCES ai_training_images(id)
                ON DELETE RESTRICT
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # --------------------------------------------------------
    # JOBS PERSISTENTES (sin worker en Fase 1)
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_jobs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ai_model_id INT NULL,
            dataset_id INT NULL,
            kind VARCHAR(30) NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'PENDIENTE',
            progress DECIMAL(5,2) NOT NULL DEFAULT 0,
            log_path VARCHAR(500) NULL,
            error_message TEXT NULL,
            created_by INT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME NULL,
            finished_at DATETIME NULL,

            KEY idx_ai_jobs_model (ai_model_id),
            KEY idx_ai_jobs_dataset (dataset_id),
            KEY idx_ai_jobs_status (status),
            KEY idx_ai_jobs_kind (kind),

            CONSTRAINT fk_ai_jobs_model
                FOREIGN KEY (ai_model_id)
                REFERENCES garment_ai_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_jobs_dataset
                FOREIGN KEY (dataset_id)
                REFERENCES ai_datasets(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_jobs_user
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # --------------------------------------------------------
    # CONTROL DE EJECUCIÓN DE JOBS (FASE 3A)
    # MySQL es la fuente de verdad: progreso, etapa, worker y
    # heartbeat sobreviven a reinicios de Flask y del worker.
    # --------------------------------------------------------
    add_column(
        "ai_jobs",
        "stage",
        "stage VARCHAR(60) NULL AFTER log_path",
    )
    add_column(
        "ai_jobs",
        "stage_label",
        "stage_label VARCHAR(160) NULL AFTER stage",
    )
    add_column(
        "ai_jobs",
        "worker_id",
        "worker_id VARCHAR(120) NULL AFTER stage_label",
    )
    add_column(
        "ai_jobs",
        "heartbeat_at",
        "heartbeat_at DATETIME NULL AFTER worker_id",
    )
    add_column(
        "ai_jobs",
        "config_json",
        "config_json JSON NULL AFTER progress",
    )
    add_column(
        "ai_jobs",
        "artifacts_json",
        "artifacts_json JSON NULL AFTER config_json",
    )
    add_column(
        "ai_jobs",
        "updated_at",
        "updated_at DATETIME NULL AFTER finished_at",
    )

    # Un solo job TRAINING activo por versión de IA: la restricción
    # vive en la BD (doble clic o dos peticiones no la evitan).
    # NULL permite varios jobs terminados (reintentos con versión nueva).
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = 'ai_jobs'
          AND COLUMN_NAME = 'training_active_key'
        """,
        (db_name,),
    )

    if int(_scalar(cur.fetchone()) or 0) == 0:
        cur.execute(
            """
            ALTER TABLE ai_jobs
            ADD COLUMN training_active_key INT
            GENERATED ALWAYS AS (
                IF(kind = 'TRAINING'
                   AND status IN ('PENDIENTE', 'EN_CURSO'),
                   ai_model_id,
                   NULL)
            ) STORED
            """
        )

    _ensure_index(
        cur,
        db_name,
        "ai_jobs",
        "uq_ai_jobs_one_training",
        "(training_active_key)",
        unique=True,
    )

    # --------------------------------------------------------
    # AUDITORÍA PERSISTENTE DE IA (no existe AuditEvent previo)
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_events (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ai_model_id INT NULL,
            dataset_id INT NULL,
            capture_session_id INT NULL,
            actor_id INT NULL,
            event_type VARCHAR(50) NOT NULL,
            payload_json JSON NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            KEY idx_ai_events_model (ai_model_id),
            KEY idx_ai_events_dataset (dataset_id),
            KEY idx_ai_events_session (capture_session_id),
            KEY idx_ai_events_actor (actor_id),
            KEY idx_ai_events_type (event_type),
            KEY idx_ai_events_created (created_at),

            CONSTRAINT fk_ai_events_model
                FOREIGN KEY (ai_model_id)
                REFERENCES garment_ai_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_events_dataset
                FOREIGN KEY (dataset_id)
                REFERENCES ai_datasets(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_events_session
                FOREIGN KEY (capture_session_id)
                REFERENCES ai_capture_sessions(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_events_actor
                FOREIGN KEY (actor_id)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # --------------------------------------------------------
    # EXTENSIONES DE garment_ai_models (solo lo que falta)
    # --------------------------------------------------------
    add_column(
        "garment_ai_models",
        "checkpoint_hash",
        "checkpoint_hash CHAR(64) NULL AFTER checkpoint_path",
    )
    add_column(
        "garment_ai_models",
        "input_size",
        "input_size VARCHAR(30) NULL AFTER checkpoint_hash",
    )
    add_column(
        "garment_ai_models",
        "image_threshold",
        "image_threshold DECIMAL(6,3) NULL AFTER input_size",
    )
    add_column(
        "garment_ai_models",
        "dataset_id",
        "dataset_id INT NULL AFTER dataset_path",
    )
    add_column(
        "garment_ai_models",
        "retired_at",
        "retired_at DATETIME NULL AFTER activated_at",
    )

    _ensure_index(
        cur,
        db_name,
        "garment_ai_models",
        "idx_garment_ai_dataset",
        "(dataset_id)",
    )
    _ensure_foreign_key(
        cur,
        db_name,
        "garment_ai_models",
        "fk_garment_ai_dataset",
        "FOREIGN KEY (dataset_id) REFERENCES ai_datasets(id) "
        "ON DELETE RESTRICT",
    )

    # --------------------------------------------------------
    # UN SOLO ACTIVO POR MODELO DE PRENDA (constraint de BD)
    # active_key = garment_model_id si status='ACTIVO', NULL en
    # otro caso; el índice UNIQUE acepta múltiples NULL.
    # --------------------------------------------------------
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = 'garment_ai_models'
          AND COLUMN_NAME = 'active_key'
        """,
        (db_name,),
    )

    if int(_scalar(cur.fetchone()) or 0) == 0:
        cur.execute(
            """
            ALTER TABLE garment_ai_models
            ADD COLUMN active_key INT
            GENERATED ALWAYS AS (
                IF(status = 'ACTIVO', garment_model_id, NULL)
            ) STORED
            """
        )

    cur.execute(
        """
        SELECT garment_model_id, COUNT(*) AS total
        FROM garment_ai_models
        WHERE status = 'ACTIVO'
        GROUP BY garment_model_id
        HAVING total > 1
        """
    )

    duplicated_active = cur.fetchall()

    if duplicated_active:
        # No se limpia, no se elige un ganador y no se omite el índice
        # en silencio: la instalación queda bloqueada hasta resolver.
        conflicted = ", ".join(
            str(row["garment_model_id"])
            if isinstance(row, dict)
            else str(row[0])
            for row in duplicated_active
        )
        raise AIDomainError(
            "CONFLICTO ACTIVE: existen múltiples versiones ACTIVO "
            "para garment_model_id="
            f"{conflicted}. No se creó uq_garment_ai_one_active. "
            "Resuelva el conflicto manualmente (sin borrar historia) "
            "y vuelva a inicializar."
        )

    _ensure_index(
        cur,
        db_name,
        "garment_ai_models",
        "uq_garment_ai_one_active",
        "(active_key)",
        unique=True,
    )

    # --------------------------------------------------------
    # INMUTABILIDAD DE DATASETS CERRADOS (constraints de BD)
    # --------------------------------------------------------
    _ensure_trigger(
        cur,
        db_name,
        "trg_ai_dataset_images_before_insert",
        """
        CREATE TRIGGER trg_ai_dataset_images_before_insert
        BEFORE INSERT ON ai_dataset_images
        FOR EACH ROW
        BEGIN
            DECLARE v_status VARCHAR(30);
            SELECT status INTO v_status
            FROM ai_datasets
            WHERE id = NEW.dataset_id;

            IF v_status IS NULL OR v_status <> 'ABIERTO' THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset no ABIERTO: no admite nuevas imagenes';
            END IF;
        END
        """,
    )
    _ensure_trigger(
        cur,
        db_name,
        "trg_ai_dataset_images_before_update",
        """
        CREATE TRIGGER trg_ai_dataset_images_before_update
        BEFORE UPDATE ON ai_dataset_images
        FOR EACH ROW
        BEGIN
            DECLARE v_status VARCHAR(30);
            SELECT status INTO v_status
            FROM ai_datasets
            WHERE id = OLD.dataset_id;

            IF v_status IS NULL OR v_status <> 'ABIERTO' THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset no ABIERTO: relacion inmutable';
            END IF;
        END
        """,
    )
    _ensure_trigger(
        cur,
        db_name,
        "trg_ai_dataset_images_before_delete",
        """
        CREATE TRIGGER trg_ai_dataset_images_before_delete
        BEFORE DELETE ON ai_dataset_images
        FOR EACH ROW
        BEGIN
            DECLARE v_status VARCHAR(30);
            SELECT status INTO v_status
            FROM ai_datasets
            WHERE id = OLD.dataset_id;

            IF v_status IS NULL OR v_status <> 'ABIERTO' THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset no ABIERTO: no admite eliminaciones';
            END IF;
        END
        """,
    )

    # No reabrir datasets CERRADOS/ARCHIVADOS ni alterar su snapshot.
    _ensure_trigger(
        cur,
        db_name,
        "trg_ai_datasets_before_update",
        """
        CREATE TRIGGER trg_ai_datasets_before_update
        BEFORE UPDATE ON ai_datasets
        FOR EACH ROW
        BEGIN
            IF OLD.status IN ('CERRADO', 'ARCHIVADO')
               AND NEW.status = 'ABIERTO' THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset cerrado no puede reabrirse';
            END IF;

            IF OLD.status = 'CERRADO'
               AND NEW.status NOT IN ('CERRADO', 'ARCHIVADO') THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset cerrado solo puede archivarse';
            END IF;

            IF OLD.status = 'ARCHIVADO'
               AND NEW.status <> 'ARCHIVADO' THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset archivado es terminal';
            END IF;

            IF OLD.status IN ('CERRADO', 'ARCHIVADO')
               AND NOT (
                   OLD.garment_model_id <=> NEW.garment_model_id
                   AND OLD.version <=> NEW.version
                   AND OLD.image_count <=> NEW.image_count
                   AND OLD.manifest_path <=> NEW.manifest_path
                   AND OLD.manifest_hash <=> NEW.manifest_hash
                   AND OLD.closed_at <=> NEW.closed_at
                   AND OLD.created_by <=> NEW.created_by
                   AND OLD.created_at <=> NEW.created_at
               ) THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Dataset cerrado: snapshot inmutable';
            END IF;
        END
        """,
    )

    # Imágenes que ya pertenecen a un dataset CLOSED no cambian ni se
    # eliminan (la representación reproducible del manifest queda fija).
    _ensure_trigger(
        cur,
        db_name,
        "trg_ai_training_images_before_update",
        """
        CREATE TRIGGER trg_ai_training_images_before_update
        BEFORE UPDATE ON ai_training_images
        FOR EACH ROW
        BEGIN
            DECLARE v_closed INT DEFAULT 0;

            SELECT COUNT(*) INTO v_closed
            FROM ai_dataset_images di
            JOIN ai_datasets ds ON ds.id = di.dataset_id
            WHERE di.image_id = OLD.id
              AND ds.status <> 'ABIERTO'
            LIMIT 1;

            IF v_closed > 0 THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Imagen en dataset cerrado: inmutable';
            END IF;
        END
        """,
    )
    _ensure_trigger(
        cur,
        db_name,
        "trg_ai_training_images_before_delete",
        """
        CREATE TRIGGER trg_ai_training_images_before_delete
        BEFORE DELETE ON ai_training_images
        FOR EACH ROW
        BEGIN
            DECLARE v_closed INT DEFAULT 0;

            SELECT COUNT(*) INTO v_closed
            FROM ai_dataset_images di
            JOIN ai_datasets ds ON ds.id = di.dataset_id
            WHERE di.image_id = OLD.id
              AND ds.status <> 'ABIERTO'
            LIMIT 1;

            IF v_closed > 0 THEN
                SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT =
                    'Imagen en dataset cerrado: no se elimina';
            END IF;
        END
        """,
    )

    # --------------------------------------------------------
    # FASE 3B — SESIONES Y CASOS DE VALIDACIÓN
    # Banco de imágenes de validación separado del training: una
    # imagen de validación jamás se registra en ai_training_images
    # ni en ai_dataset_images.
    # --------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_validation_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            garment_model_id INT NOT NULL,
            ai_model_id INT NOT NULL,
            status VARCHAR(30) NOT NULL DEFAULT 'ABIERTA',
            threshold_candidate DECIMAL(8,3) NULL,
            metrics_json JSON NULL,
            created_by INT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            evaluated_at DATETIME NULL,
            closed_at DATETIME NULL,

            KEY idx_ai_val_session_garment (garment_model_id),
            KEY idx_ai_val_session_model (ai_model_id),
            KEY idx_ai_val_session_status (status),

            CONSTRAINT fk_ai_val_session_garment
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_val_session_model
                FOREIGN KEY (ai_model_id)
                REFERENCES garment_ai_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_val_session_user
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_validation_cases (
            id INT AUTO_INCREMENT PRIMARY KEY,
            validation_session_id INT NOT NULL,
            ai_model_id INT NOT NULL,
            garment_model_id INT NOT NULL,
            category VARCHAR(20) NOT NULL,
            image_path VARCHAR(500) NOT NULL,
            image_sha256 CHAR(64) NOT NULL,
            anomaly_score DECIMAL(10,4) NULL,
            threshold_used DECIMAL(8,3) NULL,
            prediction VARCHAR(30) NULL,
            result VARCHAR(30) NULL,
            observation TEXT NULL,
            heatmap_path VARCHAR(500) NULL,
            comparison_path VARCHAR(500) NULL,
            created_by INT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            KEY idx_ai_val_case_session (validation_session_id),
            KEY idx_ai_val_case_model (ai_model_id),
            KEY idx_ai_val_case_garment (garment_model_id),
            KEY idx_ai_val_case_category (category),
            KEY idx_ai_val_case_sha256 (image_sha256),

            CONSTRAINT fk_ai_val_case_session
                FOREIGN KEY (validation_session_id)
                REFERENCES ai_validation_sessions(id)
                ON DELETE CASCADE,

            CONSTRAINT fk_ai_val_case_model
                FOREIGN KEY (ai_model_id)
                REFERENCES garment_ai_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_val_case_garment
                FOREIGN KEY (garment_model_id)
                REFERENCES garment_models(id)
                ON DELETE RESTRICT,

            CONSTRAINT fk_ai_val_case_user
                FOREIGN KEY (created_by)
                REFERENCES users(id)
                ON DELETE SET NULL
        ) ENGINE=InnoDB
          DEFAULT CHARSET=utf8mb4
          COLLATE=utf8mb4_unicode_ci
        """
    )

    # «PENDIENTE_CALIBRACION» mide 21 caracteres: una primera versión de
    # FASE 3B creó la columna con VARCHAR(20) y se amplía aquí.
    cur.execute(
        """
        SELECT CHARACTER_MAXIMUM_LENGTH
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = 'ai_validation_cases'
          AND COLUMN_NAME = 'result'
        """,
        (db_name,),
    )

    if int(_scalar(cur.fetchone()) or 0) < 30:
        cur.execute(
            "ALTER TABLE ai_validation_cases "
            "MODIFY COLUMN result VARCHAR(30) NULL"
        )

    # Raíz conceptual de artefactos (sin escribir checkpoints).
    root = get_ai_artifacts_root()
    root.mkdir(parents=True, exist_ok=True)

    for name in AI_ARTIFACT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# HELPERS DE DOMINIO PERSISTENTES
# ------------------------------------------------------------

def _lock_garment_model(cur, garment_model_id: int) -> None:
    cur.execute(
        "SELECT id FROM garment_models WHERE id = %s FOR UPDATE",
        (garment_model_id,),
    )

    if cur.fetchone() is None:
        raise AIDomainError(
            f"El modelo de prenda {garment_model_id} no existe."
        )


def _fetch_ai_model_for_update(cur, ai_model_id: int) -> dict:
    cur.execute(
        """
        SELECT
            id,
            garment_model_id,
            version,
            status,
            active,
            notes
        FROM garment_ai_models
        WHERE id = %s
        FOR UPDATE
        """,
        (ai_model_id,),
    )

    row = cur.fetchone()

    if row is None:
        raise AIDomainError(
            f"La versión de IA {ai_model_id} no existe."
        )

    return row


def create_next_ai_model_version(
    cur,
    garment_model_id: int,
    actor_id: int | None,
    *,
    model_type: str = "PatchCore",
    dataset_name: str | None = None,
    dataset_path: str | None = None,
) -> dict:
    """Crea la siguiente versión vN en PREPARACION con bloqueo del modelo.

    Debe ejecutarse dentro de una transacción abierta por el llamador.
    """
    _lock_garment_model(cur, garment_model_id)

    cur.execute(
        """
        SELECT id, version, status, notes
        FROM garment_ai_models
        WHERE garment_model_id = %s
        ORDER BY id ASC
        """,
        (garment_model_id,),
    )

    versions = [dict(row) for row in cur.fetchall()]

    # Una versión invalidada técnicamente es histórica: no activable y
    # tampoco impide entrenar una versión nueva que la reemplace.
    pending = [
        item
        for item in versions
        if item["status"] in AI_MODEL_PENDING_STATUSES
        and not is_technically_invalidated(item.get("notes"))
    ]

    if pending:
        raise AIDomainError(
            "Ya existe una versión de IA en proceso para este modelo."
        )

    version = next_version_label(versions)

    cur.execute(
        """
        INSERT INTO garment_ai_models (
            garment_model_id,
            version,
            model_type,
            dataset_name,
            dataset_path,
            checkpoint_path,
            status,
            normal_images_count,
            notes,
            created_by,
            active
        )
        VALUES (%s, %s, %s, %s, %s, NULL, 'PREPARACION', 0, NULL, %s, 0)
        """,
        (
            garment_model_id,
            version,
            model_type,
            dataset_name,
            dataset_path,
            actor_id,
        ),
    )

    ai_model_id = cur.lastrowid

    return {
        "id": ai_model_id,
        "garment_model_id": garment_model_id,
        "version": version,
        "status": AI_MODEL_STATUS_PREPARACION,
    }


def transition_ai_model_status(
    cur,
    ai_model_id: int,
    new_status: str,
    actor_id: int | None,
    *,
    notes: str | None = None,
) -> dict:
    """Transición validada de estado de una versión de IA.

    Debe ejecutarse dentro de una transacción abierta por el llamador.
    """
    new_raw = str(new_status or "").strip().upper()
    row = _fetch_ai_model_for_update(cur, ai_model_id)

    validate_ai_status_transition(row["status"], new_raw)

    # Una versión invalidada técnicamente nunca vuelve a entrar al
    # circuito de validación ni a activarse: es histórica.
    if new_raw in (
        AI_MODEL_STATUS_VALIDACION,
        AI_MODEL_STATUS_ACTIVO,
    ) and is_technically_invalidated(row.get("notes")):
        raise AIDomainError(
            "Esta versión está marcada NO VALIDADA / NO APTO PARA "
            "ACTIVACIÓN. Entrene una versión nueva."
        )

    _lock_garment_model(cur, row["garment_model_id"])

    sets = ["status = %s"]
    params = [new_raw]

    if new_raw == AI_MODEL_STATUS_ENTRENADO:
        sets.append("trained_by = %s")
        params.append(actor_id)
        sets.append("trained_at = NOW()")

    elif new_raw == AI_MODEL_STATUS_VALIDADO:
        sets.append("validated_by = %s")
        params.append(actor_id)
        sets.append("validated_at = NOW()")

    elif new_raw == AI_MODEL_STATUS_ACTIVO:
        # Si esta configuración ya tiene conflicto histórico de ACTIVE
        # (2+), no se activa ni se retira nada automáticamente.
        cur.execute(
            """
            SELECT id, version
            FROM garment_ai_models
            WHERE garment_model_id = %s
              AND status = 'ACTIVO'
              AND id <> %s
            """,
            (row["garment_model_id"], ai_model_id),
        )
        other_active = cur.fetchall()

        if len(other_active) > 1:
            versions = ", ".join(
                str(item["version"])
                if isinstance(item, dict)
                else str(item[1])
                for item in other_active
            )
            raise AIDomainError(
                "CONFLICTO ACTIVE: la configuración "
                f"garment_model_id={row['garment_model_id']} ya tiene "
                f"múltiples versiones ACTIVO ({versions}). "
                "Actuación bloqueada hasta resolver el conflicto."
            )

        cur.execute(
            """
            UPDATE garment_ai_models
            SET
                status = 'RETIRADO',
                active = 0,
                retired_at = NOW()
            WHERE garment_model_id = %s
              AND status = 'ACTIVO'
              AND id <> %s
            """,
            (row["garment_model_id"], ai_model_id),
        )
        sets.append("active = 1")
        sets.append("activated_by = %s")
        params.append(actor_id)
        sets.append("activated_at = NOW()")
        sets.append("retired_at = NULL")

    elif new_raw == AI_MODEL_STATUS_RETIRADO:
        sets.append("active = 0")
        sets.append("retired_at = NOW()")

    elif new_raw == AI_MODEL_STATUS_RECHAZADO:
        sets.append("validated_by = %s")
        params.append(actor_id)
        sets.append("validated_at = NOW()")

    if notes is not None:
        sets.append("notes = %s")
        params.append(notes)

    params.append(ai_model_id)

    cur.execute(
        f"""
        UPDATE garment_ai_models
        SET {', '.join(sets)}
        WHERE id = %s
        """,
        tuple(params),
    )

    return {
        "id": ai_model_id,
        "garment_model_id": row["garment_model_id"],
        "version": row["version"],
        "status": new_raw,
    }


def annotate_technical_invalidation(
    cur,
    ai_model_id: int,
    reason: str,
    *,
    actor_id: int | None = None,
) -> dict:
    """Marca una versión como NO VALIDADA / NO APTA PARA ACTIVACIÓN.

    Usa los mecanismos de dominio existentes: la columna notes de
    garment_ai_models y el registro de eventos. No cambia el estado
    (no existe transición de invalidación técnica) y se niega a tocar
    una versión que ya esté ACTIVO.
    """
    row = _fetch_ai_model_for_update(cur, ai_model_id)

    if row["status"] == AI_MODEL_STATUS_ACTIVO:
        raise AIDomainError(
            "No se puede invalidar técnicamente una versión ACTIVO; "
            "retírela primero."
        )

    detail = " ".join(str(reason or "").split())[:400]
    notes = f"{AI_MODEL_TECHNICAL_INVALIDATION_PREFIX}: {detail}"

    cur.execute(
        "UPDATE garment_ai_models SET notes = %s WHERE id = %s",
        (notes, int(ai_model_id)),
    )

    record_ai_event(
        cur,
        "TECHNICAL_INVALIDATION",
        actor_id=actor_id,
        ai_model_id=int(ai_model_id),
        payload={
            "notes": notes,
            "previous_status": row["status"],
            "version": row["version"],
        },
    )

    return {
        "id": int(ai_model_id),
        "garment_model_id": row["garment_model_id"],
        "version": row["version"],
        "status": row["status"],
        "notes": notes,
    }


def open_ai_capture_session(
    cur,
    garment_model_id: int,
    actor_id: int | None,
    *,
    notes: str | None = None,
) -> dict:
    """Abre una sesión de captura (una ABIERTA por modelo)."""
    _lock_garment_model(cur, garment_model_id)

    # Exclusión a nivel estación: solo una ABIERTA en todo el station.
    cur.execute(
        """
        SELECT id, garment_model_id
        FROM ai_capture_sessions
        WHERE status = 'ABIERTA'
        LIMIT 1
        """
    )
    station_open = cur.fetchone()

    if station_open:
        raise AIDomainError(
            "Ya existe una sesión de captura ABIERTA en la estación "
            f"(id={station_open['id']}, "
            f"garment_model_id={station_open['garment_model_id']})."
        )

    cur.execute(
        """
        INSERT INTO ai_capture_sessions (
            garment_model_id,
            status,
            created_by,
            notes
        )
        VALUES (%s, 'ABIERTA', %s, %s)
        """,
        (garment_model_id, actor_id, notes),
    )

    session_id = cur.lastrowid

    return {
        "id": session_id,
        "garment_model_id": garment_model_id,
        "status": CAPTURE_STATUS_ABIERTA,
    }


def get_open_capture_session(cur) -> dict | None:
    """Sesión ABIERTA de la estación (si existe)."""
    cur.execute(
        """
        SELECT id, garment_model_id, status, started_at, created_by, notes
        FROM ai_capture_sessions
        WHERE status = 'ABIERTA'
        ORDER BY id DESC
        LIMIT 1
        """
    )
    return cur.fetchone()


def get_capture_session(cur, capture_session_id: int) -> dict | None:
    cur.execute(
        """
        SELECT
            id,
            garment_model_id,
            status,
            started_at,
            finished_at,
            created_by,
            notes
        FROM ai_capture_sessions
        WHERE id = %s
        """,
        (capture_session_id,),
    )
    return cur.fetchone()


def start_ai_capture_session(
    cur,
    garment_model_id: int,
    actor_id: int | None,
    *,
    notes: str | None = None,
    config: dict | None = None,
) -> dict:
    """Inicia sesión de captura en la estación (idempotente si ya ABIERTA igual)."""
    cfg = dict(config or get_ai_capture_config())

    # Lock a nivel servidor: evita dos sesiones ABIERTA concurrentes.
    cur.execute("SELECT GET_LOCK(%s, 5) AS got", ("astrid_ai_capture_station",))
    lock_row = cur.fetchone()
    got = int(_scalar(lock_row) or 0)
    if got != 1:
        raise AIDomainError(
            "No se pudo adquirir el lock de captura de la estación."
        )

    try:
        existing = get_open_capture_session(cur)

        if existing is not None:
            if int(existing["garment_model_id"]) == int(garment_model_id):
                return {
                    "id": int(existing["id"]),
                    "garment_model_id": int(existing["garment_model_id"]),
                    "status": existing["status"],
                    "started_at": existing["started_at"],
                    "already_open": True,
                    "target_images": int(cfg["target_images"]),
                }
            raise AIDomainError(
                "Ya hay una sesión de captura activa en la estación "
                f"para garment_model_id={existing['garment_model_id']}."
            )

        session = open_ai_capture_session(
            cur,
            garment_model_id,
            actor_id,
            notes=notes,
        )
        session["already_open"] = False
        session["target_images"] = int(cfg["target_images"])
        session["started_at"] = None
        return session
    finally:
        cur.execute(
            "SELECT RELEASE_LOCK(%s) AS released",
            ("astrid_ai_capture_station",),
        )
        cur.fetchone()


def stop_ai_capture_session(
    cur,
    capture_session_id: int,
) -> dict:
    """Finaliza sesión (idempotente si ya COMPLETADA)."""
    row = get_capture_session(cur, capture_session_id)

    if row is None:
        raise AIDomainError(
            f"La sesión de captura {capture_session_id} no existe."
        )

    counts = count_session_images(cur, capture_session_id)
    minimum = int(get_ai_capture_config()["min_images"])
    accepted = int(counts.get("accepted_images") or 0)
    if accepted < minimum:
        missing = minimum - accepted
        raise AIDomainError(
            "No se puede finalizar la preparación: "
            f"faltan {missing} imágenes válidas para alcanzar el mínimo de {minimum}."
        )

    if row["status"] == CAPTURE_STATUS_COMPLETADA:
        return {
            "id": int(row["id"]),
            "garment_model_id": int(row["garment_model_id"]),
            "status": row["status"],
            "finished_at": row["finished_at"],
            "already_finished": True,
        }

    result = transition_capture_session_status(
        cur,
        capture_session_id,
        CAPTURE_STATUS_COMPLETADA,
    )
    result["already_finished"] = False
    return result


def cancel_ai_capture_session(
    cur,
    capture_session_id: int,
) -> dict:
    """Cancela sesión (idempotente si ya CANCELADA)."""
    row = get_capture_session(cur, capture_session_id)

    if row is None:
        raise AIDomainError(
            f"La sesión de captura {capture_session_id} no existe."
        )

    if row["status"] == CAPTURE_STATUS_CANCELADA:
        return {
            "id": int(row["id"]),
            "garment_model_id": int(row["garment_model_id"]),
            "status": row["status"],
            "finished_at": row["finished_at"],
            "already_cancelled": True,
        }

    result = transition_capture_session_status(
        cur,
        capture_session_id,
        CAPTURE_STATUS_CANCELADA,
    )
    result["already_cancelled"] = False
    return result


def claim_frame_sequence(
    cur,
    capture_session_id: int,
    frame_sequence: int,
) -> None:
    """Reserva frame_sequence en la sesión (único por sesión)."""
    try:
        seq = int(frame_sequence)
    except (TypeError, ValueError) as error:
        raise AIDomainError(
            "frame_sequence debe ser un entero."
        ) from error

    cur.execute(
        """
        SELECT id
        FROM ai_training_images
        WHERE capture_session_id = %s
          AND frame_sequence = %s
        LIMIT 1
        """,
        (capture_session_id, seq),
    )

    if cur.fetchone() is not None:
        raise AIDomainError(
            f"frame_sequence {seq} ya fue persistido en esta sesión."
        )


def find_accepted_sha256(
    cur,
    garment_model_id: int,
    sha256: str,
) -> dict | None:
    """Imagen ACEPTADA con el mismo SHA-256 para el modelo."""
    digest = normalize_sha256(sha256)
    cur.execute(
        """
        SELECT id, capture_session_id, image_path, sha256
        FROM ai_training_images
        WHERE garment_model_id = %s
          AND sha256 = %s
          AND status = %s
        LIMIT 1
        """,
        (
            garment_model_id,
            digest,
            TRAINING_IMAGE_STATUS_ACEPTADA,
        ),
    )
    return cur.fetchone()


def load_session_accepted_hashes(
    cur,
    capture_session_id: int,
) -> dict:
    """SHA-256 y (si se usara) conjunto de hashes aceptados de la sesión."""
    cur.execute(
        """
        SELECT sha256
        FROM ai_training_images
        WHERE capture_session_id = %s
          AND status = %s
        """,
        (
            capture_session_id,
            TRAINING_IMAGE_STATUS_ACEPTADA,
        ),
    )
    sha256_set = {
        str(row["sha256"]).lower()
        for row in cur.fetchall()
    }
    return {"sha256": sha256_set}


def capture_session_status_payload(
    cur,
    capture_session_id: int,
    *,
    config: dict | None = None,
    last_capture=None,
) -> dict:
    """Payload de progreso para endpoints de estado."""
    cfg = dict(config or get_ai_capture_config())
    row = get_capture_session(cur, capture_session_id)

    if row is None:
        raise AIDomainError(
            f"La sesión de captura {capture_session_id} no existe."
        )

    counts = count_session_images(cur, capture_session_id)

    cur.execute(
        """
        SELECT id, code
        FROM garment_models
        WHERE id = %s
        """,
        (row["garment_model_id"],),
    )
    garment = cur.fetchone() or {
        "id": row["garment_model_id"],
        "code": None,
    }

    return {
        "session_id": int(row["id"]),
        "garment_model": {
            "id": int(garment["id"]),
            "code": garment.get("code"),
        },
        "status": row["status"],
        "accepted_count": int(counts["accepted_images"]),
        "rejected_count": int(counts["rejected_images"]),
        "target_count": int(cfg["target_images"]),
        "min_count": int(cfg["min_images"]),
        "last_capture": last_capture,
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "config": {
            "min_coverage": cfg["min_coverage"],
            "min_sharpness": cfg["min_sharpness"],
            "duplicate_max_distance": cfg["duplicate_max_distance"],
            "keep_rejected": cfg["keep_rejected"],
        },
    }


# ============================================================
# FASE 2B — ESTADOS DE UI Y MENSAJES PARA LA USUARIA
# ============================================================

CAPTURE_UI_STATE_SIN_PREPARAR = "SIN_PREPARAR"
CAPTURE_UI_STATE_PREPARANDO = "PREPARANDO"
CAPTURE_UI_STATE_CAPTURANDO = "CAPTURANDO"
CAPTURE_UI_STATE_CAPTURA_COMPLETADA = "CAPTURA_COMPLETADA"
CAPTURE_UI_STATE_LISTO_PARA_ENTRENAR = "LISTO_PARA_ENTRENAR"

CAPTURE_UI_STATES = (
    CAPTURE_UI_STATE_SIN_PREPARAR,
    CAPTURE_UI_STATE_PREPARANDO,
    CAPTURE_UI_STATE_CAPTURANDO,
    CAPTURE_UI_STATE_CAPTURA_COMPLETADA,
    CAPTURE_UI_STATE_LISTO_PARA_ENTRENAR,
)

CAPTURE_UI_STATE_LABELS = {
    CAPTURE_UI_STATE_SIN_PREPARAR: "SIN PREPARAR",
    CAPTURE_UI_STATE_PREPARANDO: "PREPARANDO",
    CAPTURE_UI_STATE_CAPTURANDO: "CAPTURANDO",
    CAPTURE_UI_STATE_CAPTURA_COMPLETADA: "CAPTURA COMPLETADA",
    CAPTURE_UI_STATE_LISTO_PARA_ENTRENAR: "LISTO PARA ENTRENAR",
}

# Motivos internos -> texto que sí se le explica a la usuaria.
CAPTURE_REJECT_LABELS = {
    "INVALID_ROI": (
        "La prenda no quedó dentro del área de inspección."
    ),
    "INVALID_COVERAGE": (
        "No se pudo medir la prenda por completo."
    ),
    "LOW_COVERAGE": (
        "La prenda no cubrió suficiente el área de inspección."
    ),
    "INVALID_SHARPNESS": (
        "No se pudo evaluar la nitidez de la imagen."
    ),
    "BLUR": "Movimiento excesivo.",
    "DUPLICATE_SHA256": "Imagen repetida.",
    "DUPLICATE_PERCEPTUAL": "Imagen repetida.",
    "DUPLICATE": "Imagen repetida.",
    "MANUAL_REPEAT": "Marcada para repetir por la operadora.",
    "MANUAL_DISCARD": "Descartada manualmente por la operadora.",
}

CAPTURE_REJECT_FALLBACK = "La imagen no cumplió la calidad requerida."


def humanize_reject_reason(reject_reason) -> str | None:
    """Traduce un motivo técnico de descarte a texto comprensible."""
    if reject_reason in (None, ""):
        return None

    key = str(reject_reason).strip().upper()
    return CAPTURE_REJECT_LABELS.get(key, CAPTURE_REJECT_FALLBACK)


def resolve_capture_ui_state(
    *,
    session_status,
    accepted_count=0,
    min_images=20,
    has_preparation_version: bool = False,
) -> str:
    """
    Estado visible para la ficha del modelo.

    Solo expone lo que le aporta a la usuaria; nunca estados internos.
    """
    status = str(session_status or "").strip().upper()

    if status == CAPTURE_STATUS_ABIERTA:
        return CAPTURE_UI_STATE_CAPTURANDO

    if status == CAPTURE_STATUS_COMPLETADA:
        if int(accepted_count or 0) >= int(min_images or 20):
            return CAPTURE_UI_STATE_LISTO_PARA_ENTRENAR
        return CAPTURE_UI_STATE_CAPTURA_COMPLETADA

    # Sin sesión o sesión cancelada: la preparación sigue en curso.
    if has_preparation_version:
        return CAPTURE_UI_STATE_PREPARANDO

    return CAPTURE_UI_STATE_SIN_PREPARAR


def humanize_capture_error(message) -> str:
    """
    Convierte mensajes internos de captura en mensajes para la usuaria.

    Nunca expone ids internos, locks ni SQL.
    """
    text = str(message or "").strip()

    if not text:
        return "No se pudo completar la operación de captura."

    lowered = text.lower()

    if "lock" in lowered or "get_lock" in lowered:
        return (
            "La estación está ocupada por otro proceso de captura. "
            "Espere unos segundos e intente de nuevo."
        )

    if "sesión de captura" in lowered and (
        "activa" in lowered or "abierta" in lowered
    ):
        return (
            "Ya hay una sesión de captura activa en la estación. "
            "Finalícela o cancélela antes de iniciar otra."
        )

    if "sesión de captura" in lowered and "no existe" in lowered:
        return "La sesión de captura ya no existe."

    if "no existe" in lowered and "modelo" in lowered:
        return "El modelo de prenda no existe."

    return text


def transition_capture_session_status(
    cur,
    capture_session_id: int,
    new_status: str,
) -> dict:
    new_raw = str(new_status or "").strip().upper()

    cur.execute(
        """
        SELECT id, garment_model_id, status
        FROM ai_capture_sessions
        WHERE id = %s
        FOR UPDATE
        """,
        (capture_session_id,),
    )

    row = cur.fetchone()

    if row is None:
        raise AIDomainError(
            f"La sesión de captura {capture_session_id} no existe."
        )

    validate_capture_status_transition(row["status"], new_raw)

    if new_raw == CAPTURE_STATUS_ABIERTA:
        raise AIDomainError(
            "Una sesión completada o cancelada no puede reabrirse."
        )

    cur.execute(
        """
        UPDATE ai_capture_sessions
        SET status = %s,
            finished_at = NOW()
        WHERE id = %s
        """,
        (new_raw, capture_session_id),
    )

    return {
        "id": capture_session_id,
        "garment_model_id": row["garment_model_id"],
        "status": new_raw,
    }


def register_training_image(
    cur,
    *,
    garment_model_id: int,
    image_path: str,
    sha256: str,
    capture_session_id: int | None = None,
    frame_sequence: int | None = None,
    coverage: float | None = None,
    quality_score: float | None = None,
    status: str = TRAINING_IMAGE_STATUS_ACEPTADA,
    reject_reason: str | None = None,
) -> dict:
    """Registra una imagen de entrenamiento con hash y ruta controlada."""
    _lock_garment_model(cur, garment_model_id)

    safe_path = ensure_safe_relative_path(image_path)
    digest = normalize_sha256(sha256)

    status_raw = str(status or "").strip().upper()

    if status_raw not in TRAINING_IMAGE_STATUSES:
        raise AIDomainError(
            f"Estado de imagen no soportado: {status!r}"
        )

    if capture_session_id is not None:
        cur.execute(
            """
            SELECT id, garment_model_id, status
            FROM ai_capture_sessions
            WHERE id = %s
            FOR UPDATE
            """,
            (capture_session_id,),
        )

        session = cur.fetchone()

        if session is None:
            raise AIDomainError(
                "La sesión de captura indicada no existe."
            )

        if int(session["garment_model_id"]) != int(garment_model_id):
            raise AIDomainError(
                "La sesión no pertenece al modelo de prenda indicado."
            )

        if session["status"] != CAPTURE_STATUS_ABIERTA:
            raise AIDomainError(
                "Solo se aceptan imágenes en sesiones ABIERTA."
            )

    cur.execute(
        """
        INSERT INTO ai_training_images (
            capture_session_id,
            garment_model_id,
            image_path,
            sha256,
            frame_sequence,
            coverage,
            quality_score,
            status,
            reject_reason
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            capture_session_id,
            garment_model_id,
            safe_path,
            digest,
            frame_sequence,
            coverage,
            quality_score,
            status_raw,
            reject_reason,
        ),
    )

    return {
        "id": cur.lastrowid,
        "garment_model_id": garment_model_id,
        "capture_session_id": capture_session_id,
        "image_path": safe_path,
        "sha256": digest,
        "status": status_raw,
    }


def count_session_images(cur, capture_session_id: int) -> dict:
    """Conteos derivados de la verdad primaria (sin contadores en sesión)."""
    cur.execute(
        """
        SELECT
            status,
            COUNT(*) AS total
        FROM ai_training_images
        WHERE capture_session_id = %s
        GROUP BY status
        """,
        (capture_session_id,),
    )

    counts = {
        TRAINING_IMAGE_STATUS_ACEPTADA: 0,
        TRAINING_IMAGE_STATUS_RECHAZADA: 0,
    }

    for row in cur.fetchall():
        counts[row["status"]] = int(row["total"])

    return {
        "accepted_images": counts[TRAINING_IMAGE_STATUS_ACEPTADA],
        "rejected_images": counts[TRAINING_IMAGE_STATUS_RECHAZADA],
        "total": sum(counts.values()),
    }


def create_ai_dataset(
    cur,
    garment_model_id: int,
    actor_id: int | None,
) -> dict:
    """Crea un dataset ABIERTO con versión dN única por modelo."""
    _lock_garment_model(cur, garment_model_id)

    cur.execute(
        """
        SELECT version
        FROM ai_datasets
        WHERE garment_model_id = %s
        ORDER BY id ASC
        """,
        (garment_model_id,),
    )

    versions = [row for row in cur.fetchall()]
    version = next_dataset_version_label(versions)

    cur.execute(
        """
        INSERT INTO ai_datasets (
            garment_model_id,
            version,
            status,
            image_count,
            created_by
        )
        VALUES (%s, %s, 'ABIERTO', 0, %s)
        """,
        (garment_model_id, version, actor_id),
    )

    dataset_id = cur.lastrowid

    return {
        "id": dataset_id,
        "garment_model_id": garment_model_id,
        "version": version,
        "status": DATASET_STATUS_ABIERTO,
        "image_count": 0,
    }


def add_images_to_dataset(
    cur,
    dataset_id: int,
    image_ids,
) -> int:
    """Agrega imágenes ACEPTADAS al dataset (solo si está ABIERTO)."""
    cur.execute(
        """
        SELECT id, garment_model_id, status
        FROM ai_datasets
        WHERE id = %s
        FOR UPDATE
        """,
        (dataset_id,),
    )

    dataset = cur.fetchone()

    if dataset is None:
        raise AIDomainError(f"El dataset {dataset_id} no existe.")

    if dataset["status"] != DATASET_STATUS_ABIERTO:
        raise AIDomainError(
            "Solo se pueden agregar imágenes a un dataset ABIERTO."
        )

    added = 0

    for image_id in dict.fromkeys(int(value) for value in (image_ids or [])):
        if image_id <= 0:
            raise AIDomainError("image_id inválido.")

        cur.execute(
            """
            SELECT id, garment_model_id, status
            FROM ai_training_images
            WHERE id = %s
            FOR UPDATE
            """,
            (image_id,),
        )

        image = cur.fetchone()

        if image is None:
            raise AIDomainError(f"La imagen {image_id} no existe.")

        if int(image["garment_model_id"]) != int(
            dataset["garment_model_id"]
        ):
            raise AIDomainError(
                "La imagen pertenece a otro modelo de prenda."
            )

        if image["status"] != TRAINING_IMAGE_STATUS_ACEPTADA:
            raise AIDomainError(
                "Solo las imágenes ACEPTADA pueden entrar al dataset."
            )

        cur.execute(
            """
            INSERT INTO ai_dataset_images (dataset_id, image_id)
            VALUES (%s, %s)
            """,
            (dataset_id, image_id),
        )
        added += 1

    return added


def close_ai_dataset(
    cur,
    dataset_id: int,
    actor_id: int | None,
    *,
    artifacts_root: Path | None = None,
) -> dict:
    """Cierra un dataset: congela relación, cuenta y guarda manifest."""
    cur.execute(
        """
        SELECT
            id,
            garment_model_id,
            version,
            status
        FROM ai_datasets
        WHERE id = %s
        FOR UPDATE
        """,
        (dataset_id,),
    )

    dataset = cur.fetchone()

    if dataset is None:
        raise AIDomainError(f"El dataset {dataset_id} no existe.")

    validate_dataset_status_transition(
        dataset["status"],
        DATASET_STATUS_CERRADO,
    )

    _lock_garment_model(cur, dataset["garment_model_id"])

    cur.execute(
        """
        SELECT ti.id, ti.sha256
        FROM ai_dataset_images di
        JOIN ai_training_images ti
          ON ti.id = di.image_id
        WHERE di.dataset_id = %s
        """,
        (dataset_id,),
    )

    rows = cur.fetchall()
    manifest_hash = compute_manifest_hash(rows)
    manifest_path = compute_manifest_path(
        dataset["garment_model_id"],
        dataset_id,
    )

    try:
        root = Path(artifacts_root) if artifacts_root else get_ai_artifacts_root()
        target = resolve_under_root(root, manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "garment_model_id": dataset["garment_model_id"],
                    "version": dataset["version"],
                    "image_count": len(rows),
                    "manifest_hash": manifest_hash,
                    "images": [
                        {
                            "image_id": int(row["id"]),
                            "sha256": str(row["sha256"]).lower(),
                        }
                        for row in sorted(
                            rows,
                            key=lambda item: int(item["id"]),
                        )
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        manifest_path = None

    cur.execute(
        """
        UPDATE ai_datasets
        SET
            status = 'CERRADO',
            image_count = %s,
            manifest_path = %s,
            manifest_hash = %s,
            closed_at = NOW()
        WHERE id = %s
        """,
        (len(rows), manifest_path, manifest_hash, dataset_id),
    )

    return {
        "id": dataset_id,
        "garment_model_id": dataset["garment_model_id"],
        "version": dataset["version"],
        "status": DATASET_STATUS_CERRADO,
        "image_count": len(rows),
        "manifest_path": manifest_path,
        "manifest_hash": manifest_hash,
    }


def archive_ai_dataset(cur, dataset_id: int) -> dict:
    cur.execute(
        """
        SELECT id, garment_model_id, version, status
        FROM ai_datasets
        WHERE id = %s
        FOR UPDATE
        """,
        (dataset_id,),
    )

    dataset = cur.fetchone()

    if dataset is None:
        raise AIDomainError(f"El dataset {dataset_id} no existe.")

    validate_dataset_status_transition(
        dataset["status"],
        DATASET_STATUS_ARCHIVADO,
    )

    cur.execute(
        """
        UPDATE ai_datasets
        SET status = 'ARCHIVADO'
        WHERE id = %s
        """,
        (dataset_id,),
    )

    return {
        "id": dataset_id,
        "garment_model_id": dataset["garment_model_id"],
        "version": dataset["version"],
        "status": DATASET_STATUS_ARCHIVADO,
    }


def create_ai_job(
    cur,
    kind: str,
    *,
    ai_model_id: int | None = None,
    dataset_id: int | None = None,
    created_by: int | None = None,
    log_path: str | None = None,
) -> dict:
    """Crea un job PENDIENTE (sin worker en Fase 1)."""
    kind_raw = validate_job_kind(kind)

    if kind_raw == JOB_KIND_TRAINING:
        if ai_model_id is None:
            raise AIDomainError(
                "TRAINING requiere ai_model_id."
            )
        if dataset_id is None:
            raise AIDomainError(
                "TRAINING requiere dataset_id."
            )

    if kind_raw == JOB_KIND_VALIDATION and ai_model_id is None:
        raise AIDomainError("VALIDATION requiere ai_model_id.")

    if ai_model_id is not None:
        cur.execute(
            "SELECT id FROM garment_ai_models WHERE id = %s",
            (ai_model_id,),
        )

        if cur.fetchone() is None:
            raise AIDomainError(
                f"La versión de IA {ai_model_id} no existe."
            )

    if dataset_id is not None:
        cur.execute(
            "SELECT id FROM ai_datasets WHERE id = %s",
            (dataset_id,),
        )

        if cur.fetchone() is None:
            raise AIDomainError(f"El dataset {dataset_id} no existe.")

    safe_log_path = None

    if log_path:
        safe_log_path = ensure_safe_relative_path(log_path)

    cur.execute(
        """
        INSERT INTO ai_jobs (
            ai_model_id,
            dataset_id,
            kind,
            status,
            progress,
            log_path,
            created_by
        )
        VALUES (%s, %s, %s, 'PENDIENTE', 0, %s, %s)
        """,
        (
            ai_model_id,
            dataset_id,
            kind_raw,
            safe_log_path,
            created_by,
        ),
    )

    return {
        "id": cur.lastrowid,
        "ai_model_id": ai_model_id,
        "dataset_id": dataset_id,
        "kind": kind_raw,
        "status": JOB_STATUS_PENDIENTE,
        "progress": 0,
        "log_path": safe_log_path,
    }


def transition_ai_job_status(
    cur,
    job_id: int,
    new_status: str,
    *,
    progress: float | None = None,
    error_message: str | None = None,
    worker_id: str | None = None,
    stage: str | None = None,
    stage_label: str | None = None,
) -> dict:
    new_raw = str(new_status or "").strip().upper()

    cur.execute(
        """
        SELECT id, status, progress
        FROM ai_jobs
        WHERE id = %s
        FOR UPDATE
        """,
        (job_id,),
    )

    job = cur.fetchone()

    if job is None:
        raise AIDomainError(f"El job {job_id} no existe.")

    validate_job_status_transition(job["status"], new_raw)

    sets = ["status = %s", "updated_at = NOW()"]
    params = [new_raw]

    if new_raw == JOB_STATUS_EN_CURSO:
        sets.append("started_at = COALESCE(started_at, NOW())")
        sets.append("heartbeat_at = NOW()")

        if worker_id is not None:
            sets.append("worker_id = %s")
            params.append(str(worker_id)[:120])

    if new_raw in {
        JOB_STATUS_COMPLETADO,
        JOB_STATUS_FALLIDO,
        JOB_STATUS_CANCELADO,
    }:
        sets.append("finished_at = NOW()")

    if progress is not None:
        value = float(progress)

        if not 0.0 <= value <= 100.0:
            raise AIDomainError(
                "progress debe estar entre 0 y 100."
            )

        sets.append("progress = %s")
        params.append(value)

    if stage is not None:
        sets.append("stage = %s")
        params.append(str(stage)[:60])

    if stage_label is not None:
        sets.append("stage_label = %s")
        params.append(str(stage_label)[:160])

    if error_message is not None:
        sets.append("error_message = %s")
        params.append(str(error_message))

    params.append(job_id)

    cur.execute(
        f"UPDATE ai_jobs SET {', '.join(sets)} WHERE id = %s",
        tuple(params),
    )

    return {
        "id": job_id,
        "status": new_raw,
    }


def record_ai_event(
    cur,
    event_type: str,
    *,
    actor_id: int | None = None,
    ai_model_id: int | None = None,
    dataset_id: int | None = None,
    capture_session_id: int | None = None,
    payload=None,
) -> dict:
    """Inserta un evento de auditoría persistente de IA."""
    type_raw = validate_event_type(event_type)
    clean_payload = sanitize_event_payload(payload)

    cur.execute(
        """
        INSERT INTO ai_events (
            ai_model_id,
            dataset_id,
            capture_session_id,
            actor_id,
            event_type,
            payload_json
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            ai_model_id,
            dataset_id,
            capture_session_id,
            actor_id,
            type_raw,
            json.dumps(clean_payload, ensure_ascii=False)
            if clean_payload is not None
            else None,
        ),
    )

    return {
        "id": cur.lastrowid,
        "event_type": type_raw,
        "ai_model_id": ai_model_id,
        "dataset_id": dataset_id,
        "capture_session_id": capture_session_id,
        "actor_id": actor_id,
    }

# ============================================================
# FASE 3A — COLA, PROGRESO Y RECUPERACIÓN DE JOBS DE ENTRENAMIENTO
#
# MySQL es la fuente de verdad. No existe cola en memoria: si Flask
# o el worker se reinician, el job sigue existiendo en la tabla.
# ============================================================

JOB_ACTIVE_STATUSES = (JOB_STATUS_PENDIENTE, JOB_STATUS_EN_CURSO)

JOB_ERROR_ORPHAN = (
    "El worker se reinició durante el entrenamiento. "
    "El trabajo se marcó como fallido; no se reanuda un proceso "
    "parcial."
)


def claim_next_training_job(cur, worker_id: str) -> dict | None:
    """Reclama el job TRAINING PENDIENTE más antiguo de forma atómica.

    Debe ejecutarse dentro de una transacción abierta por el llamador.
    ``FOR UPDATE SKIP LOCKED`` garantiza que dos workers jamás reclaman
    el mismo job (doble entrenamiento).
    """
    worker = str(worker_id or "").strip()[:120] or "worker"

    cur.execute(
        """
        SELECT id, ai_model_id, dataset_id, kind, status, progress
        FROM ai_jobs
        WHERE kind = %s AND status = %s
        ORDER BY id ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        """,
        (JOB_KIND_TRAINING, JOB_STATUS_PENDIENTE),
    )

    row = cur.fetchone()

    if row is None:
        return None

    job_id = int(row["id"])

    transition_ai_job_status(
        cur,
        job_id,
        JOB_STATUS_EN_CURSO,
        progress=0,
        worker_id=worker,
        stage="INICIANDO",
        stage_label="Iniciando entrenamiento",
    )

    claimed = dict(row)
    claimed["status"] = JOB_STATUS_EN_CURSO
    claimed["worker_id"] = worker

    return claimed


def update_job_progress(
    cur,
    job_id: int,
    progress: float,
    *,
    stage: str | None = None,
    stage_label: str | None = None,
) -> dict:
    """Actualiza el progreso real (etapas, no porcentajes inventados)."""
    value = float(progress)

    if not 0.0 <= value <= 100.0:
        raise AIDomainError("progress debe estar entre 0 y 100.")

    sets = ["progress = %s", "updated_at = NOW()", "heartbeat_at = NOW()"]
    params = [value]

    if stage is not None:
        sets.append("stage = %s")
        params.append(str(stage)[:60])

    if stage_label is not None:
        sets.append("stage_label = %s")
        params.append(str(stage_label)[:160])

    params.append(int(job_id))

    cur.execute(
        f"UPDATE ai_jobs SET {', '.join(sets)} WHERE id = %s",
        tuple(params),
    )

    if cur.rowcount == 0:
        raise AIDomainError(f"El job {job_id} no existe.")

    return {
        "id": int(job_id),
        "progress": value,
        "stage": stage,
        "stage_label": stage_label,
    }


def heartbeat_job(cur, job_id: int, worker_id: str | None = None) -> None:
    """Marca vida del worker sobre el job EN_CURSO."""
    if worker_id is None:
        cur.execute(
            "UPDATE ai_jobs SET heartbeat_at = NOW(), updated_at = NOW() "
            "WHERE id = %s",
            (int(job_id),),
        )
    else:
        cur.execute(
            "UPDATE ai_jobs SET heartbeat_at = NOW(), updated_at = NOW(), "
            "worker_id = %s WHERE id = %s",
            (str(worker_id)[:120], int(job_id)),
        )


def get_active_training_job(cur, ai_model_id: int) -> dict | None:
    """Job TRAINING PENDIENTE/EN_CURSO de una versión de IA (si existe)."""
    cur.execute(
        """
        SELECT id, ai_model_id, dataset_id, kind, status, progress,
               stage, stage_label, worker_id, created_at, started_at,
               heartbeat_at, log_path, error_message
        FROM ai_jobs
        WHERE ai_model_id = %s AND kind = %s AND status IN (%s, %s)
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            int(ai_model_id),
            JOB_KIND_TRAINING,
            JOB_STATUS_PENDIENTE,
            JOB_STATUS_EN_CURSO,
        ),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_latest_training_job_for_garment(
    cur,
    garment_model_id: int,
) -> dict | None:
    """Último job TRAINING (cualquier estado) del modelo de prenda."""
    cur.execute(
        """
        SELECT j.id, j.ai_model_id, j.dataset_id, j.kind, j.status,
               j.progress, j.stage, j.stage_label, j.worker_id,
               j.created_at, j.started_at, j.finished_at, j.updated_at,
               j.heartbeat_at, j.log_path, j.error_message,
               m.version AS model_version, m.status AS model_status
        FROM ai_jobs j
        JOIN garment_ai_models m ON m.id = j.ai_model_id
        WHERE m.garment_model_id = %s AND j.kind = %s
        ORDER BY j.id DESC
        LIMIT 1
        """,
        (int(garment_model_id), JOB_KIND_TRAINING),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def reclaim_orphaned_jobs(
    cur,
    *,
    stale_seconds: int,
    reason: str = JOB_ERROR_ORPHAN,
) -> list[dict]:
    """Marca como FALLIDO los jobs EN_CURSO huérfanos (worker caído).

    Política documentada (FASE 3A): nunca se reanuda un entrenamiento
    parcial. El job fallido conserva su log y el dataset intacto; el
    reintento crea un job nuevo.
    """
    seconds = max(1, int(stale_seconds))

    cur.execute(
        """
        SELECT id, ai_model_id, worker_id, heartbeat_at
        FROM ai_jobs
        WHERE kind = %s
          AND status = %s
          AND (
                heartbeat_at IS NULL
                OR heartbeat_at < NOW() - INTERVAL %s SECOND
          )
        FOR UPDATE
        """,
        (JOB_KIND_TRAINING, JOB_STATUS_EN_CURSO, seconds),
    )

    stale = [dict(row) for row in cur.fetchall()]

    for job in stale:
        transition_ai_job_status(
            cur,
            int(job["id"]),
            JOB_STATUS_FALLIDO,
            error_message=str(reason)[:500],
            stage="FALLIDO",
            stage_label="Entrenamiento interrumpido",
        )

        ai_model_id = job.get("ai_model_id")

        if ai_model_id:
            cur.execute(
                "SELECT status FROM garment_ai_models WHERE id = %s "
                "FOR UPDATE",
                (int(ai_model_id),),
            )
            model = cur.fetchone()

            if model and model.get("status") == AI_MODEL_STATUS_ENTRENANDO:
                try:
                    transition_ai_model_status(
                        cur,
                        int(ai_model_id),
                        AI_MODEL_STATUS_FALLIDO,
                        None,
                        notes=str(reason)[:500],
                    )
                except AIDomainError:
                    # Estado ya resuelto por otro proceso: el job fallido
                    # es lo que importa para no reanudar a ciegas.
                    pass

    return stale


def count_open_training_jobs(cur, ai_model_id: int) -> int:
    """Cuántos jobs TRAINING activos existen para esa versión (0/1)."""
    cur.execute(
        """
        SELECT COUNT(*) AS total
        FROM ai_jobs
        WHERE ai_model_id = %s AND kind = %s AND status IN (%s, %s)
        """,
        (
            int(ai_model_id),
            JOB_KIND_TRAINING,
            JOB_STATUS_PENDIENTE,
            JOB_STATUS_EN_CURSO,
        ),
    )
    row = cur.fetchone()
    return int(_scalar(row) or 0)
