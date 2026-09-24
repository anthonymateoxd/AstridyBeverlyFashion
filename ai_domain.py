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
    "DATASET_CLOSED",
    "DATASET_ARCHIVED",
    "TRAINING_STARTED",
    "TRAINING_COMPLETED",
    "TRAINING_FAILED",
    "VALIDATION_STARTED",
    "VALIDATED",
    "REJECTED",
    "ACTIVATED",
    "RETIRED",
    "ROLLBACK",
    "VERSION_PREPARED",
    "JOB_CREATED",
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
            active
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
        SELECT version
        FROM garment_ai_models
        WHERE garment_model_id = %s
        ORDER BY id ASC
        """,
        (garment_model_id,),
    )

    versions = [row for row in cur.fetchall()]

    cur.execute(
        """
        SELECT COUNT(*) AS total
        FROM garment_ai_models
        WHERE garment_model_id = %s
          AND status IN (
              'PREPARACION',
              'ENTRENANDO',
              'ENTRENADO',
              'VALIDACION',
              'VALIDADO'
          )
        """,
        (garment_model_id,),
    )

    pending = int(_scalar(cur.fetchone()) or 0)

    if pending > 0:
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


def open_ai_capture_session(
    cur,
    garment_model_id: int,
    actor_id: int | None,
    *,
    notes: str | None = None,
) -> dict:
    """Abre una sesión de captura (una ABIERTA por modelo)."""
    _lock_garment_model(cur, garment_model_id)

    cur.execute(
        """
        SELECT id
        FROM ai_capture_sessions
        WHERE garment_model_id = %s
          AND status = 'ABIERTA'
        LIMIT 1
        """,
        (garment_model_id,),
    )

    existing = cur.fetchone()

    if existing:
        raise AIDomainError(
            "Ya existe una sesión de captura ABIERTA para este modelo."
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

    sets = ["status = %s"]
    params = [new_raw]

    if new_raw == JOB_STATUS_EN_CURSO:
        sets.append("started_at = COALESCE(started_at, NOW())")

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
