"""FASE 2A — Pipeline de captura de dataset IA en la estación.

Reutiliza la arquitectura latest_frame (un único productor) sin abrir
cámara ni RTSP. No entrena PatchCore ni cambia el checkpoint productivo.

El módulo no importa Flask ni app.py: recibe frames y callbacks de
persistencia para poder probarse sin cámara física.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from ai_domain import (
    AIDomainError,
    compute_quality_score,
    evaluate_capture_metrics,
    get_ai_capture_config,
    hamming_distance,
    is_perceptual_duplicate,
    normalize_sha256,
)


# ------------------------------------------------------------
# Mutex de estación: modo preparación vs producción
# ------------------------------------------------------------

_STATION_MODE_LOCK = threading.Lock()
_AI_CAPTURE_MODE_ACTIVE = False
_AI_CAPTURE_SESSION_ID: int | None = None
_AI_CAPTURE_GARMENT_MODEL_ID: int | None = None


def set_ai_capture_mode(
    active: bool,
    *,
    session_id: int | None = None,
    garment_model_id: int | None = None,
) -> None:
    """Activa/desactiva el modo preparación IA en memoria (estación)."""
    global _AI_CAPTURE_MODE_ACTIVE
    global _AI_CAPTURE_SESSION_ID
    global _AI_CAPTURE_GARMENT_MODEL_ID

    with _STATION_MODE_LOCK:
        _AI_CAPTURE_MODE_ACTIVE = bool(active)
        if active:
            _AI_CAPTURE_SESSION_ID = session_id
            _AI_CAPTURE_GARMENT_MODEL_ID = garment_model_id
        else:
            _AI_CAPTURE_SESSION_ID = None
            _AI_CAPTURE_GARMENT_MODEL_ID = None


def is_ai_capture_mode_active() -> bool:
    with _STATION_MODE_LOCK:
        return _AI_CAPTURE_MODE_ACTIVE


def get_ai_capture_mode_state() -> dict:
    with _STATION_MODE_LOCK:
        return {
            "active": _AI_CAPTURE_MODE_ACTIVE,
            "session_id": _AI_CAPTURE_SESSION_ID,
            "garment_model_id": _AI_CAPTURE_GARMENT_MODEL_ID,
        }


def assert_production_allowed(context: str = "produccion") -> None:
    """Bloquea registros de producción mientras hay modo preparación IA."""
    if is_ai_capture_mode_active():
        raise AIDomainError(
            f"Modo preparación IA activo: no se permite {context}."
        )


def assert_ai_capture_allowed(context: str = "captura_ia") -> None:
    """Bloquea iniciar captura IA si producción automática está activa."""
    # La flag de producción se consulta desde la app (AUTO_INSPECTION_ENABLED)
    # vía set_production_probe; aquí solo validamos el modo IA ya activo.
    if is_ai_capture_mode_active():
        # ya activo: el caller decide si es idempotente o error
        return


# Probe inyectado por app.py para no importar app en tests.
_production_probe = None


def set_production_probe(fn) -> None:
    """fn() -> bool: True si la inspección automática productiva está activa."""
    global _production_probe
    _production_probe = fn


def is_production_auto_active() -> bool:
    if _production_probe is None:
        return False
    try:
        return bool(_production_probe())
    except Exception:
        return False


def ensure_ai_capture_can_start() -> None:
    if is_production_auto_active():
        raise AIDomainError(
            "La inspección automática de producción está activa. "
            "Detenla antes de iniciar la captura IA."
        )


# ------------------------------------------------------------
# Métricas de candidato (ROI compartido con producción)
# ------------------------------------------------------------


def compute_roi_coverage(frame, *, get_roi_bounds, create_garment_mask, cv2_mod) -> float:
    """Cobertura prenda/ROI — MISMA fórmula que auto_inspection_worker."""
    garment_mask = create_garment_mask(frame)
    roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_bounds(frame)
    roi_mask = garment_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_area = float(
        max(1, roi_mask.shape[0] * roi_mask.shape[1])
    )
    return float(cv2_mod.countNonZero(roi_mask) / roi_area)


def compute_sharpness(frame, *, get_roi_bounds, cv2_mod) -> float:
    """Varianza del Laplaciano sobre el ROI (nitidez)."""
    x1, y1, x2, y2 = get_roi_bounds(frame)
    roi = frame[y1:y2, x1:x2]
    if roi is None or getattr(roi, "size", 0) == 0:
        raise AIDomainError("ROI vacío para medir nitidez.")
    if len(roi.shape) == 3:
        gray = cv2_mod.cvtColor(roi, cv2_mod.COLOR_BGR2GRAY)
    else:
        gray = roi
    lap = cv2_mod.Laplacian(gray, cv2_mod.CV_64F)
    return float(lap.var())


def compute_dhash_64(frame, *, cv2_mod) -> int:
    """dHash 64-bit (9x8) sin dependencias nuevas."""
    if len(frame.shape) == 3:
        gray = cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2GRAY)
    else:
        gray = frame
    small = cv2_mod.resize(gray, (9, 8), interpolation=cv2_mod.INTER_AREA)
    bits = 0
    for row in range(8):
        for col in range(8):
            left = int(small[row, col])
            right = int(small[row, col + 1])
            bits = (bits << 1) | (1 if left < right else 0)
    return bits


def is_roi_valid(frame, *, get_roi_bounds) -> bool:
    try:
        x1, y1, x2, y2 = get_roi_bounds(frame)
    except Exception:
        return False
    if frame is None:
        return False
    h, w = frame.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    return True


def evaluate_candidate(
    frame,
    *,
    coverage: float,
    frame_sequence: int,
    known_sha256: set[str],
    known_dhashes: list[int],
    get_roi_bounds,
    cv2_mod,
    config: dict | None = None,
    sha256_hex: str | None = None,
) -> dict:
    """Evalúa un candidato: ROI, cobertura, nitidez, duplicados."""
    cfg = dict(config or get_ai_capture_config())
    roi_valid = is_roi_valid(frame, get_roi_bounds=get_roi_bounds)

    if not roi_valid:
        decision = evaluate_capture_metrics(
            coverage=None,
            sharpness=None,
            roi_valid=False,
            config=cfg,
        )
        decision.update(
            {
                "frame_sequence": frame_sequence,
                "coverage": None,
                "sharpness": None,
                "dhash": None,
                "sha256": sha256_hex,
            }
        )
        return decision

    try:
        sharpness = compute_sharpness(
            frame,
            get_roi_bounds=get_roi_bounds,
            cv2_mod=cv2_mod,
        )
        dhash = compute_dhash_64(frame, cv2_mod=cv2_mod)
    except Exception as error:
        decision = evaluate_capture_metrics(
            coverage=coverage,
            sharpness=None,
            roi_valid=False,
            config=cfg,
        )
        decision.update(
            {
                "frame_sequence": frame_sequence,
                "coverage": float(coverage),
                "sharpness": None,
                "dhash": None,
                "sha256": sha256_hex,
                "error": str(error),
            }
        )
        return decision

    decision = evaluate_capture_metrics(
        coverage=coverage,
        sharpness=sharpness,
        roi_valid=True,
        config=cfg,
    )

    reason = decision.get("reject_reason")

    if reason is None and sha256_hex:
        digest = normalize_sha256(sha256_hex)
        if digest in known_sha256:
            decision["accepted"] = False
            decision["reject_reason"] = "DUPLICATE_SHA256"
            reason = "DUPLICATE_SHA256"

    if reason is None and known_dhashes is not None:
        max_d = int(cfg["duplicate_max_distance"])
        for existing in known_dhashes:
            if is_perceptual_duplicate(dhash, existing, max_d):
                decision["accepted"] = False
                decision["reject_reason"] = "DUPLICATE_PERCEPTUAL"
                reason = "DUPLICATE_PERCEPTUAL"
                break

    decision.update(
        {
            "frame_sequence": frame_sequence,
            "coverage": float(coverage),
            "sharpness": float(sharpness),
            "dhash": dhash,
            "sha256": sha256_hex,
            "hamming_note": (
                f"threshold={cfg['duplicate_max_distance']}"
            ),
        }
    )
    return decision


# ------------------------------------------------------------
# Seguimiento de presencia: 1 candidato por garment_token
# ------------------------------------------------------------


@dataclass
class PresenceCandidate:
    """Mejor frame observado durante una presencia física."""

    garment_token: int
    best_frame: object | None = None
    best_coverage: float = 0.0
    best_sequence: int | None = None
    present_frames: int = 0
    tracked_frames: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    finalized: bool = False

    def observe(
        self,
        frame,
        coverage: float,
        sequence: int,
        *,
        enter_coverage: float,
        confirm_frames: int,
    ) -> bool:
        """
        Alimenta un frame. Devuelve True si la presencia ya es 'confirmada'
        (útil solo para logging); la persistencia ocurre en finalize().
        """
        if self.finalized:
            return False

        if coverage >= enter_coverage:
            self.present_frames += 1
            self.tracked_frames += 1
            if coverage > self.best_coverage:
                self.best_coverage = float(coverage)
                self.best_frame = frame
                self.best_sequence = int(sequence)
            return self.present_frames >= confirm_frames

        return False

    def should_finalize_on_exit(
        self,
        coverage: float,
        *,
        exit_coverage: float,
        exit_frames_threshold: int,
        exit_frames_counter: int,
    ) -> bool:
        """True cuando la prenda salió y hay candidato que persistir."""
        if self.finalized:
            return False
        if coverage > exit_coverage:
            return False
        if self.best_frame is None or self.best_coverage <= 0:
            return False
        return exit_frames_counter >= exit_frames_threshold


def reset_presence_candidate(token: int) -> PresenceCandidate:
    return PresenceCandidate(garment_token=int(token))


# ------------------------------------------------------------
# Persistencia de un candidato (callback de app/ai_domain)
# ------------------------------------------------------------


def persist_best_candidate(
    candidate: PresenceCandidate,
    *,
    evaluate_fn,
    encode_jpeg_fn,
    persist_fn,
    config: dict | None = None,
) -> dict:
    """
    Evalúa el mejor frame de la presencia y lo persiste una sola vez.

    evaluate_fn(frame, coverage, sequence) -> decision dict
    encode_jpeg_fn(frame) -> bytes
    persist_fn(decision, jpeg_bytes, candidate) -> dict registro
    """
    if candidate is None or candidate.finalized:
        raise AIDomainError("No hay candidato pendiente de persistir.")
    if candidate.best_frame is None:
        raise AIDomainError("El candidato no tiene frame.")

    candidate.finalized = True
    cfg = dict(config or get_ai_capture_config())

    decision = evaluate_fn(
        candidate.best_frame,
        candidate.best_coverage,
        candidate.best_sequence,
    )

    jpeg_bytes = None
    if decision.get("accepted") or cfg.get("keep_rejected"):
        jpeg_bytes = encode_jpeg_fn(candidate.best_frame)

    result = persist_fn(decision, jpeg_bytes, candidate)
    return {
        "garment_token": candidate.garment_token,
        "accepted": bool(decision.get("accepted")),
        "reject_reason": decision.get("reject_reason"),
        "quality_score": decision.get("quality_score"),
        "coverage": decision.get("coverage"),
        "frame_sequence": decision.get("frame_sequence"),
        "record": result,
        "config": cfg,
    }


# ============================================================
# FASE 2C — ESTACIÓN GUIADA DE CAPTURA
#
# Posicionamiento de la prenda dentro del área de inspección y
# captura automática (1 paso físico = 1 captura).
# Solo lógica pura: no toca cámara, Flask, inspections ni producción.
# ============================================================

GUIDED_STATE_SIN_PRENDA = "SIN_PRENDA"
GUIDED_STATE_ENTRANDO = "ENTRANDO"
GUIDED_STATE_AJUSTAR = "AJUSTAR"
GUIDED_STATE_POSICION_VALIDA = "POSICION_VALIDA"
GUIDED_STATE_CAPTURANDO = "CAPTURANDO"
GUIDED_STATE_CAPTURADA = "CAPTURADA"

GUIDED_STATES = (
    GUIDED_STATE_SIN_PRENDA,
    GUIDED_STATE_ENTRANDO,
    GUIDED_STATE_AJUSTAR,
    GUIDED_STATE_POSICION_VALIDA,
    GUIDED_STATE_CAPTURANDO,
    GUIDED_STATE_CAPTURADA,
)

GUIDED_STATE_LABELS = {
    GUIDED_STATE_SIN_PRENDA: "SIN PRENDA",
    GUIDED_STATE_ENTRANDO: "COLOQUE LA PRENDA",
    GUIDED_STATE_AJUSTAR: "AJUSTAR POSICION",
    GUIDED_STATE_POSICION_VALIDA: "POSICION CORRECTA",
    GUIDED_STATE_CAPTURANDO: "GUARDANDO",
    GUIDED_STATE_CAPTURADA: "CAPTURA GUARDADA",
}

# Motivos internos -> mensaje que sí se le explica a la usuaria.
# La operadora NO ve motivos técnicos: solo qué hacer ahora mismo.
GUIDED_MESSAGES = {
    "NO_GARMENT": "Coloque la prenda dentro del área",
    "LOW_COVERAGE": "Espere, la prenda está entrando",
    "HIGH_COVERAGE": "Coloque la prenda dentro del área",
    "BLUR": "Imagen posiblemente borrosa. Mantenga la prenda quieta",
    "EDGE": "Mueva un poco para centrar la prenda",
    "MOVE_LEFT": "Mueva un poco a la izquierda",
    "MOVE_RIGHT": "Mueva un poco a la derecha",
    "MOVE_UP": "Mueva un poco hacia arriba",
    "MOVE_DOWN": "Mueva un poco hacia abajo",
    "ALIGNED": "Posición correcta",
    "READY": "Lista para capturar",
    "CAPTURING": "Guardando la captura",
    "CAPTURED": (
        "Captura revisada. Antes de capturar, retire las manos del área"
    ),
    "WAITING": "Un momento, verificando la posición",
    "NO_SIGNAL": "Esperando la señal de la cámara",
    "CAMERA_MISSING": "La cámara no está disponible",
}

# Colores del overlay (SIEMPRE acompañados de texto/ícono).
GUIDED_STATE_COLORS = {
    GUIDED_STATE_SIN_PRENDA: (150, 150, 150),
    GUIDED_STATE_ENTRANDO: (60, 140, 245),
    GUIDED_STATE_AJUSTAR: (0, 160, 255),
    GUIDED_STATE_POSICION_VALIDA: (60, 190, 90),
    GUIDED_STATE_CAPTURANDO: (40, 220, 140),
    GUIDED_STATE_CAPTURADA: (30, 170, 120),
}

# Estados en los que la prenda ya está correctamente posicionada.
GUIDED_VALID_STATES = frozenset(
    {
        GUIDED_STATE_POSICION_VALIDA,
        GUIDED_STATE_CAPTURANDO,
        GUIDED_STATE_CAPTURADA,
    }
)

AI_GUIDED_DEFAULTS = {
    # Presencia: por debajo no hay prenda en el área.
    "present_coverage": 0.28,
    # Salida: tras persistir, esperar a que la prenda se retire.
    "exit_coverage": 0.12,
    # Cobertura máxima: la prenda no debe desbordar el área.
    "max_coverage": 0.98,
    # Tolerancia del centro de la prenda respecto al centro del ROI
    # (fracción del ancho/alto del ROI). Permite variación normal.
    "center_tol_x": 0.12,
    "center_tol_y": 0.14,
    # Margen mínimo respecto a los bordes del ROI (fracción).
    "edge_margin": 0.05,
    # Frames consecutivos válidos antes de capturar automáticamente.
    "stability_frames": 4,
    # Frames de presencia antes de dar por confirmada la entrada.
    "enter_frames": 3,
    # Frames sin prenda antes de dar por completado el paso.
    "exit_frames": 3,
    # Captura automática: por defecto OFF. El flujo guiado es
    # MANUAL (botón "CAPTURAR IMAGEN") y solo se usa el automático
    # si se activa explícitamente con AI_GUIDED_AUTO_CAPTURE=true.
    "auto_capture": 0,
}


def get_ai_guided_config() -> dict:
    """Umbrales de posicionamiento (env con defaults documentados)."""
    env_map = {
        "AI_GUIDED_PRESENT_COVERAGE": ("present_coverage", float),
        "AI_GUIDED_EXIT_COVERAGE": ("exit_coverage", float),
        "AI_GUIDED_MAX_COVERAGE": ("max_coverage", float),
        "AI_GUIDED_CENTER_TOL_X": ("center_tol_x", float),
        "AI_GUIDED_CENTER_TOL_Y": ("center_tol_y", float),
        "AI_GUIDED_EDGE_MARGIN": ("edge_margin", float),
        "AI_GUIDED_STABILITY_FRAMES": ("stability_frames", int),
        "AI_GUIDED_ENTER_FRAMES": ("enter_frames", int),
        "AI_GUIDED_EXIT_FRAMES": ("exit_frames", int),
        "AI_GUIDED_AUTO_CAPTURE": ("auto_capture", int),
    }

    values = dict(AI_GUIDED_DEFAULTS)

    for env_name, (key, caster) in env_map.items():
        raw = os.getenv(env_name, "")
        if str(raw).strip() == "":
            continue
        if key == "auto_capture":
            values[key] = 1 if str(raw).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ) else 0
            continue
        try:
            values[key] = caster(raw)
        except ValueError as error:
            raise AIDomainError(
                f"{env_name} inválido: {raw!r}"
            ) from error

    base = get_ai_capture_config()
    values["min_coverage"] = float(base["min_coverage"])
    values["min_sharpness"] = float(base["min_sharpness"])

    present = float(values["present_coverage"])
    minimum = float(values["min_coverage"])
    maximum = float(values["max_coverage"])
    exit_cov = float(values["exit_coverage"])

    if not 0.0 < present <= 1.0:
        raise AIDomainError(
            "AI_GUIDED_PRESENT_COVERAGE debe estar en (0, 1]."
        )
    if not present < minimum:
        raise AIDomainError(
            "AI_GUIDED_PRESENT_COVERAGE debe ser menor que "
            "AI_CAPTURE_MIN_COVERAGE."
        )
    if not exit_cov < present:
        raise AIDomainError(
            "AI_GUIDED_EXIT_COVERAGE debe ser menor que "
            "AI_GUIDED_PRESENT_COVERAGE."
        )
    if not minimum <= maximum <= 1.0:
        raise AIDomainError(
            "AI_GUIDED_MAX_COVERAGE debe estar entre la cobertura "
            "mínima y 1."
        )
    for key in ("center_tol_x", "center_tol_y"):
        if not 0.0 <= float(values[key]) <= 0.5:
            raise AIDomainError(
                f"{key} debe estar entre 0 y 0.5."
            )
    if not 0.0 <= float(values["edge_margin"]) <= 0.25:
        raise AIDomainError("AI_GUIDED_EDGE_MARGIN debe estar entre 0 y 0.25.")
    for key in ("stability_frames", "enter_frames", "exit_frames"):
        if int(values[key]) < 1:
            raise AIDomainError(f"{key} debe ser >= 1.")

    return {
        "present_coverage": present,
        "exit_coverage": exit_cov,
        "min_coverage": minimum,
        "max_coverage": maximum,
        "min_sharpness": float(values["min_sharpness"]),
        "center_tol_x": float(values["center_tol_x"]),
        "center_tol_y": float(values["center_tol_y"]),
        "edge_margin": float(values["edge_margin"]),
        "stability_frames": int(values["stability_frames"]),
        "enter_frames": int(values["enter_frames"]),
        "exit_frames": int(values["exit_frames"]),
        "auto_capture": bool(int(values["auto_capture"])),
        "target_images": int(base["target_images"]),
        "duplicate_max_distance": int(base["duplicate_max_distance"]),
        "keep_rejected": bool(base["keep_rejected"]),
        "jpeg_quality": int(base["jpeg_quality"]),
    }


def mask_bbox(mask, cv2_mod):
    """(x1, y1, x2, y2) de los píxeles no nulos o None si está vacío."""
    if mask is None or cv2_mod is None:
        return None
    points = cv2_mod.findNonZero(mask)
    if points is None or len(points) == 0:
        return None
    x, y, w, h = cv2_mod.boundingRect(points)
    if int(w) <= 0 or int(h) <= 0:
        return None
    return (int(x), int(y), int(x + w), int(y + h))


def classify_position(
    *,
    roi,
    bbox,
    coverage,
    sharpness,
    config=None,
) -> dict:
    """
    Evalúa UN frame: ¿la prenda está bien colocada dentro del ROI?

    Criterios (en este orden):
    1. prenda presente dentro del ROI;
    2. cobertura dentro de rango configurable;
    3. nitidez mínima;
    4. margen suficiente respecto a los bordes;
    5. centro del bounding box dentro de tolerancia X/Y.

    Devuelve motivo + mensaje humano + desplazamiento normalizado.
    """
    cfg = dict(config or get_ai_guided_config())

    rx1, ry1, rx2, ry2 = (int(v) for v in roi)
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)

    try:
        cov = float(coverage)
    except (TypeError, ValueError):
        cov = -1.0

    try:
        sharp = None if sharpness is None else float(sharpness)
    except (TypeError, ValueError):
        sharp = None

    target_center = (rx1 + rw / 2.0, ry1 + rh / 2.0)

    if bbox is None or cov < float(cfg["present_coverage"]):
        return {
            "present": False,
            "aligned": False,
            "reason": "NO_GARMENT",
            "message": GUIDED_MESSAGES["NO_GARMENT"],
            "dx": 0.0,
            "dy": 0.0,
            "garment_center": None,
            "target_center": target_center,
            "coverage": max(0.0, cov),
            "sharpness": sharp,
            "roi": (rx1, ry1, rx2, ry2),
            "bbox": bbox,
        }

    bx1, by1, bx2, by2 = (int(v) for v in bbox)
    gcx = (bx1 + bx2) / 2.0
    gcy = (by1 + by2) / 2.0

    # Desplazamiento normalizado: positivo = prenda a la derecha/abajo.
    dx = (gcx - target_center[0]) / rw
    dy = (gcy - target_center[1]) / rh

    base = {
        "present": True,
        "garment_center": (gcx, gcy),
        "target_center": target_center,
        "dx": dx,
        "dy": dy,
        "coverage": cov,
        "sharpness": sharp,
        "roi": (rx1, ry1, rx2, ry2),
        "bbox": (bx1, by1, bx2, by2),
    }

    def result(reason, aligned):
        message = GUIDED_MESSAGES.get(
            reason,
            GUIDED_MESSAGES["NO_GARMENT"],
        )
        return {
            **base,
            "aligned": bool(aligned),
            "reason": reason,
            "message": message,
        }

    # 2) Cobertura dentro de rango configurable.
    if cov < float(cfg["min_coverage"]):
        return result("LOW_COVERAGE", False)
    if cov > float(cfg["max_coverage"]):
        return result("HIGH_COVERAGE", False)

    # 3) Nitidez mínima ( movimiento / desenfoque ).
    if sharp is None or sharp < float(cfg["min_sharpness"]):
        return result("BLUR", False)

    # 4) Margen suficiente respecto a los bordes del ROI.
    margin_x = float(cfg["edge_margin"]) * rw
    margin_y = float(cfg["edge_margin"]) * rh

    gaps = (
        ("MOVE_RIGHT", float(bx1 - rx1)),  # demasiado a la izquierda
        ("MOVE_LEFT", float(rx2 - bx2)),   # demasiado a la derecha
        ("MOVE_DOWN", float(by1 - ry1)),   # demasiado arriba
        ("MOVE_UP", float(ry2 - by2)),     # demasiado abajo
    )
    horizontal = gaps[:2]
    vertical = gaps[2:]

    worst_h = min(horizontal, key=lambda item: item[1])
    worst_v = min(vertical, key=lambda item: item[1])

    if worst_h[1] < margin_x:
        return result(worst_h[0], False)
    if worst_v[1] < margin_y:
        return result(worst_v[0], False)

    # 5) Centro dentro de tolerancia X/Y.
    if abs(dx) > float(cfg["center_tol_x"]):
        return result(
            "MOVE_RIGHT" if dx < 0 else "MOVE_LEFT",
            False,
        )
    if abs(dy) > float(cfg["center_tol_y"]):
        return result(
            "MOVE_DOWN" if dy < 0 else "MOVE_UP",
            False,
        )

    return result("ALIGNED", True)


class GuidedTracker:
    """
    Máquina de estados de la estación guiada.

    - 1 captura automática por paso físico (sin botón por prenda).
    - Tras persistir, espera a que la prenda salga del área.
    - Selecciona el mejor frame de cada pasada entre los válidos.
    """

    def __init__(self, config=None):
        self.config = dict(config or get_ai_guided_config())
        self.reset()

    # ----------------------------------------------------
    def reset(self) -> None:
        """Limpia el paso actual (prenda fuera o nuevo comienzo)."""
        self.state = GUIDED_STATE_SIN_PRENDA
        self.message = GUIDED_MESSAGES["NO_GARMENT"]
        self.reason = "NO_GARMENT"
        self.aligned = False
        self.ready = False
        self.dx = 0.0
        self.dy = 0.0
        self.present_streak = 0
        self.valid_streak = 0
        self.exit_streak = 0
        self.pass_captured = False
        self.awaiting_persist = False
        self.best = None
        self.last_decision = None
        self.last_result = None

    # ----------------------------------------------------
    @property
    def stability_frames(self) -> int:
        return int(self.config["stability_frames"])

    @property
    def capture_reached(self) -> bool:
        """True cuando ya se pidió/realizó la captura de este paso."""
        return bool(self.pass_captured or self.awaiting_persist)

    def _score(self, coverage, sharpness, dx, dy) -> float:
        """Mejor frame = mejor calidad + prenda más centrada."""
        quality = compute_quality_score(
            coverage,
            sharpness,
            min_coverage=float(self.config["min_coverage"]),
            min_sharpness=float(self.config["min_sharpness"]),
        )
        # Penalización suave por descentrado (documentada, no mágica).
        return float(quality) - 0.10 * (abs(dx) + abs(dy))

    def _track_best(self, frame, coverage, sharpness, sequence, decision):
        if frame is None:
            return
        score = self._score(
            coverage,
            sharpness,
            decision["dx"],
            decision["dy"],
        )
        if self.best is None or score > self.best["score"]:
            self.best = {
                "frame": frame,
                "coverage": float(coverage),
                "sharpness": float(sharpness),
                "sequence": int(sequence),
                "score": float(score),
            }

    # ----------------------------------------------------
    def step(
        self,
        *,
        roi,
        bbox,
        coverage,
        sharpness,
        sequence: int = 0,
        frame=None,
    ) -> dict:
        """Alimenta un frame y devuelve el estado visible."""
        cfg = self.config
        decision = classify_position(
            roi=roi,
            bbox=bbox,
            coverage=coverage,
            sharpness=sharpness,
            config=cfg,
        )
        self.last_decision = decision
        self.aligned = decision["aligned"]
        self.reason = decision["reason"]
        self.dx = decision["dx"]
        self.dy = decision["dy"]

        # --- persistencia en curso: no decidir de nuevo. ---
        if self.state == GUIDED_STATE_CAPTURANDO:
            self.message = GUIDED_MESSAGES["CAPTURING"]
            return self._snapshot(False, decision)

        # --- tras la captura: esperar a que la prenda salga. ---
        if self.state == GUIDED_STATE_CAPTURADA:
            if decision["coverage"] <= float(cfg["exit_coverage"]):
                self.exit_streak += 1
            else:
                self.exit_streak = 0

            if self.exit_streak >= int(cfg["exit_frames"]):
                self.reset()
                self.message = GUIDED_MESSAGES["NO_GARMENT"]
                return self._snapshot(False, decision)

            self.message = GUIDED_MESSAGES["CAPTURED"]
            return self._snapshot(False, decision)

        # --- sin prenda en el área: nuevo paso. ---
        if not decision["present"]:
            # Señal intermitente del detector: si todavía queda
            # cobertura sobre la prenda NO volvemos a "SIN PRENDA".
            # Así una prenda visible dentro del área nunca muestra
            # un estado falso de "no hay nada".
            if (
                not self.pass_captured
                and self.present_streak
                and float(decision.get("coverage") or 0.0)
                >= float(cfg["exit_coverage"])
            ):
                self.valid_streak = 0
                self.aligned = False
                self.ready = False
                self.state = GUIDED_STATE_ENTRANDO
                self.message = GUIDED_MESSAGES["LOW_COVERAGE"]
                return self._snapshot(False, decision)

            if (
                self.pass_captured
                or self.present_streak
                or self.valid_streak
                or self.best is not None
            ):
                self.reset()
            self.message = GUIDED_MESSAGES["NO_GARMENT"]
            return self._snapshot(False, decision)

        # --- hay prenda. ---
        self.present_streak += 1

        if decision["aligned"]:
            self.valid_streak += 1
            self._track_best(
                frame,
                decision["coverage"],
                decision["sharpness"],
                sequence,
                decision,
            )
        else:
            self.valid_streak = 0

        if decision["reason"] in ("LOW_COVERAGE", "BLUR"):
            self.state = GUIDED_STATE_ENTRANDO
            self.message = decision["message"]
        elif not decision["aligned"]:
            self.state = GUIDED_STATE_AJUSTAR
            self.message = decision["message"]
        elif self.present_streak < int(cfg["enter_frames"]):
            self.state = GUIDED_STATE_ENTRANDO
            self.message = GUIDED_MESSAGES["WAITING"]
        else:
            self.state = GUIDED_STATE_POSICION_VALIDA
            self.message = GUIDED_MESSAGES["ALIGNED"]

        wants_capture = (
            self.state == GUIDED_STATE_POSICION_VALIDA
            and self.valid_streak >= int(cfg["stability_frames"])
            and not self.capture_reached
            and self.best is not None
        )

        # Lista para capturar = la posición es válida y suficientemente
        # estable. Es lo único que habilita el botón manual.
        self.ready = bool(wants_capture)

        return self._snapshot(wants_capture, decision)

    # ----------------------------------------------------
    def begin_capture(self) -> None:
        """Marca que se va a persistir el mejor frame del paso."""
        self.state = GUIDED_STATE_CAPTURANDO
        self.awaiting_persist = True
        self.ready = False
        self.message = GUIDED_MESSAGES["CAPTURING"]

    def confirm_capture(self, result=None) -> None:
        """Captura persistida: esperar a que la prenda salga."""
        self.pass_captured = True
        self.awaiting_persist = False
        self.ready = False
        self.exit_streak = 0
        self.state = GUIDED_STATE_CAPTURADA
        self.message = GUIDED_MESSAGES["CAPTURED"]
        self.last_result = result

    def cancel_capture(self) -> None:
        """
        La captura manual no se pudo guardar: se vuelve al paso de
        posicionamiento para que la operadora pueda reintentar.
        """
        self.awaiting_persist = False
        self.state = (
            GUIDED_STATE_POSICION_VALIDA
            if self.aligned and self.best is not None
            else GUIDED_STATE_ENTRANDO
        )
        self.message = (
            GUIDED_MESSAGES["ALIGNED"]
            if self.state == GUIDED_STATE_POSICION_VALIDA
            else GUIDED_MESSAGES["LOW_COVERAGE"]
        )

    def release(self) -> None:
        """
        La persistencia falló: no reintentar en el mismo paso
        (evita bucles); se espera a que la prenda salga.
        """
        self.awaiting_persist = False
        self.pass_captured = True
        self.state = GUIDED_STATE_CAPTURADA
        self.message = GUIDED_MESSAGES["CAPTURED"]

    def best_candidate(self, token: int) -> "PresenceCandidate | None":
        """Candidato (mejor frame del paso) listo para persistir."""
        if self.best is None:
            return None
        return PresenceCandidate(
            garment_token=int(token),
            best_frame=self.best["frame"],
            best_coverage=float(self.best["coverage"]),
            best_sequence=int(self.best["sequence"]),
        )

    # ----------------------------------------------------
    def snapshot(self) -> dict:
        """Estado visible actual, sin pedir una nueva captura."""
        return self._snapshot(False, self.last_decision)

    def _snapshot(self, wants_capture: bool, decision=None) -> dict:
        decision = decision or self.last_decision or {}
        return {
            "state": self.state,
            "state_label": GUIDED_STATE_LABELS.get(
                self.state,
                self.state,
            ),
            "message": self.message,
            "reason": self.reason,
            "aligned": bool(self.aligned),
            "present": bool(decision.get("present")),
            "dx": round(float(self.dx), 4),
            "dy": round(float(self.dy), 4),
            "coverage": decision.get("coverage"),
            "sharpness": decision.get("sharpness"),
            "bbox": decision.get("bbox"),
            "roi": decision.get("roi"),
            "garment_center": decision.get("garment_center"),
            "target_center": decision.get("target_center"),
            "present_streak": int(self.present_streak),
            "valid_streak": int(self.valid_streak),
            "stability_frames": int(self.config["stability_frames"]),
            "wants_capture": bool(wants_capture),
            "ready": bool(self.ready),
            "ready_message": GUIDED_MESSAGES["READY"],
            "captured_this_pass": bool(self.pass_captured),
            "has_best": self.best is not None,
            "last_result": self.last_result,
        }
