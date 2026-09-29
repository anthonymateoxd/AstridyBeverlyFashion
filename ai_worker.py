"""FASE 3A — Worker asíncrono de entrenamiento PatchCore.

Proceso separado de Flask. Ciclo:

    1. reclamar jobs huérfanos (worker caído) -> FALLIDO;
    2. reclamar el job TRAINING PENDIENTE más antiguo (SKIP LOCKED);
    3. ejecutarlo con `ai_training.execute_training_job`;
    4. mantener heartbeat mientras corre;
    5. repetir (o salir con AI_WORKER_ONCE=1 para pruebas).

Nunca entrena dentro de una petición HTTP y nunca toca
PATCHCORE_CKPT (inferencia productiva).
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time

from ai_domain import (
    claim_next_training_job,
    reclaim_orphaned_jobs,
    heartbeat_job,
)
from ai_training import default_connect, execute_training_job

log = logging.getLogger("ai_worker")

_stop = threading.Event()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)

    if raw is None or str(raw).strip() == "":
        return float(default)

    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning("Valor inválido en %s=%r; se usa %s", name, raw, default)
        return float(default)


def default_worker_id() -> str:
    base = os.environ.get("AI_WORKER_ID")

    if base and base.strip():
        return base.strip()[:120]

    return f"{socket.gethostname()}-{os.getpid()}"[:120]


def reclaim_stale_jobs(connect, stale_seconds: int) -> int:
    """Falla los jobs EN_CURSO sin heartbeat reciente (sin reanudar)."""
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            stale = reclaim_orphaned_jobs(
                cur,
                stale_seconds=int(stale_seconds),
            )
            conn.commit()

            for job in stale:
                log.warning(
                    "Job huérfano marcado como FALLIDO: job_id=%s "
                    "worker=%s",
                    job.get("id"),
                    job.get("worker_id"),
                )

            return len(stale)
        finally:
            cur.close()
    finally:
        conn.close()


def claim_job(connect, worker_id: str) -> dict | None:
    """Reclama (transaccionalmente) el siguiente job PENDIENTE."""
    conn = connect()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            conn.start_transaction()
            claimed = claim_next_training_job(cur, worker_id)
            conn.commit()
            return claimed
        finally:
            cur.close()
    finally:
        conn.close()


def start_heartbeat(connect, job_id: int, worker_id: str, interval: float):
    """Hilo que marca vida del worker sobre el job EN_CURSO."""
    stop = threading.Event()

    def _loop():
        while not stop.wait(interval):
            try:
                conn = connect()

                try:
                    cur = conn.cursor(dictionary=True)
                    try:
                        conn.start_transaction()
                        heartbeat_job(cur, int(job_id), worker_id)
                        conn.commit()
                    finally:
                        cur.close()
                finally:
                    conn.close()
            except Exception as error:  # noqa: BLE001 - el job sigue
                log.warning("heartbeat job_id=%s falló: %s", job_id, error)

    thread = threading.Thread(
        target=_loop,
        name=f"ai-heartbeat-{job_id}",
        daemon=True,
    )
    thread.start()
    return thread, stop


def run_once(connect, worker_id: str, *, orphan_seconds: int) -> bool:
    """Un ciclo completo: reclamar huérfanos, reclamar y ejecutar."""
    reclaimed = reclaim_stale_jobs(connect, orphan_seconds)

    if reclaimed:
        log.warning("%s job(s) huérfano(s) resuelto(s)", reclaimed)

    claimed = claim_job(connect, worker_id)

    if claimed is None:
        return False

    job_id = int(claimed["id"])
    log.info(
        "Job reclamado job_id=%s ai_model_id=%s dataset_id=%s",
        job_id,
        claimed.get("ai_model_id"),
        claimed.get("dataset_id"),
    )

    heartbeat, heartbeat_stop = start_heartbeat(
        connect,
        job_id,
        worker_id,
        _env_float("AI_WORKER_HEARTBEAT_SECONDS", 15.0),
    )

    try:
        result = execute_training_job(job_id, connect=connect)
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=5)

    if result.get("ok"):
        log.info("Job completado job_id=%s", job_id)
    else:
        log.error(
            "Job fallido job_id=%s motivo=%s",
            job_id,
            result.get("error"),
        )

    return True


def serve(connect=None, *, worker_id: str | None = None) -> None:
    """Bucle infinito (una sola pasada con AI_WORKER_ONCE=1)."""
    connect = connect or default_connect
    worker = worker_id or default_worker_id()
    poll = _env_float("AI_WORKER_POLL_SECONDS", 3.0)
    orphan = int(_env_float("AI_WORKER_ORPHAN_SECONDS", 900.0))
    once = str(os.environ.get("AI_WORKER_ONCE", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    log.info(
        "AI worker iniciado worker_id=%s poll=%.1fs orphan=%ss once=%s",
        worker,
        poll,
        orphan,
        once,
    )

    while not _stop.is_set():
        try:
            worked = run_once(connect, worker, orphan_seconds=orphan)
        except Exception:  # noqa: BLE001 - el worker no debe morir
            log.exception("Ciclo del worker falló; se reintenta")
            worked = False

        if once:
            break

        if not worked:
            _stop.wait(poll)

    log.info("AI worker detenido worker_id=%s", worker)


def _handle_signal(signum, _frame):
    log.info("Señal %s recibida: deteniendo worker", signum)
    _stop.set()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("AI_WORKER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    serve()


if __name__ == "__main__":
    main()
