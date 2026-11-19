"""Configuración de Gunicorn para Astrid Quality Control."""

from __future__ import annotations

import os

bind = f"0.0.0.0:{int(os.getenv('APP_PORT', '5000'))}"

# El estado de la estación, el lector RTSP y el modo automático viven en
# memoria del proceso. Más de un worker produciría lectores y estados
# independientes; por eso se usa un worker con varios threads.
workers = 1
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", "8"))

timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
capture_output = True
