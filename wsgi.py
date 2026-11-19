"""Punto de entrada WSGI para el servidor de producción.

Se mantiene un único proceso Gunicorn para garantizar exactamente un lector
RTSP y compartir el estado de la estación automática dentro del proceso.
"""

from app import app, ensure_camera_capture_worker, init_db


# CAPÍTULO V — Inicio del servidor de aplicación.
# Prepara la base de datos y el lector de cámara al iniciar el proceso
# que atiende las solicitudes web mediante Gunicorn.
def bootstrap_runtime() -> None:
    """Inicializa esquema y lector de cámara una sola vez por proceso."""
    init_db()
    ensure_camera_capture_worker()


bootstrap_runtime()
