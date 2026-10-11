import socket


DIAGNOSTICS = {
    "CAMERA_CHECKING": {
        "severity": "info",
        "title": "Comprobando cámara",
        "cause": "El sistema todavía está verificando el stream de video.",
        "action": "Espere unos segundos. Si no cambia, use Reconectar.",
    },
    "CAMERA_RECONNECTING": {
        "severity": "warning",
        "title": "Reconectando cámara",
        "cause": "La transmisión se interrumpió y el sistema está intentando abrirla de nuevo.",
        "action": "Espere el reintento automático. Si persiste, use Reconectar.",
    },
    "CAMERA_RTSP_UNREACHABLE": {
        "severity": "error",
        "title": "La cámara no responde por RTSP",
        "cause": "La IP o el puerto RTSP configurado no aceptan conexión.",
        "action": (
            "Verifique que la cámara esté encendida y conectada. "
            "Si está en otra red, revise Tailscale y la ruta hacia la IP de la cámara."
        ),
    },
    "CAMERA_RTSP_OPEN_FAILED": {
        "severity": "error",
        "title": "La cámara responde, pero el video no abre",
        "cause": (
            "El puerto RTSP está disponible, pero OpenCV/FFmpeg no pudo iniciar el stream."
        ),
        "action": (
            "Use Reconectar. Si vuelve a ocurrir, revise usuario/contraseña, "
            "canal, tipo de stream y que no haya demasiadas sesiones RTSP abiertas."
        ),
    },
    "CAMERA_AUTH_FAILED": {
        "severity": "error",
        "title": "La cámara rechazó la autenticación",
        "cause": "El servidor RTSP respondió, pero no aceptó las credenciales configuradas.",
        "action": "Revise el usuario y la contraseña de la cámara en la configuración segura.",
    },
    "CAMERA_STREAM_NO_FRAMES": {
        "severity": "error",
        "title": "El stream dejó de entregar imágenes",
        "cause": (
            "La conexión RTSP estaba abierta, pero dejaron de llegar frames válidos."
        ),
        "action": (
            "Use Reconectar. Si se repite, revise la estabilidad de red "
            "o considere usar el substream para reducir carga y ancho de banda."
        ),
    },
    "CAMERA_STREAM_READ_ERROR": {
        "severity": "error",
        "title": "Error al leer el video",
        "cause": "OpenCV/FFmpeg reportó un error mientras leía el stream RTSP.",
        "action": "Use Reconectar. Si persiste, revise red, códec y configuración RTSP.",
    },
    "CAMERA_RUNTIME_UNAVAILABLE": {
        "severity": "error",
        "title": "El componente de cámara no está disponible",
        "cause": "OpenCV no está disponible en el proceso que atiende la estación.",
        "action": "Reinicie el servicio de la aplicación y revise la instalación de OpenCV.",
    },
    "CAMERA_OPEN_ERROR": {
        "severity": "error",
        "title": "No se pudo iniciar la cámara",
        "cause": "Ocurrió un error inesperado al abrir la conexión RTSP.",
        "action": "Use Reconectar. Si persiste, revise los detalles técnicos.",
    },
    "CAMERA_UNKNOWN": {
        "severity": "error",
        "title": "Cámara no disponible",
        "cause": "El sistema no está recibiendo video y no pudo determinar una causa más específica.",
        "action": "Use Reconectar y revise los detalles técnicos.",
    },
}


def probe_tcp_port(host, port, timeout=0.8):
    """Best-effort TCP probe used only after a failed RTSP open."""
    if not host:
        return False
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout)):
            return True
    except (OSError, ValueError, TypeError):
        return False


def normalize_camera_error_code(error_code, last_error=None, reconnecting=False):
    text = str(last_error or "").lower()

    if any(token in text for token in ("401", "unauthorized", "authentication", "auth failed")):
        return "CAMERA_AUTH_FAILED"

    if error_code:
        return str(error_code)

    if reconnecting:
        return "CAMERA_RECONNECTING"

    return "CAMERA_UNKNOWN"


def build_camera_diagnostic(
    error_code,
    *,
    last_error=None,
    reconnecting=False,
    technical=None,
):
    code = normalize_camera_error_code(
        error_code,
        last_error=last_error,
        reconnecting=reconnecting,
    )
    base = dict(DIAGNOSTICS.get(code, DIAGNOSTICS["CAMERA_UNKNOWN"]))
    base["code"] = code
    # El texto crudo del error se usa solo para clasificar internamente.
    # No se envía al navegador porque algunas librerías pueden incluir
    # detalles sensibles de la URL RTSP.
    base["technical"] = dict(technical or {})
    return base
