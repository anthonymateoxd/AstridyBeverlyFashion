# Operación y diagnóstico de cámara

Este documento describe la configuración de cámara RTSP usada por la estación de inspección y el procedimiento recomendado para diagnosticar fallos sin exponer credenciales.

## Configuración recomendada

La cámara debe tener una IP fija o una reserva DHCP estable. En la instalación de tesis se usa:

```env
CAMERA_IP=192.168.0.5
CAMERA_RTSP_PORT=554
CAMERA_CHANNEL=1
CAMERA_SUBTYPE=0
CAMERA_OPEN_TIMEOUT_MS=4000
CAMERA_READ_TIMEOUT_MS=4000
CAMERA_DISCOVERY_ENABLED=0
```

`CAMERA_PASSWORD` y otras credenciales pertenecen únicamente al archivo `.env` privado de producción y nunca deben versionarse.

### Tailscale

Cuando AWS llega a la cámara mediante un subnet router y una ruta específica como `192.168.0.5/32`, el descubrimiento automático debe permanecer desactivado. El servidor solo tiene garantizado acceso a esa dirección, no a toda la red `192.168.0.0/24`.

Active `CAMERA_DISCOVERY_ENABLED=1` únicamente cuando el servidor tenga acceso real a la subred completa y exista una necesidad explícita de localizar una cámara cuya IP pueda cambiar.

## Diagnóstico mostrado en la Estación

La Estación devuelve un diagnóstico estructurado con causa probable, acción recomendada y detalles técnicos. Los códigos principales son:

| Código | Significado | Acción recomendada |
| --- | --- | --- |
| `CAMERA_OK` | Se reciben frames correctamente. | Ninguna. |
| `CAMERA_CHECKING` | El proceso está abriendo o comprobando el stream. | Esperar unos segundos; si persiste, usar Reconectar. |
| `CAMERA_RECONNECTING` | La transmisión se interrumpió y se está reabriendo. | Esperar el reintento automático o usar Reconectar. |
| `CAMERA_RTSP_UNREACHABLE` | La IP o el puerto RTSP no aceptan conexión. | Revisar energía, red local, IP, puerto y ruta Tailscale. |
| `CAMERA_RTSP_OPEN_FAILED` | El puerto responde, pero OpenCV/FFmpeg no puede abrir el stream. | Reconectar y revisar credenciales, canal, tipo de stream o sesiones RTSP. |
| `CAMERA_AUTH_FAILED` | El servidor RTSP rechazó autenticación. | Revisar credenciales en el `.env` privado. |
| `CAMERA_STREAM_NO_FRAMES` | El stream estaba abierto, pero dejó de entregar imágenes. | Reconectar; si se repite, revisar estabilidad de red o usar substream. |
| `CAMERA_STREAM_READ_ERROR` | Ocurrió un error durante la lectura del stream. | Reconectar y revisar red, códec y RTSP. |
| `CAMERA_RUNTIME_UNAVAILABLE` | OpenCV no está disponible. | Revisar el contenedor/servicio de aplicación. |
| `CAMERA_OPEN_ERROR` | Error inesperado al abrir la cámara. | Reconectar y revisar detalles técnicos. |

Los detalles técnicos visibles no contienen contraseña ni URL RTSP completa.

## Procedimiento de recuperación

1. Revise el mensaje de la Estación y abra **Ver detalles técnicos**.
2. Si aparece `CAMERA_RTSP_UNREACHABLE`, compruebe primero la cámara desde el equipo que comparte la red local.
3. Confirme que la IP esperada responde y que el puerto `554` está disponible.
4. Si la cámara funciona localmente pero AWS no la alcanza, revise Tailscale y la ruta anunciada.
5. Si RTSP responde pero no llegan frames, use **Reconectar**.
6. Si el problema se repite por ancho de banda o latencia, pruebe temporalmente `CAMERA_SUBTYPE=1` (substream) antes de cambiar otros parámetros.
7. No active descubrimiento por `/24` en una topología donde AWS solo tiene una ruta `/32`.

## Prevención

- Mantener la cámara con IP fija o reserva DHCP.
- Mantener activo el equipo que anuncia la ruta Tailscale durante la operación.
- Desactivar la expiración de la clave del subnet router cuando corresponda a un equipo operativo permanente.
- No exponer el puerto RTSP directamente a Internet.
- No versionar usuario, contraseña ni secretos.
- Conservar timeouts y reintentos en variables de entorno para poder ajustar sin modificar código.

## Nota de producción

El diagnóstico es informativo y no modifica lotes, inspecciones, datasets ni modelos IA. La acción **Reconectar** reinicia únicamente la conexión de cámara administrada por la aplicación.
