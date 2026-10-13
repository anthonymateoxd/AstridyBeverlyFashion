# Estabilidad de la cámara — diagnóstico y prueba controlada

**Proyecto:** Astrid & Beverly Fashion

**Fecha de la prueba:** 12 de octubre de 2026 (Guatemala; 13 de octubre UTC)

**Entorno:** AWS EC2 `astrid-quality-thesis`, Dahua `192.168.0.5:554`, PC Windows Anthony y Tailscale.

## Arquitectura actual y su límite

La cámara Dahua está en la LAN `192.168.0.0/24`. Windows Anthony anuncia por Tailscale exclusivamente `192.168.0.5/32`; EC2 accede a RTSP a través de este router de subred. La aplicación Flask usa un único lector OpenCV/FFmpeg TCP y entrega video MJPEG al navegador.

**El equipo Windows es un punto único de fallo.** Si se apaga, pierde Wi-Fi, entra en suspensión o Tailscale deja de anunciar la ruta, AWS no podrá recibir imágenes aunque el backend esté sano. Una cámara apagada o sin conexión de red tampoco puede recuperarse mediante software en AWS.

El servicio Tailscale de Windows está configurado con inicio automático. Su plan de energía no suspende el equipo conectado a corriente, pero en batería tiene configurada suspensión a los tres minutos (dato observado al revisar).

## Hallazgo adicional: contenedor candidato redundante

En AWS estaban activos simultáneamente `astrid_quality_app` y `astrid_quality_app_candidate`. El candidato no tenía puerto publicado, tenía política `restart=no` y registraba múltiples intentos de localizar la cámara en la subred. Se detuvo **sin eliminarlo**; el contenedor principal siguió saludable, la cámara respondió en TCP 554 y MySQL no se reinició.

Puede arrancarse manualmente con `docker start astrid_quality_app_candidate` únicamente si existiera un motivo operativo confirmado. Evitar mantener lectores RTSP o exploraciones de cámara duplicadas en producción.

## Prueba de pérdida controlada

- **Método:** regla temporal `iptables` limitada a TCP 554 entre el contenedor principal y `192.168.0.5`. No se desconectó físicamente la cámara.
- **Seguridad:** limpieza desde el mismo procedimiento y temporizador systemd de reserva para retirar la regla. No se modificaron MySQL, lotes ni modelos.
- **Inicio del corte:** `2026-10-13T04:38:12Z`.
- **Fin del corte:** `2026-10-13T04:38:28Z`.
- **Duración simulada:** 16 segundos.
- **Paquetes rechazados:** 54.
- **Último registro del lector previo al corte:** `04:38:11.488666Z`.
- **Primer registro del lector tras restablecer RTSP:** `04:38:32.844317Z`.
- **Tiempo observado hasta nueva actividad después de desbloquear:** 4.84 s.
- **Separación máxima de registros del lector:** 21.36 s.
- **Resultado:** la comunicación y los registros del lector se recuperaron sin reiniciar Docker; regla temporal ausente al finalizar. Endpoint de salud operativo y MySQL saludable.

Este resultado **no garantiza recuperación para cortes de horas, cámaras sin electricidad ni sesiones RTSP bloqueadas de forma indefinida**. El tiempo anterior se calculó con marcas de registros del lector, no con un instrumento de fotogramas de vídeo del navegador.

## Monitor de conectividad y actividad

`scripts/monitor_estabilidad_camara.py` permite medir sin abrir otra sesión RTSP:

```bash
python3 scripts/monitor_estabilidad_camara.py --duracion 120 --intervalo 5
```

Se ejecuta en el host EC2, desde un directorio de diagnóstico separado. Revisa TCP 554, `127.0.0.1:5000/health` y la frecuencia de registros del `camera_capture_worker`. Una conexión TCP exitosa no sustituye una prueba de captura de imagen ni la comprobación visual del navegador.

## Medición continua posterior (120 segundos)

Se ejecutó el monitor de solo lectura en EC2 desde `2026-10-13T04:42:29Z`, con una muestra cada cinco segundos.

- Muestras de conectividad y salud: **24**.
- Fallos TCP contra `192.168.0.5:554`: **0**.
- Fallos del endpoint local `/health`: **0**.
- Actividad del lector registrada: **57 eventos**.
- Separación máxima entre registros de actividad del lector: **5.6 segundos**.

La prueba no generó inspecciones ni abrió sesiones RTSP adicionales. Estas mediciones no sustituyen un ensayo prolongado ni la verificación visual del video y la inferencia IA.

## Arquitectura recomendada para operación permanente

1. Instalar un equipo Linux dedicado de bajo consumo (por ejemplo, Raspberry Pi 4 o mini-PC) conectado de forma estable a la LAN de la Dahua. Preferir Ethernet cableado y alimentación protegida.
2. Configurar Tailscale con autenticación y permisos restringidos; habilitar solo `192.168.0.5/32`, no anunciar toda la LAN sin necesidad.
3. Aprobar esa ruta de subred para AWS, verificar TCP 554 y una sesión de lectura RTSP real.
4. Una vez comprobado el nuevo puente, retirar la ruta anunciada por Windows Anthony o configurar una estrategia de alta disponibilidad explícita, sin crear rutas ambiguas.
5. Reservar la IP de la cámara en DHCP o documentar su configuración estática, así como probar retorno de energía y reinicio del equipo puente.
6. Mantener las credenciales RTSP exclusivamente en configuración privada; **no exponer el puerto RTSP a Internet**.
7. Configurar monitorización y alertas sobre falta de fotogramas recientes, no solamente sobre el puerto TCP; validar también la recuperación tras 5, 30 y 60 minutos de interrupción.

## Estado y límites

- El enlace actual funciona y superó un corte breve.
- Los datos productivos no se alteraron.
- No se ha implementado todavía el equipo puente permanente.
- No habilitar CD automático ni reconstruir Docker por esta prueba.
- Una prueba prolongada, bajo supervisión, y una verificación visual después de la reconexión son necesarias antes de declarar estabilidad operativa de tesis.
