# Despliegue controlado en AWS — 12 de octubre de 2026 (Guatemala)

## Estado comprobado
- AWS EC2 `i-01c3bf4359c980355`, us-east-2; aplicación `astrid_quality_app`.
- Imagen desplegada: `astrid-quality-control-app:hotfix-129449a`, ID `sha256:12233fb46dc67530417de1f65fea0550c39d86501e36397d417849cea50f62c5`.
- Fuente de la imagen: rama `fix/imagen-activa-camara-modelos`, commit `129449a`, preparado **sobre el código exacto del contenedor previo**, no sobre la rama `main` antigua.
- El contenedor web fue recreado con `docker compose --env-file .env up -d --no-deps --no-build --force-recreate app`. MySQL y `ai-worker` no se reiniciaron.
- Verificación posterior: `/health` responde `status=ok`, MySQL `healthy`, temporizadores de backup activos y sin excepciones recientes en registros revisados.
- El endpoint `/api/camera/status` está registrado y exige autenticación (HTTP 401 sin sesión).

## Cambios del código
1. Reconexión RTSP: corrección del acceso a la captura global al cerrar y volver a abrir.
2. Estado real: la interfaz deja de decir permanentemente «Cámara activa». Comprueba que haya fotogramas recientes y consulta el estado periódicamente.
3. Eliminación de modelos incompletos: `utc_now_db()` definida, evitando el `NameError` de la ruta de eliminación. No se ejecutaron eliminaciones sobre datos reales.
4. Pruebas automatizadas del conjunto: **9/9 correctas**. Plantilla Jinja y sintaxis Python verificadas.
5. Se verificó lectura RTSP desde AWS al reconectar físicamente la Dahua: imagen 2688 × 1520.

## Resguardos
- SQL transaccional predespliegue en S3: `astrid/production/snapshots/2026/10/13/20261013T035729Z-ip-172-31-2-214/`.
- Respaldo externo de evidencias, modelos y checkpoints verificado: `astrid/production/emergency/20261013T023413Z/`.
- Respaldo de código de producción previo: `astrid/production/emergency/20261013T024106Z/production-code.tar.gz`.
- Prueba de restauración MySQL anterior: **22 tablas y 539 inspecciones**.
- Imagen anterior conservada localmente con tag `astrid-quality-control-app:rollback-prehotfix-20261013`.
- Copias temporales redundantes de los archivos grandes se retiraron del disco **solo después de verificar SHA-256 y tamaño contra S3**. Los datos originales no se tocaron.
- El disco quedó con aproximadamente 8 GB libres (84 % de uso).

## Reversión
Si la nueva versión presenta errores confirmados:

```bash
cd /opt/astrid-quality-control
docker image tag \
  astrid-quality-control-app:rollback-prehotfix-20261013 \
  astrid-quality-control-app:latest
docker compose --env-file .env up -d --no-deps --no-build --force-recreate app
curl -fsS http://127.0.0.1:5000/health
```

No revertir ni restaurar la base de datos para un fallo exclusivo del código web sin analizarlo primero.

## Advertencias pendientes
- **La carpeta fuente /opt/astrid-quality-control en EC2 conserva cambios locales históricos que NO coinciden exactamente con el `app.py` del contenedor ejecutado.** La imagen activa fue creada desde la copia exacta del runtime y está versionada en `fix/imagen-activa-camara-modelos`. No ejecutar `git pull`, `git reset`, `docker compose build app` ni activar CD hasta reconciliar formalmente los cambios.
- No se realizó una prueba física de desconexión y reconexión posterior al despliegue. La estabilidad prolongada aún debe medirse.
- No se eliminó `BLUSA-761` ni otros modelos de producción; falta UAT autorizado desde navegador.
- Evaluar el estado `model_ready=false` de `/health` según lote y modelo activo antes de declarar lista la IA.
- Vigilar espacio de EBS (50 GB) y caché de compilación; una compilación completa descargó dependencias CUDA extensas. Evitar reconstrucciones generales en EC2 hasta corregir estrategia y capacidad.
- Tailscale depende de que Anthony esté encendido, en la misma red de la Dahua y mantenga conexión de subred. Se recomienda equipo dedicado y dirección IP reservada.
- Otros temas de bitácora: prendas «Aceptadas» con «Pendiente de revisión» y jornada «EN CURSO» pese a objetivo 100 %. Distinguir comportamiento esperado de defecto antes de modificar.

## Comprobaciones futuras
- Login de administrador: `/inspeccion` debe mostrar indicador real con cámara conectada.
- Inspección de prueba y registro correcto en historial sin alterar lotes importantes.
- Prueba de eliminación de un modelo borrador de demostración SIN lotes ni IA activa, verificando borrado lógico, sin modificar modelos productivos.
- Verificación de próxima copia diaria completa en S3 y restauración periódica.
