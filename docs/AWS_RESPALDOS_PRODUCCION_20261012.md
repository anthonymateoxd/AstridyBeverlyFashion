# Estado de respaldos de producción — 12 de octubre de 2026 (Guatemala)

## Entorno verificado
- AWS EC2: `astrid-quality-thesis` (`i-01c3bf4359c980355`), región `us-east-2`.
- Aplicación: `/opt/astrid-quality-control`; contenedores `astrid_quality_app`, `textile_mysql`, `astrid_ai_worker`.
- Git del EC2: `main` en `c963cb0`, **con modificaciones propias pendientes de reconciliar**. No hacer `git pull/reset/clean` ni reconstrucción global.
- Bucket privado cifrado: `astrid-quality-backups-184140788070`. Cifrado SSE-S3 AES256, acceso público bloqueado, versionado y reglas de retención habilitados.
- Rol del EC2: `AstridSSMRole`, policy específica `AstridProductionBackupsS3` con permisos limitados a S3 y al parámetro SecureString de configuración.

## Protección comprobada
1. Respaldo SQL de emergencia: `astrid/production/emergency/20261013T023216Z/database.sql.gz`.
2. Respaldo de artefactos: `astrid/production/emergency/20261013T023413Z/`, con `ai-artifacts.tar.gz`, `patchcore-results.tar.gz`, `models.tar.gz`, `captures.tar.gz`, `results.tar.gz` y `secure/env-file`.
3. Copia del código real de producción (incluye cambios del servidor): `astrid/production/emergency/20261013T024106Z/production-code.tar.gz`. No contiene `.env`, datasets ni capturas duplicadas.
4. Se verificaron tamaño y SHA-256 de los archivos cargados en S3. Una base temporal de restauración recuperó **22 tablas y 539 inspecciones**; no se modificó la BD productiva.
5. El script de `chore/ci-cd` del commit `7716deb` produjo correctamente un backup transaccional de MySQL, cifró y guardó `.env` en S3 y como SecureString SSM, generó el manifiesto y pasó `verify-latest-backup.sh` en S3.

## Instalación aislada
- Los scripts y unidades proceden de Git `chore/ci-cd@7716deb`, distribuidos como ZIP privado.
- Instalación: `/opt/astrid-backup-tools`. **No modifica el repositorio productivo.**
- Configuración privada `/etc/astrid/backup.env` (0600); no subir a Git.
- Estado systemd verificado al instalar:
  - `astrid-backup.timer`: enabled / active; copia diaria alrededor de las 03:15 (America/Guatemala).
  - `astrid-backup-verify.timer`: enabled / active; domingo alrededor de las 04:30.
- Servicios ejecutan `/opt/astrid-backup-tools/scripts/backup-production.sh` y `verify-latest-backup.sh`.
- Durante el backup completo se puede pausar brevemente `ai-worker` para mantener consistencia; la app web y MySQL no se reinician.

## Verificaciones posteriores obligatorias
- Confirmar **primera ejecución automática completa** (no solo una prueba SQL).
- Revisar `systemctl status astrid-backup.service` y `journalctl -u astrid-backup.service`.
- Verificar `astrid/production/latest/manifest.json` con todos los payloads.
- Ejecutar prueba periódica de restauración en BD temporal y recuperación de al menos un artefacto IA.
- Vigilar crecimiento de S3 y del volumen de EC2. El almacenamiento tiene costo.
- Mantener la rama `chore/ci-cd` separada de `main` hasta reconciliar producción.

## Comprobaciones útiles (solo lectura)
```bash
systemctl list-timers --all astrid-backup.timer astrid-backup-verify.timer
systemctl status astrid-backup.service --no-pager
journalctl -u astrid-backup.service -n 50 --no-pager
/opt/astrid-backup-tools/scripts/verify-latest-backup.sh
```

## Seguridad
No publicar `.env`, tokens, credenciales AWS, contraseñas de cámara ni URL RTSP completas. Un respaldo en Git no sustituye snapshots de S3 y una prueba de restauración.
