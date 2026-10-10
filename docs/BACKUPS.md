# Backups de producción

Este documento define la estrategia de respaldo para Astrid & Beverly Fashion.

## Objetivos

El sistema debe poder recuperarse aunque se pierda por completo el EC2. GitHub conserva el código, pero no conserva la base MySQL ni los artefactos generados en ejecución.

Se respaldan:

- MySQL.
- `ai_artifacts/`: datasets, modelos y artefactos IA administrados por el sistema.
- `patchcore_results/`: checkpoints PatchCore legacy que todavía puedan ser necesarios.
- `models/`: modelos adicionales persistentes.
- `static/captures/`: evidencias originales.
- `static/results/`: evidencias procesadas.
- `static/garment_models/` cuando exista.

El archivo `.env` nunca se incluye dentro de los tarballs generales ni se versiona en Git. Cada snapshot guarda una copia separada como `secure/env-file` cifrada por S3 y el valor actual también se conserva como `SecureString` en SSM Parameter Store.

## Capas

1. **Backup local MySQL:** se conservan los últimos 10 dumps en el EC2.
2. **S3 privado:** cada snapshot completo se sube fuera del EC2.
3. **Versionado S3:** protege el marcador `latest` y otros objetos frente a sobrescrituras accidentales.
4. **Configuración privada:** el `.env` queda cifrado por snapshot en S3 y como SecureString versionado en SSM.
5. **Verificación:** cada upload se comprueba mediante tamaño y SHA-256 declarado en metadata.
6. **Restore test:** existe una prueba que restaura MySQL en una base temporal sin tocar producción.
7. **Pre-deploy:** cuando S3 esté configurado, cada deployment crea primero un backup MySQL externo.

## Retención S3

Los snapshots se etiquetan automáticamente:

| Tipo | Retención |
| --- | ---: |
| daily | 8 días |
| weekly | 35 días |
| monthly | 190 días |
| predeploy | 35 días |
| manual | 35 días |

El día 1 del mes prevalece como `monthly`. Los domingos normales se etiquetan `weekly`. El resto de ejecuciones programadas son `daily`.

Los objetos se guardan bajo:

```text
s3://BUCKET/astrid/production/
├── snapshots/
│   └── YYYY/MM/DD/TIMESTAMP-HOST/
│       ├── database.sql.gz
│       ├── secure/
│       │   └── env-file
│       ├── ai-artifacts.tar.gz
│       ├── patchcore-results.tar.gz
│       ├── models.tar.gz
│       ├── evidence.tar.gz
│       └── manifest.json
└── latest/
    └── manifest.json
```
## 1. Crear y proteger el bucket S3

Ejecute con una sesión AWS administrativa válida:

```bash
cd /opt/astrid-quality-control
bash scripts/setup-backup-s3.sh
```

Por defecto se genera un nombre único:

```text
astrid-beverly-backups-<AWS_ACCOUNT_ID>-us-east-2
```

El script configura:

- bloqueo de acceso público;
- cifrado SSE-S3 AES-256 por defecto;
- versionado;
- lifecycle por etiquetas de retención;
- cancelación de multipart uploads incompletos;
- policy que rechaza conexiones S3 sin TLS;
- permisos mínimos de S3 sobre el IAM Role del EC2 `astrid-aws`;
- permisos restringidos para guardar/leer `/astrid/production/env-file` como SecureString en SSM.

Puede especificar valores:

```bash
bash scripts/setup-backup-s3.sh \
  --bucket MI_BUCKET_PRIVADO \
  --region us-east-2 \
  --ec2-name astrid-aws
```

No guarde `AWS_ACCESS_KEY_ID` ni `AWS_SECRET_ACCESS_KEY` en el EC2. El servidor debe acceder a S3 mediante su IAM Role.

## 2. Instalar ejecución automática

Después de crear el bucket, en el EC2:

```bash
cd /opt/astrid-quality-control

sudo bash scripts/install-backup-timer.sh \
  --bucket astrid-beverly-backups-<AWS_ACCOUNT_ID>-us-east-2 \
  --region us-east-2 \
  --run-now
```

Esto crea:

```text
/etc/astrid/backup.env
/etc/systemd/system/astrid-backup.service
/etc/systemd/system/astrid-backup.timer
/etc/systemd/system/astrid-backup-verify.service
/etc/systemd/system/astrid-backup-verify.timer
```

`/etc/astrid/backup.env` queda con permisos `0600`.

El backup se ejecuta diariamente alrededor de las **03:15 America/Guatemala**. La verificación semanal se ejecuta los domingos alrededor de las **04:30 America/Guatemala**. Ambos timers usan `Persistent=true`, por lo que systemd ejecutará una tarea omitida cuando el servidor vuelva a estar disponible.
## 3. Revisar timers y logs

Estado de los timers:

```bash
systemctl list-timers \
  astrid-backup.timer \
  astrid-backup-verify.timer
```

Último backup:

```bash
sudo systemctl status astrid-backup.service --no-pager
sudo journalctl -u astrid-backup.service -n 200 --no-pager
```

Última verificación:

```bash
sudo systemctl status astrid-backup-verify.service --no-pager
sudo journalctl -u astrid-backup-verify.service -n 200 --no-pager
```

## 4. Backup manual

```bash
sudo /opt/astrid-quality-control/scripts/backup-production.sh \
  --reason manual
```

El script usa el mismo lock que el deployment. Si ya hay un deploy o backup en curso, la segunda operación se cancela en vez de competir por archivos o base de datos.

Durante un snapshot completo, `ai-worker` se pausa brevemente para evitar capturar un checkpoint mientras se está escribiendo. Se reanuda automáticamente incluso si el backup falla.

## 5. Verificación del último snapshot

Verificación rápida, sin descargar todos los payloads:

```bash
sudo /opt/astrid-quality-control/scripts/verify-latest-backup.sh
```

Para descargar realmente cada archivo y volver a calcular su SHA-256:

```bash
sudo env VERIFY_DOWNLOAD=1 \
  /opt/astrid-quality-control/scripts/verify-latest-backup.sh
```

La segunda opción consume más tiempo y espacio temporal, por lo que no se programa semanalmente.
## 6. Prueba real de restauración MySQL

La prueba descarga el último dump y crea una base temporal. No modifica `textile_quality_db`.

```bash
sudo /opt/astrid-quality-control/scripts/test-restore-latest.sh
```

La prueba:

1. lee `latest/manifest.json`;
2. descarga el payload `database`;
3. valida tamaño y SHA-256;
4. ejecuta `gzip -t`;
5. crea `astrid_restore_test_<timestamp>`;
6. importa el dump;
7. comprueba las tablas núcleo;
8. elimina la base temporal.

Una ejecución exitosa es evidencia de que el respaldo de base de datos es restaurable, no solamente que existe un archivo en S3.

## 7. Recuperar la configuración privada

En un servidor nuevo, el `.env` puede recuperarse desde el último snapshot sin imprimir su contenido:

```bash
sudo /opt/astrid-quality-control/scripts/restore-env-latest.sh \
  --output /opt/astrid-quality-control/.env
```

El script valida tamaño y SHA-256 y escribe el archivo con permisos `0600`. Se niega a sobrescribir un archivo existente; para una recuperación supervisada explícita puede usarse `--force`.

También existe la copia operativa como SecureString en:

```text
/astrid/production/env-file
```

El manifest registra la versión de SSM y el SHA-256 correspondiente.

## 8. Backup previo a deployment

`scripts/deploy-production.sh` comparte el lock de mantenimiento.

Si existe:

```text
/etc/astrid/backup.env
```

el deployment ejecuta antes:

```text
backup-production.sh --reason predeploy
```

en modo database-only y exige que el upload S3 se verifique antes de continuar.

Si S3 todavía no está configurado, se conserva temporalmente el comportamiento anterior: dump MySQL local antes del deployment.

## 9. Recuperación ante pérdida total del EC2

La recuperación completa se realiza en este orden:

```text
EC2 nuevo
  ↓
IAM Role + SSM
  ↓
Docker / Git
  ↓
clonar repositorio
  ↓
restaurar configuración privada
  ↓
restaurar MySQL
  ↓
restaurar ai_artifacts
  ↓
restaurar patchcore_results / models
  ↓
restaurar evidencias necesarias
  ↓
docker compose up
  ↓
/health
  ↓
prueba de inspección controlada
```

No se debe restaurar automáticamente sobre producción sin revisión. La restauración destructiva de la base real debe ser una operación supervisada.

## Pendiente adicional

S3 cubre la pérdida del EC2 y del volumen principal. Como capa adicional conviene configurar snapshots EBS/AWS Backup semanales. Eso es independiente de estos backups lógicos y no sustituye MySQL/S3.
