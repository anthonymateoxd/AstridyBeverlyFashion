#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="/opt/astrid-quality-control"
AWS_REGION_VALUE="us-east-2"
BUCKET=""
RUN_NOW=0
FORCE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bucket)
      BUCKET="${2:-}"
      shift 2
      ;;
    --region)
      AWS_REGION_VALUE="${2:-}"
      shift 2
      ;;
    --app-dir)
      APP_DIR="${2:-}"
      shift 2
      ;;
    --run-now)
      RUN_NOW=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "Opción desconocida: $1" >&2
      exit 2
      ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Ejecute este instalador como root (sudo)." >&2
  exit 1
fi

if [ -z "${BUCKET}" ]; then
  echo "--bucket es obligatorio." >&2
  exit 2
fi

for required in   "${APP_DIR}/scripts/backup-production.sh"   "${APP_DIR}/scripts/verify-latest-backup.sh"   "${APP_DIR}/ops/systemd/astrid-backup.service"   "${APP_DIR}/ops/systemd/astrid-backup.timer"   "${APP_DIR}/ops/systemd/astrid-backup-verify.service"   "${APP_DIR}/ops/systemd/astrid-backup-verify.timer"; do
  if [ ! -f "${required}" ]; then
    echo "No existe ${required}." >&2
    exit 1
  fi
done

install -d -m 0750 /etc/astrid
config_path=/etc/astrid/backup.env

if [ -f "${config_path}" ] && [ "${FORCE}" != "1" ]; then
  echo "${config_path} ya existe. Use --force para reemplazarlo." >&2
  exit 1
fi

cat >"${config_path}" <<EOF
APP_DIR=${APP_DIR}
DEPLOY_USER=ubuntu
BACKUP_ROOT=/opt/astrid-backups
BACKUP_S3_BUCKET=${BUCKET}
BACKUP_S3_PREFIX=astrid/production
AWS_REGION=${AWS_REGION_VALUE}
BACKUP_LOCAL_DB_KEEP=10
BACKUP_LOCAL_MANIFEST_KEEP=30
BACKUP_PAUSE_AI_WORKER=1
BACKUP_KMS_KEY_ID=
BACKUP_CONFIG_TO_SSM=1
BACKUP_CONFIG_TO_S3=1
BACKUP_SSM_ENV_PARAMETER=/astrid/production/env-file
BACKUP_HEALTH_URL=http://127.0.0.1:5000/health
MAINTENANCE_LOCK=/var/lock/astrid-quality-maintenance.lock
EOF

chmod 0600 "${config_path}"
install -m 0644   "${APP_DIR}/ops/systemd/astrid-backup.service"   /etc/systemd/system/astrid-backup.service
install -m 0644   "${APP_DIR}/ops/systemd/astrid-backup.timer"   /etc/systemd/system/astrid-backup.timer
install -m 0644   "${APP_DIR}/ops/systemd/astrid-backup-verify.service"   /etc/systemd/system/astrid-backup-verify.service
install -m 0644   "${APP_DIR}/ops/systemd/astrid-backup-verify.timer"   /etc/systemd/system/astrid-backup-verify.timer

systemctl daemon-reload
systemctl enable --now astrid-backup.timer astrid-backup-verify.timer

echo "Timers instalados:"
systemctl list-timers   astrid-backup.timer   astrid-backup-verify.timer   --no-pager

if [ "${RUN_NOW}" = "1" ]; then
  echo "Ejecutando primer backup..."
  systemctl start astrid-backup.service
  systemctl status astrid-backup.service --no-pager
fi
