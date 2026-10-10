#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/opt/astrid-quality-control}"
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
DEPLOY_SHA="${DEPLOY_SHA:?DEPLOY_SHA es obligatorio}"
BACKUP_DIR="${BACKUP_DIR:-/opt/astrid-backups/mysql}"
BACKUP_CONFIG_FILE="${BACKUP_CONFIG_FILE:-/etc/astrid/backup.env}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:5000/health}"
BACKUP_KEEP="${BACKUP_KEEP:-10}"
MAINTENANCE_LOCK="${MAINTENANCE_LOCK:-/var/lock/astrid-quality-maintenance.lock}"

log() {
  printf '[deploy] %s\n' "$*"
}

git_as_deploy_user() {
  sudo -u "${DEPLOY_USER}" git -C "${APP_DIR}" "$@"
}

compose() {
  docker compose --project-directory "${APP_DIR}" --env-file "${APP_DIR}/.env" "$@"
}

wait_for_health() {
  local attempt
  for attempt in $(seq 1 36); do
    if curl --fail --silent --show-error --max-time 5 "${HEALTH_URL}" >/tmp/astrid-health.json 2>/dev/null; then
      if python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/astrid-health.json").read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(1)
PY
      then
        cat /tmp/astrid-health.json
        printf '\n'
        return 0
      fi
    fi

    log "Esperando health de la aplicación (${attempt}/36)..."
    sleep 5
  done
  return 1
}
rollback_code() {
  local previous_sha="$1"

  log "Health falló. Restaurando código ${previous_sha}."
  git_as_deploy_user reset --hard "${previous_sha}"
  compose build app ai-worker
  compose up -d --remove-orphans

  if wait_for_health; then
    log "Rollback de código completado."
  else
    log "ADVERTENCIA: el rollback también quedó sin health válido."
  fi
}

if [ ! -d "${APP_DIR}/.git" ]; then
  echo "No existe un repositorio Git en ${APP_DIR}." >&2
  exit 1
fi

if [ ! -f "${APP_DIR}/.env" ]; then
  echo "Falta ${APP_DIR}/.env. Se cancela el despliegue." >&2
  exit 1
fi

exec 9>"${MAINTENANCE_LOCK}"
if ! flock -n 9; then
  echo "Ya existe otro backup o despliegue de Astrid en ejecución." >&2
  exit 1
fi

cd "${APP_DIR}"

if [ -n "$(git_as_deploy_user status --porcelain --untracked-files=no)" ]; then
  echo "El servidor tiene cambios tracked sin confirmar. No se sobrescriben." >&2
  git_as_deploy_user status --short >&2
  exit 1
fi

previous_sha="$(git_as_deploy_user rev-parse HEAD)"
remote_sha="$(git_as_deploy_user rev-parse origin/main)"

if [ "${remote_sha}" != "${DEPLOY_SHA}" ]; then
  echo "origin/main=${remote_sha} no coincide con DEPLOY_SHA=${DEPLOY_SHA}." >&2
  exit 1
fi

log "Versión anterior: ${previous_sha}"
log "Versión objetivo:  ${DEPLOY_SHA}"

if [ -f "${BACKUP_CONFIG_FILE}" ]; then
  log "Creando backup externo pre-deploy."
  backup_script="/tmp/astrid-backup-production.sh"
  git_as_deploy_user show     "origin/main:scripts/backup-production.sh"     >"${backup_script}"
  chmod 700 "${backup_script}"

  BACKUP_CONFIG_FILE="${BACKUP_CONFIG_FILE}"   BACKUP_DATABASE_ONLY=1   BACKUP_SKIP_LOCK=1   bash "${backup_script}" --reason predeploy
else
  log "S3 aún no está configurado; creando respaldo MySQL local de fallback."
  mkdir -p "${BACKUP_DIR}"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup_file="${BACKUP_DIR}/textile_quality_${timestamp}_${previous_sha:0:8}.sql.gz"

  compose exec -T mysql sh -lc     'exec mysqldump --single-transaction --routines --triggers -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE"'     | gzip -9 >"${backup_file}"

  if [ ! -s "${backup_file}" ]; then
    echo "El respaldo MySQL quedó vacío. Se cancela el despliegue." >&2
    rm -f "${backup_file}"
    exit 1
  fi
fi

log "Actualizando código."
git_as_deploy_user reset --hard "${DEPLOY_SHA}"

log "Construyendo imágenes."
if ! compose build app ai-worker; then
  log "Falló el build. Restaurando el commit anterior."
  git_as_deploy_user reset --hard "${previous_sha}"
  exit 1
fi

log "Levantando servicios."
if ! compose up -d --remove-orphans; then
  rollback_code "${previous_sha}"
  exit 1
fi

if ! wait_for_health; then
  rollback_code "${previous_sha}"
  exit 1
fi

log "Servicios desplegados."
compose ps
mapfile -t old_backups < <(
  find "${BACKUP_DIR}" -maxdepth 1 -type f -name 'textile_quality_*.sql.gz'     -printf '%T@ %p\n'     | sort -nr     | awk -v keep="${BACKUP_KEEP}" 'NR > keep {sub(/^[^ ]+ /, ""); print}'
)

if [ "${#old_backups[@]}" -gt 0 ]; then
  rm -f -- "${old_backups[@]}"
fi

log "Despliegue completado: ${DEPLOY_SHA}"
