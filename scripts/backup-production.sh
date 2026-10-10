#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

CONFIG_FILE="${BACKUP_CONFIG_FILE:-/etc/astrid/backup.env}"
if [ -f "${CONFIG_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

APP_DIR="${APP_DIR:-/opt/astrid-quality-control}"
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/astrid-backups}"
BACKUP_S3_BUCKET="${BACKUP_S3_BUCKET:-}"
BACKUP_S3_PREFIX="${BACKUP_S3_PREFIX:-astrid/production}"
BACKUP_S3_PREFIX="${BACKUP_S3_PREFIX%/}"
AWS_REGION="${AWS_REGION:-us-east-2}"
BACKUP_LOCAL_DB_KEEP="${BACKUP_LOCAL_DB_KEEP:-10}"
BACKUP_LOCAL_MANIFEST_KEEP="${BACKUP_LOCAL_MANIFEST_KEEP:-30}"
BACKUP_PAUSE_AI_WORKER="${BACKUP_PAUSE_AI_WORKER:-1}"
BACKUP_DATABASE_ONLY="${BACKUP_DATABASE_ONLY:-0}"
BACKUP_LOCAL_ONLY="${BACKUP_LOCAL_ONLY:-0}"
BACKUP_SKIP_LOCK="${BACKUP_SKIP_LOCK:-0}"
BACKUP_KMS_KEY_ID="${BACKUP_KMS_KEY_ID:-}"
BACKUP_SSM_ENV_PARAMETER="${BACKUP_SSM_ENV_PARAMETER:-/astrid/production/env-file}"
BACKUP_CONFIG_TO_SSM="${BACKUP_CONFIG_TO_SSM:-1}"
BACKUP_CONFIG_TO_S3="${BACKUP_CONFIG_TO_S3:-1}"
BACKUP_HEALTH_URL="${BACKUP_HEALTH_URL:-http://127.0.0.1:5000/health}"
MAINTENANCE_LOCK="${MAINTENANCE_LOCK:-/var/lock/astrid-quality-maintenance.lock}"

reason="scheduled"
if [ "${1:-}" = "--reason" ]; then
  reason="${2:-}"
fi

case "${reason}" in
  scheduled|manual|predeploy) ;;
  *)
    echo "Motivo de backup no válido: ${reason}" >&2
    exit 2
    ;;
esac

log() {
  printf '[backup] %s\n' "$*"
}

fail() {
  printf '[backup] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Falta el comando requerido: $1"
}

for command_name in aws curl docker flock git gzip hostname python3 sha256sum stat sudo tar; do
  require_command "${command_name}"
done
[ -d "${APP_DIR}/.git" ] || fail "No existe repositorio Git en ${APP_DIR}."
[ -f "${APP_DIR}/.env" ] || fail "Falta ${APP_DIR}/.env."

if [ "${BACKUP_LOCAL_ONLY}" != "1" ] && [ -z "${BACKUP_S3_BUCKET}" ]; then
  fail "BACKUP_S3_BUCKET es obligatorio para un backup externo."
fi

if [ "${BACKUP_SKIP_LOCK}" != "1" ]; then
  exec 9>"${MAINTENANCE_LOCK}"
  flock -n 9 || fail "Hay otro backup o despliegue en ejecución."
fi

LOCAL_DB_DIR="${BACKUP_ROOT}/mysql"
LOCAL_MANIFEST_DIR="${BACKUP_ROOT}/manifests"
STAGING_ROOT="${BACKUP_ROOT}/staging"
mkdir -p "${LOCAL_DB_DIR}" "${LOCAL_MANIFEST_DIR}" "${STAGING_ROOT}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
created_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
snapshot_id="${timestamp}-$(hostname -s)"
snapshot_date_path="$(date -u +%Y/%m/%d)"
snapshot_prefix="${BACKUP_S3_PREFIX}/snapshots/${snapshot_date_path}/${snapshot_id}"
work_dir="$(mktemp -d "${STAGING_ROOT}/${snapshot_id}.XXXXXX")"
files_tsv="${work_dir}/files.tsv"
touch "${files_tsv}"

worker_paused=0
backup_succeeded=0

compose() {
  docker compose --project-directory "${APP_DIR}" --env-file "${APP_DIR}/.env" "$@"
}

git_as_deploy_user() {
  sudo -u "${DEPLOY_USER}" git -C "${APP_DIR}" "$@"
}

cleanup() {
  if [ "${worker_paused}" = "1" ]; then
    log "Reanudando ai-worker."
    compose unpause ai-worker >/dev/null 2>&1 || true
  fi

  if [ "${backup_succeeded}" = "1" ]; then
    rm -rf "${work_dir}"
  else
    log "Se conserva staging para diagnóstico: ${work_dir}"
  fi
}
trap cleanup EXIT
local_day="$(TZ=America/Guatemala date +%d)"
local_weekday="$(TZ=America/Guatemala date +%u)"

if [ "${reason}" = "predeploy" ]; then
  retention_class="predeploy"
elif [ "${reason}" = "manual" ]; then
  retention_class="manual"
elif [ "${local_day}" = "01" ]; then
  retention_class="monthly"
elif [ "${local_weekday}" = "7" ]; then
  retention_class="weekly"
else
  retention_class="daily"
fi

if [ "${BACKUP_LOCAL_ONLY}" != "1" ]; then
  log "Validando identidad AWS e ingreso al bucket S3."
  aws sts get-caller-identity --region "${AWS_REGION}" >/dev/null
  aws s3api head-bucket     --bucket "${BACKUP_S3_BUCKET}"     --region "${AWS_REGION}" >/dev/null
fi

if [ "${BACKUP_DATABASE_ONLY}" != "1" ] && [ "${BACKUP_PAUSE_AI_WORKER}" = "1" ]; then
  worker_id="$(compose ps -q ai-worker 2>/dev/null || true)"
  if [ -n "${worker_id}" ]; then
    worker_state="$(docker inspect -f '{{.State.Running}} {{.State.Paused}}' "${worker_id}" 2>/dev/null || true)"
    if [ "${worker_state}" = "true false" ]; then
      log "Pausando ai-worker durante la instantánea de archivos IA."
      compose pause ai-worker >/dev/null
      worker_paused=1
    fi
  fi
fi

git_sha="$(git_as_deploy_user rev-parse HEAD)"
if [ -n "$(git_as_deploy_user status --porcelain --untracked-files=no)" ]; then
  git_dirty=1
else
  git_dirty=0
fi

health_file="${work_dir}/health.json"
if ! curl --fail --silent --show-error --max-time 5 "${BACKUP_HEALTH_URL}" >"${health_file}" 2>/dev/null; then
  printf '{"status":"unavailable"}\n' >"${health_file}"
fi

db_archive="${work_dir}/database.sql.gz"
log "Generando dump transaccional de MySQL."
compose exec -T mysql sh -lc   'exec mysqldump --single-transaction --quick --routines --triggers --events --hex-blob --set-gtid-purged=OFF -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE"'   | gzip -9 >"${db_archive}"

[ -s "${db_archive}" ] || fail "El dump MySQL quedó vacío."

local_db_archive="${LOCAL_DB_DIR}/database_${snapshot_id}.sql.gz"
cp "${db_archive}" "${local_db_archive}"
chmod 600 "${local_db_archive}"
archive_group() {
  local output="$1"
  shift
  local -a existing=()
  local relative

  for relative in "$@"; do
    if [ -e "${APP_DIR}/${relative}" ]; then
      existing+=("${relative}")
    fi
  done

  if [ "${#existing[@]}" -eq 0 ]; then
    return 1
  fi

  if ! tar -C "${APP_DIR}" -czf "${output}" -- "${existing[@]}"; then
    log "El árbol cambió durante el tar; reintentando una vez."
    rm -f "${output}"
    sleep 2
    tar -C "${APP_DIR}" -czf "${output}" -- "${existing[@]}"
  fi

  [ -s "${output}" ] || fail "Archivo vacío: ${output}"
  return 0
}

declare -a payloads=()
payloads+=("database|${db_archive}|database.sql.gz")

if [ "${BACKUP_LOCAL_ONLY}" != "1" ] && [ "${BACKUP_CONFIG_TO_S3}" = "1" ]; then
  payloads+=("secure_env|${APP_DIR}/.env|secure/env-file")
fi

if [ "${BACKUP_DATABASE_ONLY}" != "1" ]; then
  ai_archive="${work_dir}/ai-artifacts.tar.gz"
  if archive_group "${ai_archive}" ai_artifacts; then
    payloads+=("ai_artifacts|${ai_archive}|ai-artifacts.tar.gz")
  fi

  patchcore_archive="${work_dir}/patchcore-results.tar.gz"
  if archive_group "${patchcore_archive}" patchcore_results; then
    payloads+=("patchcore_results|${patchcore_archive}|patchcore-results.tar.gz")
  fi

  models_archive="${work_dir}/models.tar.gz"
  if archive_group "${models_archive}" models; then
    payloads+=("models|${models_archive}|models.tar.gz")
  fi

  evidence_archive="${work_dir}/evidence.tar.gz"
  if archive_group "${evidence_archive}"       static/captures       static/results       static/garment_models; then
    payloads+=("evidence|${evidence_archive}|evidence.tar.gz")
  fi
fi
declare -a sse_args=()
if [ -n "${BACKUP_KMS_KEY_ID}" ]; then
  sse_args=(--sse aws:kms --sse-kms-key-id "${BACKUP_KMS_KEY_ID}")
else
  sse_args=(--sse AES256)
fi

upload_payload() {
  local label="$1"
  local local_path="$2"
  local object_name="$3"
  local sha256
  local size
  local key
  local remote_sha

  sha256="$(sha256sum "${local_path}" | awk '{print $1}')"
  size="$(stat -c '%s' "${local_path}")"
  key="${snapshot_prefix}/${object_name}"

  if [ "${BACKUP_LOCAL_ONLY}" != "1" ]; then
    log "Subiendo ${label}: s3://${BACKUP_S3_BUCKET}/${key}"
    aws s3 cp       "${local_path}"       "s3://${BACKUP_S3_BUCKET}/${key}"       --region "${AWS_REGION}"       --only-show-errors       --metadata "sha256=${sha256},snapshot=${snapshot_id}"       "${sse_args[@]}"

    aws s3api put-object-tagging       --bucket "${BACKUP_S3_BUCKET}"       --key "${key}"       --region "${AWS_REGION}"       --tagging "TagSet=[{Key=Retention,Value=${retention_class}},{Key=Reason,Value=${reason}}]"       >/dev/null

    remote_sha="$(
      aws s3api head-object         --bucket "${BACKUP_S3_BUCKET}"         --key "${key}"         --region "${AWS_REGION}"         --query 'Metadata.sha256'         --output text
    )"

    [ "${remote_sha}" = "${sha256}" ] || fail "SHA metadata no coincide para ${key}."
  else
    key="LOCAL_ONLY/${object_name}"
  fi

  printf '%s\t%s\t%s\t%s\n'     "${label}" "${key}" "${sha256}" "${size}" >>"${files_tsv}"
}

for payload in "${payloads[@]}"; do
  IFS='|' read -r label local_path object_name <<<"${payload}"
  upload_payload "${label}" "${local_path}" "${object_name}"
done

secure_config_meta="${work_dir}/secure-config.json"
printf '{"enabled":false}\n' >"${secure_config_meta}"

if [ "${BACKUP_LOCAL_ONLY}" != "1" ] && [ "${BACKUP_CONFIG_TO_SSM}" = "1" ]; then
  env_size="$(stat -c '%s' "${APP_DIR}/.env")"
  if [ "${env_size}" -gt 3900 ]; then
    fail ".env supera el límite seguro de SecureString Standard (3900 bytes usados como margen)."
  fi

  env_sha="$(sha256sum "${APP_DIR}/.env" | awk '{print $1}')"
  ssm_input="${work_dir}/.ssm-put-parameter.json"

  ENV_FILE="${APP_DIR}/.env" \
  PARAMETER_NAME="${BACKUP_SSM_ENV_PARAMETER}" \
  python3 - <<'PY' >"${ssm_input}"
import json
import os
from pathlib import Path

payload = {
    "Name": os.environ["PARAMETER_NAME"],
    "Description": "Astrid production .env backup managed by backup-production.sh",
    "Value": Path(os.environ["ENV_FILE"]).read_bytes().decode("utf-8"),
    "Type": "SecureString",
    "Overwrite": True,
    "Tier": "Standard",
}

print(json.dumps(payload, ensure_ascii=False))
PY

  if ! ssm_response="$(
    aws ssm put-parameter \
      --region "${AWS_REGION}" \
      --cli-input-json "file://${ssm_input}" \
      --output json
  )"; then
    rm -f "${ssm_input}"
    fail "No se pudo respaldar .env en SSM Parameter Store."
  fi
  rm -f "${ssm_input}"

  ssm_version="$(
    SSM_RESPONSE="${ssm_response}" python3 - <<'PY'
import json
import os

print(json.loads(os.environ["SSM_RESPONSE"])["Version"])
PY
  )"

  PARAMETER_NAME="${BACKUP_SSM_ENV_PARAMETER}" \
  PARAMETER_VERSION="${ssm_version}" \
  ENV_SHA="${env_sha}" \
  python3 - <<'PY' >"${secure_config_meta}"
import json
import os

print(
    json.dumps(
        {
            "enabled": True,
            "parameter": os.environ["PARAMETER_NAME"],
            "version": int(os.environ["PARAMETER_VERSION"]),
            "sha256": os.environ["ENV_SHA"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
PY
fi

manifest="${work_dir}/manifest.json"
SNAPSHOT_ID="${snapshot_id}" CREATED_AT_UTC="${created_at_utc}" RETENTION_CLASS="${retention_class}" BACKUP_REASON="${reason}" HOSTNAME_VALUE="$(hostname -f 2>/dev/null || hostname)" GIT_SHA="${git_sha}" GIT_DIRTY="${git_dirty}" FILES_TSV="${files_tsv}" HEALTH_FILE="${health_file}" SECURE_CONFIG_META="${secure_config_meta}" python3 - <<'PY' >"${manifest}"
import json
import os
from pathlib import Path

files = []
for raw in Path(os.environ["FILES_TSV"]).read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    label, key, sha256, size = raw.split("\t", 3)
    files.append(
        {
            "label": label,
            "key": key,
            "sha256": sha256,
            "size_bytes": int(size),
        }
    )

health_path = Path(os.environ["HEALTH_FILE"])
try:
    health = json.loads(health_path.read_text(encoding="utf-8"))
except Exception:
    health = {"status": "unparseable"}

secure_config = json.loads(
    Path(os.environ["SECURE_CONFIG_META"]).read_text(encoding="utf-8")
)

manifest = {
    "schema_version": 1,
    "snapshot_id": os.environ["SNAPSHOT_ID"],
    "created_at_utc": os.environ["CREATED_AT_UTC"],
    "retention_class": os.environ["RETENTION_CLASS"],
    "reason": os.environ["BACKUP_REASON"],
    "hostname": os.environ["HOSTNAME_VALUE"],
    "git": {
        "sha": os.environ["GIT_SHA"],
        "dirty_tracked_files": os.environ["GIT_DIRTY"] == "1",
    },
    "health": health,
    "secure_config": secure_config,
    "files": files,
}

print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
PY

manifest_sha="$(sha256sum "${manifest}" | awk '{print $1}')"

if [ "${BACKUP_LOCAL_ONLY}" != "1" ]; then
  manifest_key="${snapshot_prefix}/manifest.json"
  log "Subiendo manifest del snapshot."
  aws s3 cp     "${manifest}"     "s3://${BACKUP_S3_BUCKET}/${manifest_key}"     --region "${AWS_REGION}"     --only-show-errors     --metadata "sha256=${manifest_sha},snapshot=${snapshot_id}"     "${sse_args[@]}"

  aws s3api put-object-tagging     --bucket "${BACKUP_S3_BUCKET}"     --key "${manifest_key}"     --region "${AWS_REGION}"     --tagging "TagSet=[{Key=Retention,Value=${retention_class}},{Key=Reason,Value=${reason}}]"     >/dev/null

  latest_key="${BACKUP_S3_PREFIX}/latest/manifest.json"
  aws s3 cp     "${manifest}"     "s3://${BACKUP_S3_BUCKET}/${latest_key}"     --region "${AWS_REGION}"     --only-show-errors     --metadata "sha256=${manifest_sha},snapshot=${snapshot_id}"     "${sse_args[@]}"

  latest_remote_sha="$(
    aws s3api head-object       --bucket "${BACKUP_S3_BUCKET}"       --key "${latest_key}"       --region "${AWS_REGION}"       --query 'Metadata.sha256'       --output text
  )"
  [ "${latest_remote_sha}" = "${manifest_sha}" ] || fail "No se pudo verificar latest/manifest.json."
fi
local_manifest="${LOCAL_MANIFEST_DIR}/manifest_${snapshot_id}.json"
cp "${manifest}" "${local_manifest}"
chmod 600 "${local_manifest}"

prune_local() {
  local directory="$1"
  local pattern="$2"
  local keep="$3"

  find "${directory}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n'     | sort -nr     | awk -v keep="${keep}" 'NR > keep {sub(/^[^ ]+ /, ""); print}'     | while IFS= read -r stale; do
        [ -n "${stale}" ] && rm -f -- "${stale}"
      done
}

prune_local "${LOCAL_DB_DIR}" 'database_*.sql.gz' "${BACKUP_LOCAL_DB_KEEP}"
prune_local "${LOCAL_MANIFEST_DIR}" 'manifest_*.json' "${BACKUP_LOCAL_MANIFEST_KEEP}"

find "${STAGING_ROOT}"   -mindepth 1   -maxdepth 1   -type d   -mtime +3   -exec rm -rf -- {} + 2>/dev/null || true

backup_succeeded=1

log "Backup completado."
log "Snapshot: ${snapshot_id}"
log "Retención: ${retention_class}"
log "Git: ${git_sha}"
if [ "${BACKUP_LOCAL_ONLY}" = "1" ]; then
  log "Modo local: no se subió a S3."
else
  log "S3: s3://${BACKUP_S3_BUCKET}/${snapshot_prefix}/"
fi
