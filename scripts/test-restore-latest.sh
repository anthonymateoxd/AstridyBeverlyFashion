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
BACKUP_S3_BUCKET="${BACKUP_S3_BUCKET:-}"
BACKUP_S3_PREFIX="${BACKUP_S3_PREFIX:-astrid/production}"
BACKUP_S3_PREFIX="${BACKUP_S3_PREFIX%/}"
AWS_REGION="${AWS_REGION:-us-east-2}"

if [ -z "${BACKUP_S3_BUCKET}" ]; then
  echo "BACKUP_S3_BUCKET no está configurado." >&2
  exit 1
fi

for command_name in aws docker gzip python3 sha256sum stat; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Falta el comando ${command_name}." >&2
    exit 1
  }
done

[ -f "${APP_DIR}/.env" ] || {
  echo "Falta ${APP_DIR}/.env." >&2
  exit 1
}

compose() {
  docker compose --project-directory "${APP_DIR}" --env-file "${APP_DIR}/.env" "$@"
}

tmp_dir="$(mktemp -d)"
restore_db="astrid_restore_test_$(date -u +%Y%m%d%H%M%S)_$$"
created_db=0

cleanup() {
  if [ "${created_db}" = "1" ]; then
    compose exec -T -e RESTORE_DB="${restore_db}" mysql sh -lc       'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "DROP DATABASE IF EXISTS $RESTORE_DB"'       >/dev/null 2>&1 || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

latest_key="${BACKUP_S3_PREFIX}/latest/manifest.json"
manifest="${tmp_dir}/manifest.json"

echo "[restore-test] Descargando manifest más reciente."
aws s3 cp   "s3://${BACKUP_S3_BUCKET}/${latest_key}"   "${manifest}"   --region "${AWS_REGION}"   --only-show-errors

db_info="$(
  python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in payload.get("files", []):
    if item.get("label") == "database":
        print(
            item["key"],
            item["sha256"],
            item["size_bytes"],
            sep="\t",
        )
        break
else:
    raise SystemExit("El manifest no contiene payload database.")
PY
)"

IFS=$'\t' read -r db_key expected_sha expected_size <<<"${db_info}"
db_archive="${tmp_dir}/database.sql.gz"

echo "[restore-test] Descargando dump MySQL."
aws s3 cp   "s3://${BACKUP_S3_BUCKET}/${db_key}"   "${db_archive}"   --region "${AWS_REGION}"   --only-show-errors

actual_sha="$(sha256sum "${db_archive}" | awk '{print $1}')"
actual_size="$(stat -c '%s' "${db_archive}")"

if [ "${actual_sha}" != "${expected_sha}" ]; then
  echo "[restore-test] SHA-256 del dump no coincide." >&2
  exit 1
fi

if [ "${actual_size}" != "${expected_size}" ]; then
  echo "[restore-test] Tamaño del dump no coincide." >&2
  exit 1
fi

gzip -t "${db_archive}"

echo "[restore-test] Creando BD temporal ${restore_db}."
compose exec -T -e RESTORE_DB="${restore_db}" mysql sh -lc   'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "CREATE DATABASE $RESTORE_DB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"'
created_db=1

echo "[restore-test] Restaurando dump en la BD temporal."
gzip -dc "${db_archive}"   | compose exec -T -e RESTORE_DB="${restore_db}" mysql sh -lc       'exec mysql -uroot -p"$MYSQL_ROOT_PASSWORD" "$RESTORE_DB"'

table_list="$(
  compose exec -T -e RESTORE_DB="${restore_db}" mysql sh -lc     'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" "$RESTORE_DB" -Nse "SHOW TABLES"'     | tr -d '\r'
)"

table_count="$(printf '%s\n' "${table_list}" | sed '/^$/d' | wc -l | tr -d ' ')"

if [ "${table_count:-0}" -lt 5 ]; then
  echo "[restore-test] La restauración produjo muy pocas tablas: ${table_count:-0}." >&2
  exit 1
fi

for required_table in users batches inspections garment_models garment_ai_models; do
  if ! printf '%s\n' "${table_list}" | grep -Fxq "${required_table}"; then
    echo "[restore-test] Falta tabla núcleo: ${required_table}." >&2
    exit 1
  fi
done

echo "[restore-test] Restauración válida."
echo "[restore-test] Tablas restauradas: ${table_count}"
echo "[restore-test] La BD temporal será eliminada; producción no fue modificada."
