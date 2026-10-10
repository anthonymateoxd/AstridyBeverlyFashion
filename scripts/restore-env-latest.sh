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

BACKUP_S3_BUCKET="${BACKUP_S3_BUCKET:-}"
BACKUP_S3_PREFIX="${BACKUP_S3_PREFIX:-astrid/production}"
BACKUP_S3_PREFIX="${BACKUP_S3_PREFIX%/}"
AWS_REGION="${AWS_REGION:-us-east-2}"
OUTPUT=""
FORCE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output)
      OUTPUT="${2:-}"
      shift 2
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

if [ -z "${BACKUP_S3_BUCKET}" ]; then
  echo "BACKUP_S3_BUCKET no está configurado." >&2
  exit 1
fi

if [ -z "${OUTPUT}" ]; then
  echo "--output es obligatorio." >&2
  exit 2
fi

if [ -e "${OUTPUT}" ] && [ "${FORCE}" != "1" ]; then
  echo "${OUTPUT} ya existe. Use --force para reemplazarlo." >&2
  exit 1
fi

for command_name in aws python3 sha256sum stat; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Falta el comando ${command_name}." >&2
    exit 1
  }
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

latest_key="${BACKUP_S3_PREFIX}/latest/manifest.json"
manifest="${tmp_dir}/manifest.json"

aws s3 cp   "s3://${BACKUP_S3_BUCKET}/${latest_key}"   "${manifest}"   --region "${AWS_REGION}"   --only-show-errors

env_info="$(
  python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in payload.get("files", []):
    if item.get("label") == "secure_env":
        print(
            item["key"],
            item["sha256"],
            item["size_bytes"],
            sep="\t",
        )
        break
else:
    raise SystemExit("El snapshot no contiene secure_env.")
PY
)"

IFS=$'\t' read -r env_key expected_sha expected_size <<<"${env_info}"
temp_env="${tmp_dir}/env-file"

aws s3 cp   "s3://${BACKUP_S3_BUCKET}/${env_key}"   "${temp_env}"   --region "${AWS_REGION}"   --only-show-errors

actual_sha="$(sha256sum "${temp_env}" | awk '{print $1}')"
actual_size="$(stat -c '%s' "${temp_env}")"

if [ "${actual_sha}" != "${expected_sha}" ]; then
  echo "SHA-256 de configuración privada no coincide." >&2
  exit 1
fi

if [ "${actual_size}" != "${expected_size}" ]; then
  echo "Tamaño de configuración privada no coincide." >&2
  exit 1
fi

install -m 0600 "${temp_env}" "${OUTPUT}"

echo "Configuración privada restaurada en ${OUTPUT}."
echo "Contenido no mostrado por seguridad."
