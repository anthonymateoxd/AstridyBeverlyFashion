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
VERIFY_DOWNLOAD="${VERIFY_DOWNLOAD:-0}"

if [ -z "${BACKUP_S3_BUCKET}" ]; then
  echo "BACKUP_S3_BUCKET no está configurado." >&2
  exit 1
fi

for command_name in aws python3 sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Falta el comando ${command_name}." >&2
    exit 1
  }
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

latest_key="${BACKUP_S3_PREFIX}/latest/manifest.json"
manifest="${tmp_dir}/manifest.json"

echo "[verify] Descargando s3://${BACKUP_S3_BUCKET}/${latest_key}"
aws s3 cp   "s3://${BACKUP_S3_BUCKET}/${latest_key}"   "${manifest}"   --region "${AWS_REGION}"   --only-show-errors

python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {
    "schema_version",
    "snapshot_id",
    "created_at_utc",
    "retention_class",
    "files",
}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"Manifest incompleto: faltan {missing}")

if payload["schema_version"] != 1:
    raise SystemExit(
        f"schema_version no soportado: {payload['schema_version']!r}"
    )

print(
    f"[verify] Snapshot={payload['snapshot_id']} "
    f"fecha={payload['created_at_utc']} "
    f"retención={payload['retention_class']}"
)
PY

secure_info="$(
  python3 - "${manifest}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
secure = payload.get("secure_config") or {}
if secure.get("enabled"):
    print(
        secure["parameter"],
        secure["version"],
        secure["sha256"],
        sep="\t",
    )
PY
)"

if [ -n "${secure_info}" ]; then
  IFS=$'\t' read -r secure_parameter secure_version secure_sha <<<"${secure_info}"
  secure_response="${tmp_dir}/secure-parameter.json"
  secure_value="${tmp_dir}/secure-env"

  aws ssm get-parameter     --name "${secure_parameter}:${secure_version}"     --with-decryption     --region "${AWS_REGION}"     --output json >"${secure_response}"

  python3 - "${secure_response}" "${secure_value}" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(sys.argv[2]).write_text(
    response["Parameter"]["Value"],
    encoding="utf-8",
)
PY

  actual_secure_sha="$(sha256sum "${secure_value}" | awk '{print $1}')"
  rm -f "${secure_response}" "${secure_value}"

  if [ "${actual_secure_sha}" != "${secure_sha}" ]; then
    echo "[verify] SecureString .env no coincide con el manifest." >&2
    exit 1
  fi

  echo "[verify] OK configuración privada SSM versión ${secure_version}"
fi

files_tsv="${tmp_dir}/files.tsv"
python3 - "${manifest}" >"${files_tsv}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in payload.get("files", []):
    print(
        item["label"],
        item["key"],
        item["sha256"],
        item["size_bytes"],
        sep="\t",
    )
PY

verified=0

while IFS=$'\t' read -r label key expected_sha expected_size; do
  [ -n "${key}" ] || continue

  remote_sha="$(
    aws s3api head-object       --bucket "${BACKUP_S3_BUCKET}"       --key "${key}"       --region "${AWS_REGION}"       --query 'Metadata.sha256'       --output text
  )"

  if [ "${remote_sha}" != "${expected_sha}" ]; then
    echo "[verify] SHA metadata inválido para ${label}: ${key}" >&2
    exit 1
  fi

  remote_size="$(
    aws s3api head-object       --bucket "${BACKUP_S3_BUCKET}"       --key "${key}"       --region "${AWS_REGION}"       --query 'ContentLength'       --output text
  )"

  if [ "${remote_size}" != "${expected_size}" ]; then
    echo "[verify] Tamaño inválido para ${label}: ${key}" >&2
    exit 1
  fi

  if [ "${VERIFY_DOWNLOAD}" = "1" ]; then
    local_copy="${tmp_dir}/verify-${verified}.bin"
    aws s3 cp       "s3://${BACKUP_S3_BUCKET}/${key}"       "${local_copy}"       --region "${AWS_REGION}"       --only-show-errors

    actual_sha="$(sha256sum "${local_copy}" | awk '{print $1}')"
    if [ "${actual_sha}" != "${expected_sha}" ]; then
      echo "[verify] SHA descargado inválido para ${label}: ${key}" >&2
      exit 1
    fi
  fi

  verified=$((verified + 1))
  echo "[verify] OK ${label} (${expected_size} bytes)"
done <"${files_tsv}"

if [ "${verified}" -lt 1 ]; then
  echo "[verify] El manifest no contiene payloads." >&2
  exit 1
fi

echo "[verify] Backup válido: ${verified} payload(s) verificados."
