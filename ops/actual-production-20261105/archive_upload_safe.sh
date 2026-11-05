#!/usr/bin/env bash
set -euo pipefail
expected="sha256:51ad3eb55636eb5e89aa125ca4e658dca861e5ba67c3519bc9d73cdd5acabb19"
actual="$(/usr/bin/docker inspect -f '{{.Image}}' astrid_quality_app)"
test "$actual" = "$expected" || { echo 'ERROR_UNEXPECTED_APP_IMAGE' >&2; exit 2; }
test "$(/usr/bin/docker inspect -f '{{.State.Running}}' astrid_quality_app)" = true || exit 2
envnow="$(/usr/bin/docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' astrid_quality_app)"
for requirement in 'PHOTO_ARCHIVE_ENABLED=1' 'PHOTO_ARCHIVE_ALLOW_LOCAL_DELETE=0' 'PHOTO_ARCHIVE_BUCKET=astrid-quality-backups-184140788070' 'PHOTO_ARCHIVE_PREFIX=astrid/inspection-images/v1'; do
 grep -Fxq "$requirement" <<<"$envnow" || { echo 'ERROR_UNEXPECTED_PHOTO_CONFIG' >&2; exit 2; }
done
exec 9>/run/lock/astrid-photo-archive.lock
flock -n 9 || { echo 'ALREADY_RUNNING_SKIP=1'; exit 0; }
echo "ARCHIVE_UPLOAD_ONLY_START=$(date -u +%FT%TZ)"
/usr/bin/docker exec -i astrid_quality_app python - --run --limit 200 < /opt/astrid-release-d733420/archive_upload_safe.py
echo "ARCHIVE_UPLOAD_ONLY_COMPLETE=$(date -u +%FT%TZ)"

