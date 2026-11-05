"""ASTRID upload-only daily archive, never removes local data."""
import argparse
import hashlib
import os
import sys
from pathlib import Path
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import mysql.connector
from photo_archive import eligible_paths, local_photo, upload_photo, save_manifest, lookup_manifest, s3_client

ROOT=Path("/app")
EXPECTED_BUCKET="astrid-quality-backups-184140788070"
EXPECTED_PREFIX="astrid/inspection-images/v1"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    args=parser.parse_args()
    if not 1<=args.limit<=200:
        raise RuntimeError("BAD_LIMIT")
    if os.getenv("PHOTO_ARCHIVE_ENABLED")!="1" or os.getenv("PHOTO_ARCHIVE_ALLOW_LOCAL_DELETE")!="0":
        raise RuntimeError("ARCHIVE_FLAGS_UNSAFE")
    if os.getenv("PHOTO_ARCHIVE_BUCKET")!=EXPECTED_BUCKET or os.getenv("PHOTO_ARCHIVE_PREFIX")!=EXPECTED_PREFIX:
        raise RuntimeError("S3_DESTINATION_UNEXPECTED")
    cfg=dict(host=os.getenv("MYSQL_HOST","127.0.0.1"),port=int(os.getenv("MYSQL_PORT","3306")),user=os.environ["MYSQL_USER"],password=os.environ["MYSQL_PASSWORD"],database=os.environ["MYSQL_DATABASE"],connection_timeout=10)
    def connect(): return mysql.connector.connect(**cfg)
    # Solo se SUBEN las fotos de inspecciones del dia anterior o mas antiguas.
    # El modulo compartido aplica una barrera minima de 7 dias; se ajusta
    # su reloj de consulta para que el corte efectivo sea la medianoche GT.
    midnight_gt = datetime.combine(datetime.now(ZoneInfo("America/Guatemala")).date(), time.min, tzinfo=ZoneInfo("America/Guatemala"))
    query_now = midnight_gt.astimezone(timezone.utc) + timedelta(days=7)
    paths=eligible_paths(connect,days=7,limit=500,now=query_now,stage="upload")
    candidates=[]
    missing=0
    for path in paths:
        if local_photo(ROOT,path).is_file():
            if len(candidates)<args.limit:
                candidates.append(path)
        else:
            missing+=1
    print("ARCHIVE_MODE="+("UPLOAD_ONLY" if args.run else "DRY_RUN"),flush=True)
    print("ELIGIBLE_SCANNED="+str(len(paths))+" AVAILABLE_SELECTED="+str(len(candidates))+" MISSING_REFERENCES="+str(missing),flush=True)
    if not args.run:
        return 0
    client=s3_client()
    done=0
    skipped=0
    for path in candidates:
        file=local_photo(ROOT,path)
        if not file.is_file():
            skipped+=1
            continue
        original=file.read_bytes()
        if not original:
            raise RuntimeError("EMPTY_PHOTO")
        digest=hashlib.sha256(original).hexdigest()
        m=upload_photo(client,root=ROOT,path=path,bucket=EXPECTED_BUCKET,prefix=EXPECTED_PREFIX)
        if m["sha256"]!=digest:
            raise RuntimeError("DIGEST_MISMATCH")
        save_manifest(connect,m)
        record=lookup_manifest(connect,path)
        if not record or record["sha256"]!=digest or record["local_removed_at"] is not None:
            raise RuntimeError("MANIFEST_ERROR")
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("LOCAL_CHANGED")
        done+=1
    print("UPLOAD_VERIFIED="+str(done)+" SKIPPED_MISSING_DURING_RUN="+str(skipped),flush=True)
    print("LOCAL_REMOVAL_PERFORMED=0",flush=True)
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ARCHIVE_FAILED="+type(exc).__name__+":"+str(exc),file=sys.stderr,flush=True)
        sys.exit(1)

