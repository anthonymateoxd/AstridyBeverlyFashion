"""Archivador manual de fotos de inspecciones (NO de entrenamiento).
La ejecucion sin argumentos es DRY RUN y no abre conexion AWS.
La retirada de copias locales requiere doble habilitacion separada.

Ejemplos:
  python archive_inspection_photos.py
  PHOTO_ARCHIVE_ENABLED=1 PHOTO_ARCHIVE_BUCKET=<bucket> python archive_inspection_photos.py --mode upload
  PHOTO_ARCHIVE_ALLOW_LOCAL_DELETE=1 python archive_inspection_photos.py --mode remove-local --confirm-local-removal
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
import mysql.connector
from photo_archive import (
    eligible_paths, local_photo, lookup_manifest, remove_local_verified,
    safe_photo_path, save_manifest, s3_client, upload_photo,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "upload", "remove-local"),
                        default="dry-run")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--confirm-local-removal", action="store_true")
    args = parser.parse_args(argv)

    if args.days < 7 or not 1 <= args.limit <= 500:
        parser.error("Solo se permiten al menos 7 dias y maximo 500 archivos")
    if args.mode != "dry-run" and os.getenv("PHOTO_ARCHIVE_ENABLED") != "1":
        parser.error("Activar PHOTO_ARCHIVE_ENABLED=1 para transferencias")
    if args.mode == "remove-local" and not (
        args.confirm_local_removal
        and os.getenv("PHOTO_ARCHIVE_ALLOW_LOCAL_DELETE") == "1"
    ):
        parser.error("La eliminacion local requiere dos autorizaciones explicitas")

    config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "textile_quality_db"),
        "connection_timeout": 10,
    }
    if not config["database"] or not config["user"]:
        parser.error("La BD debe estar configurada")
    connect = lambda: mysql.connector.connect(**config)
    paths = []
    seen = set()
    for path in eligible_paths(
        connect, days=args.days, limit=args.limit,
        stage="remove" if args.mode == "remove-local" else "upload",
    ):
        try:
            clean = safe_photo_path(path)
        except ValueError:
            continue
        if clean not in seen:
            seen.add(clean)
            paths.append(clean)
    print("MODO=", args.mode, "CANDIDATOS=", len(paths), "DIAS=", args.days)
    if args.mode == "dry-run":
        for path in paths:
            local = local_photo(ROOT, path)
            print("CANDIDATO", path, "LOCAL=", local.is_file())
        return 0

    bucket = os.getenv("PHOTO_ARCHIVE_BUCKET", "").strip()
    prefix = os.getenv("PHOTO_ARCHIVE_PREFIX", "astrid/inspection-images/v1")
    if not bucket:
        parser.error("PHOTO_ARCHIVE_BUCKET es obligatorio")
    client = s3_client()
    done, skipped = 0, 0
    for path in paths:
        if args.mode == "upload":
            existing = lookup_manifest(connect, path)
            if existing and not local_photo(ROOT, path).is_file():
                skipped += 1
                continue
            manifest = upload_photo(client, root=ROOT, path=path,
                                    bucket=bucket, prefix=prefix)
            save_manifest(connect, manifest)
        else:
            if not remove_local_verified(
                client, connect=connect, root=ROOT, path=path,
                days=args.days,
            ):
                skipped += 1
                continue
        done += 1
        print("OK", args.mode, path)
    print("PROCESADOS=", done, "OMITIDOS=", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
