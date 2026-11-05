"""Archivado conservador de fotografias de inspeccion: NUNCA de entrenamiento.
Sin operaciones AWS ni eliminaciones al importar este modulo.

Etapas: descubrir -> subir/verificar -> registrar en MySQL -> (operacion
SEPARADA y explicita) comprobar nuevamente y retirar solo copia local.
"""
import hashlib
import mimetypes
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

ARCHIVE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspection_photo_archive (
    photo_path VARCHAR(255) NOT NULL PRIMARY KEY,
    bucket VARCHAR(255) NOT NULL,
    object_key VARCHAR(512) NOT NULL,
    sha256 CHAR(64) NOT NULL,
    byte_size BIGINT UNSIGNED NOT NULL,
    archived_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    local_removed_at DATETIME NULL,
    INDEX idx_inspection_photo_archive_removed (local_removed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# Nunca se admiten rutas de modelos, datasets, entrenamiento, IA o secretos.
ALLOWED_FOLDERS = frozenset(("captures", "results"))
ALLOWED_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".webp"))


def safe_photo_path(value):
    if not isinstance(value, str) or not value or len(value) > 255:
        raise ValueError("Ruta de fotografia no valida")
    if "\\" in value or "\x00" in value or "%" in value:
        raise ValueError("Ruta de fotografia no valida")
    path = PurePosixPath(value)
    if (path.is_absolute() or len(path.parts) != 2
            or path.parts[0] not in ALLOWED_FOLDERS
            or path.suffix.lower() not in ALLOWED_SUFFIXES
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", path.name)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Fuera de directorio de inspecciones")
    return path.as_posix()


def local_photo(root, path):
    clean = safe_photo_path(path)
    base = (Path(root) / "static").resolve()
    target = base.joinpath(*PurePosixPath(clean).parts)
    if target.is_symlink() or not target.resolve().is_relative_to(base):
        raise ValueError("Enlace fuera de static")
    return target


def object_key(path, prefix="astrid/inspection-images/v1"):
    relative = safe_photo_path(path)
    prefix = prefix.strip("/")
    if not prefix or any(part in ("", ".", "..") for part in prefix.split("/")):
        raise ValueError("Prefijo S3 invalido")
    return prefix + "/" + relative


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_remote(client, bucket, key, expected_sha, expected_size):
    """Comprueba bytes realmente recuperables, no solo metadata declarada."""
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        body = response["Body"].read()
    finally:
        response["Body"].close()
    if len(body) != expected_size or sha256(body) != expected_sha:
        raise IOError("S3 devolvio contenido diferente del original")
    return True


def upload_photo(client, *, root, path, bucket, prefix):
    """Sube con cifrado, valida lectura S3 y devuelve manifiesto. Sin borrar."""
    clean = safe_photo_path(path)
    file = local_photo(root, clean)
    if not file.is_file():
        raise FileNotFoundError(clean)
    data = file.read_bytes()
    if not data:
        raise ValueError("Fotografia vacia")
    digest = sha256(data)
    key = object_key(clean, prefix)
    # Objetos existentes NO se sobreescriben si no son identicos.
    exists = False
    try:
        client.head_object(Bucket=bucket, Key=key)
        exists = True
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if str(code) not in ("404", "NoSuchKey", "NotFound"):
            raise
    if not exists:
        client.put_object(
            Bucket=bucket, Key=key, Body=data,
            ContentType=mimetypes.guess_type(clean)[0] or "application/octet-stream",
            ServerSideEncryption="AES256",
            Metadata={"sha256": digest},
        )
    verify_remote(client, bucket, key, digest, len(data))
    return {
        "photo_path": clean,
        "bucket": bucket,
        "object_key": key,
        "sha256": digest,
        "byte_size": len(data),
    }


def save_manifest(connect, manifest):
    """Graba referencia solo tras confirmar que S3 permite la recuperacion."""
    conn = connect()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT bucket, object_key, sha256, byte_size "
            "FROM inspection_photo_archive WHERE photo_path=%s FOR UPDATE",
            (manifest["photo_path"],),
        )
        prior = cursor.fetchone()
        if prior:
            for field in ("bucket", "object_key", "sha256", "byte_size"):
                if str(prior[field]) != str(manifest[field]):
                    raise ValueError("El manifiesto existente es distinto")
        else:
            cursor.execute(
                "INSERT INTO inspection_photo_archive "
                "(photo_path,bucket,object_key,sha256,byte_size) "
                "VALUES (%s,%s,%s,%s,%s)",
                tuple(manifest[field] for field in (
                    "photo_path", "bucket", "object_key", "sha256", "byte_size"
                )),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def lookup_manifest(connect, path):
    clean = safe_photo_path(path)
    conn = connect()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT photo_path,bucket,object_key,sha256,byte_size,local_removed_at "
            "FROM inspection_photo_archive WHERE photo_path=%s", (clean,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()
        conn.close()


def eligible_paths(connect, *, days=7, limit=50, now=None, stage="upload"):
    """Devuelve lotes de trabajo progresivos, excluyendo los ya procesados."""
    if days < 7 or not 1 <= limit <= 500:
        raise ValueError("Politica minima: 7 dias, maximo 500 archivos")
    if stage not in ("upload", "remove"):
        raise ValueError("Etapa del archivador invalida")
    now = now or datetime.now(timezone.utc)
    # Las fechas MySQL se almacenan en UTC sin tzinfo.
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(days=days)
    # La condicion se selecciona de dos constantes, nunca del SQL del usuario.
    stage_filter = (
        "a.photo_path IS NULL"
        if stage == "upload"
        else "a.photo_path IS NOT NULL AND a.local_removed_at IS NULL"
    )
    conn = connect()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT c.path AS path FROM (
                SELECT i.image_original AS path, MAX(i.created_at) AS last_use
                FROM inspections i
                WHERE i.image_original IS NOT NULL AND i.image_original != ''
                GROUP BY i.image_original
                UNION ALL
                SELECT i.image_result AS path, MAX(i.created_at) AS last_use
                FROM inspections i
                WHERE i.image_result IS NOT NULL AND i.image_result != ''
                GROUP BY i.image_result
            ) c
            LEFT JOIN inspection_photo_archive a ON a.photo_path=(c.path COLLATE utf8mb4_unicode_ci)
            WHERE """ + stage_filter + """
              AND NOT EXISTS (
                  SELECT 1 FROM inspections active
                  LEFT JOIN batches b ON b.id=active.batch_id
                  WHERE (active.image_original=c.path OR active.image_result=c.path)
                    AND (b.id IS NULL OR b.status NOT IN
                         ('COMPLETADO','REVISION_PENDIENTE'))
              )
            GROUP BY c.path
            HAVING MAX(c.last_use) < %s
            ORDER BY MIN(c.last_use), c.path LIMIT %s
            """,
            (cutoff, limit),
        )
        return [row["path"] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()


def remove_local_verified(client, *, connect, root, path, days=7, now=None):
    """Elimina copia LOCAL solo si fue archivada y su contenido no cambio.

    Debe invocarse solamente despues de --remove-local confirmado. Prohibe
    borrado de imagenes no elegibles, no referenciadas o demasiado recientes.
    """
    clean = safe_photo_path(path)
    if clean not in set(eligible_paths(connect, days=days, limit=500, now=now, stage="remove")):
        raise ValueError("Imagen no elegible para retirada")
    item = lookup_manifest(connect, clean)
    if item is None:
        raise ValueError("Sin copia S3 verificada")
    file = local_photo(root, clean)
    present = file.is_file()
    if present:
        contents = file.read_bytes()
        if len(contents) != item["byte_size"] or sha256(contents) != item["sha256"]:
            raise ValueError("Fotografia local cambio desde el archivado")
    verify_remote(client, item["bucket"], item["object_key"],
                  item["sha256"], item["byte_size"])
    if present:
        file.unlink()
    conn = connect()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE inspection_photo_archive "
            "SET local_removed_at=UTC_TIMESTAMP() WHERE photo_path=%s",
            (clean,),
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    return present


def s3_client():
    import boto3
    return boto3.client("s3", region_name=os.getenv("PHOTO_ARCHIVE_REGION", "us-east-2"))
