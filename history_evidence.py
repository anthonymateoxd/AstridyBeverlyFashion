"""Read-only history presentation: checkpoint availability and per-inspection files."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=64)
def _checkpoint_hash(path, size, modified_ns, changed_ns):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_is_available(root, relative, expected_hash):
    try:
        root = Path(root).resolve()
        path = (root / str(relative or '')).resolve()
        expected = str(expected_hash or '').strip().lower()
        if not path.is_relative_to(root) or not path.is_file() or len(expected) != 64:
            return False
        stat = path.stat()
        return stat.st_size >= 1024 and _checkpoint_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == expected
    except (OSError, ValueError):
        return False


def inspection_evidence_path(root, relative):
    """Resolve only the field belonging to this inspection; never a catalog image."""
    root = Path(root).resolve()
    relative = str(relative or '').replace('\\', '/')
    if not relative or relative.startswith(('results/model_', 'garment_models/')):
        return None
    try:
        source = (root / relative).resolve()
        if (source.is_relative_to(root) and source.is_file()
                and source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
                and source.relative_to(root).parts[0] in {'captures', 'results'}):
            return source.relative_to(root).as_posix()
    except (OSError, ValueError):
        pass
    return None
