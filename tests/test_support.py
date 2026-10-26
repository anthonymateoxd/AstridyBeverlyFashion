"""Configuracion segura para las pruebas que necesitan la base de datos.

El proyecto tiene un `.env` con la base real, la camara y el checkpoint de
PatchCore. Las pruebas NO deben heredar nada de ese archivo: talk to the
throwaway container created for the test run and nothing else.

Order of precedence:
  1. Environment variables (MYSQL_HOST, MYSQL_PORT, MYSQL_USER,
     MYSQL_PASSWORD, MYSQL_DATABASE).
  2. The local file `.env.test` (never tracked by git).

The helper refuses to continue when the target does not look like a
disposable test database, so a test run can never write to real data.
"""
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ENV_FILE = ROOT / ".env.test"

# Databases that hold real data: a test run must never touch them.
PRODUCTION_DATABASES = {
    "textile_quality_db",
}

# A disposable test database always carries one of these markers.
DISPOSABLE_MARKERS = (
    "test",
    "prueba",
    "validation",
    "validacion",
    "pytest",
    "local",
)

# Local disposable container started for the test run (see diagnostics).
DEFAULT_TEST_HOST = "127.0.0.1"
DEFAULT_TEST_PORT = "43316"
DEFAULT_TEST_USER = "root"
DEFAULT_TEST_DATABASE = "astrid_ai_validation"

# Variables that describe the station geometry and the algorithm thresholds.
# The tests need them to exercise the same configuration as production, but
# they carry no credential, no service and no model.
SAFE_ENV_KEYS = (
    "ROI_X1",
    "ROI_Y1",
    "ROI_X2",
    "ROI_Y2",
    "PATCHCORE_IMAGE_SIZE",
    "PATCHCORE_SCORE_THRESHOLD",
    "PATCHCORE_PIXEL_THRESHOLD",
    "PATCHCORE_MAX_BOX_RATIO",
    "FRAME_SELECTION_BENCHMARK",
)

_CONFIGURED = False


class UnsafeTestDatabase(RuntimeError):
    """Raised when the configured database is not clearly disposable."""


def _read_env_file():
    """Values of the project `.env` (never written back to the environment)."""
    values = {}
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return values
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _safe_load_dotenv(*args, **kwargs):
    """Replacement for `dotenv.load_dotenv` used during the test run.

    Only the station geometry and the thresholds are taken from the project
    `.env`. The real database, the camera credentials, the session secret and
    the PatchCore checkpoint never reach the process.
    """
    values = _read_env_file()
    for key in SAFE_ENV_KEYS:
        if key in values and not os.environ.get(key):
            os.environ[key] = values[key]
    return False


def _read_env_test_file():
    values = {}
    if not TEST_ENV_FILE.is_file():
        return values
    for line in TEST_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _settings():
    file_values = _read_env_test_file()

    def pick(key, default=None):
        value = os.environ.get(key)
        if value is None or value == "":
            value = file_values.get(key)
        if value is None or value == "":
            return default
        return value

    return {
        "host": pick("MYSQL_HOST", DEFAULT_TEST_HOST),
        "port": int(pick("MYSQL_PORT", DEFAULT_TEST_PORT)),
        "user": pick("MYSQL_USER", DEFAULT_TEST_USER),
        "password": pick("MYSQL_PASSWORD", ""),
        "database": pick("MYSQL_DATABASE", DEFAULT_TEST_DATABASE),
    }


def assert_disposable(settings):
    """Refuse to run against anything but a throwaway database."""
    name = str(settings["database"]).strip().lower()
    if name in PRODUCTION_DATABASES:
        raise UnsafeTestDatabase(
            "La base configurada contiene datos reales; las pruebas no pueden "
            "escribir en ella."
        )
    if not any(marker in name for marker in DISPOSABLE_MARKERS):
        raise UnsafeTestDatabase(
            "La base configurada no se reconoce como base de pruebas "
            "desechable; revise MYSQL_DATABASE."
        )
    return settings


def configure_test_environment():
    """Prepare the process to import the app against the test database.

    * The project's `.env` is not loaded (no real credentials, no camera and
      no PatchCore checkpoint reach the test run).
    * The connection points explicitly at the disposable database.
    * Returns the effective settings for the assertions of the tests.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return _settings()

    import dotenv

    # La app lee su geometria de estacion y sus umbrales del `.env`: las
    # pruebas deben ejercitar la misma configuracion. Lo que NO debe llegar
    # a la corrida es la base real, la camara, la clave de sesion ni el
    # checkpoint de PatchCore.
    dotenv.load_dotenv = _safe_load_dotenv

    settings = assert_disposable(_settings())

    os.environ["MYSQL_HOST"] = str(settings["host"])
    os.environ["MYSQL_PORT"] = str(settings["port"])
    os.environ["MYSQL_USER"] = str(settings["user"])
    os.environ["MYSQL_PASSWORD"] = str(settings["password"])
    os.environ["MYSQL_DATABASE"] = str(settings["database"])
    # La app exige una clave de sesion: se usa una exclusiva de pruebas,
    # nunca la del `.env` de produccion.
    os.environ.setdefault(
        "SECRET_KEY",
        "clave-solo-de-pruebas-no-usar-en-produccion",
    )
    # The tests never drive the camera nor load a checkpoint: the camera
    # points to an unreachable local address and carries no credentials.
    os.environ["CAMERA_IP"] = "127.0.0.1"
    for key in (
        "CAMERA_USER",
        "CAMERA_PASSWORD",
        "CAMERA_RTSP_PORT",
        "PATCHCORE_CKPT",
    ):
        os.environ.pop(key, None)

    _CONFIGURED = True
    return settings


def require_test_database():
    """Return the settings or skip the module with an explicit message."""
    try:
        return configure_test_environment()
    except UnsafeTestDatabase as error:
        raise unittest.SkipTest(str(error)) from error
