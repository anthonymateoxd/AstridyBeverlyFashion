"""Pruebas aisladas del ciclo de reconexión RTSP: sin cámara ni base de datos."""

import ast
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


class StopWorker(Exception):
    """Detiene el bucle infinito después de un ciclo de prueba."""


class CaptureSimulado:
    def __init__(self):
        self.liberaciones = 0

    def read(self):
        return False, None

    def release(self):
        self.liberaciones += 1


def construir_worker(captura, *, forzar=False):
    ruta = Path(__file__).resolve().parents[1] / "app.py"
    modulo = ast.parse(ruta.read_text(encoding="utf-8"))
    funcion = next(
        nodo for nodo in modulo.body
        if isinstance(nodo, ast.FunctionDef)
        and nodo.name == "camera_capture_worker"
    )
    entorno = {
        "CAMERA_CONNECTED": True,
        "CAMERA_LAST_OK_AT": None,
        "CAMERA_LAST_ERROR": None,
        "CAMERA_FAILURE_COUNT": 0,
        "CAMERA_RECONNECTING": False,
        "CAMERA_FORCE_REOPEN": forzar,
        "CAMERA_LAST_CONNECT_ATTEMPT": 18.0,
        "CAMERA_RETRY_SECONDS": 1,
        "CAMERA_READ_FAIL_RELEASE_LIMIT": 1,
        "CAMERA_RECONNECT_LOCK": threading.Lock(),
        "CAMERA_CAPTURE_WORKER_LOCK": threading.Lock(),
        "CAMERA_CURRENT_IP": "127.0.0.1",
        "CAMERA_CHANNEL": 1,
        "CAMERA_SUBTYPE": 0,
        "camera_lock": threading.Lock(),
        "camera_capture": captura,
        "cv2": object(),
        "app": SimpleNamespace(logger=SimpleNamespace(info=lambda *a: None)),
        "get_camera": lambda: entorno["camera_capture"],
        "_clear_latest_frame": lambda: None,
        "_note_camera_signal_lost": lambda error: None,
        "_try_discover_camera_after_failures": lambda count: False,
        "time": SimpleNamespace(sleep=lambda _: (_ for _ in ()).throw(StopWorker())),
    }
    solo_funcion = ast.fix_missing_locations(ast.Module(body=[funcion], type_ignores=[]))
    exec(compile(solo_funcion, str(ruta), "exec"), entorno)
    return entorno


class PruebasReconectador(unittest.TestCase):
    def test_fallo_de_lectura_libera_captura_global(self):
        captura = CaptureSimulado()
        entorno = construir_worker(captura)
        with self.assertRaises(StopWorker):
            entorno["camera_capture_worker"]()
        self.assertEqual(captura.liberaciones, 1)
        self.assertIsNone(entorno["camera_capture"])
        self.assertEqual(entorno["CAMERA_LAST_CONNECT_ATTEMPT"], 0.0)

    def test_reconexion_forzada_libera_captura_global(self):
        captura = CaptureSimulado()
        entorno = construir_worker(captura, forzar=True)
        with self.assertRaises(StopWorker):
            entorno["camera_capture_worker"]()
        self.assertEqual(captura.liberaciones, 1)
        self.assertIsNone(entorno["camera_capture"])
        self.assertFalse(entorno["CAMERA_FORCE_REOPEN"])


class PruebasEstadoCamara(unittest.TestCase):
    def _estado(self, antiguedad):
        ruta = Path(__file__).resolve().parents[1] / "app.py"
        modulo = ast.parse(ruta.read_text(encoding="utf-8"))
        funcion = next(
            n for n in modulo.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "get_camera_status"
        )
        entorno = {
            "ensure_camera_capture_worker": lambda: None,
            "latest_frame_lock": threading.Lock(),
            "latest_frame_timestamp_monotonic": 20.0 - antiguedad,
            "CAMERA_CONNECTED": True,
            "CAMERA_LAST_OK_AT": "2026-10-12 19:00:00",
            "CAMERA_RECONNECT_LOCK": threading.Lock(),
            "CAMERA_RECONNECTING": False,
            "CAMERA_LAST_ERROR": None,
            "time": SimpleNamespace(perf_counter=lambda: 20.0),
        }
        arbol = ast.fix_missing_locations(ast.Module(body=[funcion], type_ignores=[]))
        exec(compile(arbol, str(ruta), "exec"), entorno)
        return entorno["get_camera_status"]()

    def test_fotograma_reciente_si_conectado(self):
        self.assertTrue(self._estado(0.5)["connected"])

    def test_fotograma_antiguo_no_indica_conectado(self):
        estado = self._estado(12.0)
        self.assertFalse(estado["connected"])
        self.assertEqual(estado["state"], "STALE")


if __name__ == "__main__":
    unittest.main()
