"""Pruebas del estado de salud PatchCore sin importar Flask ni abrir la BD."""
import ast
import tempfile
import unittest
from pathlib import Path


def health_classifier():
    path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "summarize_patchcore_readiness"
    )
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    class InvalidPath(Exception):
        pass

    def safe_resolve(root, name):
        value = Path(name)
        if value.is_absolute() or ".." in value.parts:
            raise InvalidPath("Ruta no permitida")
        return (Path(root) / value).resolve()

    namespace = {
        "resolve_under_root": safe_resolve,
        "AIDomainError": InvalidPath,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[fn.name]


class PatchCoreHealth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classify = staticmethod(health_classifier())

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        (self.root / "a.pt").write_bytes(b"x" * 2048)
        (self.root / "short.pt").write_bytes(b"x")

    def version(self, ident=1, path="a.pt", status="ACTIVO",
                active=1, threshold=50.0, frozen="2026-10-13"):
        return {
            "id": ident, "checkpoint_path": path,
            "status": status, "active": active,
            "threshold_final": threshold,
            "threshold_frozen_at": frozen,
        }

    def test_modelos_disponibles_sin_lote(self):
        result = self.classify(
            [self.version(), self.version(2, None)], None, self.root
        )
        self.assertTrue(result["model_ready"])
        self.assertEqual(result["ready_patchcore_models"], 1)
        self.assertEqual(result["active_patchcore_models"], 2)
        self.assertEqual(result["model_ready_scope"], "available_models")
        self.assertIsNone(result["active_batch_model_ready"])
        self.assertEqual(result["detection_mode"], "patchcore")

    def test_lote_activo_valido(self):
        result = self.classify(
            [self.version()],
            {"ai_model_id": 1, "garment_model_id": 4,
             "ai_garment_model_id": 4},
            self.root,
        )
        self.assertTrue(result["active_batch_model_ready"])
        self.assertEqual(result["model_ready_scope"], "active_batch")

    def test_lote_activo_con_modelo_ausente(self):
        result = self.classify(
            [self.version()],
            {"ai_model_id": 7, "garment_model_id": 4,
             "ai_garment_model_id": 4},
            self.root,
        )
        self.assertFalse(result["model_ready"])
        self.assertEqual(result["ready_patchcore_models"], 1)

    def test_lote_activo_con_modelo_de_otra_prenda(self):
        result = self.classify(
            [self.version()],
            {"ai_model_id": 1, "garment_model_id": 4,
             "ai_garment_model_id": 8},
            self.root,
        )
        self.assertFalse(result["active_batch_model_ready"])

    def test_faltan_umbral_archivo_o_congelacion(self):
        versions = [
            self.version(1, "short.pt"),
            self.version(2, "missing.pt"),
            self.version(3, threshold=None),
            self.version(4, frozen=None),
            self.version(5, path="../escape.pt"),
            self.version(6, status="RETIRADO"),
        ]
        result = self.classify(versions, None, self.root)
        self.assertEqual(result["ready_patchcore_models"], 0)
        self.assertFalse(result["model_ready"])

    def test_salud_conserva_status_ok_monitor(self):
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        handler = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "health"
        )
        body = ast.get_source_segment(source, handler)
        self.assertIn('"status": "ok"', body)
        self.assertIn("summarize_patchcore_readiness(", body)
        self.assertNotIn("model_ready = MODEL_PATH", body)


if __name__ == "__main__":
    unittest.main()
