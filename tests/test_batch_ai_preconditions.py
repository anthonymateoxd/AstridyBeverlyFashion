"""Pruebas de las precondiciones de IA antes de iniciar un lote productivo."""
import ast
import unittest
from pathlib import Path


def helper_de_aplicacion():
    path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "ai_model_ready_for_batch"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    symbols = {}
    exec(compile(module, str(path), "exec"), symbols)
    return symbols[node.name]


class PreparacionModeloIA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready = staticmethod(helper_de_aplicacion())

    def test_activo_con_checkpoint_y_umbral_congelado(self):
        self.assertTrue(self.ready("ACTIVO", 1, "garment_1/checkpoint.pt", 53.3, "2026-10-12"))

    def test_modelo_de_prueba_sin_checkpoint_no_puede_iniciarse(self):
        self.assertFalse(self.ready("ACTIVO", 1, None, None, None))

    def test_modelo_retirado_no_puede_iniciarse(self):
        self.assertFalse(self.ready("RETIRADO", 0, "garment_1/checkpoint.pt", 53.3, "2026-10-12"))

    def test_activo_sin_umbral_no_puede_iniciarse(self):
        self.assertFalse(self.ready("ACTIVO", 1, "garment_1/checkpoint.pt", None, None))

    def test_activo_sin_fecha_de_congelacion_no_puede_iniciarse(self):
        self.assertFalse(self.ready("ACTIVO", 1, "garment_1/checkpoint.pt", 53.3, None))

    def test_checkpoint_vacio_no_puede_iniciarse(self):
        self.assertFalse(self.ready("ACTIVO", 1, "   ", 53.3, "2026-10-12"))

    def test_validacion_se_aplica_en_la_ruta_de_inicio(self):
        path = Path(__file__).resolve().parents[1] / "app.py"
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        handler = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "start_batch")
        body = ast.get_source_segment(src, handler)
        self.assertIn("ai_model_ready_for_batch(", body)
        self.assertIn("checkpoint.is_file()", body)
        self.assertIn("conn.rollback()", body)


if __name__ == "__main__":
    unittest.main()
