"""Regresión de los estados visibles en el historial de inspecciones.

Se extrae la función pura del código real sin importar Flask o conectar MySQL.
"""
import ast
import unittest
from pathlib import Path


def cargar_clasificador():
    app = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app.read_text(encoding="utf-8"))
    target = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "classify_record_final_decision"
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[target], type_ignores=[])
    )
    context = {}
    exec(compile(isolated, str(app), "exec"), context)
    return context[target.name]


class EstadosHistorial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classify = staticmethod(cargar_clasificador())

    def verificar(self, ai_decision, status, review_status, human, text, style):
        self.assertEqual(
            self.classify(ai_decision, status, review_status, human),
            (text, style),
        )

    def test_normal_sin_revision(self):
        self.verificar(
            "NORMAL", "Aprobado", "NO_REQUIERE_REVISION", "No requerida",
            "Aceptada", "passed",
        )

    def test_aceptacion_legacy(self):
        self.verificar(None, "Aprobado", None, "Pendiente", "Aceptada", "passed")

    def test_anomalia_pendiente(self):
        self.verificar(
            "ANOMALIA", "Defecto", "PENDIENTE", "Pendiente",
            "Pendiente de revisión", "pending",
        )

    def test_defecto_confirmado(self):
        self.verificar(
            "ANOMALIA", "Defecto", "DEFECTO_CONFIRMADO", "Correcto",
            "Rechazada por defecto", "rejected",
        )

    def test_alerta_descartada(self):
        self.verificar(
            "ANOMALIA", "Defecto", "ALERTA_DESCARTADA", "Incorrecto",
            "Aceptada - alerta descartada", "passed",
        )

    def test_prioridad_revision_humana_en_estado_contradictorio(self):
        self.verificar(
            "NORMAL", "Aprobado", "DEFECTO_CONFIRMADO", "Correcto",
            "Rechazada por defecto", "rejected",
        )

    def test_template_muestra_responsable_para_normal(self):
        template = (Path(__file__).resolve().parents[1] / "templates" /
                    "records.html").read_text(encoding="utf-8-sig")
        self.assertIn("No requiere revisi&oacute;n humana", template)
        self.assertIn("{{ row.decision_label }}", template)


if __name__ == "__main__":
    unittest.main()
