"""Regresión: la función UTC usada por la eliminación debe existir y ser consistente."""

import ast
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


class FechaUTCParaMySQL(unittest.TestCase):
    def test_helper_devuelve_datetime_naive_en_utc(self):
        ruta = Path(__file__).resolve().parents[1] / "app.py"
        modulo = ast.parse(ruta.read_text(encoding="utf-8"))
        funcion = next(
            nodo for nodo in modulo.body
            if isinstance(nodo, ast.FunctionDef) and nodo.name == "utc_now_db"
        )
        codigo = ast.fix_missing_locations(ast.Module(body=[funcion], type_ignores=[]))
        contexto = {"datetime": datetime, "ZoneInfo": ZoneInfo}
        exec(compile(codigo, str(ruta), "exec"), contexto)
        antes = datetime.now(timezone.utc).replace(tzinfo=None)
        fecha = contexto["utc_now_db"]()
        despues = datetime.now(timezone.utc).replace(tzinfo=None)
        self.assertIsInstance(fecha, datetime)
        self.assertIsNone(fecha.tzinfo)
        self.assertLessEqual(antes, fecha)
        self.assertLessEqual(fecha, despues)


if __name__ == "__main__":
    unittest.main()
