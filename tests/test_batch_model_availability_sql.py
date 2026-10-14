"""Comprueba el contrato SQL de los modelos elegibles para crear lotes."""
import ast
import re
import unittest
from pathlib import Path


class ModelosDeLoteDisponibles(unittest.TestCase):
    def test_get_y_post_exigen_ia_productiva(self):
        source = (
            Path(__file__).resolve().parents[1] / "app.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        handler = next(
            item for item in module.body
            if isinstance(item, ast.FunctionDef) and item.name == "batches"
        )
        query_strings = [
            " ".join(node.value.split())
            for node in ast.walk(handler)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "JOIN garment_ai_models ai" in node.value
        ]
        self.assertEqual(len(query_strings), 2)
        for sql in query_strings:
            self.assertIn("ai.status = 'ACTIVO'", sql)
            self.assertIn("ai.active = 1", sql)
            self.assertIn("NULLIF(TRIM(ai.checkpoint_path), '') IS NOT NULL", sql)
            self.assertIn("ai.threshold_final IS NOT NULL", sql)
            self.assertIn("ai.threshold_frozen_at IS NOT NULL", sql)
        self.assertIn(
            "NULLIF(TRIM(ai2.checkpoint_path), '') IS NOT NULL",
            query_strings[0],
        )


if __name__ == "__main__":
    unittest.main()
