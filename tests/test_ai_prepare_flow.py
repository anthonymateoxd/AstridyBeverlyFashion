"""No-database regression tests for first AI preparation UX and locking."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FirstVersionPreparationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (ROOT / "templates/garment_model_detail.html").read_text(
            encoding="utf-8"
        )
        cls.source = (ROOT / "app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_new_models_get_direct_start_capture_action(self):
        self.assertIn("Iniciar captura de fotografías", self.template)
        self.assertIn("if not has_ai_history", self.template)

    def test_existing_preparation_exposes_continue_and_hides_new_version(self):
        self.assertIn("if can_prepare_ai and latest_ai_preparation", self.template)
        self.assertIn("Continuar captura", self.template)
        self.assertIn("not latest_ai_preparation and not ai_in_progress", self.template)

    def test_prepare_post_is_permissioned_transactional_and_reuses_preparation(self):
        route = next(node for node in ast.walk(self.tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "garment_ai_prepare")
        route_source = ast.get_source_segment(self.source, route)
        decorators = [ast.get_source_segment(self.source, item)
                      for item in route.decorator_list]
        self.assertTrue(any("role_required" in item for item in decorators))
        self.assertIn("can_manage_garment_model", route_source)
        self.assertIn("conn.start_transaction()", route_source)
        self.assertIn("FOR UPDATE", route_source)
        self.assertIn("existing_preparation", route_source)
        self.assertIn('"guided_capture_page"', route_source)

    def test_first_version_number_is_derived_from_version_history(self):
        route = next(node for node in ast.walk(self.tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "garment_ai_prepare")
        text = ast.get_source_segment(self.source, route)
        self.assertIn("max(version_numbers, default=0) + 1", text)
        self.assertIn("'PREPARACION'", text)

    def test_version_number_and_patchcore_model_type_are_first_version_contract(self):
        route = next(node for node in ast.walk(self.tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "garment_ai_prepare")
        body = ast.get_source_segment(self.source, route)
        self.assertIn("version = f\"v{next_number}\"", body)
        self.assertIn("'PatchCore'", body)
        self.assertIn("active\n            )", body)

    def test_minimum_gate_is_existing_capture_config_and_training_is_explicit(self):
        config = (ROOT / "ai_domain.py").read_text(encoding="utf-8")
        self.assertIn('"min_images": 20', config)
        self.assertIn("accepted >= minimum", config)
        train_ui = self.template[self.template.find("function actionsHtml"):
                                 self.template.find("function bodyHtml")]
        self.assertIn("if (state.can_train)", train_ui)
        self.assertIn('data-action="train"', train_ui)

    def test_no_production_db_or_auto_training_added_to_approval_handler(self):
        route = next(node for node in ast.walk(self.tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "garment_model_approve")
        text = ast.get_source_segment(self.source, route)
        self.assertNotIn("garment_ai_models", text)
        self.assertNotIn("request_training_job", text)


if __name__ == "__main__":
    unittest.main()
