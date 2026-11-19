"""Preview changes stay separate from capture, inference and training paths."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CatalogPreviewIsolationTests(unittest.TestCase):
    def test_guided_capture_keeps_legacy_mask_only_call(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "garment_visual_preview"]
        self.assertTrue(any(any(k.arg == "mask_only" and
                                isinstance(k.value, ast.Constant) and
                                k.value.value is True for k in call.keywords)
                            for call in calls))

    def test_preview_is_not_used_by_training_or_patchcore_modules(self):
        for name in ("ai_training.py", "ai_capture.py", "ai_validation.py",
                     "patchcore_inference.py", "patchcore_preprocess.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("garment_visual_preview", text, name)

    def test_failed_segmentation_is_not_reported_as_success(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        routes = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.name == "garment_reference_upload_cutout"]
        self.assertEqual(len(routes), 1)
        text = ast.get_source_segment(source, routes[0])
        self.assertIn("if visual is None:", text)
        self.assertIn("}), 422", text)

    def test_reference_fallback_exposes_original_label(self):
        template = (ROOT / "templates/garment_model_detail.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("image.visual_label", template)
        self.assertIn("image.has_visual_cutout", template)

    def test_failed_reference_cutout_uses_original_image_in_detail(self):
        template = (ROOT / "templates/garment_model_detail.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('src="{{ image.visual_url }}"', template)
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        resolver = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "garment_reference_visual_url")
        resolver_source = ast.get_source_segment(source, resolver)
        self.assertIn("_safe_reference_original_url(relative)", resolver_source)

    def test_catalog_canvas_preserves_native_garment_size_without_empty_portrait_frame(self):
        import cv2
        import numpy as np
        from garment_background_preview import _catalog_canvas
        frame = np.full((300, 400, 3), 255, np.uint8)
        mask = np.zeros((300, 400), np.uint8)
        mask[80:220, 100:300] = 1
        frame[mask > 0] = [110, 70, 120]
        png = _catalog_canvas(frame, mask, (100, 80, 200, 140))
        rendered = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        x, y, width, height = cv2.boundingRect(np.any(rendered < 245, axis=2).astype(np.uint8))
        self.assertEqual((width, height), (200, 140))
        self.assertGreater(width / rendered.shape[1], .90)
        self.assertGreater(height / rendered.shape[0], .88)
        self.assertTrue(np.all(rendered[0] == 255))

    def test_reference_url_falls_back_only_to_safe_original_path(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        resolver = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_safe_reference_original_url")
        text = ast.get_source_segment(source, resolver)
        self.assertIn('"results", "model_references"', text)
        self.assertIn("relative_to", text)

    def test_new_ai_version_button_and_idempotent_backend(self):
        template = (ROOT / "templates/garment_model_detail.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("Iniciar captura de fotografías", template)
        self.assertIn("Preparar nueva versión", template)
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        route = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "garment_ai_prepare")
        body = ast.get_source_segment(source, route)
        self.assertIn("FOR UPDATE", body)
        self.assertIn("existing_preparation", body)
        self.assertIn('"guided_capture_page"', body)

    def test_capture_progress_uses_valid_count_not_zeroed_placeholder(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"accepted_count": accepted_count', source)
        self.assertIn("cs.target_ai_model_id IS NULL", source)
        template = (ROOT / "templates/garment_model_detail.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("{{ ai_capture_count }} / {{ ai_capture_minimum }}", template)

    def test_catalog_caption_is_from_same_preview_status_as_url(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn(
            'image["has_visual_cutout"], image["visual_url"] = garment_reference_preview_status(',
            source,
        )
        self.assertIn(
            'model["cover_url"], model["cover_has_cutout"] = garment_cover_display_status(',
            source,
        )
        catalog_template = (ROOT / "templates/garment_models.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("model.cover_display_label", catalog_template)


if __name__ == "__main__":
    unittest.main()
