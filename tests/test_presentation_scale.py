"""Guard against nested white margins, distorted silhouettes and raw fallbacks."""
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from garment_background_preview import _catalog_canvas, garment_manual_preview


class PresentationScaleTests(unittest.TestCase):
    def assert_fills_frame(self, image, minimum=.88):
        x, y, width, height = cv2.boundingRect(np.any(image < 245, axis=2).astype(np.uint8))
        self.assertGreater(width / image.shape[1], minimum)
        self.assertGreater(height / image.shape[0], minimum)
        self.assertTrue(np.all(image[:8] == 255))
        self.assertTrue(np.all(image[-8:] == 255))
        self.assertTrue(np.all(image[:, :8] == 255))
        self.assertTrue(np.all(image[:, -8:] == 255))
        self.assertLessEqual(abs(x - (image.shape[1] - x - width)), 2)
        self.assertLessEqual(abs(y - (image.shape[0] - y - height)), 2)

    def test_padding_depends_on_garment_not_camera_frame(self):
        outputs = []
        for shape in [(500, 640), (1000, 1400)]:
            frame = np.full((*shape, 3), 50, np.uint8)
            mask = np.zeros(shape, np.uint8)
            mask[100:340, 100:280] = 1
            frame[mask > 0] = [160, 70, 150]
            before = frame.copy()
            encoded = _catalog_canvas(frame, mask, (100, 100, 180, 240))
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            self.assert_fills_frame(image)
            self.assertTrue(np.array_equal(frame, before))
            self.assertLessEqual(max(image.shape[:2]), 260)
            x, y, width, height = cv2.boundingRect(np.any(image < 245, axis=2).astype(np.uint8))
            self.assertAlmostEqual(width / height, 180 / 240, places=3)
            outputs.append(image)
        self.assertTrue(np.array_equal(*outputs))

    def test_real_purple_manual_outline_is_large_and_complete(self):
        root = Path(__file__).resolve().parents[1]
        fixtures = root / 'diagnostics/catalog-presentation-20261020-172735'
        raw = (fixtures / 'purple-2076-original.jpg').read_bytes()
        options = json.loads((fixtures / 'purple-2076-manual-options.json').read_text())
        encoded = garment_manual_preview(raw, options['points'], options['turns'])
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        self.assert_fills_frame(image, .91)
        foreground = np.any(image < 245, axis=2)
        # Both sleeve extremes, collar and hem survive the original outline.
        self.assertTrue(np.any(foreground[:image.shape[0] // 4, :image.shape[1] // 4]))
        self.assertTrue(np.any(foreground[:image.shape[0] // 3, -image.shape[1] // 4:]))
        self.assertTrue(np.any(foreground[:image.shape[0] // 8, image.shape[1] // 3:2 * image.shape[1] // 3]))
        self.assertTrue(np.any(foreground[-image.shape[0] // 8:]))

    def test_history_uses_only_valid_catalog_derivatives_and_one_bulk_read(self):
        from tests.test_support import configure_test_environment
        configure_test_environment()
        m = importlib.import_module('app')
        with patch.object(m, 'fetch_all', return_value=[
            {'id': 2074, 'reference_image_path': 'results/model_references/2074/photo.jpg'},
            {'id': 2076, 'reference_image_path': 'results/model_references/2076/photo.jpg'},
        ]) as query, patch.object(m, 'garment_cover_display_status', side_effect=[
            ('/static/results/model_previews/2074/photo.png', True),
            ('/static/results/model_references/2076/photo.jpg', False),
        ]):
            urls = m.get_garment_presentation_urls([2074, 2074, 2076, None])
        self.assertEqual(urls, {2074: '/static/results/model_previews/2074/photo.png'})
        query.assert_called_once()
        self.assertEqual(query.call_args.args[1], (2074, 2076))
        self.assertTrue(query.call_args.args[0].lstrip().startswith('SELECT'))


if __name__ == '__main__':
    unittest.main()
