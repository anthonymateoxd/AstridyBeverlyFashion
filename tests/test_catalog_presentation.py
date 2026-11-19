"""Reference presentation: real camera bytes remain unchanged and display is isolated."""
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from garment_background_preview import garment_visual_preview, garment_manual_preview

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "diagnostics/deployment-preparation/current-reference-original.jpg"
LIGHTBOX = ROOT / "diagnostics/authorized_blusa_1218_original.jpg"


class PresentationRegressionTests(unittest.TestCase):
    @unittest.skipUnless(CURRENT.exists(), "local reference fixture unavailable")
    def test_current_reference_generates_white_presentation_without_fallback(self):
        data = CURRENT.read_bytes()
        before = hashlib.sha256(data).hexdigest()
        png = garment_visual_preview(data)
        self.assertIsNotNone(png)
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[2], 3)
        self.assertLessEqual(max(image.shape[:2]), max(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2]) * 1.08 + 2)
        self.assertTrue(np.all(image[:16] == 255))
        self.assertTrue(np.all(image[-16:] == 255))
        self.assertTrue(np.all(image[:, :16] == 255))
        self.assertTrue(np.all(image[:, -16:] == 255))
        self.assertGreater(np.count_nonzero(np.all(image < 90, axis=2)), 300)
        foreground = np.any(image < 245, axis=2).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        self.assertEqual(sum(stats[i, cv2.CC_STAT_AREA] > 20 for i in range(1, count)), 1)
        self.assertEqual(before, hashlib.sha256(CURRENT.read_bytes()).hexdigest())

    def test_real_purple_catalog_photo_is_white_and_original_unchanged(self):
        root = ROOT / "diagnostics/catalog-presentation-20261020-172735"
        source = root / "purple-2074-original.jpg"
        if not source.exists():
            self.skipTest("authorized purple fixture unavailable")
        raw = source.read_bytes()
        result = garment_visual_preview(raw)
        self.assertIsNotNone(result)
        image = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[2], 3)
        self.assertLessEqual(max(image.shape[:2]), max(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[:2]) * 1.08 + 2)
        self.assertTrue(np.all(image[:16] == 255))
        self.assertTrue(np.all(image[:, :16] == 255))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), "180059c1242f88983be3f9956f870016472875ed2a9bdc74a38bd2f178527426")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gold = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > 50)
        self.assertGreater(np.count_nonzero(gold), 500, "Original golden badge and label must survive segmentation")
        self.assertGreater(np.count_nonzero((image[:, :, 0] > image[:, :, 1] + 10) & (image[:, :, 2] > image[:, :, 1] + 5)), 20000)

    def test_real_lightbox_purple_manual_outline_survives_preview_and_save(self):
        root = ROOT / "diagnostics/catalog-presentation-20261020-172735"
        source = root / "purple-2076-original.jpg"
        if not source.exists():
            self.skipTest("authorized purple fixture unavailable")
        raw = source.read_bytes()
        options = json.loads((root / "purple-2076-manual-options.json").read_text())
        self.assertTrue(garment_visual_preview(raw) is not None, 'Real camera reference must be segmented automatically')
        png = garment_manual_preview(raw, options['points'], options['turns'])
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[2], 3)
        self.assertLessEqual(max(image.shape[:2]), max(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[:2]) * 1.08 + 2)
        self.assertTrue(np.all(image[:16] == 255))
        self.assertTrue(np.all(image[-16:] == 255))
        self.assertTrue(np.all(image[:, :16] == 255))
        self.assertTrue(np.all(image[:, -16:] == 255))
        from tests.test_support import configure_test_environment
        configure_test_environment()
        m = importlib.import_module('app')
        client = m.app.test_client()
        with client.session_transaction() as session:
            session.update(user_id=41, role='ADMIN', username='qa')
        user = dict(id=41, username='qa', role='ADMIN', active=1)
        settings = json.dumps({hashlib.sha256(raw).hexdigest():options})
        with patch.object(m, 'fetch_one', return_value=user):
            result = client.post('/modelos-prenda/camara-referencia/recorte', data={'photo':(io.BytesIO(raw),'purple.jpg'),'reference_presentation':settings})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.data, png)
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.app, '_static_folder', tmp):
            with m.app.test_request_context(method='POST', data={'reference_presentation':settings}):
                m.session['user_id'] = 41
                prepared = m.prepare_garment_reference_images([__import__('werkzeug').datastructures.FileStorage(io.BytesIO(raw),filename='purple.jpg')])
                self.assertEqual(prepared[0][1], raw)
                cursor = Mock()
                if 'cur' in inspect.signature(m.save_garment_reference_images).parameters:
                    m.save_garment_reference_images(2076, prepared, cur=cursor)
                else:
                    with patch.object(m, 'execute', side_effect=cursor.execute):
                        m.save_garment_reference_images(2076, prepared)
                originals = list((Path(tmp)/'results/model_references/2076').glob('*'))
                previews = list((Path(tmp)/'results/model_previews/2076').glob('*'))
                self.assertEqual(originals[0].read_bytes(), raw)
                self.assertEqual(previews[0].read_bytes(), png)
                self.assertNotIn('ai_',cursor.execute.call_args.args[0])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), 'b9857acddd26fb0b8cb6b30c14356415e705031152ace46e41d8f3a167eeb529')

    def test_invalid_manual_outline_is_rejected(self):
        raw = CURRENT.read_bytes()
        for points in [[], [[0,0],[1,0],[.5,1]], [[.5,.5]]*3, [[float('nan'),.2],[.3,.3],[.4,.4]]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                garment_manual_preview(raw, points)

    @unittest.skipUnless(LIGHTBOX.exists(), "local lightbox fixture unavailable")
    def test_new_camera_photo_persists_original_and_catalog_derivative(self):
        from tests.test_support import configure_test_environment
        configure_test_environment()
        m = importlib.import_module("app")
        client = m.app.test_client()
        user = {"id": 41, "username": "qa", "role": "ADMIN", "active": 1}
        with client.session_transaction() as session:
            session.update(user_id=41, role="ADMIN", username="qa")
        frame = cv2.imread(str(LIGHTBOX))
        original_frame = frame.copy()
        with patch.object(m, "fetch_one", return_value=user), \
             patch.object(m, "ensure_camera_capture_worker"), \
             patch.object(m, "get_latest_camera_frame", return_value=(frame, time.perf_counter(), 1)):
            captured = client.post("/modelos-prenda/camara-referencia/foto")
        self.assertEqual(captured.status_code, 200)
        self.assertTrue(np.array_equal(frame, original_frame))
        with patch.object(m, "fetch_one", return_value=user):
            preview = client.post("/modelos-prenda/camara-referencia/recorte", data={
                "photo": (io.BytesIO(captured.data), "camara.jpg"),
            }, content_type="multipart/form-data")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.mimetype, "image/png")
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.app, "_static_folder", tmp):
            with m.app.test_request_context():
                m.session["user_id"] = 41
                cursor = Mock()
                if 'cur' in inspect.signature(m.save_garment_reference_images).parameters:
                    paths = m.save_garment_reference_images(2071, [(".jpg", captured.data)], cur=cursor)
                else:
                    with patch.object(m, 'execute', side_effect=cursor.execute):
                        paths = m.save_garment_reference_images(2071, [(".jpg", captured.data)])
                paths = paths or list(Path(tmp).rglob("*"))
                source = next(p for p in paths if p.is_file() and "model_references" in str(p))
                derivative = next(p for p in paths if p.is_file() and "model_previews" in str(p))
                self.assertEqual(source.read_bytes(), captured.data)
                self.assertNotEqual(derivative.read_bytes(), captured.data)
                relative = source.relative_to(Path(tmp)).as_posix()
                catalog, has_cutout = m.garment_cover_display_status(2071, relative)
                self.assertTrue(has_cutout)
                self.assertIn("model_previews/2071/", catalog)
                self.assertEqual(m.garment_reference_visual_url(relative), catalog)
                self.assertEqual(m.garment_cover_original_url(2071, relative).split("?", 1)[0], "/static/" + relative)
                cursor.execute.assert_called_once()
                self.assertNotIn("ai_", cursor.execute.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
