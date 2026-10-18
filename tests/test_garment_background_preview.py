"""Background removal is display-only. Original reference files remain unchanged."""
from pathlib import Path
from unittest.mock import patch
from io import BytesIO
import importlib
import tempfile
import unittest

import cv2
import numpy as np

from garment_background_preview import garment_visual_preview


def sample(color=(105, 115, 135)):
    frame = np.full((520, 820, 3), 234, dtype=np.uint8)
    frame[:, :38] = 31
    frame[:, -38:] = 31
    outline = np.array([
        [290, 105], [340, 144], [410, 158], [480, 142],
        [532, 101], [592, 145], [622, 205], [579, 231],
        [562, 206], [560, 410], [320, 410], [317, 212],
        [285, 240], [253, 201],
    ], np.int32)
    cv2.fillPoly(frame, [outline], color)
    ok, enc = cv2.imencode(".jpg", frame)
    assert ok
    return enc.tobytes()


class GarmentPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = importlib.import_module("app")

    def test_colorful_garment_crops_with_alpha(self):
        raw = sample()
        png = garment_visual_preview(raw)
        self.assertIsNotNone(png)
        cut = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(cut.shape[2], 4)
        self.assertLess(cut.shape[1], 820)
        self.assertTrue(np.any(cut[:, :, 3] == 0))
        self.assertTrue(np.any(cut[:, :, 3] == 255))
        self.assertTrue(raw.startswith(b"\xff\xd8"))

    def test_near_white_background_without_object_falls_back(self):
        blank = np.full((480, 720, 3), 240, np.uint8)
        _, jpg = cv2.imencode(".jpg", blank)
        self.assertIsNone(garment_visual_preview(jpg.tobytes()))

    def test_dark_scene_falls_back(self):
        dark = np.full((480, 720, 3), 30, np.uint8)
        cv2.rectangle(dark, (230, 120), (520, 400), (60, 70, 85), -1)
        _, jpg = cv2.imencode(".jpg", dark)
        self.assertIsNone(garment_visual_preview(jpg.tobytes()))

    def test_invalid_bytes_do_not_produce_fake_cutouts(self):
        self.assertIsNone(garment_visual_preview(b"invalid-image"))
        self.assertIsNone(garment_visual_preview(b""))

    def test_preserve_original_and_write_derivative_with_no_db_change(self):
        m=self.app_module
        original=m.app.static_folder
        with tempfile.TemporaryDirectory() as temporary:
            m.app.static_folder=temporary
            try:
                with m.app.test_request_context():
                    with patch.dict(m.session, {"user_id": 15}):
                        from unittest.mock import Mock
                        cursor=Mock()
                        raw=sample()
                        import inspect
                        if "cur" in inspect.signature(
                            m.save_garment_reference_images
                        ).parameters:
                            originals=m.save_garment_reference_images(
                                417, [(".jpg", raw)], cur=cursor
                            )
                            cursor.execute.assert_called_once()
                            sql, values=cursor.execute.call_args.args
                            self.assertEqual(len(originals),2)
                        else:
                            with patch.object(m,"execute") as execute:
                                m.save_garment_reference_images(417,[(".jpg",raw)])
                                execute.assert_called_once()
                                sql, values=execute.call_args.args
                        self.assertIn("INSERT INTO garment_model_images",sql)
                        stored=Path(temporary)/values[1]
                        self.assertEqual(stored.read_bytes(),raw)
                        visual=m.garment_reference_preview_path(values[1])
                        self.assertTrue(visual.is_file())
                        self.assertTrue(visual.read_bytes().startswith(b"\x89PNG"))
                        url=m.garment_cover_url(417,values[1])
                        self.assertIn("model_previews/417",url)
            finally:
                m.app.static_folder=original

    def test_preview_endpoint_requires_authorized_session(self):
        m=self.app_module
        client=m.app.test_client()
        response=client.post("/modelos-prenda/camara-referencia/recorte")
        self.assertNotEqual(response.status_code,200)

    def test_preview_endpoint_returns_png_for_authorized_user(self):
        m=self.app_module
        client=m.app.test_client()
        with client.session_transaction() as session:
            session.update(user_id=35,username="qa",role="ADMIN")
        with patch.object(m,"fetch_one",return_value={
            "id":35,"username":"qa","role":"ADMIN","full_name":"Prueba","active":1
        }):
            response=client.post(
                "/modelos-prenda/camara-referencia/recorte",
                data={"photo":(BytesIO(sample()),"foto.jpg")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.mimetype,"image/png")
        self.assertTrue(response.data.startswith(b"\x89PNG"))

    def test_existing_cover_wins_if_newer_than_reference_preview(self):
        m=self.app_module
        original=m.app.static_folder
        with tempfile.TemporaryDirectory() as temporary:
            m.app.static_folder=temporary
            try:
                with m.app.test_request_context():
                    raw=sample()
                    ref=Path(temporary)/"results/model_references/44/a.jpg"
                    ref.parent.mkdir(parents=True,exist_ok=True)
                    ref.write_bytes(raw)
                    visual=m.garment_reference_preview_path("results/model_references/44/a.jpg")
                    visual.parent.mkdir(parents=True,exist_ok=True)
                    visual.write_bytes(garment_visual_preview(raw))
                    cover=Path(temporary)/"results/model_covers/44.jpg"
                    cover.parent.mkdir(parents=True,exist_ok=True)
                    cover.write_bytes(raw)
                    import os
                    os.utime(ref,ns=(10**12,10**12))
                    os.utime(cover,ns=(2*10**12,2*10**12))
                    url=m.garment_cover_url(44,"results/model_references/44/a.jpg")
                    self.assertIn("model_covers/44.jpg",url)
            finally:
                m.app.static_folder=original

if __name__=="__main__":
    unittest.main()
