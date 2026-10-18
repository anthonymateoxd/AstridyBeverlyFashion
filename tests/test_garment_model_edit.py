from __future__ import annotations

import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from werkzeug.datastructures import FileStorage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


class GarmentModelEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = importlib.import_module("app")

    def setUp(self):
        self.client = self.a.app.test_client()
        self.user = {"id": 11, "username": "test-admin", "role": "ADMIN", "active": 1}
        self.model = {
            "id": 12345, "code": "BLUSA-TEST",
            "name": "Blusa rosa", "color": "Rosa",
            "description": "Original", "status": "BORRADOR",
            "active": 1, "created_by": 11,
        }
        with self.client.session_transaction() as s:
            s.update(user_id=11, role="ADMIN", username="test-admin")
        self.temp = tempfile.TemporaryDirectory()
        self.old_static = self.a.app.static_folder
        self.a.app.static_folder = self.temp.name

    def tearDown(self):
        self.a.app.static_folder = self.old_static
        self.temp.cleanup()

    def _post(self, payload, role=None):
        if role:
            self.user["role"] = role
            with self.client.session_transaction() as s:
                s["role"] = role
        with (
            patch.object(self.a, "fetch_one", return_value=self.user),
            patch.object(self.a, "get_garment_model", return_value=self.model),
            patch.object(self.a, "execute") as execute,
        ):
            response = self.client.post(
                f"/modelos-prenda/{self.model['id']}/editar",
                data=payload, content_type="multipart/form-data",
            )
        return response, execute

    @staticmethod
    def photo():
        from cv2 import imencode
        from numpy import full
        image = full((240, 180, 3), 175, dtype="uint8")
        ok, jpg = imencode(".jpg", image)
        assert ok
        return io.BytesIO(jpg.tobytes())

    def test_edit_draft_name_color_and_photo(self):
        response, execute = self._post({
            "name": "Blusa nueva", "color": "Azul",
            "description": "Portada actualizada",
            "cover_photo": (self.photo(), "blusa.jpg"),
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn("/modelos-prenda", response.location)
        execute.assert_called_once()
        self.assertIn("UPDATE garment_models", execute.call_args.args[0])
        image = Path(self.temp.name)/"results"/"model_covers"/"12345.jpg"
        self.assertTrue(image.is_file())
        self.assertTrue(image.read_bytes().startswith(b"\xff\xd8"))
        # La portada se verifica sin consultar una base de datos real.
        with (
            self.a.app.test_request_context(),
            patch.object(self.a, "fetch_one", return_value=None),
        ):
            self.assertIn("model_covers/12345.jpg", self.a.garment_cover_url(12345))

    def test_approved_model_accepts_cover_but_cannot_change_metadata(self):
        self.model["status"] = "APROBADO"
        response, execute = self._post({
            "name": "Nombre alterado", "color": "Negro",
            "cover_photo": (self.photo(), "blusa.jpg"),
        })
        self.assertEqual(response.status_code, 302)
        execute.assert_not_called()
        self.assertTrue((Path(self.temp.name)/"results"/"model_covers"/"12345.jpg").exists())

    def test_invalid_photo_keeps_existing_photo_and_metadata(self):
        old = Path(self.temp.name)/"results"/"model_covers"/"12345.jpg"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"previous_photo")
        response, execute = self._post({
            "name": "Alteracion", "color": "Rojo",
            "cover_photo": (io.BytesIO(b"not-a-photo"), "virus.png"),
        })
        self.assertEqual(response.status_code, 302)
        execute.assert_not_called()
        self.assertEqual(old.read_bytes(), b"previous_photo")

    def test_model_manager_cannot_edit_other_persons_model(self):
        self.model["created_by"] = 99
        response, execute = self._post({
            "name": "Alteracion", "color": "Rojo",
            "cover_photo": (self.photo(), "blusa.jpg"),
        }, role="MODEL_MANAGER")
        self.assertEqual(response.status_code, 403)
        execute.assert_not_called()

    def test_quality_manager_cannot_edit_model(self):
        response, execute = self._post({
            "name": "Alteracion", "color": "Rojo",
        }, role="QUALITY_MANAGER")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.location)
        execute.assert_not_called()

    def test_photo_is_rejected_above_five_megabytes(self):
        upload = FileStorage(
            stream=io.BytesIO(b"0" * (5*1024*1024+1)),
            filename="oversize.jpg",
        )
        with self.assertRaises(ValueError):
            self.a.prepare_garment_cover_photo(upload)

    def test_get_edit_form_shows_existing_model(self):
        with (
            patch.object(self.a, "fetch_one", return_value=self.user),
            patch.object(self.a, "get_garment_model", return_value=self.model),
        ):
            response = self.client.get("/modelos-prenda/12345/editar")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Editar modelo de prenda", response.data)
        self.assertIn(b"Blusa rosa", response.data)


if __name__ == "__main__":
    unittest.main()
