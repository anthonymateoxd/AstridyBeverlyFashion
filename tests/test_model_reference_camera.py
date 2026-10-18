"""Reference photo from existing camera, without adding inspections or AI captures."""
import io
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

class ModelReferenceCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module=importlib.import_module("app")

    def client(self,role="MODEL_MANAGER"):
        app=self.module.app
        c=app.test_client()
        with c.session_transaction() as s:
            s.update(user_id=15,username="reference-qa",role=role)
        return c,{"id":15,"username":"reference-qa","role":role,"active":1}

    def test_camera_snapshot_returns_raw_jpeg_with_no_inspection(self):
        m=self.module
        client,user=self.client("MODEL_MANAGER")
        rgb=np.zeros((300,400,3),dtype=np.uint8)
        rgb[:]=[35,120,210]
        with (patch.object(m,"fetch_one",return_value=user),
              patch.object(m,"ensure_camera_capture_worker",return_value=True) as worker,
              patch.object(m,"get_latest_camera_frame",return_value=(rgb,4000,2)),
              patch.object(m.time,"perf_counter",return_value=4001),
              patch.object(m,"draw_inspection_overlay") as overlay,
              patch.object(m,"get_active_batch") as batches):
            result=client.post("/modelos-prenda/camara-referencia/foto")
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.mimetype,"image/jpeg")
        self.assertTrue(result.data.startswith(b"\xff\xd8"))
        self.assertIn("no-store",result.headers["Cache-Control"])
        overlay.assert_not_called()
        batches.assert_not_called()
        worker.assert_called_once()

    def test_stale_frame_is_rejected_without_fake_photo(self):
        m=self.module
        client,user=self.client()
        with (patch.object(m,"fetch_one",return_value=user),
              patch.object(m,"ensure_camera_capture_worker",return_value=True),
              patch.object(m,"get_latest_camera_frame",return_value=(np.zeros((50,60,3),dtype=np.uint8),2000,2)),
              patch.object(m.time,"perf_counter",return_value=2005)):
            result=client.post("/modelos-prenda/camara-referencia/foto")
        self.assertEqual(result.status_code,503)
        self.assertIn("imagen reciente",result.json["error"])

    def test_no_frame_is_rejected(self):
        m=self.module
        client,user=self.client()
        with (patch.object(m,"fetch_one",return_value=user),
              patch.object(m,"ensure_camera_capture_worker",return_value=False),
              patch.object(m,"get_latest_camera_frame",return_value=(None,None,0))):
            result=client.post("/modelos-prenda/camara-referencia/foto")
        self.assertEqual(result.status_code,503)
        self.assertFalse(result.json["ok"])

    def test_quality_manager_cannot_capture_model_reference(self):
        m=self.module
        client,user=self.client("QUALITY_MANAGER")
        with (patch.object(m,"fetch_one",return_value=user),
              patch.object(m,"get_latest_camera_frame") as frame):
            response=client.post("/modelos-prenda/camara-referencia/foto")
        self.assertNotEqual(response.status_code,200)
        frame.assert_not_called()

    def test_unauthenticated_redirected(self):
        client=self.module.app.test_client()
        resp=client.post("/modelos-prenda/camara-referencia/foto")
        self.assertNotEqual(resp.status_code,200)

    def test_camera_live_preview_access_for_model_manager(self):
        m=self.module
        client,user=self.client()
        def stream():
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nx\r\n"
        with (patch.object(m,"fetch_one",return_value=user),
              patch.object(m,"generate_video_feed",side_effect=stream) as feed):
            response=client.get("/modelos-prenda/camara-referencia/video")
        self.assertEqual(response.status_code,200)
        self.assertIn("multipart/x-mixed-replace",response.mimetype)
        self.assertIn("no-store",response.headers["Cache-Control"])

    def test_form_supports_upload_capture_and_removal(self):
        doc=(ROOT/"templates/garment_model_form.html").read_text(encoding="utf-8")
        for phrase in ["Subir im&aacute;genes","Tomar fotograf&iacute;a","Capturar fotograf&iacute;a",
                       'name="reference_images"',"new DataTransfer()","photos.splice(index,1)",
                       "fetch(capturedEndpoint"]:
            self.assertIn(phrase,doc)

    def test_backend_limits_photos_to_eight(self):
        m=self.module
        client,user=self.client("ADMIN")
        payload={"name":"Prueba limite", "color":"Blanco",
                 "reference_images":[(io.BytesIO(b"fake photo"),f"image-{n}.jpg") for n in range(9)]}
        with (patch.object(m,"fetch_one",return_value=user),
              patch.object(m,"generate_garment_model_code",return_value="BLUSA-999999"),
              patch.object(m,"db") as db):
            response=client.post("/modelos-prenda/nuevo",data=payload,content_type="multipart/form-data")
        self.assertEqual(response.status_code,200)
        self.assertIn("8 imagenes",response.get_data(as_text=True))
        db.assert_not_called()

if __name__=="__main__":
    unittest.main()
