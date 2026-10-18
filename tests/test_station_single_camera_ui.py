"""Regression tests for single-camera workflow and station overlay."""
import importlib
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

class CameraLotUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.appmodule=importlib.import_module("app")

    def setUp(self):
        self.client=self.appmodule.app.test_client()
        self.user={"id":17,"username":"admin-test","role":"ADMIN","active":1}
        with self.client.session_transaction() as s:
            s.update(user_id=17,username="admin-test",role="ADMIN")

    def test_lots_displays_active_ai_capture_with_owner(self):
        m=self.appmodule
        captured={"id":3280,"garment_model_id":2069,
                  "model_code":"BLUSA-1216","model_name":"Blusa modelo blanco",
                  "model_active":1}
        plan={"date":date(2026,10,16)}
        with (
            patch.object(m,"fetch_one",side_effect=[self.user,captured]),
            patch.object(m,"get_today_production_plan",return_value=plan),
            patch.object(m,"fetch_all",return_value=[]),
            patch.object(m,"get_inspection_lines",return_value=[]),
            patch.object(m,"get_batches",return_value=[]),
            patch.object(m,"get_active_batch",return_value=None),
            patch.object(m,"render_template",return_value="OK") as render,
        ):
            response=self.client.get("/lotes")
        self.assertEqual(response.status_code,200)
        self.assertEqual(render.call_args.args[0],"batches.html")
        self.assertEqual(render.call_args.kwargs["occupied_capture"],captured)

    def test_lots_page_has_explicit_pause_form(self):
        content=(ROOT/"templates"/"batches.html").read_text(encoding="utf-8")
        self.assertIn("url_for('pause_busy_capture')",content)
        self.assertIn('name="return_to_lots" value="1"',content)
        self.assertIn('name="capture_session_id" value="{{ occupied_capture.id }}"',content)
        self.assertIn("Pausar captura de IA y liberar camara",content)

    def test_admin_return_after_pause_goes_back_to_lots(self):
        m=self.appmodule
        cursor=Mock()
        connection=Mock()
        connection.cursor.return_value=cursor
        with (
            patch.object(m,"fetch_one",return_value=self.user),
            patch.object(m,"db",return_value=connection),
            patch.object(m,"get_open_capture_session",return_value={"id":3280,"garment_model_id":2069}),
            patch.object(m,"_ai_capture_finish",return_value=m.app.response_class('{"ok":true}',mimetype="application/json")) as pause,
            patch.object(m,"_stop_ai_guided_worker"),
            patch.object(m,"_ai_guided_reset_state"),
        ):
            response=self.client.post("/modelos-prenda/captura-ia/pausar-ocupada",
                data={"capture_session_id":"3280","return_to_lots":"1"})
        self.assertEqual(response.status_code,302)
        self.assertEqual(response.location,"/lotes")
        pause.assert_called_once_with("pause",session_id_override=3280)

    def test_station_removes_center_line_but_roi_overlay_remains(self):
        station=(ROOT/"templates"/"station.html").read_text(encoding="utf-8")
        self.assertNotIn("station-scan-line",station)
        appcode=(ROOT/"app.py").read_text(encoding="utf-8")
        start=appcode.index("def draw_inspection_overlay(")
        end=appcode.index("def generate_video_feed(",start)
        self.assertIn("cv2.rectangle(",appcode[start:end])
        self.assertIn("get_roi_bounds(",appcode[start:end])

if __name__=="__main__": unittest.main()
