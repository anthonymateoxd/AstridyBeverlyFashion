"""Station occupancy must be actionable from the guided capture page."""
from __future__ import annotations
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

class OccupiedCapturePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=importlib.import_module("app")

    def setUp(self):
        self.client=self.a.app.test_client()
        self.model={"id":2069,"code":"BLUSA-1216","name":"Blusa modelo blanco",
                    "status":"APROBADO","active":1,"created_by":3}

    def get_page(self,role,station):
        with self.client.session_transaction() as sess:
            sess.update(user_id=3,role=role,username="testowner")
        user={"id":3,"username":"testowner","role":role,"active":1}
        cursor=Mock()
        cursor.fetchone.return_value={"code":"BLUSA-1214","active":0}
        conn=Mock()
        conn.cursor.return_value=cursor
        state={"ok":True,"has_preparation_version":True,
               "active":False,"station_busy":bool(station)}
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"_guarded_model_or_none",return_value=(self.model,None)),
            patch.object(self.a,"_ai_capture_status_payload",return_value=state),
            patch.object(self.a,"db",return_value=conn),
            patch.object(self.a,"get_open_capture_session",return_value=station),
        ):
            response=self.client.get("/modelos-prenda/2069/captura-ia")
        return response,conn,cursor

    def test_admin_sees_one_click_action_for_archived_model(self):
        resp,conn,cur=self.get_page(
            "ADMIN",{"id":3277,"garment_model_id":2067}
        )
        self.assertEqual(resp.status_code,200)
        page=resp.get_data(as_text=True)
        self.assertIn("BLUSA-1214",page)
        self.assertIn("modelo retirado del cat\u00e1logo",page)
        self.assertIn("Pausar sesi\u00f3n anterior y continuar",page)
        self.assertIn('name="capture_session_id" value="3277"',page)
        self.assertIn('name="return_to_capture" value="1"',page)
        self.assertIn('"stationBusy": true',page)
        self.assertNotIn("FINALIZAR PREPARACI\u00d3N",page)
        self.assertNotIn("CANCELAR SESI\u00d3N",page)
        conn.close.assert_called_once()
        cur.close.assert_called_once()

    def test_model_manager_sees_explanation_not_admin_action(self):
        resp,_,_=self.get_page(
            "MODEL_MANAGER",{"id":3277,"garment_model_id":2067}
        )
        self.assertEqual(resp.status_code,200)
        page=resp.get_data(as_text=True)
        self.assertIn("Solicite al administrador",page)
        self.assertNotIn('name="capture_session_id" value="3277"',page)

    def test_when_station_free_guided_session_controls_remain(self):
        resp,_,_=self.get_page("ADMIN",None)
        self.assertEqual(resp.status_code,200)
        page=resp.get_data(as_text=True)
        self.assertNotIn("Pausar sesi\u00f3n anterior y continuar",page)
        self.assertIn("FINALIZAR PREPARACI\u00d3N",page)
        self.assertIn('"stationBusy": false',page)

    def test_style_uses_sidebar_palette(self):
        template=(ROOT/"templates"/"batches.html").read_text(encoding="utf-8")
        self.assertIn("var(--ui-navy, #0d2a66)",template)
        self.assertIn("var(--ui-text, #10275a)",template)
        self.assertIn("color: #667794",template)

    def test_admin_redirects_back_to_guided_after_pause(self):
        with self.client.session_transaction() as sess:
            sess.update(user_id=3,role="ADMIN",username="testowner")
        user={"id":3,"username":"testowner","role":"ADMIN","active":1}
        conn=Mock()
        cur=Mock()
        conn.cursor.return_value=cur
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"db",return_value=conn),
            patch.object(self.a,"get_open_capture_session",
                         return_value={"id":3277,"garment_model_id":2067}),
            patch.object(self.a,"_ai_capture_finish",
                         return_value=self.a.app.response_class(
                             '{"ok": true}',mimetype="application/json")) as finish,
            patch.object(self.a,"_stop_ai_guided_worker"),
            patch.object(self.a,"_ai_guided_reset_state"),
        ):
            response=self.client.post("/modelos-prenda/captura-ia/pausar-ocupada",
                                      data={"capture_session_id":"3277",
                                            "return_model_id":"2069",
                                            "return_to_capture":"1"})
        self.assertEqual(response.status_code,302)
        self.assertTrue(response.location.endswith("/modelos-prenda/2069/captura-ia"))
        finish.assert_called_once_with("pause",session_id_override=3277)

if __name__=="__main__":
    unittest.main()
