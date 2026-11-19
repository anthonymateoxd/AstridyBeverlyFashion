"""El administrador puede pausar una sesion ajena aun si su modelo fue archivado."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from flask import jsonify

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

class BusyCaptureAdminTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=importlib.import_module("app")

    def prepare(self,role="ADMIN"):
        client=self.a.app.test_client()
        with client.session_transaction() as sess:
            sess.update(user_id=7,username="owner-test",role=role)
        cursor=Mock()
        conn=Mock()
        conn.cursor.return_value=cursor
        return client,cursor,conn,{"id":7,"username":"owner-test","role":role,"active":1}

    def test_archived_model_session_can_be_paused_explicitly(self):
        client,cur,conn,user=self.prepare()
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"db",return_value=conn),
            patch.object(self.a,"get_open_capture_session",return_value={"id":3277,"garment_model_id":2067}),
            patch.object(self.a,"_ai_capture_finish",return_value=self.a.app.response_class('{"ok":true}', mimetype="application/json")) as finish,
            patch.object(self.a,"_stop_ai_guided_worker"),
            patch.object(self.a,"_ai_guided_reset_state"),
        ):
            response=client.post(
                "/modelos-prenda/captura-ia/pausar-ocupada",
                data={"capture_session_id":"3277","return_model_id":"2069"},
            )
        self.assertEqual(response.status_code,302)
        self.assertIn("/modelos-prenda/2069",response.location)
        finish.assert_called_once_with("pause",session_id_override=3277)
        cur.close.assert_called_once()

    def test_cannot_pause_a_different_new_session(self):
        client,cur,conn,user=self.prepare()
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"db",return_value=conn),
            patch.object(self.a,"get_open_capture_session",return_value={"id":3300,"garment_model_id":2068}),
            patch.object(self.a,"_ai_capture_finish") as finish,
        ):
            response=client.post(
                "/modelos-prenda/captura-ia/pausar-ocupada",
                data={"capture_session_id":"3277"},
            )
        self.assertEqual(response.status_code,302)
        finish.assert_not_called()

    def test_non_admin_cannot_pause_another_session(self):
        client,cur,conn,user=self.prepare(role="MODEL_MANAGER")
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"db",return_value=conn),
            patch.object(self.a,"_ai_capture_finish") as finish,
        ):
            response=client.post(
                "/modelos-prenda/captura-ia/pausar-ocupada",
                data={"capture_session_id":"3277"},
            )
        self.assertEqual(response.status_code,302)
        finish.assert_not_called()
        conn.cursor.assert_not_called()

if __name__=="__main__":
    unittest.main()
