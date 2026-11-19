from __future__ import annotations
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


class Cursor:
    def __init__(self, session_status="PAUSADA", photo_status="ACEPTADA",
                 reject_reason=None, in_dataset=False, foreign_image=False):
        self.session_status=session_status
        self.photo_status=photo_status
        self.reject_reason=reject_reason
        self.in_dataset=in_dataset
        self.foreign_image=foreign_image
        self.return_type=None
        self.sql=[]
        self.params=[]
        self.rowcount=1
        self.lastrowid=501
        self.rows=[]
        self.closed=False

    def execute(self,sql,params=()):
        self.sql.append(sql)
        self.params.append(params)
        if "FROM ai_capture_sessions" in sql:
            self.return_type="session"
        elif "FROM ai_training_images" in sql and "FOR UPDATE" in sql:
            self.return_type="photo"
        elif "FROM ai_training_images" in sql:
            self.return_type="list"
        elif "FROM ai_dataset_images" in sql:
            self.return_type="dataset"
        else:
            self.return_type="other"
        if sql.strip().startswith("UPDATE ai_training_images"):
            self.photo_status=params[0]
            self.reject_reason=params[1]

    def fetchone(self):
        if self.return_type=="session":
            return {"id":3279,"status":self.session_status}
        if self.return_type=="photo":
            return None if self.foreign_image else {
                "id":29942,"status":self.photo_status,
                "reject_reason":self.reject_reason
            }
        if self.return_type=="dataset":
            return {"id":10} if self.in_dataset else None
        return None

    def fetchall(self):
        if self.return_type=="list":
            return [{
                "id":29942, "status":self.photo_status,
                "reject_reason":self.reject_reason,
                "captured_at":None,"in_dataset":self.in_dataset
            }]
        return []

    def close(self):self.closed=True


class Conn:
    def __init__(self,cur):
        self.cur=cur
        self.in_transaction=False
        self.commits=0
        self.rollbacks=0
        self.closed=False
    def cursor(self,dictionary=False):return self.cur
    def start_transaction(self):self.in_transaction=True
    def commit(self):self.commits+=1;self.in_transaction=False
    def rollback(self):self.rollbacks+=1;self.in_transaction=False
    def close(self):self.closed=True


class TrainingGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=importlib.import_module("app")
        cls.model={"id":2069,"code":"BLUSA-1216","name":"Blusa modelo blanco",
                   "active":1,"status":"APROBADO","created_by":3}
    def client_as(self,role="ADMIN"):
        client=self.a.app.test_client()
        with client.session_transaction() as s:
            s.update(user_id=3,role=role,username="gallery-test")
        return client,{"id":3,"username":"gallery-test","role":role,"active":1}
    def submit(self,operation="exclude",status="ACEPTADA",reason=None,
               session_status="PAUSADA",in_dataset=False,
               foreign_image=False,role="ADMIN",confirm=29942):
        client,user=self.client_as(role=role)
        cursor=Cursor(session_status,status,reason,in_dataset,foreign_image)
        conn=Conn(cursor)
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"get_garment_model",return_value=self.model),
            patch.object(self.a,"can_manage_garment_model",return_value=(role=="ADMIN")),
            patch.object(self.a,"db",return_value=conn),
            patch.object(self.a,"record_ai_event") as audit,
        ):
            response=client.post(
                "/modelos-prenda/2069/captura-ia/imagenes/29942/revisar",
                data={"action":operation,"confirm_image_id":str(confirm)},
            )
        return response,conn,cursor,audit

    def test_gallery_shows_image_thumbnail_and_sequence(self):
        client,user=self.client_as()
        cursor=Cursor()
        conn=Conn(cursor)
        with (
            patch.object(self.a,"fetch_one",return_value=user),
            patch.object(self.a,"get_garment_model",return_value=self.model),
            patch.object(self.a,"can_manage_garment_model",return_value=True),
            patch.object(self.a,"db",return_value=conn)
        ):
            resp=client.get("/modelos-prenda/2069/captura-ia/imagenes")
        self.assertEqual(resp.status_code,200)
        html=resp.get_data(as_text=True)
        self.assertIn("Captura 1",html)
        self.assertIn("/modelos-prenda/2069/captura-ia/imagen/29942",html)
        self.assertIn("Excluir del entrenamiento",html)
        self.assertTrue(conn.closed)

    def test_excluding_one_accepted_photo_preserves_file_and_others(self):
        res,conn,cur,audit=self.submit()
        self.assertEqual(res.status_code,302)
        self.assertEqual(conn.commits,1)
        sql="\n".join(cur.sql)
        self.assertIn("UPDATE ai_training_images",sql)
        self.assertNotIn("DELETE FROM",sql)
        self.assertEqual(cur.photo_status,"RECHAZADA")
        self.assertEqual(cur.reject_reason,"MANUAL_TRAINING_EXCLUDED")
        audit.assert_called_once()

    def test_restoring_excluded_photo(self):
        res,conn,cur,audit=self.submit(
            operation="restore",status="RECHAZADA",reason="MANUAL_TRAINING_EXCLUDED",
        )
        self.assertEqual(res.status_code,302)
        self.assertEqual(conn.commits,1)
        self.assertEqual(cur.photo_status,"ACEPTADA")
        self.assertIsNone(cur.reject_reason)
        audit.assert_called_once()

    def test_cannot_modify_completed_training(self):
        res,conn,cur,audit=self.submit(session_status="COMPLETADA")
        self.assertEqual(res.status_code,302)
        self.assertEqual(conn.commits,0)
        self.assertEqual(conn.rollbacks,1)
        audit.assert_not_called()

    def test_cannot_modify_image_already_in_dataset(self):
        res,conn,cur,audit=self.submit(in_dataset=True)
        self.assertEqual(conn.commits,0)
        self.assertEqual(conn.rollbacks,1)
        audit.assert_not_called()

    def test_cannot_edit_other_capture_image(self):
        res,conn,cur,audit=self.submit(foreign_image=True)
        self.assertEqual(conn.commits,0)
        self.assertEqual(conn.rollbacks,1)
        audit.assert_not_called()

    def test_model_manager_cannot_edit_other_model(self):
        res,conn,cur,audit=self.submit(role="MODEL_MANAGER")
        self.assertEqual(res.status_code,403)
        self.assertEqual(conn.commits,0)
        audit.assert_not_called()

    def test_prevent_tampered_confirmation(self):
        res,conn,cur,audit=self.submit(confirm=9999)
        self.assertEqual(res.status_code,302)
        self.assertEqual(conn.commits,0)
        audit.assert_not_called()

    def test_cannot_restore_other_rejected_reason(self):
        res,conn,cur,audit=self.submit(
            operation="restore",status="RECHAZADA",reason="BLUR"
        )
        self.assertEqual(conn.commits,0)
        self.assertEqual(conn.rollbacks,1)
        audit.assert_not_called()

if __name__=="__main__":
    unittest.main()
