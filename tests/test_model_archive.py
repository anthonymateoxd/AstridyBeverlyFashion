from __future__ import annotations
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

class Cursor:
    def __init__(self, row):
        self.row=row;self.rowcount=0;self.updated=False;self.updates=[];self.closed=False
    def execute(self, sql, params=()):
        if sql.strip().startswith("UPDATE garment_models"):
            self.updated=True
            self.updates.append((sql,params))
            self.rowcount=1
    def fetchone(self):return self.row
    def close(self):self.closed=True

class Conn:
    def __init__(self, c):
        self.c=c;self.in_transaction=False;self.commits=0;self.rollbacks=0;self.closed=False
    def cursor(self,dictionary=False):return self.c
    def start_transaction(self):self.in_transaction=True
    def commit(self):self.in_transaction=False;self.commits+=1
    def rollback(self):self.in_transaction=False;self.rollbacks+=1
    def close(self):self.closed=True

class ModelArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.A=importlib.import_module("app")
    def call(self,model,confirm="BLUSA-1215",role="ADMIN"):
        client=self.A.app.test_client()
        with client.session_transaction() as s:
            s["user_id"]=23;s["username"]="admin-test";s["role"]=role
        c=Cursor(model)
        conn=Conn(c)
        admin={"id":23,"username":"admin-test","role":role,"active":1}
        with patch.object(self.A,"fetch_one",return_value=admin),patch.object(self.A,"db",return_value=conn):
            result=client.post("/modelos-prenda/555/eliminar",data={"confirm_code":confirm},follow_redirects=False)
        return result,c,conn
    def model(self,**kw):
        row={"id":555,"code":"BLUSA-1215","active":1,"has_inspecting_batch":0,"ai_training_now":0}
        row.update(kw)
        return row
    def test_approved_used_model_archived_without_deleting_related_data(self):
        res,cur,conn=self.call(self.model())
        self.assertEqual(res.status_code,302)
        self.assertTrue(cur.updated)
        self.assertIn("SET active = 0",cur.updates[0][0])
        self.assertEqual(cur.updates[0][1],(555,))
        self.assertEqual(conn.commits,1)
        self.assertEqual(conn.rollbacks,0)
    def test_rejects_wrong_confirmation(self):
        res,cur,conn=self.call(self.model(),confirm="wrong")
        self.assertEqual(res.status_code,302)
        self.assertFalse(cur.updated)
        self.assertEqual(conn.rollbacks,1)
    def test_rejects_lot_in_inspection(self):
        res,cur,conn=self.call(self.model(has_inspecting_batch=1))
        self.assertEqual(res.status_code,302)
        self.assertFalse(cur.updated)
        self.assertEqual(conn.rollbacks,1)
    def test_preparation_and_review_lots_are_not_blockers(self):
        res,cur,conn=self.call(self.model(has_inspecting_batch=0))
        self.assertEqual(res.status_code,302)
        self.assertTrue(cur.updated)
        self.assertEqual(conn.commits,1)
    def test_rejects_training_in_progress(self):
        res,cur,conn=self.call(self.model(ai_training_now=1))
        self.assertEqual(res.status_code,302)
        self.assertFalse(cur.updated)
        self.assertEqual(conn.rollbacks,1)
    def test_rejects_already_archived(self):
        res,cur,conn=self.call(self.model(active=0))
        self.assertEqual(res.status_code,302)
        self.assertFalse(cur.updated)
    def test_model_manager_cannot_remove(self):
        res,cur,conn=self.call(self.model(),role="MODEL_MANAGER")
        self.assertEqual(res.status_code,302)
        self.assertFalse(cur.updated)
    def test_get_does_not_change_model(self):
        cli=self.A.app.test_client()
        with cli.session_transaction() as s:
            s["user_id"]=23;s["role"]="ADMIN";s["username"]="admin-test"
        self.assertEqual(cli.get("/modelos-prenda/555/eliminar").status_code,405)
if __name__=="__main__":unittest.main()
