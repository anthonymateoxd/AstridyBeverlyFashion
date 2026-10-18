from __future__ import annotations
import io,importlib,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np,cv2
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root))

class Conn:
    def __init__(self,cur):
        self.cur=cur;self.in_transaction=False;self.commits=0;self.rollbacks=0
    def cursor(self,dictionary=False):return self.cur
    def start_transaction(self):self.in_transaction=True
    def commit(self):self.commits+=1;self.in_transaction=False
    def rollback(self):self.rollbacks+=1;self.in_transaction=False
    def close(self):pass
class Cur:
    def __init__(self,previous=True,fail=False):
        self.lastrowid=8;self.cur_action=None;self.calls=[];self.previous=previous;self.fail=fail
    def execute(self,sql,params=()):
        self.calls.append((sql,params))
        if sql.strip().startswith("SELECT"):
            self.cur_action="fetch"
        elif self.fail and sql.strip().startswith("UPDATE"):
            raise RuntimeError("simulated db update failure")
    def fetchone(self):
        return {"id":8,"image_path":"garment_models/2069/missing.png"} if self.previous else None
    def close(self):pass
class ReferenceUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.a=importlib.import_module("app")
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.original=self.a.app.static_folder
        self.a.app.static_folder=self.temp.name
        self.model={"id":2069,"code":"BLUSA-1216","name":"Blusa modelo blanco","active":1,"status":"APROBADO","created_by":5}
        self.user={"id":5,"username":"testadmin","role":"ADMIN","active":1}
        self.client=self.a.app.test_client()
        with self.client.session_transaction() as s:s.update(user_id=5,username="testadmin",role="ADMIN")
    def tearDown(self):
        self.a.app.static_folder=self.original
        self.temp.cleanup()
    def image(self):
        ok,buf=cv2.imencode(".png",np.zeros((240,300,3),np.uint8))
        assert ok
        return io.BytesIO(buf.tobytes())
    def submit(self,content,previous=True,fail=False,authorized=True):
        cur=Cur(previous,fail);conn=Conn(cur)
        with (
            patch.object(self.a,"fetch_one",return_value=self.user),
            patch.object(self.a,"get_garment_model",return_value=self.model),
            patch.object(self.a,"can_manage_garment_model",return_value=authorized),
            patch.object(self.a,"db",return_value=conn)
        ):
            response=self.client.post("/modelos-prenda/2069/imagen-referencia",data=content,content_type="multipart/form-data")
        return response,conn,cur
    def test_replacement_is_persisted_inside_results(self):
        resp,conn,cur=self.submit({"reference_photo":(self.image(),"blusa.png")})
        self.assertEqual(resp.status_code,302)
        self.assertEqual(conn.commits,1)
        updates=[(sql,p) for sql,p in cur.calls if sql.startswith("UPDATE garment_model_images")]
        self.assertEqual(len(updates),1)
        self.assertTrue(updates[0][1][0].startswith("results/model_references/2069/"))
        files=list(Path(self.temp.name).rglob("*.jpg"))
        self.assertEqual(len(files),1)
        self.assertTrue(files[0].read_bytes().startswith(b"\xff\xd8"))
    def test_new_ref_if_legacy_missing_row(self):
        resp,conn,cur=self.submit({"reference_photo":(self.image(),"blusa.png")},previous=False)
        self.assertEqual(conn.commits,1)
        self.assertTrue(any("INSERT INTO garment_model_images" in sql for sql,p in cur.calls))
    def test_invalid_upload_does_not_write_db(self):
        resp,conn,cur=self.submit({"reference_photo":(io.BytesIO(b'not-photo'),"blusa.png")})
        self.assertEqual(resp.status_code,302)
        self.assertEqual(conn.commits,0)
        self.assertEqual(list(Path(self.temp.name).rglob("*.jpg")),[])
    def test_disallowed_extension_does_not_write_db(self):
        resp,conn,cur=self.submit({"reference_photo":(self.image(),"blusa.svg")})
        self.assertEqual(conn.commits,0)
        self.assertEqual(list(Path(self.temp.name).rglob("*.jpg")),[])
    def test_unauthorized_cannot_modify(self):
        resp,conn,cur=self.submit({"reference_photo":(self.image(),"blusa.png")},authorized=False)
        self.assertEqual(resp.status_code,403)
        self.assertEqual(conn.commits,0)
    def test_failed_db_update_cleans_staged_photo(self):
        resp,conn,cur=self.submit({"reference_photo":(self.image(),"blusa.png")},fail=True)
        self.assertEqual(conn.commits,0)
        self.assertEqual(conn.rollbacks,1)
        self.assertFalse(list(Path(self.temp.name).rglob("*.jpg")))
if __name__=="__main__":unittest.main()
