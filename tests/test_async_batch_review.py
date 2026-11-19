"""Isolated MySQL checks for the two existing manual-review decisions."""
import importlib,os,uuid,unittest
from unittest.mock import patch
import mysql.connector
from tests.test_support import configure_test_environment

CONFIG=configure_test_environment()
A=importlib.import_module('app')

class ReviewDatabase:
    def __init__(self):
        self.cfg=dict(CONFIG);self.cfg['database']='astrid_review_validation_'+uuid.uuid4().hex[:12]
        c=mysql.connector.connect(**{k:v for k,v in self.cfg.items() if k!='database'});q=c.cursor();q.execute('CREATE DATABASE `'+self.cfg['database']+'`');c.close()
        c=self.connect();q=c.cursor()
        q.execute('CREATE TABLE users(id INT PRIMARY KEY,username VARCHAR(50),full_name VARCHAR(100),role VARCHAR(40),active INT)')
        q.execute("INSERT INTO users VALUES(41,'qa','QA','ADMIN',1)")
        q.execute('CREATE TABLE batches(id INT PRIMARY KEY,status VARCHAR(40),planned_quantity INT,active INT NOT NULL DEFAULT 1)')
        q.execute('''CREATE TABLE inspections(id INT PRIMARY KEY,batch_id INT,batch_position INT,ai_decision VARCHAR(30),review_status VARCHAR(40),human_validation VARCHAR(40),reviewed_by INT,reviewed_at DATETIME,review_notes TEXT,status VARCHAR(40),code VARCHAR(100),created_at DATETIME,image_original VARCHAR(255),image_result VARCHAR(255),defect_type VARCHAR(100),zone VARCHAR(100))''')
        c.commit();c.close();self.next_batch=9000
    def connect(self,**kwargs):return mysql.connector.connect(**self.cfg)
    def fixture(self,count=2):
        self.next_batch+=1;b=self.next_batch;c=self.connect();q=c.cursor();q.execute("INSERT INTO batches VALUES(%s,'REVISION_PENDIENTE',%s,1)",(b,count))
        for pos in range(1,count+1):q.execute("INSERT INTO inspections(id,batch_id,batch_position,ai_decision,review_status,status,code,created_at,defect_type) VALUES(%s,%s,%s,'ANOMALIA','PENDIENTE','Alerta',%s,UTC_TIMESTAMP(),'Mancha')",(b*100+pos,b,pos,'QA-'+str(pos)))
        c.commit();c.close();return b
    def row(self,ident):
        c=self.connect();q=c.cursor(dictionary=True);q.execute('SELECT * FROM inspections WHERE id=%s',(ident,));row=q.fetchone();c.close();return row

class AsyncReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.database=ReviewDatabase()
    def setUp(self):
        self.patch=patch.object(A,'db',side_effect=self.database.connect);self.patch.start();self.addCleanup(self.patch.stop);self.batch=self.database.fixture();self.ident=self.batch*100+1;self.client=A.app.test_client()
        with self.client.session_transaction() as s:s.update(user_id=41,role='ADMIN',username='qa')
    def post(self,decision='confirmar',note='Motivo de prueba',headers=True):
        return self.client.post(f'/lotes/{self.batch}/revision/{self.ident}/{decision}',data={'review_notes':note,'csrf_token':'existing-form-token'},headers={'Accept':'application/json','X-Requested-With':'XMLHttpRequest'} if headers else {})
    def test_confirm_persists_and_counters_are_read_from_database(self):
        r=self.post();self.assertEqual(r.status_code,200);self.assertTrue(r.json['ok']);row=self.database.row(self.ident);self.assertEqual(row['review_status'],'DEFECTO_CONFIRMADO');self.assertEqual(row['review_notes'],'Motivo de prueba');self.assertEqual(row['reviewed_by'],41)
        summary=A.get_batch_review_summary(self.batch);self.assertEqual(summary['pending_alerts'],1);self.assertEqual(summary['confirmed_defects'],1)
    def test_false_alarm_persists_and_updates_its_counter(self):
        r=self.post('descartar');self.assertEqual(r.status_code,200);self.assertEqual(self.database.row(self.ident)['review_status'],'ALERTA_DESCARTADA');s=A.get_batch_review_summary(self.batch);self.assertEqual((s['pending_alerts'],s['discarded_alerts']),(1,1))
    def test_duplicate_cannot_replace_saved_decision_or_note(self):
        self.post();r=self.post('descartar','Otro motivo distinto');self.assertEqual(r.status_code,409);self.assertEqual(self.database.row(self.ident)['review_notes'],'Motivo de prueba');self.assertEqual(self.database.row(self.ident)['review_status'],'DEFECTO_CONFIRMADO')
    def test_validation_preserves_pending_record(self):
        r=self.post(note='abc');self.assertEqual(r.status_code,400);self.assertFalse(r.json['ok']);self.assertEqual(self.database.row(self.ident)['review_status'],'PENDIENTE')
    def test_traditional_submit_keeps_redirect(self):
        self.assertEqual(self.post(headers=False).status_code,302);self.assertEqual(self.database.row(self.ident)['review_status'],'DEFECTO_CONFIRMADO')
    def test_permissions_are_not_bypassed_by_json(self):
        with self.client.session_transaction() as s:s['role']='VIEWER'
        with patch.object(A,'fetch_one',return_value={'id':41,'username':'qa','full_name':'QA','active':1,'role':'VIEWER'}):r=self.post()
        self.assertNotEqual(r.status_code,200);self.assertEqual(self.database.row(self.ident)['review_status'],'PENDIENTE')
    def test_stale_batch_and_invalid_action_do_not_change_data(self):
        self.assertEqual(self.post('inventada').status_code,400)
        c=self.database.connect();q=c.cursor();q.execute("UPDATE batches SET status='COMPLETADO' WHERE id=%s",(self.batch,));c.commit();c.close();self.assertEqual(self.post().status_code,409);self.assertEqual(self.database.row(self.ident)['review_status'],'PENDIENTE')
