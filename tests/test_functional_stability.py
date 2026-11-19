import inspect
import os
import threading
import time
import unittest
import uuid
from unittest.mock import patch

import mysql.connector
import numpy as np

from tests.test_support import configure_test_environment
configure_test_environment()
import app as A
from inspection_request_guard import InspectionBusy, ManualInspectionGate
from production_totals import daily_production_totals


class Cursor:
    def execute(self, sql, params=()):
        self.sql = sql
    def fetchone(self):
        return (0 if 'COUNT' in self.sql else 1,)
    def close(self):
        pass


class Connection:
    def cursor(self):
        return Cursor()
    def close(self):
        pass


class GateTests(unittest.TestCase):
    def test_parallel_request_rejected_and_failure_releases_gate(self):
        gate = ManualInspectionGate()
        with self.assertRaisesRegex(ValueError, 'failure'):
            with gate.acquire(Connection):
                errors = []
                def attempt():
                    try:
                        with gate.acquire(Connection):
                            self.fail('parallel request entered')
                    except InspectionBusy as exc:
                        errors.append(str(exc))
                t = threading.Thread(target=attempt); t.start(); t.join(2)
                self.assertEqual(len(errors), 1)
                raise ValueError('failure')
        with gate.acquire(Connection):
            pass

    def test_cooldown_and_stale_or_repeated_frame(self):
        now = [100.0]
        gate = ManualInspectionGate(clock=lambda: now[0])
        frame = np.zeros((3, 3, 3), dtype=np.uint8)
        gate.validate_frame(frame, 100, 1)
        gate.completed(1)
        with self.assertRaises(InspectionBusy):
            with gate.acquire(Connection):
                pass
        now[0] = 103
        with gate.acquire(Connection):
            with self.assertRaises(InspectionBusy):
                gate.validate_frame(frame, 100, 2)
            with self.assertRaises(InspectionBusy):
                gate.validate_frame(frame, 103, 1)
            gate.validate_frame(frame, 103, 2)


class ManualRouteTests(unittest.TestCase):
    def setUp(self):
        self.gate = ManualInspectionGate()
        self.patches = [patch.object(A, 'MANUAL_REQUEST_GATE', self.gate),
            patch.object(A, 'MANUAL_INSPECTION_IN_PROGRESS', False),
            patch.object(A, 'AUTO_INSPECTION_ENABLED', False),
            patch.object(A, 'AUTO_THREAD', None), patch.object(A, 'db', Connection),
            patch.object(A, 'get_active_batch', return_value={'id':114}),
            patch.object(A, 'get_latest_camera_frame', side_effect=lambda:(np.zeros((4,4,3),np.uint8),time.monotonic(),7))]
        for p in self.patches: p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def call(self):
        with A.app.test_request_context('/api/station/manual-inspect',method='POST'):
            result = inspect.unwrap(A.station_manual_inspect)()
            return result if isinstance(result,tuple) else (result,200)

    def test_burst_generates_one_real_decision_including_anomaly(self):
        entered=threading.Event();release=threading.Event();first=[]
        def inference(*args,**kw):
            entered.set();release.wait(3)
            return {'status':'Defecto','confidence':55.64,'ai_decision':'ANOMALIA'}
        with patch.object(A,'register_inspection_from_frame',side_effect=inference) as run:
            thread=threading.Thread(target=lambda:first.append(self.call()))
            thread.start();self.assertTrue(entered.wait(2))
            self.assertEqual(self.call()[1],409)
            release.set();thread.join(4)
            self.assertEqual(first[0][1],200)
            self.assertEqual(first[0][0].get_json()['result']['confidence'],55.64)
            self.assertEqual(self.call()[1],429)
            self.assertEqual(run.call_count,1)
        self.assertFalse(A.MANUAL_INSPECTION_IN_PROGRESS)

    def test_stale_camera_never_generates_record(self):
        with patch.object(A,'get_latest_camera_frame',return_value=(np.zeros((3,3,3)),time.monotonic()-5,1)), \
             patch.object(A,'register_inspection_from_frame') as run:
            self.assertEqual(self.call()[1],429)
            run.assert_not_called()

    def test_automatic_and_manual_are_mutually_exclusive(self):
        with patch.object(A,'AUTO_INSPECTION_ENABLED',True), patch.object(A,'register_inspection_from_frame') as run:
            self.assertEqual(self.call()[1],409)
            run.assert_not_called()


class BatchRouteTests(unittest.TestCase):
    def test_covered_goal_still_rejects_invalid_quantity_without_write(self):
        plan=dict(has_goal=True,remaining_garments=0,suggested_quantity=0)
        with A.app.test_request_context('/lotes',method='POST',data={'planned_quantity':'0'}), \
             patch.object(A,'has_role',return_value=True), \
             patch.object(A,'get_today_production_plan',return_value=plan), \
             patch.object(A,'fetch_all',return_value=[]), \
             patch.object(A,'get_inspection_lines',return_value=[]), \
             patch.object(A,'db') as connect:
            inspect.unwrap(A.batches)()
            self.assertIn('entre 1 y 5000',A.session['_flashes'][-1][1])
            connect.assert_not_called()

    def test_extra_production_keeps_explicit_valid_quantity_and_pinned_ai(self):
        class WriteCursor(Cursor):
            lastrowid=9901
            def execute(self,sql,params=()):
                calls.append((sql,params))
        class WriteConnection(Connection):
            in_transaction=True
            def cursor(self,**kw):return WriteCursor()
            def start_transaction(self):pass
            def commit(self):pass
        calls=[]
        def fetch(sql,params):
            if 'FROM inspection_lines' in sql:return dict(id=1)
            return dict(id=2074,ai_model_id=2960)
        with A.app.test_request_context('/lotes',method='POST',data={'planned_quantity':'10','quantity_override':'1','garment_model_id':'2074','inspection_line_id':'1'}), \
             patch.object(A,'has_role',return_value=True), \
             patch.object(A,'get_today_production_plan',return_value=dict(has_goal=True,remaining_garments=0,suggested_quantity=0)), \
             patch.object(A,'fetch_all',return_value=[]),patch.object(A,'fetch_one',side_effect=fetch), \
             patch.object(A,'get_inspection_lines',return_value=[]),patch.object(A,'db',WriteConnection):
            A.session['user_id']=1
            response=inspect.unwrap(A.batches)()
            self.assertEqual(response.status_code,302)
            inserted=next(params for sql,params in calls if 'INSERT INTO batches' in sql)
            self.assertEqual(inserted[1:3],(2074,2960))
            self.assertEqual(inserted[5],10)

    def test_inactive_model_blocks_resume_without_changing_historical_links(self):
        calls=[]
        class ReadCursor(Cursor):
            def execute(self,sql,params=()):self.sql=sql;calls.append(sql)
            def fetchone(self):
                if 'GET_LOCK' in self.sql:return {'acquired':1}
                if 'SELECT b.id' in self.sql:return dict(id=109,status='PAUSADO',model_active=0,ai_active=1,ai_status='ACTIVO')
                return {'released':1}
        class ReadConnection(Connection):
            in_transaction=True
            def cursor(self,**kw):return ReadCursor()
            def start_transaction(self):pass
            def rollback(self):pass
        with A.app.test_request_context('/lotes/109/reanudar',method='POST'), \
             patch.object(A,'db',ReadConnection),patch.object(A,'AUTO_INSPECTION_ENABLED',False), \
             patch.object(A,'AUTO_THREAD',None),patch.object(A,'MANUAL_INSPECTION_IN_PROGRESS',False):
            inspect.unwrap(A.resume_batch)(109)
            self.assertIn('inactivo',A.session['_flashes'][-1][1])
        self.assertFalse(any(sql.lstrip().startswith(('UPDATE','DELETE','INSERT')) for sql in calls))


class DatabaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from datetime import date
        cls.today=date(2026,10,20)
        cls.schema='astrid_functional_'+uuid.uuid4().hex[:10]+'_test'
        cls.settings=dict(host=os.environ['MYSQL_HOST'],port=int(os.environ['MYSQL_PORT']),
            user=os.environ['MYSQL_USER'],password=os.environ['MYSQL_PASSWORD'])
        c=mysql.connector.connect(**cls.settings);q=c.cursor()
        q.execute('CREATE DATABASE '+cls.schema);q.close();c.close()
        c=cls.connect();q=c.cursor()
        q.execute('CREATE TABLE daily_production_goals(goal_date DATE,target_garments INT,target_batches INT)')
        q.execute('CREATE TABLE batches(id INT PRIMARY KEY,planned_quantity INT,status VARCHAR(40),production_date DATE,active INT NOT NULL DEFAULT 1)')
        q.execute('CREATE TABLE inspections(id INT PRIMARY KEY,batch_id INT,created_at DATETIME,ai_decision VARCHAR(30),review_status VARCHAR(40))')
        q.execute("INSERT INTO daily_production_goals VALUES('2026-10-20',200,2)")
        q.executemany("INSERT INTO batches VALUES(%s,10,%s,'2026-10-20',1)",[(i,'REVISION_PENDIENTE' if i<=18 else 'PAUSADO') for i in range(1,21)])
        q.executemany("INSERT INTO inspections VALUES(%s,%s,'2026-10-20 12:00:00',%s,%s)",
            [(i,(i-1)//10+1,'ANOMALIA' if i<=41 else 'NORMAL','PENDIENTE' if i<=41 else 'NO_REQUIERE_REVISION') for i in range(1,190)])
        c.commit();q.close();c.close()

    @classmethod
    def connect(cls):
        return mysql.connector.connect(**cls.settings,database=cls.schema)

    @classmethod
    def tearDownClass(cls):
        # Kept as isolated evidence for reproduction; never a real production schema.
        pass

    def fetch(self,sql,params=()):
        c=self.connect();q=c.cursor(dictionary=True);q.execute(sql,params)
        row=q.fetchone();q.close();c.close();return row

    def test_real_sql_distinguishes_plan_goal_inspection_and_review(self):
        totals=daily_production_totals(self.fetch,self.today)
        expected=dict(target_batches=2,target_garments=200,planned_batches=20,planned_garments=200,
            inspected_today=189,remaining_to_goal=11,remaining_to_plan=0,pending_inspection=11,
            pending_batches=2,pending_alerts=41,pending_closure=18)
        self.assertEqual(totals,expected)
        self.assertEqual(self.fetch('SELECT status FROM batches WHERE id=18')['status'],'REVISION_PENDIENTE')

    def test_mysql_lock_protects_independent_workers_and_is_released(self):
        a,b=ManualInspectionGate(),ManualInspectionGate()
        with a.acquire(self.connect):
            with self.assertRaises(InspectionBusy):
                with b.acquire(self.connect):
                    self.fail('another process entered')
        with b.acquire(self.connect):
            pass

    def test_mysql_cooldown_protects_new_process(self):
        c=self.connect();q=c.cursor();q.execute("INSERT INTO inspections VALUES(9001,20,UTC_TIMESTAMP(),'ANOMALIA','PENDIENTE')");c.commit()
        try:
            with self.assertRaises(InspectionBusy):
                with ManualInspectionGate().acquire(self.connect):
                    self.fail('burst entered')
        finally:
            q.execute('DELETE FROM inspections WHERE id=9001');c.commit();q.close();c.close()


if __name__=='__main__':
    unittest.main()
