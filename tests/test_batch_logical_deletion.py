"""Pruebas aisladas de la eliminacion logica reversible de lotes.

Todo corre contra una base MySQL desechable creada por test con datos
ficticios; no se toca la base de produccion ni credenciales reales.
"""
import importlib
import threading
import unittest
import uuid

import mysql.connector
from unittest.mock import patch

from tests.test_support import require_test_database

CONFIG = require_test_database()
A = importlib.import_module("app")


class DeletionDatabase:
    """Base desechable con el esquema minimo que exige la caracteristica."""

    def __init__(self):
        settings = dict(CONFIG)
        settings["database"] = (
            "astrid_batch_deletion_" + uuid.uuid4().hex[:10] + "_test"
        )
        self.settings = settings
        admin = mysql.connector.connect(
            **{k: v for k, v in settings.items() if k != "database"}
        )
        cursor = admin.cursor()
        cursor.execute("CREATE DATABASE `" + settings["database"] + "`")
        cursor.close()
        admin.close()
        self._create_schema()
        self.batch_sequence = 7000

    def connect(self, **kwargs):
        return mysql.connector.connect(**self.settings)

    def _create_schema(self):
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE users(id INT PRIMARY KEY,username VARCHAR(50),"
            "full_name VARCHAR(100),role VARCHAR(40),active INT)"
        )
        cursor.executemany(
            "INSERT INTO users VALUES(%s,%s,%s,%s,1)",
            [
                (41, "admin_qa", "Admin QA", "ADMIN"),
                (42, "calidad_qa", "Calidad QA", "QUALITY_MANAGER"),
                (43, "solo_lectura", "Solo lectura", "VIEWER"),
            ],
        )
        cursor.execute(
            "CREATE TABLE garment_models(id INT PRIMARY KEY,code VARCHAR(80),"
            "name VARCHAR(120))"
        )
        cursor.execute(
            "CREATE TABLE garment_ai_models(id INT PRIMARY KEY,version VARCHAR(40),"
            "model_type VARCHAR(60))"
        )
        cursor.execute(
            "CREATE TABLE inspection_lines(id INT PRIMARY KEY,code VARCHAR(60),"
            "name VARCHAR(100),withdrawal_distance_cm DECIMAL(10,2),"
            "conveyor_speed_cm_s DECIMAL(10,2))"
        )
        cursor.execute(
            "CREATE TABLE batches(id INT PRIMARY KEY,code VARCHAR(80),"
            "planned_quantity INT,status VARCHAR(40) DEFAULT 'PREPARACION',"
            "active TINYINT(1) NOT NULL DEFAULT 1,started_at DATETIME NULL,"
            "inspection_completed_at DATETIME NULL,closed_at DATETIME NULL,"
            "production_date DATE NULL,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
            "deleted_at DATETIME NULL,deleted_by INT NULL,deletion_reason VARCHAR(500) NULL,"
            "garment_model_id INT NULL,ai_model_id INT NULL,inspection_line_id INT NULL,"
            "created_by INT NULL,notes TEXT)"
        )
        cursor.execute(
            "CREATE TABLE inspections(id INT PRIMARY KEY,batch_id INT,batch_position INT,"
            "code VARCHAR(100),created_at DATETIME,status VARCHAR(30),"
            "ai_decision VARCHAR(30),review_status VARCHAR(40),"
            "human_validation VARCHAR(40),image_original VARCHAR(255),"
            "image_result VARCHAR(255),defect_type VARCHAR(100),zone VARCHAR(100),"
            "confidence DECIMAL(5,4),reviewed_by INT NULL,reviewed_at DATETIME NULL,"
            "review_notes TEXT,garment_type VARCHAR(60),size VARCHAR(20),notes TEXT)"
        )
        cursor.execute(
            "CREATE TABLE batch_deletion_audit(id INT AUTO_INCREMENT PRIMARY KEY,"
            "batch_id INT NOT NULL,batch_code VARCHAR(80) NOT NULL,action VARCHAR(20) NOT NULL,"
            "batch_status VARCHAR(40) NULL,planned_quantity INT NOT NULL DEFAULT 0,"
            "processed_quantity INT NOT NULL DEFAULT 0,reason VARCHAR(500) NOT NULL,"
            "admin_id INT NOT NULL,admin_username VARCHAR(80) NULL,"
            "active_after TINYINT(1) NOT NULL DEFAULT 1,"
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
            "INDEX idx_batch_deletion_audit_batch(batch_id),"
            "INDEX idx_batch_deletion_audit_created(created_at))"
        )
        cursor.execute(
            "CREATE TABLE daily_production_goals(goal_date DATE PRIMARY KEY,"
            "target_garments INT,target_batches INT,shift_start TIME NULL,shift_end TIME NULL,"
            "created_by INT NULL)"
        )
        connection.commit()
        cursor.close()
        connection.close()

    def batch_row(self, batch_id, columns="*"):
        connection = self.connect()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT " + columns + " FROM batches WHERE id=%s", (batch_id,)
        )
        row = cursor.fetchone()
        cursor.fetchall()
        cursor.close()
        connection.close()
        return row

    def scalar(self, sql, params=()):
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        # El cursor sin buffer exige consumir el resto del result set.
        cursor.fetchall()
        cursor.close()
        connection.close()
        return row[0] if row else None

    def audit_rows(self, batch_id):
        connection = self.connect()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM batch_deletion_audit WHERE batch_id=%s ORDER BY id",
            (batch_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return rows

    def goal_row(self, day):
        connection = self.connect()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM daily_production_goals WHERE goal_date=%s", (day,)
        )
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        return row

    def add_batch(self, status="REVISION_PENDIENTE", planned=40, active=1, day=None):
        self.batch_sequence += 1
        batch_id = self.batch_sequence
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO batches(id,code,planned_quantity,status,active,production_date) "
            "VALUES(%s,%s,%s,%s,%s,COALESCE(%s,CURDATE()))",
            (batch_id, "LOT-QA-%05d" % batch_id, planned, status, active, day),
        )
        connection.commit()
        cursor.close()
        connection.close()
        return batch_id

    def add_inspections(self, batch_id, count=3):
        created = []
        for position in range(1, count + 1):
            inspection_id = batch_id * 10 + position
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO inspections(id,batch_id,batch_position,code,created_at,status,"
                "ai_decision,review_status,human_validation,image_original,image_result,"
                "defect_type,zone) VALUES(%s,%s,%s,%s,UTC_TIMESTAMP(),'Defecto','ANOMALIA',"
                "'PENDIENTE','Pendiente',%s,%s,'Mancha','pecho')",
                (
                    inspection_id,
                    batch_id,
                    position,
                    "QA-%s-%s" % (batch_id, position),
                    "results/%s_original.jpg" % inspection_id,
                    "results/%s_result.jpg" % inspection_id,
                ),
            )
            connection.commit()
            cursor.close()
            connection.close()
            created.append(inspection_id)
        return created


class BatchDeletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = DeletionDatabase()

    def setUp(self):
        self.db_patch = patch.object(
            A, "db", side_effect=self.database.connect
        )
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.client = A.app.test_client()
        with self.client.session_transaction() as session:
            session.update(
                user_id=41, role="ADMIN", username="admin_qa", display_name="Admin QA"
            )

    def post(self, batch_id, data=None, user=None):
        payload = {"csrf_token": "qa-token"}
        if data:
            payload.update(data)
        if user is not None:
            with self.client.session_transaction() as session:
                session.update(user_id=user[0], role=user[1], username=user[2])
        return self.client.post(
            "/lotes/%s/eliminar" % batch_id,
            data=payload,
            headers={
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            },
        )

    def restore(self, batch_id, data=None):
        payload = {"csrf_token": "qa-token"}
        if data:
            payload.update(data)
        return self.client.post(
            "/lotes/%s/restaurar" % batch_id,
            data=payload,
            headers={
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            },
        )

    # ------------------------------------------------------------------
    # Autorizacion y validacion
    # ------------------------------------------------------------------
    def test_reason_is_mandatory_and_nothing_changes_without_it(self):
        batch_id = self.database.add_batch()
        for payload in ({}, {"reason": "  "}, {"reason": "abc"}):
            response = self.post(batch_id, payload)
            self.assertEqual(response.status_code, 400, payload)
            self.assertFalse(response.get_json()["ok"])
        row = self.database.batch_row(batch_id, "active,deleted_at,deletion_reason")
        self.assertEqual(int(row["active"]), 1)
        self.assertIsNone(row["deleted_at"])
        self.assertEqual(self.database.audit_rows(batch_id), [])

    def test_only_admin_can_delete_and_open_the_audit_screen(self):
        batch_id = self.database.add_batch()
        with self.client.session_transaction() as session:
            session.update(
                user_id=42, role="QUALITY_MANAGER", username="calidad_qa"
            )
        response = self.post(batch_id, {"reason": "Lote duplicado por error de captura"})
        # role_required redirige con flash en rutas web y responde 403 en API.
        self.assertIn(response.status_code, (302, 403))
        self.assertEqual(int(self.database.batch_row(batch_id, "active")["active"]), 1)
        self.assertEqual(self.database.audit_rows(batch_id), [])

        with self.client.session_transaction() as session:
            session.update(user_id=41, role="ADMIN", username="admin_qa")
        self.assertEqual(self.client.get("/lotes/eliminados").status_code, 200)
        self.assertEqual(
            self.client.get(
                "/lotes/eliminados/%s/auditoria" % batch_id
            ).status_code,
            200,
        )

        with self.client.session_transaction() as session:
            session.update(
                user_id=42, role="QUALITY_MANAGER", username="calidad_qa"
            )
        blocked = self.client.get("/lotes/eliminados")
        self.assertNotEqual(blocked.status_code, 200)

    # ------------------------------------------------------------------
    # Eliminacion logica y preservacion de evidencia
    # ------------------------------------------------------------------
    def test_lot_with_inspections_is_deleted_logically_and_evidence_survives(self):
        batch_id = self.database.add_batch(status="REVISION_PENDIENTE", planned=40)
        inspections = self.database.add_inspections(batch_id, 3)
        before_images = [
            self.database.scalar(
                "SELECT image_original FROM inspections WHERE id=%s", (item,)
            )
            for item in inspections
        ]

        response = self.post(batch_id, {"reason": "Lote duplicado por error de captura"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["processed_quantity"], 3)
        self.assertEqual(body["planned_quantity"], 40)

        # La fila sigue existiendo: es una baja logica, nunca un DELETE.
        row = self.database.batch_row(batch_id)
        self.assertEqual(int(row["active"]), 0)
        self.assertIsNotNone(row["deleted_at"])
        self.assertEqual(int(row["deleted_by"]), 41)
        self.assertEqual(row["deletion_reason"], "Lote duplicado por error de captura")
        self.assertEqual(row["status"], "REVISION_PENDIENTE")

        # Inventario fisico intacto.
        self.assertEqual(
            self.database.scalar("SELECT COUNT(*) FROM batches WHERE id=%s", (batch_id,)),
            1,
        )
        self.assertEqual(
            self.database.scalar(
                "SELECT COUNT(*) FROM inspections WHERE batch_id=%s", (batch_id,)
            ),
            3,
        )
        for item, image in zip(inspections, before_images):
            after = self.database.scalar(
                "SELECT image_original FROM inspections WHERE id=%s", (item,)
            )
            self.assertEqual(after, image)
        self.assertEqual(
            self.database.scalar(
                "SELECT COUNT(*) FROM inspections "
                "WHERE batch_id=%s AND review_status='PENDIENTE'",
                (batch_id,),
            ),
            3,
        )

        # Auditoria: administrador, fecha, motivo y cantidades.
        rows = self.database.audit_rows(batch_id)
        self.assertEqual(len(rows), 1)
        entry = rows[0]
        self.assertEqual(entry["action"], "ELIMINACION")
        self.assertEqual(entry["admin_id"], 41)
        self.assertEqual(entry["admin_username"], "admin_qa")
        self.assertEqual(entry["reason"], "Lote duplicado por error de captura")
        self.assertEqual(int(entry["planned_quantity"]), 40)
        self.assertEqual(int(entry["processed_quantity"]), 3)
        self.assertEqual(int(entry["active_after"]), 0)
        self.assertIsNotNone(entry["created_at"])

        # El objetivo diario configurado no se toca.
        goal_before = self.database.goal_row("2026-01-01")
        self.assertIsNone(goal_before)

    def test_batch_in_active_inspection_cannot_be_deleted(self):
        for status in ("EN_INSPECCION", "PAUSADO"):
            batch_id = self.database.add_batch(status=status, planned=10)
            response = self.post(
                batch_id, {"reason": "Intento durante inspeccion activa"}
            )
            self.assertEqual(response.status_code, 409, status)
            self.assertIn("inspeccion activa", response.get_json()["error"])
            self.assertEqual(int(self.database.batch_row(batch_id, "active")["active"]), 1)
            self.assertEqual(self.database.audit_rows(batch_id), [])

    def test_duplicate_requests_are_detected_and_only_one_wins(self):
        batch_id = self.database.add_batch(status="COMPLETADO", planned=25)
        self.database.add_inspections(batch_id, 2)

        first = self.post(batch_id, {"reason": "Duplicado por error de captura"})
        second = self.post(batch_id, {"reason": "Duplicado por error de captura"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertTrue(second.get_json()["already_deleted"])
        rows = self.database.audit_rows(batch_id)
        self.assertEqual(len(rows), 1)

    def test_concurrent_deletes_allow_exactly_one_success(self):
        batch_id = self.database.add_batch(status="REVISION_PENDIENTE", planned=15)
        self.database.add_inspections(batch_id, 4)

        results = []
        barrier = threading.Barrier(2)

        def worker():
            client = A.app.test_client()
            with client.session_transaction() as session:
                session.update(
                    user_id=41, role="ADMIN", username="admin_qa"
                )
            barrier.wait()
            response = client.post(
                "/lotes/%s/eliminar" % batch_id,
                data={"reason": "Carrera entre dos peticiones simultaneas"},
                headers={
                    "Accept": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            results.append(response.status_code)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(sorted(results), [200, 409], results)
        self.assertEqual(int(self.database.batch_row(batch_id, "active")["active"]), 0)
        self.assertEqual(len(self.database.audit_rows(batch_id)), 1)

    # ------------------------------------------------------------------
    # Restauracion
    # ------------------------------------------------------------------
    def test_restore_brings_the_lot_back_and_is_audited(self):
        batch_id = self.database.add_batch(status="COMPLETADO", planned=33)
        inspections = self.database.add_inspections(batch_id, 5)
        self.post(batch_id, {"reason": "Baja temporal para auditoria interna"})
        self.assertEqual(int(self.database.batch_row(batch_id, "active")["active"]), 0)

        response = self.restore(batch_id, {"reason": "Auditoria finalizada"})
        self.assertEqual(response.status_code, 200)
        row = self.database.batch_row(batch_id)
        self.assertEqual(int(row["active"]), 1)
        self.assertIsNone(row["deleted_at"])
        self.assertIsNone(row["deletion_reason"])
        self.assertEqual(row["status"], "COMPLETADO")
        self.assertEqual(
            self.database.scalar(
                "SELECT COUNT(*) FROM inspections WHERE batch_id=%s", (batch_id,)
            ),
            len(inspections),
        )

        actions = [entry["action"] for entry in self.database.audit_rows(batch_id)]
        self.assertEqual(actions, ["ELIMINACION", "RESTAURACION"])
        restored = self.database.audit_rows(batch_id)[-1]
        self.assertEqual(int(restored["active_after"]), 1)
        self.assertEqual(restored["reason"], "Auditoria finalizada")

        # Restaurar dos veces no duplica registros.
        again = self.restore(batch_id)
        self.assertEqual(again.status_code, 409)
        self.assertEqual(len(self.database.audit_rows(batch_id)), 2)

    # ------------------------------------------------------------------
    # Indicadores operativos
    # ------------------------------------------------------------------
    def test_operational_totals_exclude_deleted_lots_until_restored(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from production_totals import daily_production_totals

        today = datetime.now(ZoneInfo("America/Guatemala")).date()
        keep = self.database.add_batch(status="REVISION_PENDIENTE", planned=40, day=today)
        drop = self.database.add_batch(status="REVISION_PENDIENTE", planned=60, day=today)
        self.database.add_inspections(keep, 4)
        self.database.add_inspections(drop, 6)

        # Meta diaria configurada: debe permanecer intacta tras la baja.
        connection = self.database.connect()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO daily_production_goals(goal_date,target_garments,target_batches) "
            "VALUES(%s,500,5) ON DUPLICATE KEY UPDATE target_garments=500,target_batches=5",
            (today,),
        )
        connection.commit()
        cursor.close()
        connection.close()

        before = daily_production_totals(A.fetch_one, today)
        self.assertEqual(before["target_garments"], 500)
        self.assertEqual(before["target_batches"], 5)

        self.post(drop, {"reason": "Lote de prueba fuera de indicadores"})

        during = daily_production_totals(A.fetch_one, today)
        self.assertEqual(
            during["planned_batches"], before["planned_batches"] - 1
        )
        self.assertEqual(
            during["planned_garments"], before["planned_garments"] - 60
        )
        self.assertEqual(
            during["inspected_today"], before["inspected_today"] - 6
        )
        self.assertEqual(
            during["pending_alerts"], before["pending_alerts"] - 6
        )

        # El lote eliminado tampoco aparece en el listado operativo.
        active_codes = [row["code"] for row in A.get_batches()]
        keep_code = self.database.batch_row(keep, "code")["code"]
        drop_code = self.database.batch_row(drop, "code")["code"]
        self.assertIn(keep_code, active_codes)
        self.assertNotIn(drop_code, active_codes)
        deleted_codes = [row["code"] for row in A.get_deleted_batches()]
        self.assertIn(drop_code, deleted_codes)

        # La meta diaria configurada sigue intacta.
        goal_after_delete = daily_production_totals(A.fetch_one, today)
        self.assertEqual(goal_after_delete["target_garments"], 500)
        self.assertEqual(goal_after_delete["target_batches"], 5)

        self.restore(drop)
        after = daily_production_totals(A.fetch_one, today)
        self.assertEqual(after["planned_batches"], before["planned_batches"])
        self.assertEqual(after["planned_garments"], before["planned_garments"])
        self.assertEqual(after["inspected_today"], before["inspected_today"])
        self.assertEqual(after["pending_alerts"], before["pending_alerts"])

    def test_dashboard_indicators_exclude_deleted_lots(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("America/Guatemala")).date()
        keep = self.database.add_batch(status="EN_INSPECCION", planned=40, day=today)
        drop = self.database.add_batch(status="REVISION_PENDIENTE", planned=60, day=today)
        self.database.add_inspections(keep, 2)
        self.database.add_inspections(drop, 3)

        connection = self.database.connect()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO daily_production_goals("
            "goal_date,target_garments,target_batches,shift_start,shift_end) "
            "VALUES(%s,500,5,'08:00','17:00') ON DUPLICATE KEY UPDATE "
            "target_garments=500,target_batches=5,"
            "shift_start='08:00',shift_end='17:00'",
            (today,),
        )
        connection.commit()
        cursor.close()
        connection.close()

        self.post(drop, {"reason": "Lote fuera de los indicadores del panel"})

        # 1) Panel principal (render del servidor): 2 prendas y 2 alertas
        #    pendientes, no las 5 del lote eliminado.
        page = self.client.get("/dashboard")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn('<strong id="dashPendingAlerts">2</strong>', html)
        self.assertIn('<strong id="dashInspected">2 / 500</strong>', html)

        # 2) Panel en vivo (JSON): mismas cifras, tampoco cuenta el lote bajo.
        live = self.client.get("/api/dashboard/live")
        self.assertEqual(live.status_code, 200)
        payload = live.get_json()
        self.assertEqual(payload["inspected"], 2)
        self.assertEqual(payload["pending_alerts"], 2)

        # 3) La meta diaria se valida con prendas de lotes vivos: con 5
        #    inspecciones en total pero 2 de lotes activos, bajar la meta a 2
        #    prendas supera la validacion (si contara el lote eliminado, el
        #    servidor rechazaria la peticion).
        with patch.object(A, "flash") as flash_mock:
            response = self.client.post(
                "/dashboard/meta",
                data={
                    "csrf_token": "qa-token",
                    "target_batches": "1",
                    "target_garments": "2",
                    "shift_start": "08:00",
                    "shift_end": "17:00",
                },
            )
        self.assertEqual(response.status_code, 302)
        messages = " | ".join(
            str(call.args[0]) for call in flash_mock.call_args_list
        )
        self.assertNotIn("por debajo de las 5 prendas", messages)
        # Dime que llego hasta la regla de "meta ya definida" del dia.
        self.assertIn("ya est", messages.lower())

        # Restaurar vuelve a contar sus inspecciones.
        self.restore(drop)
        live = self.client.get("/api/dashboard/live").get_json()
        self.assertEqual(live["inspected"], 5)
        self.assertEqual(live["pending_alerts"], 5)

    def test_deleted_lot_alerts_leave_the_station_buffer(self):
        batch_id = self.database.add_batch(status="REVISION_PENDIENTE", planned=10)
        self.database.add_inspections(batch_id, 1)
        other = self.database.add_batch(status="REVISION_PENDIENTE", planned=12)

        with A.QUALITY_ALERT_LOCK:
            backup = list(A.QUALITY_ALERT_EVENTS)
        self.addCleanup(
            lambda: A.QUALITY_ALERT_EVENTS.__setitem__(slice(None), backup)
        )

        A.QUALITY_ALERT_EVENTS[:] = [
            {"alert_id": "ae-keep", "seq": 1, "batch_id": other},
            {"alert_id": "ae-drop", "seq": 2, "batch_id": batch_id},
        ]

        self.post(batch_id, {"reason": "Lote fuera de la estacion"})

        remaining = [event.get("batch_id") for event in A.QUALITY_ALERT_EVENTS]
        self.assertNotIn(batch_id, remaining)
        self.assertIn(other, remaining)

    def test_deleted_lot_disappears_from_records_and_report_queries(self):
        batch_id = self.database.add_batch(status="COMPLETADO", planned=20)
        self.database.add_inspections(batch_id, 2)
        self.post(batch_id, {"reason": "Lote fuera del historial operativo"})

        # Consulta de opciones de lote usada por registros e informes.
        options = A.fetch_all(
            "SELECT id, code FROM batches WHERE active = 1 ORDER BY id DESC LIMIT 200"
        )
        self.assertNotIn(batch_id, [row["id"] for row in options])

        # Resumen historico: el JOIN con active excluye sus inspecciones.
        excluded = A.fetch_all(
            "SELECT COUNT(i.id) AS inspected FROM inspections i "
            "LEFT JOIN batches b ON b.id = i.batch_id "
            "WHERE i.batch_id = %s AND COALESCE(b.active, 1) = 1",
            (batch_id,),
        )
        self.assertEqual(int(excluded[0]["inspected"]), 0)

        # Conservacion: siguen existiendo y siendo legibles en auditoria.
        kept = A.fetch_one(
            "SELECT COUNT(*) AS total FROM inspections WHERE batch_id = %s",
            (batch_id,),
        )
        self.assertEqual(int(kept["total"]), 2)

        # Conservacion: la evidencia sigue consultable en la auditoria.
        detail = self.client.get(
            "/lotes/eliminados/%s/auditoria" % batch_id
        )
        self.assertEqual(detail.status_code, 200)
        page = detail.get_data(as_text=True)
        self.assertIn("LOT-QA-%05d" % batch_id, page)
        self.assertIn(
            self.database.scalar(
                "SELECT code FROM inspections WHERE batch_id=%s", (batch_id,)
            ),
            page,
        )
        self.assertIn("no se han borrado", page)

    # ------------------------------------------------------------------
    # UI del modal y reglas del repositorio
    # ------------------------------------------------------------------
    def test_lots_template_uses_a_confirmation_modal_with_required_reason(self):
        from pathlib import Path

        template = (
            Path(__file__).resolve().parents[1] / "templates" / "batches.html"
        ).read_text(encoding="utf-8")
        self.assertIn("data-batch-delete", template)
        self.assertIn('id="batchDeleteDialog"', template)
        self.assertIn('name="reason"', template)
        self.assertIn('minlength="5"', template)
        self.assertIn("required", template)
        self.assertIn("url_for('delete_batch'", template)
        self.assertIn("url_for('restore_batch'", template)
        self.assertIn("deleted_lots", template)
        self.assertNotIn("delete_accidental_batch", template)
        self.assertNotIn("return confirm('?Eliminar este lote?", template)

    def test_repository_never_issues_a_physical_batch_delete(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "app.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("DELETE FROM batches", source)


if __name__ == "__main__":
    unittest.main()
