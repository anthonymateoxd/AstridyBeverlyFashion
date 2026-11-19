"""Renombrar un modelo de prenda sin tocar su identidad tecnica."""
import importlib
import os
import sys
import unittest
from pathlib import Path

import mysql.connector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_support import require_test_database


class GarmentModelRenameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Base de pruebas aislada y desechable; nunca el `.env` del proyecto.
        cls.settings = require_test_database()
        cls.m = importlib.import_module("app")
        cls.database = cls.settings["database"]
        cls.conn = mysql.connector.connect(
            host=cls.settings["host"],
            port=cls.settings["port"],
            user=cls.settings["user"],
            password=cls.settings["password"],
            database=cls.database,
        )
        cls.conn.autocommit = True
        cls.cur = cls.conn.cursor(dictionary=True)
        cls.client = cls.m.app.test_client()

    def setUp(self):
        self._cleanup()
        self.cur.execute(
            "INSERT INTO users (username, password_hash, role, active) "
            "VALUES ('qa-ren-admin', 'x', 'ADMIN', 1)"
        )
        self.admin_id = int(self.cur.lastrowid)
        self.cur.execute(
            "INSERT INTO users (username, password_hash, role, active) "
            "VALUES ('qa-ren-qm', 'x', 'QUALITY_MANAGER', 1)"
        )
        self.qm_id = int(self.cur.lastrowid)
        self.cur.execute(
            "INSERT INTO garment_models (code, name, color, status, active, "
            "created_by) VALUES ('BLUSA-REN-TEST', 'Blusa nombre viejo', "
            "'Rosa', 'APROBADO', 1, %s)",
            (self.admin_id,),
        )
        self.model_id = int(self.cur.lastrowid)
        self.cur.execute(
            "INSERT INTO garment_ai_models (garment_model_id, version, "
            "model_type, status, active) VALUES (%s, 'v1', 'PatchCore', "
            "'PREPARACION', 0)",
            (self.model_id,),
        )
        self.ai_model_id = int(self.cur.lastrowid)
        self._login(self.admin_id, "ADMIN")

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        self.cur.execute(
            "DELETE FROM garment_ai_models WHERE garment_model_id IN "
            "(SELECT id FROM garment_models WHERE code = 'BLUSA-REN-TEST')"
        )
        self.cur.execute(
            "DELETE FROM garment_models WHERE code = 'BLUSA-REN-TEST'"
        )
        self.cur.execute(
            "DELETE FROM users WHERE username IN ('qa-ren-admin', 'qa-ren-qm')"
        )

    def _login(self, user_id, role):
        with self.client.session_transaction() as session:
            session.update(user_id=user_id, username="qa", role=role)

    def _model_row(self):
        self.cur.execute(
            "SELECT id, code, name, color, status, active, created_by "
            "FROM garment_models WHERE id = %s",
            (self.model_id,),
        )
        return self.cur.fetchone()

    def test_admin_can_rename_an_approved_model(self):
        response = self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "  Blusa lila amal "},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(f"/modelos-prenda/{self.model_id}", response.headers["Location"])

        row = self._model_row()
        self.assertEqual(row["name"], "Blusa lila amal")
        # La identidad tecnica no cambia.
        self.assertEqual(row["code"], "BLUSA-REN-TEST")
        self.assertEqual(row["id"], self.model_id)
        self.assertEqual(row["color"], "Rosa")
        self.assertEqual(row["status"], "APROBADO")

    def test_rename_keeps_ai_versions_and_history(self):
        self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "Blusa renombrada"},
            follow_redirects=False,
        )

        self.cur.execute(
            "SELECT id, version, status FROM garment_ai_models "
            "WHERE id = %s",
            (self.ai_model_id,),
        )
        row = self.cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["version"], "v1")
        self.assertEqual(row["status"], "PREPARACION")

    def test_rename_is_visible_on_the_detail_sheet(self):
        self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "Blusa lila de catálogo"},
            follow_redirects=False,
        )

        html = self.client.get(
            f"/modelos-prenda/{self.model_id}"
        ).get_data(as_text=True)

        self.assertIn("Blusa lila de catálogo", html)
        self.assertIn("Editar nombre", html)
        self.assertIn("BLUSA-REN-TEST", html)

    def test_empty_name_is_rejected(self):
        response = self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "   "},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Ingrese el nombre del modelo de blusa.",
                      response.get_data(as_text=True))
        self.assertEqual(self._model_row()["name"], "Blusa nombre viejo")

    def test_too_long_name_is_rejected(self):
        response = self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "B" * 151},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("no puede superar", response.get_data(as_text=True))
        self.assertEqual(self._model_row()["name"], "Blusa nombre viejo")

    def test_quality_manager_cannot_rename(self):
        self._login(self.qm_id, "QUALITY_MANAGER")

        response = self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "Nombre no autorizado"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self._model_row()["name"], "Blusa nombre viejo")

    def test_anonymous_user_is_rejected(self):
        with self.client.session_transaction() as session:
            session.clear()

        response = self.client.post(
            f"/modelos-prenda/{self.model_id}/nombre",
            data={"name": "Sin sesión"},
            follow_redirects=False,
        )

        # Sin sesión la petición se rechaza y el nombre del modelo no cambia.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self._model_row()["name"], "Blusa nombre viejo")

    def test_name_validation_helper(self):
        name, error = self.m.garment_model_name_from_form({"name": "  Blusa  "})
        self.assertEqual(name, "Blusa")
        self.assertIsNone(error)

        name, error = self.m.garment_model_name_from_form({"name": ""})
        self.assertEqual(name, "")
        self.assertIsNotNone(error)

        name, error = self.m.garment_model_name_from_form({"name": "B" * 151})
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
