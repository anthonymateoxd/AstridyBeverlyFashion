"""Administrator-only full-name changes; no account or permission mutation."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

class DisplayNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module=importlib.import_module("app")

    def client(self, role="ADMIN"):
        client=self.module.app.test_client()
        with client.session_transaction() as s:
            s.update(user_id=15,username="test-admin",role=role)
        return client

    def auth_user(self, role="ADMIN", fullname="Antiguo Nombre"):
        return dict(id=15,username="test-admin",
                    full_name=fullname,role=role,active=1)

    def target(self,name="Antiguo Nombre"):
        return dict(id=25,username="prueba-25",full_name=name)

    def test_admin_can_rename_other_user_without_changing_login(self):
        m=self.module
        client=self.client()
        with (patch.object(m,"fetch_one",side_effect=[
            self.auth_user(),self.target()
        ]) as lookup,patch.object(m,"execute") as execute):
            response=client.post("/usuarios/25/nombre",
                data={"full_name":"Beverly Cordon"})
        self.assertEqual(response.status_code,302)
        self.assertIn("/usuarios",response.location)
        execute.assert_called_once_with(
            "UPDATE users SET full_name = %s WHERE id = %s",
            ("Beverly Cordon",25),
        )
        self.assertNotIn("password",execute.call_args.args[0])
        self.assertNotIn("role",execute.call_args.args[0])
        self.assertNotIn("username",execute.call_args.args[0])
        self.assertEqual(lookup.call_count,2)

    def test_admin_can_rename_own_account_and_refresh_session(self):
        m=self.module
        client=self.client()
        with (patch.object(m,"fetch_one",side_effect=[
            self.auth_user(),dict(id=15,username="test-admin",full_name="Viejo")
        ]),patch.object(m,"execute") as execute):
            response=client.post("/usuarios/15/nombre",
                data={"full_name":"Nombre Nuevo"})
        self.assertEqual(response.status_code,302)
        execute.assert_called_once()
        with client.session_transaction() as sess:
            self.assertEqual(sess["display_name"],"Nombre Nuevo")
            self.assertEqual(sess["username"],"test-admin")
            self.assertEqual(sess["role"],"ADMIN")

    def test_model_manager_cannot_change_other_name(self):
        m=self.module
        client=self.client("MODEL_MANAGER")
        with (patch.object(m,"fetch_one",return_value=self.auth_user("MODEL_MANAGER")),
              patch.object(m,"execute") as execute):
            response=client.post("/usuarios/25/nombre",
                data={"full_name":"Cambiado"})
        self.assertEqual(response.status_code,302)
        execute.assert_not_called()

    def test_empty_name_rejected(self):
        m=self.module
        client=self.client()
        with (patch.object(m,"fetch_one",return_value=self.auth_user()),
              patch.object(m,"execute") as execute):
            response=client.post("/usuarios/25/nombre",data={"full_name":"   "})
        self.assertEqual(response.status_code,302)
        execute.assert_not_called()

    def test_overlong_name_rejected(self):
        m=self.module
        client=self.client()
        with (patch.object(m,"fetch_one",return_value=self.auth_user()),
              patch.object(m,"execute") as execute):
            response=client.post("/usuarios/25/nombre",
                                 data={"full_name":"Z"*151})
        self.assertEqual(response.status_code,302)
        execute.assert_not_called()

    def test_missing_user_no_update(self):
        m=self.module
        client=self.client()
        with (patch.object(m,"fetch_one",side_effect=[self.auth_user(),None]),
              patch.object(m,"execute") as execute):
            response=client.post("/usuarios/4040/nombre",
                                 data={"full_name":"Sin Cuenta"})
        self.assertEqual(response.status_code,302)
        execute.assert_not_called()

    def test_repeat_name_no_update(self):
        m=self.module
        client=self.client()
        with (patch.object(m,"fetch_one",side_effect=[
            self.auth_user(),self.target("Beverly Cordon")
        ]),patch.object(m,"execute") as execute):
            response=client.post("/usuarios/25/nombre",
                data={"full_name":"Beverly Cordon"})
        self.assertEqual(response.status_code,302)
        execute.assert_not_called()

    def test_dashboard_shows_name_and_role_separately(self):
        p=ROOT/"templates/dashboard.html"
        contents=p.read_text(encoding="utf-8")
        self.assertIn("session.get('display_name')",contents)
        self.assertIn('class="dashboard-user-role"',contents)
        self.assertIn("Administrador",contents)
        userlist=(ROOT/"templates/users.html").read_text(encoding="utf-8")
        self.assertIn("Mi nombre para mostrar",userlist)
        self.assertIn("user_change_full_name",userlist)
        # Legacy hidden accounts should not return to the public list.
        self.assertIn("'admin', 'juan', 'paco'",userlist)

if __name__=="__main__":
    unittest.main()
