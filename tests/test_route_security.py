from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"
PUBLIC_ROUTE_FUNCTIONS = {
    "login",
    "logout",
    "health",
}


def _route_functions():
    tree = ast.parse(
        APP_PATH.read_text(
            encoding="utf-8",
        )
    )

    rows = []
    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        decorators = []
        is_route = False
        route_has_mutation = False

        for decorator in node.decorator_list:
            rendered = ast.unparse(decorator)
            decorators.append(rendered)

            if not rendered.startswith("app.route("):
                continue

            is_route = True
            if any(
                method in rendered
                for method in (
                    "'POST'",
                    '"POST"',
                    "'PUT'",
                    '"PUT"',
                    "'PATCH'",
                    '"PATCH"',
                    "'DELETE'",
                    '"DELETE"',
                )
            ):
                route_has_mutation = True

        if is_route:
            rows.append(
                (
                    node.name,
                    decorators,
                    route_has_mutation,
                )
            )

    return rows


class RouteSecurityTests(unittest.TestCase):
    def test_every_private_route_requires_authentication(self):
        unprotected = []

        for name, decorators, _ in _route_functions():
            if name in PUBLIC_ROUTE_FUNCTIONS:
                continue

            if "login_required" not in decorators:
                unprotected.append(name)

        self.assertEqual(
            unprotected,
            [],
            f"Rutas privadas sin login_required: {unprotected}",
        )

    def test_every_mutating_route_requires_authentication_except_login(self):
        unprotected = []

        for name, decorators, mutating in _route_functions():
            if not mutating or name == "login":
                continue

            if "login_required" not in decorators:
                unprotected.append(name)

        self.assertEqual(
            unprotected,
            [],
            f"Rutas mutables sin login_required: {unprotected}",
        )

    def test_session_cookie_baseline_is_hardened(self):
        import app as A

        self.assertTrue(A.app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(
            A.app.config["SESSION_COOKIE_SAMESITE"],
            "Lax",
        )
        self.assertIsInstance(
            A.app.config["SESSION_COOKIE_SECURE"],
            bool,
        )

    def test_secure_cookie_setting_is_forwarded_to_container(self):
        compose = (ROOT / "docker-compose.yaml").read_text(
            encoding="utf-8"
        )
        env_example = (ROOT / ".env.example").read_text(
            encoding="utf-8"
        )

        self.assertIn("SESSION_COOKIE_SECURE:", compose)
        self.assertIn("SESSION_COOKIE_SECURE=0", env_example)


if __name__ == "__main__":
    unittest.main()
