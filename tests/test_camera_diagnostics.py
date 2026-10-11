import unittest
from unittest.mock import patch

from camera_diagnostics import (
    build_camera_diagnostic,
    normalize_camera_error_code,
    probe_tcp_port,
)


class CameraDiagnosticsTests(unittest.TestCase):
    def test_auth_error_has_priority(self):
        code = normalize_camera_error_code(
            "CAMERA_RTSP_OPEN_FAILED",
            last_error="RTSP server returned 401 Unauthorized",
            reconnecting=False,
        )
        self.assertEqual(code, "CAMERA_AUTH_FAILED")

    def test_reconnecting_without_code_is_explicit(self):
        diagnostic = build_camera_diagnostic(
            None,
            reconnecting=True,
            technical={"ip": "192.168.0.5"},
        )
        self.assertEqual(diagnostic["code"], "CAMERA_RECONNECTING")
        self.assertEqual(diagnostic["technical"]["ip"], "192.168.0.5")
        self.assertNotIn("last_error", diagnostic)

    @patch("camera_diagnostics.socket.create_connection")
    def test_tcp_probe_success(self, create_connection):
        class DummyConnection:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        create_connection.return_value = DummyConnection()
        self.assertTrue(probe_tcp_port("192.168.0.5", 554, 0.1))

    @patch("camera_diagnostics.socket.create_connection", side_effect=OSError("down"))
    def test_tcp_probe_failure(self, _create_connection):
        self.assertFalse(probe_tcp_port("192.168.0.5", 554, 0.1))


if __name__ == "__main__":
    unittest.main()
