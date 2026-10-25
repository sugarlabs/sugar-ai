# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import hashlib
import hmac
import subprocess
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.responses import JSONResponse

import app.routes.webhook as webhook_module
from app.routes.webhook import verify_github_signature, webhook


class TestWebhook(unittest.TestCase):
    """Test suite for the GitHub Webhook endpoint security and functionality."""

    def setUp(self):
        self.secret = "test-webhook-secret-key"
        self.repo_path = "/home/sugar/sugar-ai"
        self.git_path = "/usr/bin/git"

        webhook_module.WEBHOOK_SECRET = self.secret
        webhook_module.REPO_PATH_LOCALLY = self.repo_path
        webhook_module.GIT_PATH = self.git_path
        webhook_module._webhook_configured = True

    def _generate_signature(self, payload: bytes) -> str:
        mac = hmac.new(self.secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
        return f"sha256={mac.hexdigest()}"

    def test_verify_github_signature_valid(self):
        """Test that valid HMAC signatures pass verification."""
        payload = b'{"ref": "refs/heads/main"}'
        signature = self._generate_signature(payload)
        self.assertTrue(verify_github_signature(payload, signature))

    def test_verify_github_signature_invalid(self):
        """Test that invalid signatures are rejected."""
        payload = b'{"ref": "refs/heads/main"}'
        self.assertFalse(verify_github_signature(payload, "sha256=invalid_hash_value"))
        self.assertFalse(verify_github_signature(payload, "sha1=invalid_algorithm"))
        self.assertFalse(verify_github_signature(payload, "invalid_format_without_equal"))
        self.assertFalse(verify_github_signature(payload, "sha256="))
        self.assertFalse(verify_github_signature(payload, ""))

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_successful_execution_security(self, mock_subprocess_run):
        """Verify safe execution of subprocess.run without shell=True, with timeout=30 and cwd set."""
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="success", stderr="")

        payload = b'{"ref": "refs/heads/main"}'
        signature = self._generate_signature(payload)

        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": signature}

        response = asyncio.run(webhook(mock_request))

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 200)

        # Ensure all 3 commands were executed with list arguments, timeout=30, and NO shell=True
        self.assertEqual(mock_subprocess_run.call_count, 3)

        expected_calls = [
            (([self.git_path, "fetch", "origin", "main"],), {"cwd": self.repo_path, "capture_output": True, "text": True, "check": True, "timeout": 30}),
            (([self.git_path, "reset", "--hard", "origin/main"],), {"cwd": self.repo_path, "capture_output": True, "text": True, "check": True, "timeout": 30}),
            ((["sudo", "systemctl", "restart", "sugarai"],), {"capture_output": True, "text": True, "check": True, "timeout": 30}),
        ]

        for idx, (call_args, call_kwargs) in enumerate(mock_subprocess_run.call_args_list):
            exp_args, exp_kwargs = expected_calls[idx]
            self.assertEqual(call_args, exp_args)
            for k, v in exp_kwargs.items():
                self.assertEqual(call_kwargs.get(k), v)
            # Crucial security check: shell MUST NOT be True
            self.assertFalse(call_kwargs.get("shell", False))

    @patch("app.routes.webhook.subprocess.run")
    def test_security_subprocess_never_called_on_auth_failure(self, mock_subprocess_run):
        """Security test asserting that subprocess.run is NEVER called on invalid or missing signatures."""
        payload = b'{"ref": "refs/heads/main"}'

        # 1. Missing header
        mock_request_missing = MagicMock()
        mock_request_missing.body = AsyncMock(return_value=payload)
        mock_request_missing.headers = {}
        with self.assertRaises(HTTPException):
            asyncio.run(webhook(mock_request_missing))

        # 2. Malformed header
        mock_request_malformed = MagicMock()
        mock_request_malformed.body = AsyncMock(return_value=payload)
        mock_request_malformed.headers = {"X-Hub-Signature-256": "malformed_no_equals"}
        with self.assertRaises(HTTPException):
            asyncio.run(webhook(mock_request_malformed))

        # 3. Invalid signature
        mock_request_invalid = MagicMock()
        mock_request_invalid.body = AsyncMock(return_value=payload)
        mock_request_invalid.headers = {"X-Hub-Signature-256": "sha256=invalid"}
        with self.assertRaises(HTTPException):
            asyncio.run(webhook(mock_request_invalid))

        # Ensure no subprocess was ever invoked during authentication failures
        mock_subprocess_run.assert_not_called()

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_git_fetch_called_process_error(self, mock_subprocess_run):
        """Test error handling when git fetch raises CalledProcessError."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="git fetch", stderr="Connection timed out"
        )

        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": self._generate_signature(payload)}

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("Git fetch failed", str(response.body))

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_git_fetch_timeout_expired(self, mock_subprocess_run):
        """Test error handling when git fetch raises TimeoutExpired."""
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(
            cmd="git fetch", timeout=30
        )

        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": self._generate_signature(payload)}

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("Git fetch timed out", str(response.body))

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_git_reset_called_process_error(self, mock_subprocess_run):
        """Test error handling when git reset raises CalledProcessError."""
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            subprocess.CalledProcessError(returncode=1, cmd="git reset", stderr="Cannot lock ref"),
        ]

        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": self._generate_signature(payload)}

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("Git reset failed", str(response.body))

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_git_reset_timeout_expired(self, mock_subprocess_run):
        """Test error handling when git reset raises TimeoutExpired."""
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            subprocess.TimeoutExpired(cmd="git reset", timeout=30),
        ]

        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": self._generate_signature(payload)}

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("Git reset timed out", str(response.body))

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_service_restart_called_process_error(self, mock_subprocess_run):
        """Test error handling when systemctl restart raises CalledProcessError."""
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout=""),
            subprocess.CalledProcessError(returncode=1, cmd="systemctl restart", stderr="Unit not found"),
        ]

        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": self._generate_signature(payload)}

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("Service restart failed", str(response.body))

    @patch("app.routes.webhook.subprocess.run")
    def test_webhook_service_restart_timeout_expired(self, mock_subprocess_run):
        """Test error handling when systemctl restart raises TimeoutExpired."""
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout=""),
            subprocess.TimeoutExpired(cmd="systemctl restart", timeout=30),
        ]

        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": self._generate_signature(payload)}

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 500)
        self.assertIn("Service restart timed out", str(response.body))

    def test_webhook_unconfigured(self):
        """Test that unconfigured webhook returns 503 Service Unavailable."""
        webhook_module._webhook_configured = False
        mock_request = MagicMock()

        response = asyncio.run(webhook(mock_request))
        self.assertEqual(response.status_code, 503)
        self.assertIn("Webhook is not configured", str(response.body))

    def test_webhook_missing_signature_raises_403(self):
        """Test that missing signature header raises 403 Forbidden."""
        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {}

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(webhook(mock_request))
        self.assertEqual(ctx.exception.status_code, 403)

    def test_webhook_invalid_signature_raises_403(self):
        """Test that invalid signature raises 403 Forbidden."""
        payload = b'{"ref": "refs/heads/main"}'
        mock_request = MagicMock()
        mock_request.body = AsyncMock(return_value=payload)
        mock_request.headers = {"X-Hub-Signature-256": "sha256=invalid_signature"}

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(webhook(mock_request))
        self.assertEqual(ctx.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
