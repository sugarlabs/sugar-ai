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

"""Tests for API Quota Enforcement and Validation (Issue #143)."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app import create_app
from app.routes import api
from app.routes.api import check_quota, verify_api_key
from app.config import settings


class TestQuotaUnits:
    """Unit tests for quota calculation and reset logic."""

    @pytest.fixture(autouse=True)
    def setup_quota_state(self):
        api.user_quotas.clear()
        settings.MAX_DAILY_REQUESTS = 3
        self.test_key = "user_key_unit"
        self.admin_key = "admin_key_unit"

    def test_quota_decrements_properly(self):
        allowed1, remaining1, limit1 = check_quota(self.test_key)
        assert allowed1 is True
        assert remaining1 == 2
        assert limit1 == 3

        allowed2, remaining2, limit2 = check_quota(self.test_key)
        assert allowed2 is True
        assert remaining2 == 1
        assert limit2 == 3

        allowed3, remaining3, limit3 = check_quota(self.test_key)
        assert allowed3 is True
        assert remaining3 == 0
        assert limit3 == 3

        # Next call should exceed quota
        allowed4, remaining4, limit4 = check_quota(self.test_key)
        assert allowed4 is False
        assert remaining4 == 0
        assert limit4 == 3

    def test_quota_resets_on_new_day(self):
        # Fill up quota for yesterday
        yesterday = datetime.now().date() - timedelta(days=1)
        api.user_quotas[self.test_key] = {"count": 3, "date": yesterday}

        # Today's check should reset count
        allowed, remaining, limit = check_quota(self.test_key)
        assert allowed is True
        assert remaining == 2
        assert limit == 3
        assert api.user_quotas[self.test_key]["count"] == 1
        assert api.user_quotas[self.test_key]["date"] == datetime.now().date()

    def test_admin_bypasses_quota(self):
        admin_info = {"name": "admin_user", "can_change_model": True}
        for _ in range(10):
            allowed, remaining, limit = check_quota(self.admin_key, user_info=admin_info)
            assert allowed is True
            assert remaining == settings.MAX_DAILY_REQUESTS


class TestQuotaAPIIntegration:
    """Integration tests for FastAPI endpoints verifying 429 status and headers."""

    @pytest.fixture(autouse=True)
    def setup_api_and_client(self):
        self.app = create_app()
        self.client = TestClient(self.app)
        self.mock_agent = MagicMock()
        self.mock_provider = MagicMock()
        self.mock_agent.provider = self.mock_provider
        self.mock_agent.run.return_value = "Mock answer"
        self.mock_provider.generate.return_value = "Mock LLM answer"
        self.mock_agent.run_chat_completion.return_value = "Mock chat answer"
        self.mock_agent.run_with_custom_prompt.return_value = "Mock prompt answer"
        self.mock_agent.debug.return_value = "Mock debug answer"
        api.agent = self.mock_agent

        # Set up test keys
        self.user_key = "user_api_key_123"
        self.admin_key = "admin_api_key_456"
        settings.API_KEYS = {
            self.user_key: {"name": "Regular Student", "can_change_model": False},
            self.admin_key: {"name": "Admin User", "can_change_model": True},
        }
        settings.MAX_DAILY_REQUESTS = 2
        api.user_quotas.clear()

    def test_missing_api_key_returns_401(self):
        response = self.client.post("/ask?question=Hello")
        assert response.status_code == 401
        assert response.json()["detail"] == "API key is missing"

    def test_invalid_api_key_returns_401(self):
        response = self.client.post(
            "/ask?question=Hello",
            headers={"X-API-Key": "invalid_unknown_key"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"

    def test_valid_requests_decrement_and_set_headers(self):
        # 1st Request
        response1 = self.client.post(
            "/ask?question=What%20is%20Python",
            headers={"X-API-Key": self.user_key},
        )
        assert response1.status_code == 200
        assert response1.headers["X-Quota-Remaining"] == "1"
        assert response1.headers["X-Quota-Limit"] == "2"
        data1 = response1.json()
        assert data1["quota"]["remaining"] == 1
        assert data1["quota"]["total"] == 2

        # 2nd Request
        response2 = self.client.post(
            "/ask-llm?question=What%20is%20Python",
            headers={"X-API-Key": self.user_key},
        )
        assert response2.status_code == 200
        assert response2.headers["X-Quota-Remaining"] == "0"
        assert response2.headers["X-Quota-Limit"] == "2"
        data2 = response2.json()
        assert data2["quota"]["remaining"] == 0
        assert data2["quota"]["total"] == 2

    def test_exhausted_quota_returns_429_with_headers(self):
        # Exhaust 2 requests
        self.client.post("/ask?question=Q1", headers={"X-API-Key": self.user_key})
        self.client.post("/ask?question=Q2", headers={"X-API-Key": self.user_key})

        # 3rd Request -> 429 Too Many Requests
        response3 = self.client.post(
            "/ask?question=Q3",
            headers={"X-API-Key": self.user_key},
        )
        assert response3.status_code == 429
        assert "API quota exceeded" in response3.json()["detail"]
        assert response3.headers["X-Quota-Remaining"] == "0"
        assert response3.headers["X-Quota-Limit"] == "2"

    def test_admin_bypasses_quota_limits(self):
        # Make more requests than the limit
        for i in range(5):
            response = self.client.post(
                f"/ask?question=AdminQuery{i}",
                headers={"X-API-Key": self.admin_key},
            )
            assert response.status_code == 200
            assert response.headers["X-Quota-Remaining"] == "2"

    def test_prompted_chat_endpoint_quota(self):
        payload = {
            "chat": True,
            "messages": [{"role": "user", "content": "Hello bot"}],
        }
        # Request 1
        res1 = self.client.post(
            "/ask-llm-prompted",
            json=payload,
            headers={"X-API-Key": self.user_key},
        )
        assert res1.status_code == 200
        assert res1.headers["X-Quota-Remaining"] == "1"

        # Request 2
        res2 = self.client.post(
            "/ask-llm-prompted",
            json=payload,
            headers={"X-API-Key": self.user_key},
        )
        assert res2.status_code == 200
        assert res2.headers["X-Quota-Remaining"] == "0"

        # Request 3 -> 429
        res3 = self.client.post(
            "/ask-llm-prompted",
            json=payload,
            headers={"X-API-Key": self.user_key},
        )
        assert res3.status_code == 429

    def test_debug_endpoint_quota(self):
        # Request 1
        res1 = self.client.post(
            "/debug?code=print(1)&context=false",
            headers={"X-API-Key": self.user_key},
        )
        assert res1.status_code == 200
        assert res1.headers["X-Quota-Remaining"] == "1"

        # Request 2
        res2 = self.client.post(
            "/debug?code=print(2)&context=false",
            headers={"X-API-Key": self.user_key},
        )
        assert res2.status_code == 200
        assert res2.headers["X-Quota-Remaining"] == "0"

        # Request 3 -> 429
        res3 = self.client.post(
            "/debug?code=print(3)&context=false",
            headers={"X-API-Key": self.user_key},
        )
        assert res3.status_code == 429
