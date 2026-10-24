# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import pytest

from app.routes import api


@pytest.fixture(autouse=True)
def fresh_quotas(monkeypatch):
    monkeypatch.setattr(api, "user_quotas", {})
    monkeypatch.setattr(api.settings, "MAX_DAILY_REQUESTS", 3)


def test_quota_allows_exactly_max_requests():
    results = [api.check_quota("key") for _ in range(4)]

    assert results == [True, True, True, False]


def test_first_request_is_counted():
    api.check_quota("key")

    assert api.user_quotas["key"]["count"] == 1
