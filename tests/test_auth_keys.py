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
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for loading API keys from the database at startup."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.auth import load_db_keys
from app.config import settings
from app.database import APIKey, Base


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(settings, "API_KEYS", {"turned-off": {"name": "old", "can_change_model": False}})
    yield session
    session.close()


def test_keys_approved_on_the_admin_page_work_after_a_restart(db):
    db.add_all([
        APIKey(key="approved", name="teacher", approved=True, is_active=True, can_change_model=False),
        APIKey(key="admin", name="admin", approved=True, is_active=True, can_change_model=True),
        APIKey(key="waiting", name="new", approved=False, is_active=False),
        APIKey(key="turned-off", name="old", approved=True, is_active=False),
    ])
    db.commit()

    load_db_keys(db)

    assert settings.API_KEYS == {
        "approved": {"name": "teacher", "can_change_model": False},
        "admin": {"name": "admin", "can_change_model": True},
    }
