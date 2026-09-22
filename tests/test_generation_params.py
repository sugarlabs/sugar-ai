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

"""Tests for provider generation parameters."""

import pytest

from app.providers.base import GenerationParams


def test_generation_params_have_expected_defaults():
    params = GenerationParams()

    assert params.max_new_tokens == 1024
    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k == 50
    assert params.repetition_penalty == 1.1
    assert params.truncation is True
    assert params.do_sample is True


@pytest.mark.parametrize(
    ("temperature", "requested_do_sample", "expected_do_sample"),
    [
        (0.0, True, False),
        (0.1, False, True),
        (1.0, False, True),
    ],
)
def test_temperature_controls_sampling(
    temperature,
    requested_do_sample,
    expected_do_sample,
):
    params = GenerationParams(
        temperature=temperature,
        do_sample=requested_do_sample,
    )

    assert params.do_sample is expected_do_sample
