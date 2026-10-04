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

"""Pytest configuration and mocks for running test suites without heavyweight GPU dependencies."""

import sys
from unittest.mock import MagicMock

# Gracefully stub heavy accelerator/CUDA libraries for fast local unit testing
for module_name in ['torch', 'transformers', 'accelerate', 'faiss', 'fitz', 'PyMuPDF']:
    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ImportError:
            sys.modules[module_name] = MagicMock()
