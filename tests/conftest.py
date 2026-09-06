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

"""Test-only stubs for optional ML dependencies."""
import sys
import types
from types import SimpleNamespace


if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16 = object()
    torch.float32 = object()
    sys.modules["torch"] = torch


if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.pipeline = object
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.BitsAndBytesConfig = object
    sys.modules["transformers"] = transformers
