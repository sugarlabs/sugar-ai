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

import sys
import types
from types import SimpleNamespace

# Keep this test lightweight while exercising RAGAgent.run end to end.
community = types.ModuleType("langchain_community")
vectorstores = types.ModuleType("langchain_community.vectorstores")
loaders = types.ModuleType("langchain_community.document_loaders")
vectorstores.FAISS = object
loaders.PyMuPDFLoader = object
loaders.TextLoader = object
community.vectorstores = vectorstores
community.document_loaders = loaders
sys.modules["langchain_community"] = community
sys.modules["langchain_community.vectorstores"] = vectorstores
sys.modules["langchain_community.document_loaders"] = loaders

huggingface = types.ModuleType("langchain_huggingface")
huggingface.HuggingFaceEmbeddings = object
sys.modules["langchain_huggingface"] = huggingface

from app.ai import RAGAgent
from app.context import estimate_tokens


class FakeProvider:
    def __init__(self):
        self.prompts = []

    def get_model_name(self):
        return "fake"

    def get_context_window(self):
        return 128

    def count_tokens(self, text):
        return estimate_tokens(text)

    def generate(self, prompt, params=None):
        self.prompts.append(prompt)
        return "Answer: bounded response"

    def get_eos_token(self):
        return None


def test_rag_retrieved_context_is_budgeted_before_generation():
    provider = FakeProvider()
    agent = RAGAgent(provider)
    agent.get_relevant_document = lambda question: (
        SimpleNamespace(page_content="documentation " * 1000),
        1.0,
    )

    agent.run("How does this work?")

    assert len(provider.prompts) == 2
    assert len(provider.prompts[0]) < 2500
