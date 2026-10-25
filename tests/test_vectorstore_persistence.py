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

"""Tests for FAISS index persistence in RAGAgent.setup_vectorstore."""

import json
import os
from unittest.mock import MagicMock, patch

from app.ai import FAISS_FINGERPRINT_FILE, RAGAgent
from app.config import settings


def make_agent():
    agent = object.__new__(RAGAgent)
    agent.retriever = None
    return agent


def make_doc_file(tmp_path, name="doc.txt", content="Sugar is a learning platform."):
    doc = tmp_path / name
    doc.write_text(content, encoding="utf-8")
    return str(doc)


def write_cache(index_dir, fingerprint):
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "index.faiss"), "wb") as f:
        f.write(b"fake-index")
    with open(os.path.join(index_dir, FAISS_FINGERPRINT_FILE), "w", encoding="utf-8") as f:
        json.dump(fingerprint, f)


def read_fingerprint(index_dir):
    with open(os.path.join(index_dir, FAISS_FINGERPRINT_FILE), encoding="utf-8") as f:
        return json.load(f)


def test_cache_miss_builds_and_persists_index(tmp_path, monkeypatch):
    index_dir = str(tmp_path / "faiss_index")
    doc_path = make_doc_file(tmp_path)
    monkeypatch.setattr(settings, "FAISS_INDEX_PATH", index_dir)
    monkeypatch.setattr(settings, "FAISS_REBUILD_INDEX", False)

    agent = make_agent()
    store = MagicMock()
    store.save_local.side_effect = lambda path: os.makedirs(path, exist_ok=True)

    with (
        patch("app.ai.HuggingFaceEmbeddings"),
        patch("app.ai.TextLoader") as loader_cls,
        patch("app.ai.FAISS") as faiss_cls,
    ):
        loader_cls.return_value.load.return_value = [MagicMock(page_content="doc")]
        faiss_cls.from_documents.return_value = store

        retriever = agent.setup_vectorstore([doc_path])

    assert retriever is store.as_retriever.return_value
    assert agent.retriever is retriever
    loader_cls.assert_called_once_with(doc_path)
    faiss_cls.from_documents.assert_called_once()
    faiss_cls.load_local.assert_not_called()
    store.save_local.assert_called_once_with(index_dir)

    fingerprint = read_fingerprint(index_dir)
    assert fingerprint == agent._fingerprint_sources([doc_path])
    assert fingerprint["files"][doc_path]["size"] > 0


def test_cache_hit_skips_document_loaders(tmp_path, monkeypatch):
    index_dir = str(tmp_path / "faiss_index")
    doc_path = make_doc_file(tmp_path)
    monkeypatch.setattr(settings, "FAISS_REBUILD_INDEX", False)

    agent = make_agent()
    write_cache(index_dir, agent._fingerprint_sources([doc_path]))

    store = MagicMock()
    with (
        patch("app.ai.HuggingFaceEmbeddings") as embeddings_cls,
        patch("app.ai.TextLoader") as loader_cls,
        patch("app.ai.PyMuPDFLoader") as pdf_loader_cls,
        patch("app.ai.FAISS") as faiss_cls,
    ):
        faiss_cls.load_local.return_value = store

        retriever = agent.setup_vectorstore([doc_path], index_path=index_dir)

    loader_cls.assert_not_called()
    pdf_loader_cls.assert_not_called()
    faiss_cls.from_documents.assert_not_called()
    faiss_cls.load_local.assert_called_once_with(
        index_dir,
        embeddings_cls.return_value,
        allow_dangerous_deserialization=True,
    )
    assert retriever is store.as_retriever.return_value
    assert agent.retriever is retriever


def test_stale_fingerprint_triggers_rebuild(tmp_path, monkeypatch):
    index_dir = str(tmp_path / "faiss_index")
    doc_path = make_doc_file(tmp_path, content="version one")
    monkeypatch.setattr(settings, "FAISS_REBUILD_INDEX", False)

    agent = make_agent()
    write_cache(index_dir, {"embedding_model": "stale-model", "files": {}})

    store = MagicMock()
    store.save_local.side_effect = lambda path: os.makedirs(path, exist_ok=True)

    with (
        patch("app.ai.HuggingFaceEmbeddings"),
        patch("app.ai.TextLoader") as loader_cls,
        patch("app.ai.FAISS") as faiss_cls,
    ):
        loader_cls.return_value.load.return_value = [MagicMock(page_content="v1")]
        faiss_cls.from_documents.return_value = store

        retriever = agent.setup_vectorstore([doc_path], index_path=index_dir)

    faiss_cls.load_local.assert_not_called()
    loader_cls.assert_called_once_with(doc_path)
    faiss_cls.from_documents.assert_called_once()
    store.save_local.assert_called_once_with(index_dir)
    assert retriever is store.as_retriever.return_value
    assert read_fingerprint(index_dir) == agent._fingerprint_sources([doc_path])


def test_corrupt_cache_falls_back_to_rebuild(tmp_path, monkeypatch):
    index_dir = str(tmp_path / "faiss_index")
    doc_path = make_doc_file(tmp_path)
    monkeypatch.setattr(settings, "FAISS_REBUILD_INDEX", False)

    agent = make_agent()
    write_cache(index_dir, agent._fingerprint_sources([doc_path]))

    store = MagicMock()
    store.save_local.side_effect = lambda path: os.makedirs(path, exist_ok=True)

    with (
        patch("app.ai.HuggingFaceEmbeddings"),
        patch("app.ai.TextLoader") as loader_cls,
        patch("app.ai.FAISS") as faiss_cls,
    ):
        faiss_cls.load_local.side_effect = RuntimeError("bad pickle")
        loader_cls.return_value.load.return_value = [MagicMock(page_content="doc")]
        faiss_cls.from_documents.return_value = store

        retriever = agent.setup_vectorstore([doc_path], index_path=index_dir)

    faiss_cls.load_local.assert_called_once()
    faiss_cls.from_documents.assert_called_once()
    assert retriever is store.as_retriever.return_value


def test_rebuild_flag_forces_rebuild(tmp_path, monkeypatch):
    index_dir = str(tmp_path / "faiss_index")
    doc_path = make_doc_file(tmp_path)
    monkeypatch.setattr(settings, "FAISS_REBUILD_INDEX", True)

    agent = make_agent()
    write_cache(index_dir, agent._fingerprint_sources([doc_path]))

    store = MagicMock()
    store.save_local.side_effect = lambda path: os.makedirs(path, exist_ok=True)

    with (
        patch("app.ai.HuggingFaceEmbeddings"),
        patch("app.ai.TextLoader") as loader_cls,
        patch("app.ai.FAISS") as faiss_cls,
    ):
        loader_cls.return_value.load.return_value = [MagicMock(page_content="doc")]
        faiss_cls.from_documents.return_value = store

        retriever = agent.setup_vectorstore([doc_path], index_path=index_dir)

    faiss_cls.load_local.assert_not_called()
    faiss_cls.from_documents.assert_called_once()
    assert retriever is store.as_retriever.return_value


def test_no_documents_leaves_retriever_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FAISS_REBUILD_INDEX", False)

    agent = make_agent()
    with (
        patch("app.ai.HuggingFaceEmbeddings"),
        patch("app.ai.FAISS") as faiss_cls,
    ):
        result = agent.setup_vectorstore(
            [str(tmp_path / "does-not-exist.txt")],
            index_path=str(tmp_path / "faiss_index"),
        )

    assert result is None
    assert agent.retriever is None
    faiss_cls.from_documents.assert_not_called()
    faiss_cls.load_local.assert_not_called()
