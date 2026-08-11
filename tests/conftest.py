"""Pytest ortak fixture'ları ve test yardımcıları."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pytest

import database
import document_manager
import retriever


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Her test için izole SQLite veritabanı."""
    db_path = str(tmp_path / "test_knowledge_base.db")
    monkeypatch.setattr(database, "DB_NAME", db_path)
    monkeypatch.setattr(retriever, "DB_NAME", db_path)
    monkeypatch.setattr(document_manager, "DB_NAME", db_path)
    database.init_db(quiet=True)
    return db_path


@dataclass
class MockEmbedder:
    """Sabit vektörlerle embedding motorunu taklit eder."""

    query_vector: np.ndarray
    doc_vectors: dict[str, np.ndarray] | None = None

    def embed_text(self, text: str):
        if self.doc_vectors and text in self.doc_vectors:
            return self.doc_vectors[text].tolist()
        return self.query_vector.tolist()

    def vector_to_blob(self, vector_list):
        return np.array(vector_list, dtype=np.float32).tobytes()

    def blob_to_vector(self, blob_data):
        return np.frombuffer(blob_data, dtype=np.float32)


def insert_document_row(
    db_path: str,
    *,
    source: str,
    chunk_text: str,
    vector: np.ndarray,
    embedder: MockEmbedder,
) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (source_file, chunk_text, embedding_vector) VALUES (?, ?, ?)",
        (source, chunk_text, embedder.vector_to_blob(vector.tolist())),
    )
    conn.commit()
    conn.close()


def make_unit_vector(*components: float) -> np.ndarray:
    vec = np.array(components, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Sıfır vektör oluşturulamaz.")
    return vec / norm
