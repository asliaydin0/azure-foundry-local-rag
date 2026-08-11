"""document_manager.py — metin parçalama ve ingest testleri."""

import numpy as np

from document_manager import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, ingest_file
from tests.conftest import MockEmbedder


class TestChunkText:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "TechLas yerel RAG asistanı."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunks_respect_size_limit(self):
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE

    def test_overlap_preserves_continuity(self):
        text = "0123456789" * 120  # 1200 karakter
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-20:]
            assert tail in chunks[i + 1]

    def test_custom_chunk_size_and_overlap(self):
        text = "word " * 200
        chunk_size = 50
        overlap = 10
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        assert all(len(c) <= chunk_size for c in chunks)
        joined_coverage = sum(len(c) for c in chunks)
        assert joined_coverage >= len(text.strip()) - overlap * (len(chunks) - 1)


class TestIngestFile:
    def test_ingest_file_stores_chunks_with_mock_embedder(self, temp_db, tmp_path, monkeypatch):
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        monkeypatch.setattr("document_manager.DOCS_DIR", str(docs_dir))

        sample = "Parça bir. " * 150
        filepath = docs_dir / "ornek.txt"
        filepath.write_text(sample, encoding="utf-8")

        embedder = MockEmbedder(query_vector=np.array([1.0, 0.0], dtype=np.float32))
        count = ingest_file(str(filepath), embedder, source_name="ornek.txt")

        expected_chunks = len(chunk_text(sample))
        assert count == expected_chunks
        assert expected_chunks > 1

    def test_ingest_empty_file_returns_zero(self, temp_db, tmp_path, monkeypatch):
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        monkeypatch.setattr("document_manager.DOCS_DIR", str(docs_dir))

        filepath = docs_dir / "bos.txt"
        filepath.write_text("   ", encoding="utf-8")

        embedder = MockEmbedder(query_vector=np.array([1.0, 0.0], dtype=np.float32))
        assert ingest_file(str(filepath), embedder) == 0
