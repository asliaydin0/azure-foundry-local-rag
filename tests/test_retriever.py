"""retriever.py — vektör arama ve eşik filtreleme testleri."""

import numpy as np
import pytest

from retriever import SIMILARITY_THRESHOLD, VectorRetriever, cosine_similarity
from tests.conftest import MockEmbedder, insert_document_row, make_unit_vector


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestVectorSearch:
    @pytest.fixture
    def query_vector(self):
        return make_unit_vector(1.0, 0.0, 0.0)

    @pytest.fixture
    def retriever_with_mocks(self, query_vector):
        return VectorRetriever(embedder=MockEmbedder(query_vector=query_vector))

    def test_empty_database_returns_no_results(self, retriever_with_mocks, temp_db):
        results = retriever_with_mocks.search("herhangi bir soru")
        assert results == []

    def test_filters_results_below_threshold(self, temp_db, retriever_with_mocks, query_vector):
        high_vec = query_vector
        low_vec = make_unit_vector(0.0, 1.0, 0.0)  # skor ~0.0
        border_vec = make_unit_vector(0.56, 0.83, 0.0)  # skor ~0.56

        insert_document_row(
            temp_db,
            source="yuksek.txt",
            chunk_text="Yüksek benzerlik",
            vector=high_vec,
            embedder=retriever_with_mocks.embedder,
        )
        insert_document_row(
            temp_db,
            source="dusuk.txt",
            chunk_text="Düşük benzerlik",
            vector=low_vec,
            embedder=retriever_with_mocks.embedder,
        )
        insert_document_row(
            temp_db,
            source="sinir.txt",
            chunk_text="Eşik üstü",
            vector=border_vec,
            embedder=retriever_with_mocks.embedder,
        )

        results = retriever_with_mocks.search("test sorusu", top_k=5)

        assert all(r["score"] >= SIMILARITY_THRESHOLD for r in results)
        sources = {r["source"] for r in results}
        assert "yuksek.txt" in sources
        assert "sinir.txt" in sources
        assert "dusuk.txt" not in sources

    def test_returns_at_most_top_k_results(self, temp_db, retriever_with_mocks, query_vector):
        for i in range(4):
            vec = make_unit_vector(1.0 - i * 0.05, i * 0.05 + 0.01, 0.0)
            insert_document_row(
                temp_db,
                source=f"doc{i}.txt",
                chunk_text=f"Parça {i}",
                vector=vec,
                embedder=retriever_with_mocks.embedder,
            )

        results = retriever_with_mocks.search("test", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_descending_score(self, temp_db, retriever_with_mocks, query_vector):
        vectors = [
            make_unit_vector(0.6, 0.8, 0.0),
            make_unit_vector(0.9, 0.44, 0.0),
            make_unit_vector(0.7, 0.71, 0.0),
        ]
        for idx, vec in enumerate(vectors):
            insert_document_row(
                temp_db,
                source=f"s{idx}.txt",
                chunk_text=f"Metin {idx}",
                vector=vec,
                embedder=retriever_with_mocks.embedder,
            )

        results = retriever_with_mocks.search("test", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_custom_min_score_override(self, temp_db, retriever_with_mocks, query_vector):
        medium_vec = make_unit_vector(0.6, 0.8, 0.0)  # ~0.6
        insert_document_row(
            temp_db,
            source="orta.txt",
            chunk_text="Orta benzerlik",
            vector=medium_vec,
            embedder=retriever_with_mocks.embedder,
        )

        strict = retriever_with_mocks.search("test", min_score=0.95)
        relaxed = retriever_with_mocks.search("test", min_score=0.55)

        assert strict == []
        assert len(relaxed) == 1
        assert relaxed[0]["source"] == "orta.txt"
