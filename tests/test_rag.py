"""RAG öz farkındalık ve erken çıkış mantığı testleri."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_logic import (
    NO_CONTEXT_MESSAGE,
    build_rag_system_prompt,
    execute_rag_query,
    is_identity_question,
    should_early_exit,
)
from retriever import VectorRetriever
from tests.conftest import MockEmbedder, make_unit_vector


class TestIdentityDetection:
    @pytest.mark.parametrize(
        "prompt",
        [
            "TechLas nedir?",
            "TechLas Workspace nedir?",
            "Sen kimsin?",
            "Bu sistem nasıl çalışır?",
        ],
    )
    def test_identity_questions_detected(self, prompt):
        assert is_identity_question(prompt) is True

    def test_document_questions_not_treated_as_identity(self):
        assert is_identity_question("Yüklediğim belgeler hakkında ne biliyorsun?") is False

    def test_unrelated_question_not_identity(self):
        assert is_identity_question("Kuantum fiziği nedir?") is False


class TestEarlyExitDecision:
    def test_identity_question_skips_early_exit_without_context(self):
        assert should_early_exit([], "TechLas nedir?") is False

    def test_unrelated_question_triggers_early_exit_without_context(self):
        assert should_early_exit([], "Kuantum fiziği nedir?") is True

    def test_search_results_prevent_early_exit(self):
        fake_doc = {"source": "a.txt", "text": "...", "score": 0.9}
        assert should_early_exit([fake_doc], "Rastgele soru") is False


class TestExecuteRagQuery:
    @pytest.fixture
    def mock_retriever(self):
        retriever = VectorRetriever(
            embedder=MockEmbedder(query_vector=make_unit_vector(1.0, 0.0))
        )
        retriever.search = MagicMock(return_value=[])
        return retriever

    @pytest.fixture
    def mock_chat_client(self):
        return MagicMock()

    def test_techlas_question_calls_llm_without_early_exit(
        self, mock_retriever, mock_chat_client
    ):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="TechLas yerel bir RAG asistanıdır."))]
        )
        complete_fn = MagicMock(return_value=mock_response)

        result = execute_rag_query(
            "TechLas nedir?",
            mock_retriever,
            mock_chat_client,
            complete_fn,
        )

        assert result["route"] == "llm"
        assert result["identity_mode"] is True
        assert result["answer"] == "TechLas yerel bir RAG asistanıdır."
        assert result["sources"] == []
        complete_fn.assert_called_once()
        messages = complete_fn.call_args.kwargs["messages"]
        assert "KİMLİK VE SİSTEM SORULARI" in messages[0]["content"]

    def test_unrelated_question_early_exit_without_llm(
        self, mock_retriever, mock_chat_client
    ):
        complete_fn = MagicMock()

        result = execute_rag_query(
            "Kuantum fiziği nedir?",
            mock_retriever,
            mock_chat_client,
            complete_fn,
        )

        assert result["route"] == "early_exit"
        assert result["answer"] == NO_CONTEXT_MESSAGE
        assert result["identity_mode"] is False
        complete_fn.assert_not_called()

    def test_identity_mode_prompt_allows_empty_context(self):
        prompt = build_rag_system_prompt("", identity_mode=True)
        assert "BAĞLAM boş olsa bile yanıt ver" in prompt

    def test_normal_mode_prompt_requires_context(self):
        prompt = build_rag_system_prompt("Örnek bağlam", identity_mode=False)
        assert "YALNIZCA aşağıdaki BAĞLAM" in prompt
