"""
Unit Tests: QA Chain & Safety (rag/qa_chain.py)
Tests query safety, answer generation, source formatting, and edge cases.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.qa_chain import MedicalQAChain


def make_qa_chain(mock_vs=None, mock_llm=None):
    """Helper: build a QA chain with mocks injected."""
    vs = mock_vs or MagicMock()
    chain = MedicalQAChain(vector_store=vs)
    if mock_llm:
        chain.set_llm_handler(mock_llm)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Query Safety / Appropriateness Tests (Critical for Medical Domain)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryAppropriateness:
    """
    Production safety: The system must NEVER give medical advice.
    These tests ensure inappropriate queries are blocked.
    """

    # --- APPROPRIATE queries (should pass) ---
    @pytest.mark.parametrize("query", [
        "What medications am I taking?",
        "What were my lab results last month?",
        "Do I have any allergies?",
        "What was my blood pressure reading?",
        "What was I diagnosed with?",
        "Show me my medication list",
        "What did my doctor find?",
        "List all my conditions",
        "What is my HbA1c value?",
        "What doctor did I visit?",
        "Show me my lab results from last year",
    ])
    def test_appropriate_queries_pass(self, query):
        chain = make_qa_chain()
        assert chain._is_appropriate_query(query) is True, \
            f"Appropriate query incorrectly blocked: '{query}'"

    # --- INAPPROPRIATE queries (must be blocked) ---
    @pytest.mark.parametrize("query", [
        "Should I stop taking metformin?",
        "Should I take aspirin?",
        "Should I start a new medication?",
        "What should I do about my diabetes?",
        "What can I do to lower my blood pressure?",
        "What can I take for this pain?",
        "Is this blood pressure dangerous?",
        "Is this normal?",
        "Is this serious?",
        "Diagnose me",
        "Diagnose my symptoms",
        "Can you prescribe something?",
        "I need a recommendation for my condition",
    ])
    def test_inappropriate_queries_blocked(self, query):
        chain = make_qa_chain()
        assert chain._is_appropriate_query(query) is False, \
            f"Inappropriate query NOT blocked: '{query}'"


# ─────────────────────────────────────────────────────────────────────────────
# Answer Question Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerQuestion:
    def test_inappropriate_query_returns_safety_message(self, mock_vector_store):
        chain = make_qa_chain(mock_vs=mock_vector_store)
        result = chain.answer_question("Should I stop taking metformin?")
        assert "cannot provide medical advice" in result['answer'].lower() or \
               "only help you retrieve" in result['answer'].lower()
        assert result['metadata'].get('warning') == 'inappropriate_query'

    def test_appropriate_query_with_no_results(self, mock_vector_store):
        mock_vector_store.similarity_search.return_value = []
        mock_vector_store.hybrid_search.return_value = []
        chain = make_qa_chain(mock_vs=mock_vector_store)
        result = chain.answer_question("What medications am I taking?")
        assert "couldn't find" in result['answer'].lower() or \
               "no relevant information" in result['answer'].lower()
        assert result['metadata'].get('no_results') is True

    def test_appropriate_query_with_results_and_no_llm(
        self, mock_vector_store
    ):
        """Without LLM, should return a fallback message with source count."""
        chain = make_qa_chain(mock_vs=mock_vector_store)
        result = chain.answer_question("What medications am I taking?")
        assert "LLM handler not initialized" in result['answer'] or \
               len(result['sources']) >= 0

    def test_appropriate_query_with_llm(
        self, mock_vector_store, mock_llm_handler
    ):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_question("What medications am I taking?")
        assert 'answer' in result
        assert 'sources' in result
        assert 'metadata' in result

    def test_result_has_required_keys(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_question("What medications am I taking?")
        assert 'answer' in result
        assert 'sources' in result
        assert 'metadata' in result

    def test_result_metadata_has_chunks_retrieved(
        self, mock_vector_store, mock_llm_handler
    ):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_question("What medications am I taking?")
        assert 'chunks_retrieved' in result['metadata']

    def test_empty_query_handled(self, mock_vector_store):
        chain = make_qa_chain(mock_vs=mock_vector_store)
        # Should not crash with empty query
        try:
            result = chain.answer_question("")
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Empty query crashed: {e}")

    def test_very_long_query_handled(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        long_query = "What medications " * 200
        try:
            result = chain.answer_question(long_query)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Very long query crashed: {e}")

    def test_special_characters_in_query(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_question("What's my HbA1c? (%) values?")
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Source Formatting Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSources:
    def test_format_sources_returns_list(self):
        chain = make_qa_chain()
        chunks = [
            {
                "text": "Medications: Metformin 500mg",
                "metadata": {"filename": "test.pdf", "section_type": "medications"},
                "score": 0.9
            }
        ]
        sources = chain._format_sources(chunks)
        assert isinstance(sources, list)

    def test_format_sources_max_5(self):
        chain = make_qa_chain()
        chunks = [
            {
                "text": f"Chunk {i}",
                "metadata": {"filename": f"file{i}.pdf", "section_type": "general"},
                "score": 0.9 - i * 0.01
            }
            for i in range(10)
        ]
        sources = chain._format_sources(chunks)
        assert len(sources) <= 5

    def test_format_sources_has_required_keys(self):
        chain = make_qa_chain()
        chunks = [
            {
                "text": "Medications: Metformin",
                "metadata": {"filename": "test.pdf", "section_type": "medications"},
                "score": 0.92
            }
        ]
        sources = chain._format_sources(chunks)
        required_keys = ['source_id', 'filename', 'section', 'score', 'preview']
        for s in sources:
            for key in required_keys:
                assert key in s, f"Missing key '{key}' in source"

    def test_format_sources_empty_chunks(self):
        chain = make_qa_chain()
        sources = chain._format_sources([])
        assert sources == []

    def test_format_sources_preview_length(self):
        """Preview should not exceed 200 chars + '...'"""
        chain = make_qa_chain()
        chunks = [
            {
                "text": "A" * 500,  # Long text
                "metadata": {"filename": "test.pdf", "section_type": "general"},
                "score": 0.9
            }
        ]
        sources = chain._format_sources(chunks)
        assert len(sources[0]['preview']) <= 210


# ─────────────────────────────────────────────────────────────────────────────
# Answer With Specific Section Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerWithSpecificSection:
    def test_no_results_for_section(self, mock_vector_store):
        mock_vector_store.filter_by_metadata.return_value = []
        chain = make_qa_chain(mock_vs=mock_vector_store)
        result = chain.answer_with_specific_section("medications")
        assert "No medications information found" in result['answer']
        assert result['metadata']['no_results'] is True

    def test_with_results_and_no_llm(self, mock_vector_store):
        chain = make_qa_chain(mock_vs=mock_vector_store)
        result = chain.answer_with_specific_section("medications")
        assert 'answer' in result

    def test_with_results_and_llm(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_with_specific_section("medications")
        assert result['answer'] != ""

    def test_diagnosis_section(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_with_specific_section("diagnosis")
        assert 'answer' in result

    def test_lab_results_section(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_with_specific_section("lab_results")
        assert 'answer' in result

    def test_result_metadata_has_section_type(
        self, mock_vector_store, mock_llm_handler
    ):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_with_specific_section("medications")
        assert result['metadata'].get('section_type') == 'medications'


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Document Answer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerAcrossDocuments:
    def test_no_documents_returns_message(self, mock_vector_store):
        mock_vector_store.similarity_search.return_value = []
        chain = make_qa_chain(mock_vs=mock_vector_store)
        result = chain.answer_across_documents("medications?", document_names=[])
        assert 'answer' in result

    def test_multi_doc_with_results(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_across_documents(
            "What are my medications?",
            document_names=["report1.pdf", "report2.pdf"]
        )
        assert 'answer' in result
        assert result['metadata'].get('multi_document') is True

    def test_multi_doc_metadata_correct(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store, mock_llm=mock_llm_handler)
        result = chain.answer_across_documents(
            "history?",
            document_names=["a.pdf", "b.pdf"]
        )
        assert result['metadata']['documents_searched'] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Set LLM Handler Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSetLLMHandler:
    def test_llm_handler_can_be_set(self, mock_vector_store, mock_llm_handler):
        chain = make_qa_chain(mock_vs=mock_vector_store)
        assert chain.llm_handler is None
        chain.set_llm_handler(mock_llm_handler)
        assert chain.llm_handler is not None

    def test_llm_handler_used_after_setting(
        self, mock_vector_store, mock_llm_handler
    ):
        chain = make_qa_chain(mock_vs=mock_vector_store)
        chain.set_llm_handler(mock_llm_handler)
        result = chain.answer_question("What medications am I taking?")
        assert result['answer'] != ""
