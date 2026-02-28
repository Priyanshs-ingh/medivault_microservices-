"""
Unit Tests: RAG Retriever (rag/retriever.py)
Tests section detection, retrieval logic, multi-stage retrieval, and diversity reranking.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import MedicalRetriever


def make_retriever(mock_vs=None):
    """Helper: create MedicalRetriever with optional mock vector store."""
    vs = mock_vs or MagicMock()
    return MedicalRetriever(vector_store=vs)


def make_chunks(filenames: list, section_types: list, scores: list = None) -> list:
    """Helper: create fake chunks for testing."""
    chunks = []
    for i, (fn, st) in enumerate(zip(filenames, section_types)):
        chunks.append({
            "text": f"Sample text from {fn} [{st}]",
            "metadata": {
                "filename": fn,
                "section_type": st,
                "chunk_id": i,
                "page": 1
            },
            "score": (scores[i] if scores else 0.9 - i * 0.05)
        })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionDetection:
    def test_detects_medications_query(self):
        r = make_retriever()
        assert r._detect_section_type("What medications am I taking?") == 'medications'

    def test_detects_medication_synonym_prescription(self):
        r = make_retriever()
        assert r._detect_section_type("Show my prescriptions") == 'medications'

    def test_detects_medication_synonym_drug(self):
        r = make_retriever()
        assert r._detect_section_type("List all my drugs") == 'medications'

    def test_detects_diagnosis_query(self):
        r = make_retriever()
        assert r._detect_section_type("What was I diagnosed with?") == 'diagnosis'

    def test_detects_lab_results_query(self):
        r = make_retriever()
        assert r._detect_section_type("Show me my lab results") == 'lab_results'

    def test_detects_vitals_query(self):
        r = make_retriever()
        assert r._detect_section_type("What was my blood pressure?") == 'vitals'

    def test_detects_allergies_query(self):
        r = make_retriever()
        assert r._detect_section_type("Do I have any allergies?") == 'allergies'

    def test_detects_symptoms_query(self):
        r = make_retriever()
        result = r._detect_section_type("What were my symptoms?")
        assert result == 'symptoms'

    def test_detects_procedures_query(self):
        r = make_retriever()
        # Use 'procedure' keyword - confirmed in retriever's section_patterns
        result = r._detect_section_type("What procedures did I have done?")
        assert result == 'procedures'


    def test_detects_follow_up_query(self):
        r = make_retriever()
        result = r._detect_section_type("What is the follow-up plan?")
        assert result == 'follow_up'

    def test_unknown_query_returns_none(self):
        r = make_retriever()
        assert r._detect_section_type("Hello there") is None

    def test_empty_query_returns_none(self):
        r = make_retriever()
        assert r._detect_section_type("") is None

    def test_case_insensitive_detection(self):
        r = make_retriever()
        assert r._detect_section_type("WHAT MEDICATIONS AM I TAKING?") == 'medications'


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve Method Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_retrieve_returns_results(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        results = r.retrieve("What medications am I taking?", k=5)
        assert isinstance(results, list)

    def test_retrieve_calls_hybrid_for_medical_query(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        r.retrieve("What medications am I taking?", k=5, use_hybrid=True)
        # Should have tried hybrid_search
        mock_vector_store.hybrid_search.assert_called()

    def test_retrieve_without_hybrid_calls_similarity(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        r.retrieve("random query", k=5, use_hybrid=False)
        mock_vector_store.similarity_search.assert_called()

    def test_retrieve_with_section_filter(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        r.retrieve("medications", k=5, section_filter="medications")
        # hybrid_search called with section filter
        call_args = mock_vector_store.hybrid_search.call_args
        assert call_args is not None

    def test_retrieve_fallback_when_insufficient_results(self):
        """If section-filtered results are sparse, falls back to broad search."""
        mock_vs = MagicMock()
        # First call (with section filter) returns only 1 result
        mock_vs.hybrid_search.return_value = [make_chunks(["f.pdf"], ["medications"])[0]]
        # Second call (without filter) returns more
        mock_vs.similarity_search.return_value = make_chunks(
            ["f.pdf", "g.pdf"], ["medications", "general"]
        )
        r = MedicalRetriever(vector_store=mock_vs)
        results = r.retrieve("What medications am I taking?", k=5)
        assert len(results) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve All In Section Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveAllInSection:
    def test_retrieve_all_calls_filter(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        r.retrieve_all_in_section("medications", limit=10)
        mock_vector_store.filter_by_metadata.assert_called_with(
            metadata_filter={'section_type': 'medications'},
            limit=10
        )

    def test_retrieve_all_returns_list(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        results = r.retrieve_all_in_section("medications")
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve From Document Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveFromDocument:
    def test_retrieve_from_specific_document(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        r.retrieve_from_document("medications", "report.pdf", k=3)
        mock_vector_store.similarity_search.assert_called_with(
            "medications",
            k=3,
            metadata_filter={'filename': 'report.pdf'}
        )

    def test_retrieve_from_document_returns_list(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        results = r.retrieve_from_document("query", "file.pdf")
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Stage Retrieval Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveMultiStage:
    def test_multi_stage_returns_results(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        results = r.retrieve_multi_stage("medications query", initial_k=20, final_k=5)
        assert isinstance(results, list)

    def test_multi_stage_returns_at_most_final_k(self, mock_vector_store):
        """Should not return more than final_k results."""
        # 10 results from vector store
        mock_vector_store.hybrid_search.return_value = make_chunks(
            [f"file{i}.pdf" for i in range(10)],
            ["medications"] * 10
        )
        mock_vector_store.similarity_search.return_value = \
            mock_vector_store.hybrid_search.return_value

        r = MedicalRetriever(vector_store=mock_vector_store)
        results = r.retrieve_multi_stage("query", initial_k=10, final_k=5)
        assert len(results) <= 5

    def test_multi_stage_with_few_candidates(self, mock_vector_store):
        """If fewer candidates than final_k, returns all without reranking."""
        mock_vector_store.hybrid_search.return_value = make_chunks(
            ["f.pdf"], ["medications"]
        )
        mock_vector_store.similarity_search.return_value = \
            mock_vector_store.hybrid_search.return_value

        r = MedicalRetriever(vector_store=mock_vector_store)
        results = r.retrieve_multi_stage("query", initial_k=20, final_k=5)
        assert len(results) == 1  # Only 1 candidate available


# ─────────────────────────────────────────────────────────────────────────────
# Diversity Reranking Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReranking:
    def test_rerank_distributes_across_documents(self):
        """Results should be spread across documents, not all from one."""
        r = make_retriever()
        candidates = make_chunks(
            ["a.pdf", "a.pdf", "a.pdf", "b.pdf", "b.pdf"],
            ["medications"] * 5
        )
        result = r._rerank_by_diversity(candidates, top_k=4)
        # Should have results from both a.pdf and b.pdf
        filenames = {c['metadata']['filename'] for c in result}
        assert len(filenames) > 1, "Reranking should use multiple documents"

    def test_rerank_returns_correct_count(self):
        r = make_retriever()
        candidates = make_chunks(
            ["a.pdf"] * 10, ["medications"] * 10
        )
        result = r._rerank_by_diversity(candidates, top_k=3)
        assert len(result) <= 3

    def test_rerank_empty_candidates(self):
        r = make_retriever()
        result = r._rerank_by_diversity([], top_k=5)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Context Building Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFullContext:
    def test_context_is_string(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        ctx = r.get_full_context("medications query", max_tokens=500)
        assert isinstance(ctx, str)

    def test_context_within_token_limit(self, mock_vector_store):
        max_tokens = 100
        r = make_retriever(mock_vector_store)
        ctx = r.get_full_context("query", max_tokens=max_tokens)
        # Rough: 1 token ≈ 4 chars
        assert len(ctx) <= max_tokens * 4 * 2  # Some tolerance

    def test_context_has_document_headers(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        ctx = r.get_full_context("medication query")
        assert "DOCUMENT:" in ctx or len(ctx) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Get All Documents Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllDocuments:
    def test_returns_list(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        docs = r.get_all_documents()
        assert isinstance(docs, list)

    def test_delegates_to_vector_store(self, mock_vector_store):
        r = make_retriever(mock_vector_store)
        r.get_all_documents()
        mock_vector_store.get_all_filenames.assert_called_once()
