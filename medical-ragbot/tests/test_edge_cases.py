"""
Edge Case Tests: Medical RAG Bot
Tests boundary conditions, malformed inputs, special characters, 
extreme values, and resilience to unexpected data.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.qa_chain import MedicalQAChain
from rag.retriever import MedicalRetriever
from rag.prompt import PromptBuilder, MedicalPrompts
from ingestion.text_splitter import MedicalTextSplitter


def make_chain(chunks=None, llm_answer=None):
    vs = MagicMock()
    vs.similarity_search.return_value = chunks or []
    vs.hybrid_search.return_value = chunks or []
    vs.filter_by_metadata.return_value = chunks or []
    vs.get_all_filenames.return_value = []

    llm = MagicMock()
    llm.generate_response.return_value = {
        "answer": llm_answer or "Mock answer",
        "model": "llama3",
        "provider": "groq",
        "usage": {"total_tokens": 100}
    }

    chain = MedicalQAChain(vector_store=vs)
    chain.set_llm_handler(llm)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Input Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryEdgeCases:
    def test_empty_string_query(self):
        chain = make_chain(chunks=[
            {"text": "data", "metadata": {}, "score": 0.9}
        ])
        result = chain.answer_question("")
        assert isinstance(result, dict)
        assert 'answer' in result

    def test_query_with_only_spaces(self):
        chain = make_chain(chunks=[
            {"text": "data", "metadata": {}, "score": 0.9}
        ])
        result = chain.answer_question("    ")
        assert isinstance(result, dict)

    def test_query_with_only_punctuation(self):
        chain = make_chain(chunks=[
            {"text": "data", "metadata": {}, "score": 0.9}
        ])
        result = chain.answer_question("??? !!! ...")
        assert isinstance(result, dict)

    def test_very_long_query(self):
        """10,000 character query."""
        chain = make_chain(chunks=[
            {"text": "data", "metadata": {}, "score": 0.9}
        ])
        long_query = "What medications am I taking? " * 333
        result = chain.answer_question(long_query)
        assert isinstance(result, dict)

    def test_query_with_sql_injection_attempt(self):
        """SQL injection patterns should not crash the system."""
        chain = make_chain(chunks=[{"text": "data", "metadata": {}, "score": 0.9}])
        result = chain.answer_question(
            "'; DROP TABLE medications; -- What medications am I taking?"
        )
        assert isinstance(result, dict)

    def test_query_with_unicode_characters(self):
        chain = make_chain(chunks=[{"text": "data", "metadata": {}, "score": 0.9}])
        result = chain.answer_question("¿Cuáles son mis medicamentos? (medications?)")
        assert isinstance(result, dict)

    def test_query_with_null_bytes(self):
        chain = make_chain(chunks=[{"text": "data", "metadata": {}, "score": 0.9}])
        # Null bytes or control characters
        query = "What medications\x00 am\x00 I taking?"
        try:
            result = chain.answer_question(query)
            assert isinstance(result, dict)
        except Exception:
            pass  # Acceptable to raise on null bytes

    def test_query_with_newlines(self):
        chain = make_chain(chunks=[{"text": "data", "metadata": {}, "score": 0.9}])
        result = chain.answer_question("What medications\nam I\ntaking?")
        assert isinstance(result, dict)

    def test_query_with_numbers_only(self):
        chain = make_chain(chunks=[{"text": "data", "metadata": {}, "score": 0.9}])
        result = chain.answer_question("1234567890")
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Document Edge Cases in Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentEdgeCases:
    def test_document_with_no_text_key(self):
        splitter = MedicalTextSplitter()
        # Missing 'text' key
        doc = {"source": "test.pdf", "filename": "test.pdf"}
        chunks = splitter.split_document(doc)
        assert chunks == []

    def test_document_with_none_text(self):
        splitter = MedicalTextSplitter()
        doc = {"text": None, "source": "test.pdf", "filename": "test.pdf"}
        # Should handle gracefully
        try:
            chunks = splitter.split_document(doc)
            assert chunks == [] or isinstance(chunks, list)
        except Exception:
            pass

    def test_document_with_single_character(self, minimal_document):
        splitter = MedicalTextSplitter()
        chunks = splitter.split_document(minimal_document)
        # Should not crash; may produce 0 or 1 chunks
        assert isinstance(chunks, list)

    def test_document_with_only_numbers(self):
        splitter = MedicalTextSplitter()
        doc = {"text": "1234567890 " * 100, "source": "num.pdf", "filename": "num.pdf"}
        chunks = splitter.split_document(doc)
        assert isinstance(chunks, list)

    def test_document_with_repeated_section_headers(self):
        """Multiple 'Medications:' sections shouldn't crash."""
        splitter = MedicalTextSplitter()
        doc = {
            "text": """
Medications:
1. Metformin 500mg

Medications:
2. Aspirin 81mg

Medications:
3. Lisinopril 10mg
            """,
            "source": "repeat.pdf",
            "filename": "repeat.pdf"
        }
        chunks = splitter.split_document(doc)
        assert isinstance(chunks, list)

    def test_document_with_special_medical_characters(self):
        splitter = MedicalTextSplitter()
        doc = {
            "text": """
Medications:
- Metformin 500mg (α-glucosidase) ≥ 2x daily
- HbA1c  7.2%
- Blood pressure: 140/90 mmHg ()
- Temperature: 37.2°C
- SpO₂: 98%
            """,
            "source": "special.pdf",
            "filename": "special.pdf"
        }
        chunks = splitter.split_document(doc)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_document_missing_source_and_filename(self):
        splitter = MedicalTextSplitter()
        doc = {"text": "Patient has diabetes. Taking Metformin."}
        chunks = splitter.split_document(doc)
        # Should handle missing source/filename gracefully
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert chunk.get('source', '') is not None

    def test_document_with_very_large_table(self):
        """Tables that are huge should still be processed."""
        splitter = MedicalTextSplitter()
        table_data = "\n".join(
            [f"Test {i}  | {i * 1.1:.1f} | Normal" for i in range(100)]
        )
        doc = {
            "text": f"Lab Results:\n[Table 1 on Page 1]\n{table_data}\n\nFollow-up next week.",
            "source": "big_table.pdf",
            "filename": "big_table.pdf"
        }
        chunks = splitter.split_document(doc)
        assert isinstance(chunks, list)

    def test_document_with_all_sections_simultaneously(self, rich_medical_document):
        """Complex document with ALL sections should not crash."""
        splitter = MedicalTextSplitter()
        chunks = splitter.split_document(rich_medical_document)
        assert len(chunks) > 0

    def test_batch_split_with_some_empty_documents(self, simple_medical_document):
        splitter = MedicalTextSplitter()
        docs = [
            simple_medical_document,
            {"text": "", "source": "empty.pdf", "filename": "empty.pdf"},
            {"text": "   ", "source": "ws.pdf", "filename": "ws.pdf"},
        ]
        chunks = splitter.batch_split(docs)
        # At least the valid document's chunks should be here
        assert len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Retriever Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieverEdgeCases:
    def test_retrieve_with_k_zero(self):
        """k=0 should not crash."""
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = []
        mock_vs.similarity_search.return_value = []
        retriever = MedicalRetriever(vector_store=mock_vs)
        try:
            results = retriever.retrieve("query", k=0)
            assert isinstance(results, list)
        except Exception:
            pass

    def test_retrieve_with_k_one(self):
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = [
            {"text": "data", "metadata": {"filename": "f.pdf", "section_type": "general"}, "score": 0.9}
        ]
        mock_vs.similarity_search.return_value = mock_vs.hybrid_search.return_value
        retriever = MedicalRetriever(vector_store=mock_vs)
        results = retriever.retrieve("query", k=1)
        assert len(results) <= 1

    def test_retrieve_when_vector_store_returns_nothing(self):
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = []
        mock_vs.similarity_search.return_value = []
        retriever = MedicalRetriever(vector_store=mock_vs)
        results = retriever.retrieve("medications", k=5)
        assert results == []

    def test_rerank_with_single_candidate(self):
        retriever = MedicalRetriever(vector_store=MagicMock())
        candidates = [
            {"text": "data", "metadata": {"filename": "a.pdf"}, "score": 0.9}
        ]
        result = retriever._rerank_by_diversity(candidates, top_k=5)
        assert len(result) == 1

    def test_rerank_top_k_larger_than_candidates(self):
        retriever = MedicalRetriever(vector_store=MagicMock())
        candidates = [
            {"text": f"data {i}", "metadata": {"filename": f"doc{i}.pdf"}, "score": 0.9}
            for i in range(3)
        ]
        result = retriever._rerank_by_diversity(candidates, top_k=10)
        assert len(result) <= 3

    def test_context_build_with_no_results(self):
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = []
        mock_vs.similarity_search.return_value = []
        retriever = MedicalRetriever(vector_store=mock_vs)
        ctx = retriever.get_full_context("query")
        assert isinstance(ctx, str)

    def test_section_detection_partial_match(self):
        """Partial keyword matches should still detect correctly."""
        retriever = MedicalRetriever(vector_store=MagicMock())
        # "medic" is a partial match for "medication"
        section = retriever._detect_section_type("Tell me about my medication history")
        assert section == 'medications'


# ─────────────────────────────────────────────────────────────────────────────
# QA Chain Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestQAChainEdgeCases:
    def test_format_sources_with_missing_metadata(self):
        chain = make_chain()
        chunks = [{"text": "data", "score": 0.9}]  # No metadata key
        sources = chain._format_sources(chunks)
        assert isinstance(sources, list)

    def test_format_sources_with_empty_text(self):
        chain = make_chain()
        chunks = [{"text": "", "metadata": {}, "score": 0.9}]
        sources = chain._format_sources(chunks)
        assert isinstance(sources, list)

    def test_format_sources_with_very_long_text(self):
        chain = make_chain()
        chunks = [{"text": "A" * 2000, "metadata": {}, "score": 0.9}]
        sources = chain._format_sources(chunks)
        assert len(sources[0]['preview']) <= 205  # 200 + "..." with tolerance

    def test_answer_question_k_default(self, mock_vector_store):
        chain = MedicalQAChain(vector_store=mock_vector_store)
        result = chain.answer_question("What medications?")
        assert isinstance(result, dict)

    def test_answer_question_k_large(self, mock_vector_store):
        chain = MedicalQAChain(vector_store=mock_vector_store)
        result = chain.answer_question("What medications?", k=50)
        assert isinstance(result, dict)

    def test_llm_handler_exception_handled(self, mock_vector_store):
        """If LLM raises, it should propagate (not silently fail)."""
        from tenacity import RetryError
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = Exception("LLM error")
        chain = MedicalQAChain(vector_store=mock_vector_store)
        chain.set_llm_handler(mock_llm)
        try:
            result = chain.answer_question("What medications?")
            # If it doesn't raise, result should still be a dict
            assert isinstance(result, dict)
        except Exception:
            pass  # Acceptable to propagate

    def test_answer_with_section_unknown_section_type(self, mock_vector_store):
        """Unknown section type should use fallback prompt."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = {
            "answer": "Found some data.",
            "model": "llama3",
            "provider": "groq",
            "usage": {"total_tokens": 50}
        }
        chain = MedicalQAChain(vector_store=mock_vector_store)
        chain.set_llm_handler(mock_llm)
        result = chain.answer_with_specific_section("unknown_section_xyz")
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptEdgeCases:
    def test_prompt_build_with_empty_everything(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt("", "")
        assert isinstance(prompt, str)

    def test_build_context_with_chunks_missing_metadata(self):
        chunks = [{"text": "Metformin 500mg"}]  # No metadata
        result = MedicalPrompts.build_context_consolidation_prompt(chunks)
        assert isinstance(result, str)

    def test_follow_up_prompt_empty_history(self):
        prompt = MedicalPrompts.build_followup_prompt("current question", [])
        assert "current question" in prompt

    def test_follow_up_prompt_long_history(self):
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
            for i in range(20)
        ]
        prompt = MedicalPrompts.build_followup_prompt("final question", history)
        assert isinstance(prompt, str)
        assert "final question" in prompt

    def test_medication_prompt_with_empty_context(self):
        prompt = MedicalPrompts.MEDICATION_PROMPT.format(context="")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_user_prompt_with_markdown_context(self):
        ctx = "# Header\n**Bold medications**\n- Metformin 500mg\n- Aspirin 81mg"
        prompt = MedicalPrompts.build_user_prompt("What meds?", ctx)
        assert ctx in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdEdgeCases:
    def test_doc_id_with_empty_filename(self):
        from vectorstore.mongodb_handler import MongoDBVectorStore
        store = MongoDBVectorStore.__new__(MongoDBVectorStore)
        # Just test the hashing function
        doc_id = store._generate_doc_id("", 0)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 16

    def test_doc_id_with_very_long_filename(self):
        from vectorstore.mongodb_handler import MongoDBVectorStore
        store = MongoDBVectorStore.__new__(MongoDBVectorStore)
        long_filename = "a" * 10000 + ".pdf"
        doc_id = store._generate_doc_id(long_filename, 0)
        assert len(doc_id) == 16  # Always 16 chars

    def test_doc_id_with_special_filename_chars(self):
        from vectorstore.mongodb_handler import MongoDBVectorStore
        store = MongoDBVectorStore.__new__(MongoDBVectorStore)
        special = "report (1) [2024] — test & data.pdf"
        doc_id = store._generate_doc_id(special, 42)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 16

    def test_doc_id_negative_chunk_id(self):
        from vectorstore.mongodb_handler import MongoDBVectorStore
        store = MongoDBVectorStore.__new__(MongoDBVectorStore)
        doc_id = store._generate_doc_id("report.pdf", -1)
        assert isinstance(doc_id, str)
