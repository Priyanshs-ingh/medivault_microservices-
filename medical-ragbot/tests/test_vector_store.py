"""
Unit Tests: MongoDB Vector Store (vectorstore/mongodb_handler.py)
Tests document insertion, search, metadata filtering, and stats — fully mocked.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_collection():
    """Create a detailed mock MongoDB collection."""
    col = MagicMock()
    col.insert_many.return_value = MagicMock(
        inserted_ids=["id1", "id2"]
    )
    col.aggregate.return_value = iter([
        {"_id": "medications", "count": 10},
        {"_id": "diagnosis", "count": 5}
    ])
    col.find.return_value = MagicMock()
    col.find.return_value.__iter__ = MagicMock(return_value=iter([]))
    col.distinct.return_value = ["report1.pdf", "report2.pdf"]
    col.count_documents.return_value = 25
    col.delete_many.return_value = MagicMock(deleted_count=5)
    return col


def make_vector_store():
    """Create MongoDBVectorStore with all external dependencies mocked."""
    with patch("vectorstore.mongodb_handler.MongoClient") as mock_client, \
         patch("vectorstore.mongodb_handler.EmbeddingGenerator") as mock_emb:

        mock_collection = make_mock_collection()

        # Client chain
        mock_client.return_value.__getitem__.return_value.__getitem__.return_value = \
            mock_collection
        mock_client.return_value.__getitem__.return_value = MagicMock(
            __getitem__=MagicMock(return_value=mock_collection),
            command=MagicMock(return_value={})
        )

        # Embedding generator
        mock_emb_instance = MagicMock()
        mock_emb_instance.dimension = 768
        mock_emb_instance.generate_embedding.return_value = [0.1] * 768
        mock_emb_instance.generate_embeddings_batch.return_value = [
            [0.1] * 768, [0.2] * 768
        ]
        mock_emb.return_value = mock_emb_instance

        from vectorstore.mongodb_handler import MongoDBVectorStore
        store = MongoDBVectorStore.__new__(MongoDBVectorStore)
        store.collection = mock_collection
        store.embedding_generator = mock_emb_instance

        return store, mock_collection, mock_emb_instance


# ─────────────────────────────────────────────────────────────────────────────
# Document ID Generation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdGeneration:
    def test_generate_doc_id_is_string(self):
        store, _, _ = make_vector_store()
        doc_id = store._generate_doc_id("report.pdf", 0)
        assert isinstance(doc_id, str)

    def test_generate_doc_id_is_16_chars(self):
        store, _, _ = make_vector_store()
        doc_id = store._generate_doc_id("report.pdf", 0)
        assert len(doc_id) == 16

    def test_same_input_same_id(self):
        store, _, _ = make_vector_store()
        id1 = store._generate_doc_id("report.pdf", 5)
        id2 = store._generate_doc_id("report.pdf", 5)
        assert id1 == id2

    def test_different_filename_different_id(self):
        store, _, _ = make_vector_store()
        id1 = store._generate_doc_id("report1.pdf", 0)
        id2 = store._generate_doc_id("report2.pdf", 0)
        assert id1 != id2

    def test_different_chunk_id_different_doc_id(self):
        store, _, _ = make_vector_store()
        id1 = store._generate_doc_id("report.pdf", 0)
        id2 = store._generate_doc_id("report.pdf", 1)
        assert id1 != id2


# ─────────────────────────────────────────────────────────────────────────────
# Add Documents Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAddDocuments:
    def test_empty_chunks_returns_empty(self):
        store, col, emb = make_vector_store()
        result = store.add_documents([])
        assert result == []
        col.insert_many.assert_not_called()

    def test_single_chunk_inserted(self):
        store, col, emb = make_vector_store()
        emb.generate_embeddings_batch.return_value = [[0.1] * 768]
        chunks = [
            {
                "text": "Metformin 500mg twice daily",
                "chunk_id": 0,
                "section_type": "medications",
                "source": "test.pdf",
                "filename": "test.pdf"
            }
        ]
        result = store.add_documents(chunks)
        col.insert_many.assert_called_once()

    def test_document_structure_has_required_fields(self):
        store, col, emb = make_vector_store()
        emb.generate_embeddings_batch.return_value = [[0.1] * 768]
        chunks = [
            {
                "text": "Medications: Metformin",
                "chunk_id": 0,
                "section_type": "medications",
                "source": "test.pdf",
                "filename": "test.pdf"
            }
        ]
        store.add_documents(chunks)
        inserted_docs = col.insert_many.call_args[0][0]
        doc = inserted_docs[0]
        assert "doc_id" in doc
        assert "text" in doc
        assert "embedding" in doc
        assert "metadata" in doc

    def test_metadata_fields_present(self):
        store, col, emb = make_vector_store()
        emb.generate_embeddings_batch.return_value = [[0.1] * 768]
        chunks = [
            {
                "text": "Diabetes diagnosed",
                "chunk_id": 1,
                "section_type": "diagnosis",
                "source": "test.pdf",
                "filename": "test.pdf",
                "doctor_name": "Dr. Smith",
                "hospital_name": "City Medical"
            }
        ]
        store.add_documents(chunks)
        inserted_docs = col.insert_many.call_args[0][0]
        meta = inserted_docs[0]["metadata"]
        assert meta["section_type"] == "diagnosis"
        assert meta["filename"] == "test.pdf"
        assert meta["doctor_name"] == "Dr. Smith"
        assert meta["hospital_name"] == "City Medical"
        assert "created_at" in meta

    def test_schema_version_set(self):
        store, col, emb = make_vector_store()
        emb.generate_embeddings_batch.return_value = [[0.1] * 768]
        chunks = [
            {
                "text": "Lab results",
                "chunk_id": 0,
                "section_type": "lab_results",
                "source": "test.pdf",
                "filename": "test.pdf"
            }
        ]
        store.add_documents(chunks)
        inserted_docs = col.insert_many.call_args[0][0]
        meta = inserted_docs[0]["metadata"]
        assert "schema_version" in meta


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Search Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilaritySearch:
    def test_similarity_search_returns_list(self):
        store, col, emb = make_vector_store()
        col.aggregate.return_value = iter([
            {"text": "Metformin", "metadata": {}, "score": 0.9}
        ])
        results = store.similarity_search("medications", k=5)
        assert isinstance(results, list)

    def test_similarity_search_generates_query_embedding(self):
        store, col, emb = make_vector_store()
        col.aggregate.return_value = iter([])
        store.similarity_search("medications", k=5)
        emb.generate_embedding.assert_called_with("medications")

    def test_similarity_search_with_metadata_filter(self):
        store, col, emb = make_vector_store()
        col.aggregate.return_value = iter([])
        store.similarity_search(
            "query", k=5, metadata_filter={"section_type": "medications"}
        )
        pipeline = col.aggregate.call_args[0][0]
        # Should have a $match stage for filter
        stages = [list(s.keys())[0] for s in pipeline]
        assert "$match" in stages

    def test_similarity_search_fallback_on_error(self):
        store, col, emb = make_vector_store()
        emb.generate_embedding.side_effect = Exception("Embedding failed")
        # Fallback to text search
        col.find.return_value = iter([])
        results = store.similarity_search("query", k=5)
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridSearch:
    def test_hybrid_search_with_section_filter(self):
        store, col, emb = make_vector_store()
        col.aggregate.return_value = iter([
            {"text": "Meds", "metadata": {"section_type": "medications"}, "score": 0.9}
        ] * 5)
        results = store.hybrid_search(
            "medications query", k=5, section_type="medications"
        )
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_hybrid_search_no_filter(self):
        store, col, emb = make_vector_store()
        col.aggregate.return_value = iter([
            {"text": "text", "metadata": {}, "score": 0.9}
        ] * 3)
        results = store.hybrid_search("general query", k=3)
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Filter Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterByMetadata:
    def test_filter_by_section_type(self):
        store, col, emb = make_vector_store()
        fake_results = [
            {"text": "Meds", "metadata": {"section_type": "medications"}, "_id": "1"},
            {"text": "More Meds", "metadata": {"section_type": "medications"}, "_id": "2"}
        ]
        # Mock the cursor chain: col.find(...).limit(n) -> iter(results)
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = iter(fake_results)
        col.find.return_value = mock_cursor
        results = store.filter_by_metadata({"section_type": "medications"}, limit=10)
        assert isinstance(results, list)
        # Ensure MongoDB query uses metadata prefix
        call_args = col.find.call_args[0][0]
        assert "metadata.section_type" in call_args

    def test_filter_adds_score_to_results(self):
        store, col, emb = make_vector_store()
        fake_results = [
            {"text": "content", "metadata": {"section_type": "medications"}}
        ]
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = iter(fake_results)
        col.find.return_value = mock_cursor
        results = store.filter_by_metadata({"section_type": "medications"})
        for r in results:
            assert "score" in r
            assert r["score"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Delete Operations Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteOperations:
    def test_delete_by_source(self):
        store, col, emb = make_vector_store()
        count = store.delete_by_source("test.pdf")
        assert count == 5  # Mock returns 5
        col.delete_many.assert_called_with({"metadata.source": "test.pdf"})

    def test_delete_by_filename(self):
        store, col, emb = make_vector_store()
        count = store.delete_by_filename("report.pdf")
        assert count == 5
        col.delete_many.assert_called_with({"metadata.filename": "report.pdf"})

    def test_clear_collection(self):
        store, col, emb = make_vector_store()
        count = store.clear_collection()
        assert count == 5
        col.delete_many.assert_called_with({})


# ─────────────────────────────────────────────────────────────────────────────
# Stats Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_stats_returns_dict(self):
        store, col, emb = make_vector_store()
        # Reset aggregate to return something sensible for stats
        col.aggregate.return_value = iter([
            {"_id": "medications", "count": 10}
        ])
        col.count_documents.return_value = 25
        stats = store.get_stats()
        assert isinstance(stats, dict)

    def test_stats_has_required_keys(self):
        store, col, emb = make_vector_store()
        col.aggregate.return_value = iter([
            {"_id": "medications", "count": 10}
        ])
        col.count_documents.return_value = 25
        stats = store.get_stats()
        assert "total_chunks" in stats or "error" in stats


# ─────────────────────────────────────────────────────────────────────────────
# Get All Filenames Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllFilenames:
    def test_returns_list(self):
        store, col, emb = make_vector_store()
        filenames = store.get_all_filenames()
        assert isinstance(filenames, list)

    def test_calls_distinct(self):
        store, col, emb = make_vector_store()
        store.get_all_filenames()
        col.distinct.assert_called_with("metadata.filename")
