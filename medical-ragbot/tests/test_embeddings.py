"""
Unit Tests: EmbeddingGenerator (ingestion/embeddings.py)
Tests initialization, text truncation, and batch embedding logic.
All network/model calls are mocked for fast, isolated tests.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingGeneratorInit:
    @patch("ingestion.embeddings.SentenceTransformer")
    def test_local_embedding_init(self, mock_st):
        """Local embeddings initialized when use_local_embeddings=True."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        assert gen.embedding_type == "local"
        assert gen.dimension == 768

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_dimension_set_from_model(self, mock_st):
        """Dimension is read from the model, not hardcoded."""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        assert gen.get_embedding_dimension() == 384


# ─────────────────────────────────────────────────────────────────────────────
# Text Truncation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTextTruncation:
    @patch("ingestion.embeddings.SentenceTransformer")
    def test_short_text_not_truncated(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        text = "Short text"
        result = gen._truncate_text(text, max_tokens=512)
        assert result == text

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_long_text_is_truncated(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        text = "A" * 10000
        result = gen._truncate_text(text, max_tokens=512)
        assert len(result) <= 512 * 4 + 20  # +20 for "[truncated...]"
        assert "[truncated...]" in result

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_exactly_max_length_not_truncated(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        text = "A" * (512 * 4)  # Exactly at limit
        result = gen._truncate_text(text, max_tokens=512)
        assert "[truncated...]" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Generate Embedding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEmbedding:
    @patch("ingestion.embeddings.SentenceTransformer")
    def test_returns_list_of_floats(self, mock_st):
        import numpy as np
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_instance.encode.return_value = np.ones(768)
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        embedding = gen.generate_embedding("Patient has diabetes")
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_empty_text_returns_zero_vector(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        embedding = gen.generate_embedding("")
        assert embedding == [0.0] * 768

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_whitespace_only_text_returns_zero_vector(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        embedding = gen.generate_embedding("   \n\t  ")
        assert embedding == [0.0] * 768


# ─────────────────────────────────────────────────────────────────────────────
# Batch Embedding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEmbeddingsBatch:
    @patch("ingestion.embeddings.SentenceTransformer")
    def test_batch_returns_list(self, mock_st):
        import numpy as np
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_instance.encode.return_value = np.ones((3, 768))
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        texts = ["text1", "text2", "text3"]
        embeddings = gen.generate_embeddings_batch(texts)
        assert isinstance(embeddings, list)

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_empty_batch_returns_empty(self, mock_st):
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        embeddings = gen.generate_embeddings_batch([])
        assert embeddings == []

    @patch("ingestion.embeddings.SentenceTransformer")
    def test_batch_filters_empty_strings(self, mock_st):
        """Empty strings in batch should be filtered out before encoding."""
        import numpy as np
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        mock_instance.encode.return_value = np.ones((2, 768))
        mock_st.return_value = mock_instance

        from ingestion.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        texts = ["valid text", "", "  ", "another valid text"]
        # Should only encode the valid texts
        gen.generate_embeddings_batch(texts)
        call_args = mock_instance.encode.call_args
        encoded_texts = call_args[0][0]
        assert "" not in encoded_texts
