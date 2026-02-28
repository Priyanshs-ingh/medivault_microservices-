"""
Configuration settings for Medical RAG Bot
Supports LLAMA 3 and dynamic RAG
"""
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ========== LLM Configuration (LLAMA 3 - Production: 100% Local) ===========
    llm_provider: str = BaseSettings().model_config.get('LLM_PROVIDER', 'groq')
    llm_model: str = BaseSettings().model_config.get('LLM_MODEL', 'llama-3.1-8b-instant')
    llm_temperature: float = float(BaseSettings().model_config.get('LLM_TEMPERATURE', 0.1))
    llm_max_tokens: int = int(BaseSettings().model_config.get('LLM_MAX_TOKENS', 1500))

    # Ollama Configuration (for local LLAMA 3)
    ollama_base_url: str = BaseSettings().model_config.get('OLLAMA_BASE_URL', 'http://localhost:11434')

    # Cloud API Keys (if using cloud providers)
    together_api_key: Optional[str] = BaseSettings().model_config.get('TOGETHER_API_KEY', None)
    groq_api_key: Optional[str] = BaseSettings().model_config.get('GROQ_API_KEY', None)

    # ========== MongoDB Configuration ========== 
    mongodb_uri: str = BaseSettings().model_config.get('MONGODB_URI', 'mongodb://localhost:27017')
    mongodb_db_name: str = BaseSettings().model_config.get('MONGODB_DB_NAME', 'medical_ragbot')
    mongodb_collection_name: str = BaseSettings().model_config.get('MONGODB_COLLECTION_NAME', 'medical_vectors')
    
    # ========== Embedding Configuration (Production-Grade) ==========
    # Production: Use local embeddings for cost-free deployment
    use_local_embeddings: bool = True
    
    # Production model: bge-base-en-v1.5 (SOTA for retrieval, 768 dim)
    local_embedding_model: str = "BAAI/bge-base-en-v1.5"
    # BGE (BAAI General Embedding) - Best open-source model for semantic search
    # - 768 dimensions (better than MiniLM's 384)
    # - Optimized for retrieval tasks
    # - Strong performance on medical/technical text
    
    # OpenAI embeddings (only if use_local_embeddings=False)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 768  # Production standard (matches BGE)
    
    # ========== Chunking Configuration (Production-Optimized) ==========
    chunk_size: int = 600  # Production: Smaller chunks for precise retrieval
    chunk_overlap_percent: float = 0.25  # 25% overlap ensures no information loss
    use_sentence_boundaries: bool = True  # Production: Sentence-aware splitting
    
    @property
    def chunk_overlap(self) -> int:
        """Calculate chunk overlap based on percentage."""
        return int(self.chunk_size * self.chunk_overlap_percent)
    
    # ========== Data Directories ==========
    data_dir: str = "data"
    raw_pdfs_dir: str = "data/raw_pdfs"
    processed_text_dir: str = "data/processed_text"
    
    # ========== OCR Configuration ==========
    tesseract_path: Optional[str] = None   # Windows: C:\Program Files\Tesseract-OCR\tesseract.exe | Linux: /usr/bin/tesseract
    poppler_path: Optional[str] = None     # Windows: C:\poppler\...\bin | Linux: leave blank (in PATH)

    @property
    def poppler_path_or_none(self) -> Optional[str]:
        """Returns None if empty string so pdf2image uses system PATH (Linux/Docker)."""
        return self.poppler_path if self.poppler_path else None
    ocr_language: str = "eng"
    
    # ========== API Configuration ==========
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]  # Configure for production
    
    # ========== Vector Search Configuration (Production) ==========
    vector_index_name: str = "vector_index"
    vector_search_candidates_multiplier: int = 10
    retrieval_max_tokens: int = 3000
    
    # Document versioning (production requirement)
    enable_document_versioning: bool = True
    document_version: str = "v1.0"
    
    # ========== RAG Configuration ==========
    default_retrieval_k: int = 5  # Default number of chunks to retrieve
    multi_stage_initial_k: int = 20  # Initial candidates for multi-stage retrieval
    max_context_length: int = 3500  # Max characters for LLM context
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
