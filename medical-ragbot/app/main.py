"""
Medical RAG Bot - Main API Application
FastAPI application for dynamic RAG system
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from pathlib import Path
import shutil
from datetime import datetime

from config import settings
from ingestion.pdf_loader import PDFProcessor
from ingestion.text_splitter import MedicalTextSplitter
from vectorstore.mongodb_handler import MongoDBVectorStore
from rag.llm_handler import MedicalLLMHandler
from rag.qa_chain import MedicalQAChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Bot API",
    description="Dynamic RAG system for medical document analysis using LLAMA 3",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
text_splitter = MedicalTextSplitter()
vector_store = MongoDBVectorStore()
llm_handler = MedicalLLMHandler()
qa_chain = MedicalQAChain(vector_store=vector_store)
qa_chain.set_llm_handler(llm_handler)


# ========== Request/Response Models ==========

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    conversation_history: Optional[List[Dict]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    metadata: Dict


class IngestionResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    extraction_method: str


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    section_distribution: Dict[str, int]
    documents: List[Dict]


# ========== API Endpoints ==========

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical RAG Bot API",
        "version": "2.0.0",
        "llm": settings.llm_model,
        "provider": settings.llm_provider,
        "status": "running",
        "features": [
            "Dynamic document ingestion",
            "Section-aware chunking",
            "LLAMA 3 generation",
            "Multi-document queries",
            "Semantic search"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check MongoDB connection
        stats = vector_store.get_stats()
        
        return {
            "status": "healthy",
            "vector_store": "connected",
            "documents": stats.get("total_documents", 0),
            "chunks": stats.get("total_chunks", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/ingest/upload", response_model=IngestionResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a single PDF file.
    
    Fully dynamic - handles any PDF regardless of size or content.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file to raw_pdfs directory
        upload_dir = Path(settings.raw_pdfs_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {file.filename}")
        
        # Process the PDF
        document = pdf_processor.extract_text_from_pdf(str(file_path))
        
        # Split into chunks
        chunks = text_splitter.split_document(document)
        
        # Add to vector store
        vector_store.add_documents(chunks)
        
        return IngestionResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(chunks),
            extraction_method=document.get("extraction_method", "unknown")
        )
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ingest/directory")
async def ingest_directory(directory_path: Optional[str] = None):
    """
    Ingest all PDFs from a directory.
    
    Perfect for batch processing - any number of documents.
    """
    dir_to_process = directory_path or settings.raw_pdfs_dir
    
    try:
        # Extract from directory
        documents = pdf_processor.extract_from_directory(dir_to_process)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No PDF files found in directory")
        
        # Split all documents
        all_chunks = text_splitter.batch_split(documents)
        
        # Add to vector store
        vector_store.add_documents(all_chunks)
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "total_chunks_created": len(all_chunks),
            "directory": dir_to_process
        }
    
    except Exception as e:
        logger.error(f"Directory ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about the medical documents.
    
    Dynamically searches across all ingested documents.
    """
    try:
        result = qa_chain.answer_question(
            question=request.question,
            k=request.k,
            conversation_history=request.conversation_history,
            use_multi_stage=True
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/section/{section_type}")
async def query_section(section_type: str, question: Optional[str] = None):
    """
    Query a specific section type across ALL documents.
    
    Examples:
    - /query/section/medications - Lists ALL medications
    - /query/section/diagnosis - Lists ALL diagnoses
    - /query/section/lab_results - Shows ALL lab results
    """
    try:
        result = qa_chain.answer_with_specific_section(
            section_type=section_type,
            question=question
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Section query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """
    List all ingested documents.
    
    Shows the dynamic corpus of available documents.
    """
    try:
        filenames = vector_store.get_all_filenames()
        
        return {
            "total_documents": len(filenames),
            "documents": filenames
        }
    
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get vector store statistics.
    
    Shows the current state of the RAG system.
    """
    try:
        stats = vector_store.get_stats()
        return StatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a specific document from the vector store.
    
    Removes all chunks associated with the document.
    """
    try:
        deleted_count = vector_store.delete_by_filename(filename)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "filename": filename,
            "chunks_deleted": deleted_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.delete("/clear")
async def clear_all():
    """
    Clear all documents from the vector store.
    
     WARNING: This cannot be undone!
    """
    try:
        deleted_count = vector_store.clear_collection()
        
        return {
            "status": "success",
            "chunks_deleted": deleted_count,
            "warning": "All documents cleared"
        }
    
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


# ========== Startup/Shutdown Events ==========

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("="*60)
    logger.info("Medical RAG Bot API Starting...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Vector Store: MongoDB")
    logger.info(f"Chunk Size: {settings.chunk_size}")
    logger.info(f"Chunk Overlap: {settings.chunk_overlap_percent * 100}%")
    logger.info("="*60)
    
    # Ensure data directories exist
    Path(settings.raw_pdfs_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_text_dir).mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Medical RAG Bot API shutting down...")


# ========== Run Server ==========

if __name__ == "__main__":
    import os
    import uvicorn

    # reload=True is development-only — must be False in Docker/production.
    # Set APP_RELOAD=true in local .env to enable hot-reload during development.
    _reload = os.getenv("APP_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=_reload,
        log_level="info"
    )
