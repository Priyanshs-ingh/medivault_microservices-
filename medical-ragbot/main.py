
"""
Medical RAG Pipeline - Main Entry Point
Simple script to run ingestion and queries
"""
import sys
from pathlib import Path

# Load environment variables from .env before any config import
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.pdf_loader import PDFProcessor
from ingestion.text_splitter import MedicalTextSplitter
from vectorstore.mongodb_handler import MongoDBVectorStore
from rag.llm_handler import MedicalLLMHandler
from rag.qa_chain import MedicalQAChain
from config import settings


class MedicalRAGPipeline:
    """
    End-to-end RAG pipeline for medical documents.
    
    Fully dynamic:
    - Any number of PDFs
    - Any number of pages
    - Automatic section detection
    - Adaptive retrieval
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("Medical RAG Bot - Dynamic RAG Pipeline")
        print("="*60)
        
        self.pdf_processor = PDFProcessor()
        self.text_splitter = MedicalTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap_percent=settings.chunk_overlap_percent
        )
        self.vector_store = MongoDBVectorStore()
        self.llm_handler = MedicalLLMHandler()
        self.qa_chain = MedicalQAChain(vector_store=self.vector_store)
        self.qa_chain.set_llm_handler(self.llm_handler)
        
        print(f" LLM: {settings.llm_model} ({settings.llm_provider})")
        print(f" Chunk size: {settings.chunk_size}")
        print(f" Overlap: {int(settings.chunk_overlap_percent * 100)}%")
        print("="*60 + "\n")
    
    def ingest_pdf(self, pdf_path: str):
        """Ingest a single PDF."""
        print(f"Ingesting: {Path(pdf_path).name}")
        
        # Extract text
        document = self.pdf_processor.extract_text_from_pdf(pdf_path)
        print(f"   Extracted text ({document['extraction_method']})")
        
        # Split into chunks
        chunks = self.text_splitter.split_document(document)
        print(f"   Created {len(chunks)} chunks")
        
        # Add to vector store
        ids = self.vector_store.add_documents(chunks)
        print(f"   Added to vector store ({len(ids)} chunks)")
        
        return {
            "filename": document["filename"],
            "chunks": len(chunks),
            "method": document["extraction_method"]
        }
    
    def ingest_directory(self, directory: str = None):
        """Ingest all PDFs from a directory."""
        dir_path = directory or settings.raw_pdfs_dir
        
        print(f"Ingesting all PDFs from: {dir_path}\n")
        
        # Extract all PDFs
        documents = self.pdf_processor.extract_from_directory(dir_path)
        
        if not documents:
            print(" No PDFs found in directory")
            return []
        
        print(f"Found {len(documents)} PDFs\n")
        
        # Split all documents
        all_chunks = self.text_splitter.batch_split(documents)
        print(f" Created {len(all_chunks)} total chunks\n")
        
        # Add to vector store
        ids = self.vector_store.add_documents(all_chunks)
        print(f" Added {len(ids)} chunks to vector store\n")
        
        return {
            "documents": len(documents),
            "chunks": len(all_chunks)
        }
    
    def query(self, question: str, k: int = 5):
        """Ask a question about the medical documents."""
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        result = self.qa_chain.answer_question(
            question=question,
            k=k,
            use_multi_stage=True
        )
        
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Sources: {len(result['sources'])} chunks retrieved")
        print(f"Metadata: {result['metadata']}\n")
        
        return result
    
    def get_stats(self):
        """Get system statistics."""
        stats = self.vector_store.get_stats()
        
        print("\n" + "="*60)
        print("Vector Store Statistics")
        print("="*60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"\nSection distribution:")
        for section, count in stats['section_distribution'].items():
            print(f"  {section}: {count}")
        print(f"\nDocuments:")
        for doc_info in stats['documents']:
            print(f"  {doc_info['filename']}: {doc_info['chunks']} chunks")
        print("="*60 + "\n")
        
        return stats


def main():
    """Main function for interactive use."""
    pipeline = MedicalRAGPipeline()
    
    # Example usage
    print("Medical RAG Pipeline Ready!")
    print("\nExample usage:")
    print("  pipeline.ingest_directory()  # Ingest all PDFs from data/raw_pdfs/")
    print("  pipeline.query('What medications am I taking?')")
    print("  pipeline.get_stats()\n")
    
    return pipeline


if __name__ == "__main__":
    # Create pipeline instance
    pipeline = main()
    
    # Interactive mode
    import code
    code.interact(local=locals(), banner="")
