"""
Advanced Retrieval Module
Handles intelligent query processing and context retrieval with metadata filtering
"""
from typing import List, Dict, Optional
import logging
import re

from vectorstore.mongodb_handler import MongoDBVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalRetriever:
    """
    Intelligent retrieval system for medical queries.
    
    Features:
    - Query intent detection
    - Section-aware retrieval (medications, diagnosis, labs, etc.)
    - Metadata filtering
    - Hybrid search (vector + keyword)
    - Dynamic retrieval (adapts to any number of documents)
    """
    
    def __init__(self, vector_store: MongoDBVectorStore = None):
        self.vector_store = vector_store or MongoDBVectorStore()
        
        # Query patterns for section detection
        self.query_patterns = {
            'medications': [
                r'medication', r'medicine', r'drug', r'prescription',
                r'taking', r'prescribed', r'pills'
            ],
            'diagnosis': [
                r'diagnosis', r'diagnosed', r'condition', r'disease',
                r'illness', r'problem', r'disorder'
            ],
            'lab_results': [
                r'lab', r'test result', r'blood work', r'laboratory',
                r'blood test', r'screening'
            ],
            'vitals': [
                r'vital', r'blood pressure', r'temperature', r'heart rate',
                r'bp', r'pulse', r'oxygen'
            ],
            'allergies': [
                r'allerg', r'allergic', r'reaction'
            ],
            'symptoms': [
                r'symptom', r'complaint', r'pain', r'ache', r'feeling'
            ],
            'procedures': [
                r'procedure', r'surgery', r'operation', r'treatment'
            ],
            'follow_up': [
                r'follow[\-\s]?up', r'next visit', r'recommendation',
                r'plan', r'advice'
            ],
        }
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        section_filter: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            section_filter: Optional section type to filter by
            use_hybrid: Whether to use hybrid search (vector + metadata)
            
        Returns:
            List of relevant chunks with metadata
        """
        # Auto-detect section type if not provided
        if section_filter is None and use_hybrid:
            section_filter = self._detect_section_type(query)
        
        # Retrieve with appropriate strategy
        if section_filter:
            logger.info(f"Retrieving with section filter: {section_filter}")
            results = self.vector_store.hybrid_search(
                query,
                k=k,
                section_type=section_filter
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        # If not enough results, try again without section filter
        if len(results) < k // 2 and section_filter:
            logger.info("Insufficient results with filter, retrieving without filter")
            results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def retrieve_all_in_section(
        self,
        section_type: str,
        limit: int = 50
    ) -> List[Dict[str, any]]:
        """
        Retrieve all chunks of a specific section type.
        Useful for queries like "list ALL medications" across all documents.
        
        Args:
            section_type: Section type (e.g., 'medications')
            limit: Maximum chunks to retrieve
            
        Returns:
            All chunks of that section type
        """
        logger.info(f"Retrieving all chunks of type: {section_type}")
        
        return self.vector_store.filter_by_metadata(
            metadata_filter={'section_type': section_type},
            limit=limit
        )
    
    def retrieve_from_document(
        self,
        query: str,
        filename: str,
        k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks from a specific document.
        
        Args:
            query: User's question
            filename: Specific document to search in
            k: Number of chunks to retrieve
            
        Returns:
            Relevant chunks from specified document
        """
        logger.info(f"Retrieving from document: {filename}")
        
        results = self.vector_store.similarity_search(
            query,
            k=k,
            metadata_filter={'filename': filename}
        )
        
        return results
    
    def retrieve_multi_stage(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Two-stage retrieval:
        1. Retrieve more candidates initially
        2. Rerank based on relevance
        
        This ensures better coverage across multiple documents.
        """
        # Stage 1: Get initial candidates
        candidates = self.retrieve(query, k=initial_k, use_hybrid=True)
        
        if len(candidates) <= final_k:
            return candidates
        
        # Stage 2: Rerank (simple approach - by score and diversity)
        reranked = self._rerank_by_diversity(candidates, final_k)
        
        logger.info(f"Reranked {len(candidates)} candidates to top {len(reranked)}")
        return reranked
    
    def _detect_section_type(self, query: str) -> Optional[str]:
        """
        Detect which section type the query is asking about.
        Returns None if no clear section is detected.
        """
        query_lower = query.lower()
        
        for section_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return section_type
        
        return None
    
    def _rerank_by_diversity(
        self,
        candidates: List[Dict[str, any]],
        top_k: int
    ) -> List[Dict[str, any]]:
        """
        Rerank candidates to ensure diversity across documents and sections.
        Prevents all results from coming from the same document.
        """
        # Group by document
        by_document = {}
        for candidate in candidates:
            filename = candidate.get('metadata', {}).get('filename', 'unknown')
            if filename not in by_document:
                by_document[filename] = []
            by_document[filename].append(candidate)
        
        # Select top results ensuring diversity
        reranked = []
        doc_names = list(by_document.keys())
        doc_index = 0
        
        # Round-robin selection from different documents
        while len(reranked) < top_k and any(by_document.values()):
            doc_name = doc_names[doc_index % len(doc_names)]
            
            if by_document[doc_name]:
                reranked.append(by_document[doc_name].pop(0))
            
            doc_index += 1
        
        return reranked
    
    def get_full_context(
        self,
        query: str,
        max_tokens: int = 3000,
        k: int = 10
    ) -> str:
        """
        Get formatted context for LLM, staying within token limit.
        
        Args:
            query: User's question
            max_tokens: Maximum tokens for context (rough estimate: 1 token ≈ 4 chars)
            k: Initial number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        # Retrieve chunks
        chunks = self.retrieve_multi_stage(query, initial_k=k*2, final_k=k)
        
        # Build context with document grouping
        max_chars = max_tokens * 4
        context_parts = []
        current_length = 0
        
        # Group by document
        by_document = {}
        for chunk in chunks:
            filename = chunk.get('metadata', {}).get('filename', 'Unknown')
            if filename not in by_document:
                by_document[filename] = []
            by_document[filename].append(chunk)
        
        # Format context
        for filename, doc_chunks in by_document.items():
            doc_header = f"\n{'='*60}\nDOCUMENT: {filename}\n{'='*60}\n"
            
            if current_length + len(doc_header) > max_chars:
                break
            
            context_parts.append(doc_header)
            current_length += len(doc_header)
            
            for chunk in doc_chunks:
                section = chunk.get('metadata', {}).get('section_type', 'general')
                text = chunk.get('text', '')
                
                chunk_text = f"\n[{section.upper()}]\n{text}\n"
                
                if current_length + len(chunk_text) > max_chars:
                    break
                
                context_parts.append(chunk_text)
                current_length += len(chunk_text)
        
        context = "".join(context_parts)
        logger.info(f"Built context: {current_length} chars from {len(by_document)} documents")
        
        return context
    
    def get_all_documents(self) -> List[str]:
        """Get list of all unique documents in the vector store."""
        return self.vector_store.get_all_filenames()


# Example usage
if __name__ == "__main__":
    # This would require MongoDB connection
    # retriever = MedicalRetriever()
    
    # Test section detection
    retriever_test = MedicalRetriever(vector_store=None)
    
    test_queries = [
        "What medications am I taking?",
        "Show me my lab results",
        "What was I diagnosed with?",
        "Tell me about my last visit"
    ]
    
    print("Query Type Detection:")
    print("="*60)
    for query in test_queries:
        section = retriever_test._detect_section_type(query)
        print(f"Query: {query}")
        print(f"Detected section: {section}\n")
