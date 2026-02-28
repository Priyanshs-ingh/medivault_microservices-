"""
QA Chain Module
Orchestrates retrieval + generation for question answering
"""
from typing import List, Dict, Optional
import logging
import re
from datetime import datetime

from rag.retriever import MedicalRetriever
from rag.prompt import MedicalPrompts, PromptBuilder
from vectorstore.mongodb_handler import MongoDBVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalQAChain:
    """
    Question-Answering chain for medical documents.
    Combines retrieval and generation with safety checks and prompt engineering.
    
    This is the orchestrator that brings everything together:
    - Retrieves relevant context
    - Builds appropriate prompts
    - Handles conversation history
    - Formats responses
    """
    
    def __init__(
        self,
        vector_store: MongoDBVectorStore = None,
        llm_handler = None  # Will be imported to avoid circular dependency
    ):
        self.vector_store = vector_store or MongoDBVectorStore()
        self.retriever = MedicalRetriever(self.vector_store)
        self.prompt_builder = PromptBuilder()
        self.prompts = MedicalPrompts()
        
        # LLM handler will be set by the pipeline
        self.llm_handler = llm_handler
        
        logger.info("MedicalQAChain initialized")
    
    def set_llm_handler(self, llm_handler):
        """Set the LLM handler (to avoid circular imports)."""
        self.llm_handler = llm_handler
    
    def answer_question(
        self,
        question: str,
        k: int = 5,
        conversation_history: Optional[List[Dict]] = None,
        use_multi_stage: bool = True
    ) -> Dict[str, any]:
        """
        Answer a question about medical records.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            conversation_history: Optional conversation context
            use_multi_stage: Whether to use multi-stage retrieval
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Validate question appropriateness
        if not self._is_appropriate_query(question):
            return {
                "answer": (
                    "I can only help you retrieve and interpret information from your medical records. "
                    "I cannot provide medical advice, diagnose conditions, or recommend treatments. "
                    "Please rephrase your question to ask about information in your records."
                ),
                "sources": [],
                "metadata": {"warning": "inappropriate_query"}
            }
        
        # Retrieve relevant context
        if use_multi_stage:
            retrieved_chunks = self.retriever.retrieve_multi_stage(
                question,
                initial_k=k*4,
                final_k=k
            )
        else:
            retrieved_chunks = self.retriever.retrieve(question, k=k)
        
        if not retrieved_chunks:
            return {
                "answer": (
                    "I couldn't find any relevant information in your medical records "
                    "to answer this question. The information may not be available in "
                    "the uploaded documents."
                ),
                "sources": [],
                "metadata": {"no_results": True}
            }
        
        # Build context
        context = self.retriever.get_full_context(
            question,
            max_tokens=3000,
            k=k
        )
        
        # Build prompt
        query_type = self.prompt_builder.detect_query_type(question)
        user_prompt = self.prompt_builder.build_prompt(
            question,
            context,
            query_type=query_type
        )
        
        # Generate response (will be implemented with LLAMA 3)
        if self.llm_handler is None:
            # Fallback if LLM handler not set
            return {
                "answer": f"LLM handler not initialized. Retrieved {len(retrieved_chunks)} relevant chunks.",
                "sources": self._format_sources(retrieved_chunks),
                "metadata": {"chunks_retrieved": len(retrieved_chunks)}
            }
        
        # Generate with LLM
        response = self.llm_handler.generate_response(
            user_prompt,
            conversation_history
        )
        
        # Format and return
        return {
            "answer": response.get("answer", ""),
            "sources": self._format_sources(retrieved_chunks),
            "metadata": {
                "chunks_retrieved": len(retrieved_chunks),
                "query_type": query_type,
                "model": response.get("model", "unknown"),
                "tokens_used": response.get("usage", {}).get("total_tokens", 0)
            }
        }
    
    def answer_with_specific_section(
        self,
        section_type: str,
        question: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Answer a question by retrieving all content from a specific section type.
        
        Useful for queries like:
        - "List ALL my medications"
        - "Show me ALL my diagnoses"
        
        Args:
            section_type: Type of section (medications, diagnosis, lab_results, etc.)
            question: Optional specific question about that section
            
        Returns:
            Complete information from that section across all documents
        """
        logger.info(f"Retrieving all content from section: {section_type}")
        
        # Retrieve all chunks of this section type
        chunks = self.retriever.retrieve_all_in_section(section_type, limit=50)
        
        if not chunks:
            return {
                "answer": f"No {section_type} information found in your medical records.",
                "sources": [],
                "metadata": {"section_type": section_type, "no_results": True}
            }
        
        # Build comprehensive context from all chunks
        context_parts = []
        for chunk in chunks:
            filename = chunk.get('metadata', {}).get('filename', 'Unknown')
            text = chunk.get('text', '')
            context_parts.append(f"[From {filename}]\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Build appropriate prompt
        if section_type == 'medications':
            prompt = self.prompts.MEDICATION_PROMPT.format(context=context)
        elif section_type == 'diagnosis':
            prompt = self.prompts.DIAGNOSIS_PROMPT.format(context=context)
        elif section_type == 'lab_results':
            prompt = self.prompts.LAB_RESULTS_PROMPT.format(context=context)
        else:
            prompt = self.prompts.build_user_prompt(
                question or f"Summarize all {section_type} information",
                context
            )
        
        # Generate response
        if self.llm_handler is None:
            return {
                "answer": f"Found {len(chunks)} chunks of {section_type}. LLM handler not initialized.",
                "sources": self._format_sources(chunks[:10]),
                "metadata": {"section_type": section_type, "chunks_found": len(chunks)}
            }
        
        response = self.llm_handler.generate_response(prompt, None)
        
        return {
            "answer": response.get("answer", ""),
            "sources": self._format_sources(chunks[:10]),  # Top 10 sources
            "metadata": {
                "section_type": section_type,
                "chunks_retrieved": len(chunks),
                "complete_retrieval": True
            }
        }
    
    def answer_across_documents(
        self,
        question: str,
        document_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Answer a question that requires information from multiple documents.
        
        Useful for tracking changes over time or consolidating information
        from multiple visits.
        
        Args:
            question: User's question
            document_names: Optional list of specific documents to search
            
        Returns:
            Answer considering multiple documents
        """
        if document_names is None:
            # Get all documents
            document_names = self.retriever.get_all_documents()
        
        logger.info(f"Answering across {len(document_names)} documents")
        
        # Retrieve from all specified documents
        all_chunks = []
        for doc_name in document_names:
            chunks = self.retriever.retrieve_from_document(
                question,
                doc_name,
                k=3  # Get top 3 from each document
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {
                "answer": "No relevant information found across the specified documents.",
                "sources": [],
                "metadata": {"documents_searched": len(document_names)}
            }
        
        # Build multi-document context
        context = self.retriever.get_full_context(question, max_tokens=3500)
        
        # Build prompt for multi-document analysis
        multi_doc_prompt = self.prompts.build_multi_document_prompt(
            question,
            document_names
        )
        
        user_prompt = f"{multi_doc_prompt}\n\n{self.prompts.build_user_prompt(question, context)}"
        
        # Generate response
        if self.llm_handler:
            response = self.llm_handler.generate_response(user_prompt, None)
            answer = response.get("answer", "")
        else:
            answer = f"Retrieved {len(all_chunks)} chunks from {len(document_names)} documents. LLM not initialized."
        
        return {
            "answer": answer,
            "sources": self._format_sources(all_chunks[:10]),
            "metadata": {
                "documents_searched": len(document_names),
                "chunks_retrieved": len(all_chunks),
                "multi_document": True
            }
        }
    
    def _is_appropriate_query(self, query: str) -> bool:
        """
        Check if query is appropriate for a medical records assistant.
        
        Inappropriate queries:
        - Asking for medical advice
        - Asking for diagnosis of new symptoms
        - Asking for treatment recommendations
        """
        query_lower = query.lower()
        
        # Inappropriate patterns — Production Safety: NEVER give medical advice
        inappropriate_patterns = [
            # Advice/action seeking
            r'should i\s+(?:take|stop|start|increase|decrease|reduce|change|switch|use)',
            r'should i\s+',  # Any "should I" question
            r'what (?:should|can) i (?:do|take|try|use|eat|avoid)',
            r'can i\s+(?:take|start|stop|use|try|eat|drink)',
            r'is it safe',
            # Danger/normality judgments
            r'is (?:this|my|it)\s+(?:normal|dangerous|serious|okay|ok|safe|bad|good|high|low|elevated|abnormal)',
            r'is (?:my|this|the)\s+\w+\s+(?:normal|dangerous|serious|okay|ok|safe|bad|good|high|low)',
            r'(?:normal|dangerous|serious|abnormal)\s*\?',
            r'am i (?:okay|ok|fine|normal|at risk|in danger)',
            # Treatment/prescription seeking
            r'diagnose (?:me|my)',
            r'recommend(?:ation)?s?\s*(?:for|me|my)?',
            r'prescribe\b',
            r'what (?:medication|medicine|drug|treatment)\s+(?:should|can|do)',
            r'suggest\s+(?:a |any )?(?:medication|medicine|drug|treatment)',
            # Dosage advice
            r'should i (?:increase|decrease|reduce|double|halve)',
            r'(?:increase|decrease|reduce|change|adjust)\s+(?:my )?(?:dose|dosage)',
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Inappropriate query detected: {query}")
                return False
        
        return True
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format retrieved chunks as source citations."""
        sources = []
        
        for i, chunk in enumerate(chunks[:5]):  # Top 5 sources
            metadata = chunk.get('metadata', {})
            sources.append({
                "source_id": i + 1,
                "filename": metadata.get('filename', 'Unknown'),
                "section": metadata.get('section_type', 'general'),
                "score": chunk.get('score', 0.0),
                "preview": chunk.get('text', '')[:200] + "..."
            })
        
        return sources


# Example usage
if __name__ == "__main__":
    # Initialize QA chain
    # qa_chain = MedicalQAChain()
    
    # Test question appropriateness
    from rag.prompt import MedicalPrompts
    
    test_queries = [
        ("What medications am I taking?", True),
        ("Should I stop taking metformin?", False),
        ("What did my doctor diagnose me with?", True),
        ("Is my blood pressure dangerous?", False),
        ("Show me my lab results from last month", True)
    ]
    
    print("Query Appropriateness Check:")
    print("="*60)
    
    qa_chain_test = MedicalQAChain(vector_store=None)
    
    for query, expected in test_queries:
        result = qa_chain_test._is_appropriate_query(query)
        status = "" if result == expected else ""
        print(f"{status} {query} -> {'Appropriate' if result else 'Inappropriate'}")
