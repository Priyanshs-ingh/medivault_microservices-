"""
Medical Prompt Templates
Specialized prompts for medical document interpretation
"""
from typing import List, Dict
from datetime import datetime


class MedicalPrompts:
    """
    Collection of carefully crafted prompts for medical RAG system.
    Ensures accurate, safe, and helpful responses.
    """
    
    # Core system prompt
    SYSTEM_PROMPT = """You are a medical records assistant designed to help users understand their medical documentation. 

CRITICAL GUIDELINES:
1. ONLY provide information that exists in the provided medical records
2. If information is not in the records, explicitly state: "This information is not found in the provided records"
3. NEVER make medical diagnoses, recommendations, or provide medical advice
4. Your role is INFORMATION RETRIEVAL and EXPLANATION, not medical consultation
5. Always maintain HIPAA compliance and patient confidentiality

RESPONSIBILITIES:
• Extract and summarize information from medical records
• List ALL medications, diagnoses, or test results when asked (never partial lists)
• Explain medical terminology in simple language when helpful
• Cite which document/record each piece of information comes from
• Preserve exact dosages, measurements, and medical terms from source documents

WHAT YOU MUST NEVER DO:
• Provide medical advice or recommendations beyond what's in the records
• Diagnose conditions not already diagnosed in the records
• Suggest treatment changes or new medications
• Make predictions about health outcomes
• Share information that violates patient privacy

When listing items (medications, diagnoses, test results):
- Include ALL items mentioned in the context, not just a subset
- Maintain original formatting and details (dosages, frequencies, etc.)
- If a list is long, still provide the complete list

Remember: You are a helpful assistant for understanding medical records, not a medical professional."""
    
    # User prompt template for standard queries
    @staticmethod
    def build_user_prompt(query: str, context: str) -> str:
        """Build user prompt with query and retrieved context."""
        return f"""Based on the following medical records, please answer the question.

MEDICAL RECORDS:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer based ONLY on the information provided above
- If the answer is not in the records, say so clearly
- When listing items, include ALL items mentioned
- Cite which document each piece of information comes from
- Use clear, professional language

ANSWER:"""
    
    # Medication-specific prompt
    MEDICATION_PROMPT = """Based on the medical records provided, list ALL medications mentioned.

For each medication, include:
- Medication name
- Dosage
- Frequency
- Purpose (if mentioned)
- Which document it was found in

MEDICAL RECORDS:
{context}

If multiple records contain medication information, consolidate them into a complete list.
If no medications are found, state: "No medications found in the provided records."

COMPLETE MEDICATION LIST:"""
    
    # Diagnosis history prompt
    DIAGNOSIS_PROMPT = """Based on the medical records provided, list ALL diagnoses and medical conditions mentioned.

For each diagnosis, include:
- Condition name
- Date diagnosed (if available)
- Current status (active, resolved, etc.) if mentioned
- Which document it was found in

MEDICAL RECORDS:
{context}

Organize chronologically if dates are available.
If no diagnoses are found, state: "No diagnoses found in the provided records."

COMPLETE DIAGNOSIS HISTORY:"""
    
    # Lab results summary prompt
    LAB_RESULTS_PROMPT = """Based on the medical records provided, summarize ALL lab results and test findings.

For each test, include:
- Test name
- Result value
- Reference range (if provided)
- Date of test (if available)
- Which document it was found in

MEDICAL RECORDS:
{context}

Highlight any abnormal values if reference ranges are provided.
If no lab results are found, state: "No lab results found in the provided records."

COMPLETE LAB RESULTS SUMMARY:"""
    
    # Safety check prompt
    SAFETY_CHECK_PROMPT = """Analyze this query to determine if it's appropriate for a medical records assistant.

QUERY: {query}

Is this query:
1. Asking for medical advice or recommendations? (INAPPROPRIATE)
2. Asking for diagnosis of new symptoms? (INAPPROPRIATE)
3. Asking to interpret or retrieve information from medical records? (APPROPRIATE)

Respond with: APPROPRIATE or INAPPROPRIATE

If INAPPROPRIATE, explain briefly why and suggest how to rephrase."""
    
    # Context consolidation prompt
    @staticmethod
    def build_context_consolidation_prompt(chunks: List[Dict]) -> str:
        """Build a consolidated context from multiple chunks."""
        context_parts = []
        
        # Group by document
        docs = {}
        for chunk in chunks:
            filename = chunk.get('metadata', {}).get('filename', 'Unknown')
            section = chunk.get('metadata', {}).get('section_type', 'general')
            text = chunk.get('text', '')
            
            if filename not in docs:
                docs[filename] = []
            
            docs[filename].append({
                'section': section,
                'text': text
            })
        
        # Format context
        for filename, chunks_list in docs.items():
            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"DOCUMENT: {filename}")
            context_parts.append(f"{'='*60}")
            
            for chunk_info in chunks_list:
                section = chunk_info['section']
                text = chunk_info['text']
                
                context_parts.append(f"\n[{section.upper()}]")
                context_parts.append(text)
        
        return "\n".join(context_parts)
    
    # Multi-document query prompt
    @staticmethod
    def build_multi_document_prompt(query: str, documents: List[str]) -> str:
        """Build prompt for queries spanning multiple documents/visits."""
        return f"""You are analyzing medical records from multiple visits or documents.

QUERY: {query}

The user wants information compiled from ALL available documents. 
- Look across all documents provided
- Consolidate information chronologically when possible
- Note any changes or trends across visits
- Cite which document each piece of information comes from

Your answer should compile information from all {len(documents)} documents provided."""
    
    # Follow-up conversation prompt
    @staticmethod
    def build_followup_prompt(
        current_query: str,
        conversation_history: List[Dict]
    ) -> str:
        """Build prompt that maintains conversation context."""
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[-4:]  # Last 2 exchanges
        ])
        
        return f"""Continue this conversation about medical records.

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION:
{current_query}

Maintain context from the conversation while answering the current question."""


class PromptBuilder:
    """Helper class to build prompts dynamically based on query type."""
    
    def __init__(self):
        self.prompts = MedicalPrompts()
    
    def detect_query_type(self, query: str) -> str:
        """Detect what type of query this is."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['medication', 'medicine', 'drug', 'prescription', 'taking']):
            return 'medication'
        elif any(word in query_lower for word in ['diagnosis', 'diagnosed', 'condition', 'disease']):
            return 'diagnosis'
        elif any(word in query_lower for word in ['lab', 'test', 'result', 'blood work']):
            return 'lab_results'
        else:
            return 'general'
    
    def build_prompt(
        self,
        query: str,
        context: str,
        query_type: str = None
    ) -> str:
        """Build appropriate prompt based on query type."""
        if query_type is None:
            query_type = self.detect_query_type(query)
        
        if query_type == 'medication':
            return self.prompts.MEDICATION_PROMPT.format(context=context)
        elif query_type == 'diagnosis':
            return self.prompts.DIAGNOSIS_PROMPT.format(context=context)
        elif query_type == 'lab_results':
            return self.prompts.LAB_RESULTS_PROMPT.format(context=context)
        else:
            return self.prompts.build_user_prompt(query, context)


# Example usage
if __name__ == "__main__":
    builder = PromptBuilder()
    
    # Test query type detection
    queries = [
        "What medications am I currently taking?",
        "What was I diagnosed with last year?",
        "What were my recent lab results?",
        "Tell me about my last doctor visit"
    ]
    
    for query in queries:
        query_type = builder.detect_query_type(query)
        print(f"Query: {query}")
        print(f"Type: {query_type}\n")
