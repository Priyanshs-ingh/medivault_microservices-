"""
Production-Level Test Configuration & Shared Fixtures
Medical RAG Bot - MediVault
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Sample Medical Data Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_medical_document():
    return {
        "text": "Patient has Type 2 Diabetes. Taking Metformin 500mg.",
        "source": "test.pdf",
        "filename": "test.pdf"
    }

@pytest.fixture
def rich_medical_document():
    return {
        "text": """
Patient Information:
Name: Jane Doe
Age: 52 years
Gender: Female
DOB: 1972-03-15

Chief Complaint:
Patient presents with fatigue and increased thirst.

Medications:
1. Metformin 500mg - twice daily with meals
2. Lisinopril 10mg - once daily in the morning
3. Aspirin 81mg - once daily
4. Atorvastatin 20mg - once daily at bedtime
5. Omeprazole 20mg - once daily before breakfast

Diagnosis:
1. Type 2 Diabetes Mellitus - well controlled
2. Hypertension - Stage 1
3. Hyperlipidemia

Lab Results:
HbA1c: 6.8%
Fasting Glucose: 125 mg/dL
Total Cholesterol: 180 mg/dL
LDL: 100 mg/dL
HDL: 55 mg/dL

Allergies:
1. Penicillin - rash
2. Sulfa drugs - anaphylaxis

Vital Signs:
Blood Pressure: 132/84 mmHg
Heart Rate: 76 bpm
Temperature: 98.6°F
Oxygen Saturation: 98%

Follow-up:
Return in 3 months for medication review and lab work.
Continue current medications.
Lifestyle modifications recommended.
        """,
        "source": "/path/to/jane_doe_report.pdf",
        "filename": "jane_doe_report.pdf",
        "date": "2024-01-15",
        "doctor_name": "Dr. Smith",
        "hospital_name": "City Medical Center",
        "report_date": "2024-01-15",
        "report_type": "General Checkup",
        "patient_id": "P12345"
    }

@pytest.fixture
def document_with_table():
    return {
        "text": """
Patient Report

Vital Signs:
Blood Pressure: 120/80 mmHg

[Table 1 on Page 2]
Test          | Result    | Reference Range
HbA1c         | 6.8%      | < 5.7% (normal)
Glucose       | 125 mg/dL | 70-99 mg/dL
Cholesterol   | 180 mg/dL | < 200 mg/dL

Follow-up next month.
        """,
        "source": "report_with_table.pdf",
        "filename": "report_with_table.pdf"
    }

@pytest.fixture
def multi_page_document():
    return {
        "text": """
[Page 1]
Patient Information:
Name: John Smith
Age: 45

[Page 2]
Medications:
1. Metformin 500mg twice daily
2. Lisinopril 10mg once daily

[Page 3]
Lab Results:
HbA1c: 7.2%
Fasting Glucose: 140 mg/dL
        """,
        "source": "multi_page.pdf",
        "filename": "multi_page.pdf"
    }

@pytest.fixture
def empty_document():
    return {
        "text": "",
        "source": "empty.pdf",
        "filename": "empty.pdf"
    }

@pytest.fixture
def whitespace_document():
    return {
        "text": "   \n\n\t  \n  ",
        "source": "whitespace.pdf",
        "filename": "whitespace.pdf"
    }

@pytest.fixture
def minimal_document():
    return {
        "text": "A",  # Single character
        "source": "minimal.pdf",
        "filename": "minimal.pdf"
    }

@pytest.fixture
def large_document():
    """Document that requires chunking."""
    base_section = """
Medications:
1. Metformin 500mg - twice daily
2. Lisinopril 10mg - once daily
3. Aspirin 81mg - once daily
4. Atorvastatin 20mg - once daily
5. Omeprazole 20mg - once daily
6. Levothyroxine 50mcg - once daily
7. Metoprolol 25mg - twice daily
8. Gabapentin 300mg - three times daily
9. Amlodipine 5mg - once daily
10. Losartan 50mg - once daily
    """
    return {
        "text": base_section * 10,  # Repeat to force multiple chunks
        "source": "large_report.pdf",
        "filename": "large_report.pdf"
    }

@pytest.fixture
def mock_vector_store():
    """Mock MongoDB vector store for unit tests."""
    mock = MagicMock()
    mock.similarity_search.return_value = [
        {
            "text": "Medications: Metformin 500mg twice daily",
            "metadata": {
                "filename": "test.pdf",
                "section_type": "medications",
                "chunk_id": 0,
                "page": 1
            },
            "score": 0.92
        },
        {
            "text": "Diagnosis: Type 2 Diabetes Mellitus",
            "metadata": {
                "filename": "test.pdf",
                "section_type": "diagnosis",
                "chunk_id": 1,
                "page": 2
            },
            "score": 0.85
        }
    ]
    mock.hybrid_search.return_value = mock.similarity_search.return_value
    mock.filter_by_metadata.return_value = mock.similarity_search.return_value
    mock.get_all_filenames.return_value = ["test.pdf", "report2.pdf"]
    mock.get_stats.return_value = {
        "total_chunks": 50,
        "total_documents": 2,
        "section_distribution": {
            "medications": 15,
            "diagnosis": 10,
            "lab_results": 10,
            "general": 15
        },
        "documents": [
            {"filename": "test.pdf", "chunks": 30},
            {"filename": "report2.pdf", "chunks": 20}
        ]
    }
    mock.add_documents.return_value = ["id1", "id2"]
    return mock

@pytest.fixture
def mock_llm_handler():
    """Mock LLM handler for unit tests."""
    mock = MagicMock()
    mock.generate_response.return_value = {
        "answer": "Based on your records, you are taking Metformin 500mg twice daily.",
        "model": "llama-3.1-8b-instant",
        "provider": "groq",
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 50,
            "total_tokens": 250
        }
    }
    mock.validate_medical_query.return_value = True
    return mock
