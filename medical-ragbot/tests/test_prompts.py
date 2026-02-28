"""
Unit Tests: Prompt Building (rag/prompt.py)
Tests prompt template generation, query type detection, and context formatting.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.prompt import MedicalPrompts, PromptBuilder


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder - Query Type Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryTypeDetection:
    """PromptBuilder must correctly categorize medical queries."""

    @pytest.mark.parametrize("query,expected", [
        ("What medications am I taking?", "medication"),
        ("List my prescriptions", "medication"),
        ("What drugs do I take?", "medication"),
        ("What was I diagnosed with?", "diagnosis"),
        ("What conditions do I have?", "diagnosis"),
        ("Show my disease history", "diagnosis"),
        ("What were my lab results?", "lab_results"),
        ("What were my blood test results?", "lab_results"),
        ("Show my lab work", "lab_results"),
        ("Tell me about my last visit", "general"),
        ("When did I have my last appointment?", "general"),
        ("", "general"),
    ])
    def test_query_type_detection(self, query, expected):
        builder = PromptBuilder()
        result = builder.detect_query_type(query)
        assert result == expected, \
            f"Query '{query}': expected '{expected}', got '{result}'"


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder - Build Prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_medication_query_uses_medication_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            "What medications am I taking?",
            context="Metformin 500mg twice daily",
            query_type="medication"
        )
        # Medication prompt should ask for all medications
        assert "medication" in prompt.lower() or "drug" in prompt.lower()

    def test_diagnosis_query_uses_diagnosis_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            "What was I diagnosed with?",
            context="Type 2 Diabetes Mellitus",
            query_type="diagnosis"
        )
        assert "diagnosis" in prompt.lower() or "condition" in prompt.lower()

    def test_lab_results_query_uses_lab_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            "Show my lab results",
            context="HbA1c: 6.8%",
            query_type="lab_results"
        )
        assert "lab" in prompt.lower() or "test" in prompt.lower()

    def test_general_query_uses_generic_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            "Tell me about my visit",
            context="doctor's note here"
        )
        assert isinstance(prompt, str) and len(prompt) > 10

    def test_context_embedded_in_prompt(self):
        builder = PromptBuilder()
        context = "UNIQUE_CONTEXT_MARKER_FOR_TESTING"
        prompt = builder.build_prompt("query", context)
        assert context in prompt

    def test_prompt_not_empty(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt("query", "context")
        assert len(prompt) > 0

    def test_auto_detect_query_type_if_not_provided(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt("What medications am I taking?", "context")
        assert isinstance(prompt, str)


# ─────────────────────────────────────────────────────────────────────────────
# MedicalPrompts - Static Templates
# ─────────────────────────────────────────────────────────────────────────────

class TestMedicalPrompts:
    def test_system_prompt_not_empty(self):
        assert len(MedicalPrompts.SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_safety_guidelines(self):
        assert "NEVER" in MedicalPrompts.SYSTEM_PROMPT or \
               "never" in MedicalPrompts.SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_records(self):
        assert "records" in MedicalPrompts.SYSTEM_PROMPT.lower()

    def test_medication_prompt_has_context_placeholder(self):
        assert "{context}" in MedicalPrompts.MEDICATION_PROMPT

    def test_diagnosis_prompt_has_context_placeholder(self):
        assert "{context}" in MedicalPrompts.DIAGNOSIS_PROMPT

    def test_lab_results_prompt_has_context_placeholder(self):
        assert "{context}" in MedicalPrompts.LAB_RESULTS_PROMPT

    def test_medication_prompt_formatted(self):
        ctx = "Metformin 500mg twice daily"
        prompt = MedicalPrompts.MEDICATION_PROMPT.format(context=ctx)
        assert ctx in prompt

    def test_build_user_prompt_contains_query(self):
        query = "UNIQUE_QUERY_STRING"
        ctx = "context"
        prompt = MedicalPrompts.build_user_prompt(query, ctx)
        assert query in prompt

    def test_build_user_prompt_contains_context(self):
        query = "query"
        ctx = "UNIQUE_CONTEXT_STRING"
        prompt = MedicalPrompts.build_user_prompt(query, ctx)
        assert ctx in prompt

    def test_build_context_consolidation_prompt(self):
        chunks = [
            {
                "text": "Metformin 500mg",
                "metadata": {
                    "filename": "test.pdf",
                    "section_type": "medications"
                }
            }
        ]
        prompt = MedicalPrompts.build_context_consolidation_prompt(chunks)
        assert "test.pdf" in prompt
        assert "Metformin" in prompt

    def test_build_multi_document_prompt(self):
        query = "What medications am I taking?"
        docs = ["report1.pdf", "report2.pdf"]
        prompt = MedicalPrompts.build_multi_document_prompt(query, docs)
        assert query in prompt
        assert str(len(docs)) in prompt

    def test_build_followup_prompt(self):
        history = [
            {"role": "user", "content": "What are my meds?"},
            {"role": "assistant", "content": "You take Metformin."}
        ]
        prompt = MedicalPrompts.build_followup_prompt("Anything else?", history)
        assert "Anything else?" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptEdgeCases:
    def test_empty_context_in_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt("What medications?", context="")
        assert isinstance(prompt, str)

    def test_very_long_context(self):
        builder = PromptBuilder()
        context = "Metformin 500mg. " * 1000
        prompt = builder.build_prompt("query", context)
        assert context[:100] in prompt  # At least start is there

    def test_special_chars_in_context(self):
        builder = PromptBuilder()
        context = "HbA1c: 6.8% (normal <5.7%) - Dr. Smith's note [2024]"
        prompt = builder.build_prompt("query", context)
        assert context in prompt

    def test_multi_document_prompt_with_empty_docs(self):
        prompt = MedicalPrompts.build_multi_document_prompt("query?", [])
        assert "0" in prompt  # 0 documents

    def test_context_consolidation_with_no_chunks(self):
        prompt = MedicalPrompts.build_context_consolidation_prompt([])
        assert isinstance(prompt, str)
