"""
LLM Handler for LLAMA 3
Supports multiple deployment options for LLAMA 3
"""
from typing import List, Dict, Optional
import logging
import requests
import json

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLAMA3Handler:
    """
    Handler for LLAMA 3 model.
    
    Supports multiple deployment options:
    1. Ollama (local deployment) - Recommended
    2. Together AI (cloud API)
    3. Groq (cloud API)
    4. OpenAI-compatible endpoint
    """
    
    def __init__(self):
        self.model_name = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.provider = settings.llm_provider  # 'ollama', 'together', 'groq', 'openai'
        
        # Set up provider-specific configuration
        if self.provider == 'ollama':
            self.base_url = settings.ollama_base_url
            logger.info(f"Using Ollama at {self.base_url} with model {self.model_name}")
        elif self.provider == 'together':
            self.api_key = settings.together_api_key
            self.base_url = "https://api.together.xyz/v1"
            logger.info("Using Together AI")
        elif self.provider == 'groq':
            self.api_key = settings.groq_api_key
            self.base_url = "https://api.groq.com/openai/v1"
            logger.info(f"Using Groq with model {self.model_name}")
        elif self.provider == 'openai':
            # Fallback to OpenAI
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
            logger.warning("Using OpenAI instead of LLAMA 3")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate a response using LLAMA 3.
        
        Args:
            prompt: User prompt or question
            conversation_history: Optional chat history
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with response and metadata
        """
        if self.provider == 'ollama':
            return self._generate_ollama(prompt, conversation_history, system_prompt)
        elif self.provider in ['together', 'groq']:
            return self._generate_api(prompt, conversation_history, system_prompt)
        elif self.provider == 'openai':
            return self._generate_openai(prompt, conversation_history, system_prompt)
        else:
            raise ValueError(f"Provider {self.provider} not supported")
    
    def _generate_ollama(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]],
        system_prompt: Optional[str]
    ) -> Dict[str, any]:
        """Generate using Ollama (local LLAMA 3)."""
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 3 exchanges
        
        messages.append({"role": "user", "content": prompt})
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            answer = result.get("message", {}).get("content", "")
            
            return {
                "answer": answer,
                "model": self.model_name,
                "provider": "ollama",
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise
    
    def _generate_api(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]],
        system_prompt: Optional[str]
    ) -> Dict[str, any]:
        """Generate using Together AI or Groq API (OpenAI-compatible)."""
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        messages.append({"role": "user", "content": prompt})
        
        # API request
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            answer = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            
            return {
                "answer": answer,
                "model": self.model_name,
                "provider": self.provider,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            }
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating with {self.provider}: {e}")
            raise
    
    def _generate_openai(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]],
        system_prompt: Optional[str]
    ) -> Dict[str, any]:
        """Fallback: Generate using OpenAI."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name if "gpt" in self.model_name else "gpt-4-turbo-preview",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "model": response.model,
                "provider": "openai",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class MedicalLLMHandler:
    """
    Medical-specific LLM handler with safety checks and medical prompting.
    Wraps LLAMA3Handler with medical context.
    """
    
    def __init__(self):
        self.llm = LLAMA3Handler()
        
        # Medical system prompt
        self.system_prompt = """You are a medical records assistant helping to interpret medical documentation. 

Your responsibilities:
1. Provide accurate information based ONLY on the provided medical records
2. Be clear and precise when listing medications, diagnoses, or test results
3. If information is not in the provided context, clearly state "This information is not available in the provided records"
4. Do not make medical diagnoses or recommendations - only report what's in the records
5. When listing medications, include ALL medications mentioned in the context
6. Preserve exact medical terminology and dosages from the source documents

Important guidelines:
- Always cite which document the information comes from
- If asked about medications, list ALL of them, not just a subset
- Maintain patient privacy and confidentiality
- Use clear, professional language

Remember: You are assisting with information retrieval, not providing medical advice."""
    
    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """
        Generate response with medical context.
        
        Args:
            prompt: User prompt (already includes context)
            conversation_history: Optional conversation history
            
        Returns:
            Response dictionary
        """
        return self.llm.generate_response(
            prompt,
            conversation_history,
            system_prompt=self.system_prompt
        )
    
    def validate_medical_query(self, query: str) -> bool:
        """
        Validate if query is appropriate for medical records assistant.
        Must stay in sync with MedicalQAChain._is_appropriate_query.
        """
        import re
        
        query_lower = query.lower()
        
        # Inappropriate patterns — Production Safety: NEVER give medical advice
        inappropriate = [
            # Advice/action seeking
            r'should i\s+(?:take|stop|start|increase|decrease|reduce|change|switch|use)',
            r'should i\s+',
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
        
        for pattern in inappropriate:
            if re.search(pattern, query_lower):
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    print("LLAMA 3 Handler Configuration:")
    print("="*60)
    print(f"Model: {settings.llm_model}")
    print(f"Provider: {settings.llm_provider}")
    print()
    print("Supported providers:")
    print("  - ollama: Local LLAMA 3 via Ollama (Recommended)")
    print("  - together: Together AI cloud API")
    print("  - groq: Groq cloud API")
    print("  - openai: OpenAI (fallback)")
    print()
    print("To use LLAMA 3 locally:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Run: ollama pull llama3")
    print("  3. Set LLM_PROVIDER=ollama in .env")
    print("  4. Set LLM_MODEL=llama3 in .env")
