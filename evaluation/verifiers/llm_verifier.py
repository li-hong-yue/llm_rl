import os
import time
import json
from typing import Dict, Any
from google import genai
from .base import BaseVerifier


class LLMVerifier(BaseVerifier):
    """LLM-based verifier using Gemini API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get API key from environment
        api_key_env = self.verifier_config.get('api_key_env', 'GEMINI_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")

        client = genai.Client(api_key=api_key)
        
        
        # Initialize model
        self.model_name = self.verifier_config.get('model_name', 'gemini-3-pro-preview')

        
        # Get config parameters
        self.temperature = self.verifier_config.get('temperature', 0.0)
        self.max_retries = self.verifier_config.get('max_retries', 3)
        
    def get_verification_prompt(
        self,
        problem: str,
        official_solution: str,
        official_answer: str,
        generated_solution: str
    ) -> str:
        """Create prompt for LLM verification"""
        prompt = f"""You are a math teacher verifying student solutions. Evaluate if the generated solution correctly solves the problem and arrives at the right answer.

**Problem:**
{problem}

**Official Answer:**
{official_answer}

**Official Solution:**
{official_solution}

**Generated Solution:**
{generated_solution}

Please evaluate the generated solution and respond in JSON format with:
{{
    "is_correct": true/false,
    "score": 0.0 to 1.0 (0.0 = completely wrong, 1.0 = completely correct),
    "explanation": "brief explanation of your evaluation"
}}

Consider:
1. Does the solution arrive at the correct final answer?
2. Is the reasoning logically sound?
3. Are the mathematical steps valid?

Be strict but fair. Minor notation differences are acceptable if the math is correct."""
        return prompt
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            # Look for JSON block
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                result = json.loads(json_str)
                return result
            else:
                # Fallback: parse manually
                is_correct = 'true' in response_text.lower() or 'correct' in response_text.lower()
                return {
                    'is_correct': is_correct,
                    'score': 1.0 if is_correct else 0.0,
                    'explanation': response_text
                }
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                'is_correct': False,
                'score': 0.0,
                'explanation': f"Parse error: {str(e)}"
            }
    
    def verify(
        self,
        problem: str,
        official_solution: str,
        official_answer: str,
        generated_solution: str
    ) -> Dict[str, Any]:
        """Verify using LLM (Gemini)"""
        prompt = self.get_verification_prompt(
            problem, official_solution, official_answer, generated_solution
        )
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        'temperature': self.temperature,
                    },
                )
                
                result = self.parse_response(response.text)
                result['raw_response'] = response.text
                return result
                
            except Exception as e:
                print(f"Verification attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        'score': 0.0,
                        'is_correct': False,
                        'explanation': f"Verification failed after {self.max_retries} attempts: {str(e)}",
                        'error': str(e)
                    }