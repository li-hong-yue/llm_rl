import re
from typing import Dict, Any
from .base import BaseVerifier

from math_verify import verify as math_verify  # <-- math-verify
from math_verify import parse

class ExactMatchVerifier(BaseVerifier):
    """Math-Verify based verifier for answers"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def extract_answer(self, text: str) -> str:
        """Extract the final answer from solution text"""
    
        patterns = [
            # 1️⃣ Exact prompt contract (highest priority)
            r"final answer\s*:\s*(.+)",
    
            # 2️⃣ Common math formatting
            r"\\boxed\{([^}]+)\}",
            r"\$\$([^$]+)\$\$",
    
            # 3️⃣ Legacy / fallback phrasing
            r"(?:answer|result)(?:\s*is)?[\s:=]+([^\n]+)",
        ]
    
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
    
        # 4️⃣ Last-resort fallback
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else ""

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (kept for logging/debugging)"""
      #  answer = answer.strip()
      #  answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)
      #  answer = answer.replace('$', '')
      #  answer = answer.replace('\\frac', '')
       # answer = answer.lower()
        return answer

    def verify(
        self,
        problem: str,
        official_solution: str,
        official_answer: str,
        generated_solution: str
    ) -> Dict[str, Any]:
        """Verify using math-verify semantic equivalence"""
        # Extract generated answer
        generated_ans = self.extract_answer(generated_solution)

        # Run math-verify
        try:
            is_correct = math_verify(
                parse(generated_ans),
                parse(official_answer),
            )
        except Exception:
            assert 0
            is_correct = False

        return {
            'score': 1.0 if is_correct else 0.0,
            'is_correct': is_correct,
            'extracted_answer': generated_ans,
            'normalized_official': self.normalize_answer(official_answer),
            'normalized_generated': self.normalize_answer(generated_ans),
        }
