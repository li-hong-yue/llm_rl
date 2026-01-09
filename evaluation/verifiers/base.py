from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseVerifier(ABC):
    """Base class for answer verification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verifier_config = config.get('verifier', {})
    
    @abstractmethod
    def verify(
        self,
        problem: str,
        official_solution: str,
        official_answer: str,
        generated_solution: str
    ) -> Dict[str, Any]:
        """
        Verify if the generated solution is correct.
        
        Args:
            problem: The math problem text
            official_solution: The official solution
            official_answer: The official answer
            generated_solution: The LLM generated solution
            
        Returns:
            Dict containing:
                - score: float between 0 and 1
                - is_correct: bool
                - explanation: str (optional)
                - details: Any additional details
        """
        pass
    
    def batch_verify(
        self,
        problems: list,
        official_solutions: list,
        official_answers: list,
        generated_solutions: list
    ) -> list:
        """Verify a batch of solutions"""
        results = []
        for prob, off_sol, off_ans, gen_sol in zip(
            problems, official_solutions, official_answers, generated_solutions
        ):
            result = self.verify(prob, off_sol, off_ans, gen_sol)
            results.append(result)
        return results