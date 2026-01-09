from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset


class BaseDataset(ABC):
    """Base class for math evaluation datasets"""
    
    def __init__(self, config: Dict[str, Any], tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_config = config['dataset']
        self.max_length = self.dataset_config['max_length']
        assert self.tokenizer.eos_token_id is not None
        
    @abstractmethod
    def load_and_process(self) -> Dataset:
        """Load and process the dataset"""
        pass
    
    @abstractmethod
    def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single sample into the required structure"""
        pass
    
    def get_prompt_template(self) -> str:
        """Return the prompt template for the model"""
        return """Solve the following math problem step by step.
    
    Problem: {problem}
    
    At the end, write:
    Final Answer: <answer>
    
    Solution:"""
    
    def format_problem(self, problem: str) -> str:
        """Format problem text for model input"""
        template = self.get_prompt_template()
        return template.format(problem=problem)
    
    def __len__(self) -> int:
        if hasattr(self, 'dataset'):
            return len(self.dataset)
        return 0