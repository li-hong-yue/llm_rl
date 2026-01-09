from typing import Dict, Any
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from .base import BaseDataset


class OmniMathDataset(BaseDataset):
    """Dataset loader for KbsdJames/Omni-MATH"""
    
    def __init__(self, config: Dict[str, Any], tokenizer: AutoTokenizer):
        super().__init__(config, tokenizer)
        self.problem_field = self.dataset_config['problem_field']
        self.solution_field = self.dataset_config['solution_field']
        self.answer_field = self.dataset_config['answer_field']
        
    def load_and_process(self) -> Dataset:
        """Load dataset and process samples"""
        # Load dataset
        dataset = load_dataset(
            self.dataset_config['name'],
            split=self.dataset_config['split']
        )
        
        # Limit samples if specified
        max_samples = self.dataset_config.get('max_samples')
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Process dataset
        processed = dataset.map(
            self._format_sample,
            remove_columns=dataset.column_names,
            desc="Processing dataset"
        )
        
        
        self.dataset = processed
        #return processed
        
        split_ratio = 0.95     # should match train scripts
        split = processed.train_test_split(
            train_size=split_ratio,
            seed=42 #self.config['training']['seed']
        )
        self.dataset = split['test']
        return  split['test']
        # 🔹 TEMP: only use first 5 test samples
       # test_subset = split['test'].select(range(min(5, len(split['test']))))
    
       # self.dataset = test_subset
       # return test_subset
        
    
    def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single sample into standardized structure"""
        problem = sample[self.problem_field]
        solution = sample[self.solution_field]
        answer = sample[self.answer_field]
        
        # Create prompt for the model
        prompt = self.format_problem(problem)
        
        return {
            'problem': problem,
            'official_solution': solution,
            'official_answer': answer,
            'prompt': prompt,
        }