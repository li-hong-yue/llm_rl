from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from typing import Dict, Any
import yaml

class SFTDataset:
    def __init__(self, config: Dict[str, Any], tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config['dataset']['max_length']
        assert self.tokenizer.eos_token_id is not None

        
    def load_and_process(self):
        """Load dataset and create train/val splits"""
        dataset_config = self.config['dataset']
        
        # Load dataset
        dataset = load_dataset(
            dataset_config['name'],
            split=dataset_config['split']
        )
        
        # Process dataset
        processed = dataset.map(
            self._format_sample,
            remove_columns=dataset.column_names,
            desc="Processing dataset"
        )
        
        # Split into train/val
        split_ratio = dataset_config['train_test_split']
        split = processed.train_test_split(
            train_size=split_ratio,
            seed=self.config['training']['seed']
        )
        
        return split['train'], split['test']
    
    def _format_sample(self, example):
        """Format a single sample into prompt-completion format"""
        dataset_config = self.config['dataset']
        
        problem = example[dataset_config['problem_column']]
        solution = example[dataset_config['solution_column']]
        #answer = example.get(dataset_config['answer_column'], '')
        answer_col = dataset_config.get('answer_column')
        answer = example.get(answer_col, '') if answer_col else ''
        
        # Format as instruction-following prompt
        prompt = f"Problem: {problem}\n\nSolution:"
       # completion = f" {solution}\n\nAnswer: {answer}"
        completion = f" {solution}\n\nAnswer: {answer}{self.tokenizer.eos_token}"

        full_text = prompt + completion
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Create labels (mask prompt tokens)
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        labels = tokenized['input_ids'].copy()
        prompt_len = len(prompt_tokens['input_ids'])
        labels[:prompt_len] = [-100] * prompt_len  # Mask prompt
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }

def collate_fn(batch, pad_token_id):
    """Collate function with padding"""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        
        input_ids.append(
            item['input_ids'] + [pad_token_id] * pad_len
        )
        attention_mask.append(
            item['attention_mask'] + [0] * pad_len
        )
        labels.append(
            item['labels'] + [-100] * pad_len
        )
    
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels)
    }