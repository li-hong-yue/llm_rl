import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from datetime import datetime

from models.model_wrapper import ModelWrapper
from data.omnimath import OmniMathDataset
from verifiers.exact_match import ExactMatchVerifier
from verifiers.llm_verifier import LLMVerifier

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config['evaluation']
        
        # Set up logging
        log_level = config['system'].get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create save directory
        self.save_path = Path(self.eval_config['save_path'])
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info("Initializing model...")
        self.model = ModelWrapper(config)
        
        # Load dataset
        logger.info("Loading dataset...")
        self.dataset = self._load_dataset()
        
        # Load verifier
        logger.info("Initializing verifier...")
        self.verifier = self._load_verifier()
        
        logger.info("Initialization complete")
    
    def _load_dataset(self):
        """Load dataset based on config"""
        dataset_name = self.config['dataset']['name']
        
        # Map dataset names to classes
        # Add more datasets here
        dataset_map = {
            'KbsdJames/Omni-MATH': OmniMathDataset,
        }
        
        dataset_class = dataset_map.get(dataset_name)
        if dataset_class is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset = dataset_class(self.config, self.model.tokenizer)
        dataset.load_and_process()
        return dataset
    
    def _load_verifier(self):
        """Load verifier based on config"""
        verifier_type = self.config['verifier']['type']
        
        verifier_map = {
            'exact_match': ExactMatchVerifier,
            'llm_verifier': LLMVerifier,
        }
        
        verifier_class = verifier_map.get(verifier_type)
        if verifier_class is None:
            raise ValueError(f"Unknown verifier: {verifier_type}")
        
        return verifier_class(self.config)
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation"""
        logger.info("Starting evaluation...")
        
        dataset = self.dataset.dataset
        batch_size = self.eval_config['batch_size']
        
        all_results = []
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i:i + batch_size]
            batch_results = self._evaluate_batch(batch)
            all_results.extend(batch_results)
        
        # Compute metrics
        metrics = self._compute_metrics(all_results)
        
        # Save results
        self._save_results(all_results, metrics)
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.2%}")
        return metrics
    
    def _evaluate_batch(self, batch: Dict[str, List]) -> List[Dict[str, Any]]:
        """Evaluate a single batch"""
        # Generate solutions
        prompts = batch['prompt']
        generated_solutions = self.model.generate(prompts)
        
        # Verify solutions
        results = []
        for idx in range(len(prompts)):
            verification = self.verifier.verify(
                problem=batch['problem'][idx],
                official_solution=batch['official_solution'][idx],
                official_answer=batch['official_answer'][idx],
                generated_solution=generated_solutions[idx]
            )
            
            result = {
                'problem': batch['problem'][idx],
                'official_solution': batch['official_solution'][idx],
                'official_answer': batch['official_answer'][idx],
                'generated_solution': generated_solutions[idx],
                'verification': verification,
            }
            results.append(result)
        
        return results
    
    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics"""
        total = len(results)
        correct = sum(1 for r in results if r['verification']['is_correct'])
        avg_score = sum(r['verification']['score'] for r in results) / total
        
        metrics = {
            'total_samples': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'average_score': avg_score,
        }
        
        return metrics
    
    def _save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        if self.eval_config.get('save_generations', True):
            results_file = self.save_path / f"detailed_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detailed results saved to {results_file}")
        
        # Save metrics
        log_file = self.save_path / self.eval_config['log_file']
        log_entry = {
            'timestamp': timestamp,
            'config': {
                'model': self.config['model']['checkpoint_path'],
                'dataset': self.config['dataset']['name'],
                'verifier': self.config['verifier']['type'],
            },
            'metrics': metrics,
        }
        
        # Append to log file
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Metrics logged to {log_file}")