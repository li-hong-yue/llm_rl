#!/usr/bin/env python3
import argparse
import yaml
import logging
from pathlib import Path
from evaluator import Evaluator

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config with command line arguments"""
    if args.checkpoint:
        config['model']['checkpoint_path'] = args.checkpoint
    
    if args.dataset:
        config['dataset']['name'] = args.dataset
    
    if args.batch_size:
        config['evaluation']['batch_size'] = args.batch_size
    
    if args.save_path:
        config['evaluation']['save_path'] = args.save_path
    
    if args.verifier:
        config['verifier']['type'] = args.verifier
    
    if args.num_gpus:
        config['system']['num_gpus'] = args.num_gpus
        config['model']['tensor_parallel_size'] = args.num_gpus
    
    if args.max_samples:
        config['dataset']['max_samples'] = args.max_samples
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on math problems"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        help='Path to save results (overrides config)'
    )
    parser.add_argument(
        '--verifier',
        type=str,
        choices=['exact_match', 'llm_verifier'],
        help='Verifier type (overrides config)'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        help='Number of GPUs to use (overrides config)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to evaluate (for testing)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    config = override_config(config, args)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {config['model']['checkpoint_path']}")
    logger.info(f"  Dataset: {config['dataset']['name']}")
    logger.info(f"  Batch size: {config['evaluation']['batch_size']}")
    logger.info(f"  Verifier: {config['verifier']['type']}")
    logger.info(f"  Save path: {config['evaluation']['save_path']}")
    
    # Run evaluation
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate()
    
    # Print final metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Average score: {metrics['average_score']:.3f}")
    print("="*50)


if __name__ == '__main__':
    main()