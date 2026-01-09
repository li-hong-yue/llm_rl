import yaml
import torch
import argparse
from src.data import SFTDataset
from src.model import load_model_and_tokenizer
from src.trainer import SFTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load and process dataset
    print("Loading and processing dataset...")
    dataset_loader = SFTDataset(config, tokenizer)
    train_dataset, val_dataset = dataset_loader.load_and_process()
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = SFTTrainer(model, tokenizer, train_dataset, val_dataset, config)
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print("Training complete!")

if __name__ == '__main__':
    main()