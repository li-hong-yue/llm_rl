import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import wandb
import os
from tqdm import tqdm
import functools
from typing import Dict, Any

class SFTTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup distributed training
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_main_process = self.local_rank == 0
        
        if self.world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
        
        # Wrap model with FSDP if enabled
        if config['fsdp']['enabled'] and self.world_size > 1:
            self._setup_fsdp()
        else:
            self.model = self.model.to(f'cuda:{self.local_rank}')
        
        # Setup dataloaders
        self._setup_dataloaders()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup wandb
        if self.is_main_process and config['wandb']['enabled']:
            self._setup_wandb()
        
        self.global_step = 0
        
    def _setup_fsdp(self):
        """Setup FSDP wrapping"""
        # Auto-wrap policy based on model type
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
                Qwen2DecoderLayer,
            }
        )
        
        # Sharding strategy mapping
        strategy_map = {
            'FULL_SHARD': ShardingStrategy.FULL_SHARD,
            'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
            'NO_SHARD': ShardingStrategy.NO_SHARD
        }
        
        sharding_strategy = strategy_map.get(
            self.config['fsdp']['sharding_strategy'],
            ShardingStrategy.FULL_SHARD
        )
        
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True
        )
    
    def _setup_dataloaders(self):
        """Setup train and validation dataloaders"""
        from functools import partial
        from src.data import collate_fn
        
        collate = partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id)
        
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size_per_device'],
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size_per_device'],
            sampler=val_sampler,
            collate_fn=collate,
            num_workers=4,
            pin_memory=True
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config['training']['gradient_accumulation_steps']
        total_steps = steps_per_epoch * self.config['training']['num_epochs']
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
    
    def _setup_wandb(self):
        """Initialize wandb logging"""
        wandb_config = self.config['wandb']
        run_name = wandb_config.get('run_name') or f"sft-{self.config['model']['name'].split('/')[-1]}"
        
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            name=run_name,
            config=self.config
        )
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config['training']['num_epochs']):
            if self.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Train epoch
            train_loss = self._train_epoch(epoch)
            
            # Validation
            if (epoch + 1) % 1 == 0:  # Validate every epoch
                val_loss = self._validate()
                
                if self.is_main_process:
                    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    if self.config['wandb']['enabled']:
                        wandb.log({
                            'epoch': epoch + 1,
                            'train_loss_epoch': train_loss,
                            'val_loss_epoch': val_loss
                        })
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(epoch)
        
        if self.is_main_process and self.config['wandb']['enabled']:
            wandb.finish()
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        else:
            pbar = self.train_loader
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            batch = {k: v.to(f'cuda:{self.local_rank}') for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.is_main_process and self.global_step % self.config['training']['logging_steps'] == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    if isinstance(pbar, tqdm):
                        pbar.set_postfix({'loss': f"{loss.item() * grad_accum_steps:.4f}", 'lr': f"{lr:.2e}"})
                    
                    if self.config['wandb']['enabled']:
                        wandb.log({
                            'train_loss': loss.item() * grad_accum_steps,
                            'learning_rate': lr,
                            'global_step': self.global_step
                        })
            
            total_loss += loss.item() * grad_accum_steps
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(f'cuda:{self.local_rank}') for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Gather losses from all processes
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss]).to(f'cuda:{self.local_rank}')
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return avg_loss
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        save_dir = os.path.join(
            self.config['training']['output_dir'],
            f"checkpoint-epoch-{epoch + 1}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with FSDP if enabled
        if self.config['fsdp']['enabled'] and self.world_size > 1:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
                if self.is_main_process:
                    torch.save(state_dict, os.path.join(save_dir, 'pytorch_model.bin'))
        else:
            self.model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Checkpoint saved to {save_dir}")
