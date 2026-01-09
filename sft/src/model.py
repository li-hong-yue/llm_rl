from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load model and tokenizer with proper configuration"""
    model_config = config['model']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        trust_remote_code=model_config.get('trust_remote_code', True),
        padding_side='right'
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    torch_dtype = dtype_map.get(
        model_config.get('torch_dtype', 'bfloat16'),
        torch.bfloat16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get('trust_remote_code', True),
        #use_flash_attention_2=model_config.get('use_flash_attention', False)
    )
    
    # Enable gradient checkpointing for memory efficiency
    # model.gradient_checkpointing_enable()   
    # tensor size mismatch during gradient checkpointing with FSDP
    
    return model, tokenizer