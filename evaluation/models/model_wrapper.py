import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
import logging
import os

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for model loading and inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.checkpoint_path = self.model_config['checkpoint_path']
        self.use_vllm = self.model_config.get('use_vllm', True)
        
        

        if self.checkpoint_path and not os.path.exists(os.path.join(self.checkpoint_path, "config.json")):
            config = AutoConfig.from_pretrained(self.model_config['base_model'])
            config.save_pretrained(self.checkpoint_path)

        if not self.checkpoint_path:
            self.checkpoint_path = self.model_config['base_model']

        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_fsdp()
    
    def _load_vllm(self):
        """Load model using vLLM for fast inference"""
        logger.info("Loading model with vLLM")
        self.model = LLM(
            model=self.checkpoint_path,
            tensor_parallel_size=self.model_config.get('tensor_parallel_size', 1),
            max_model_len=self.model_config.get('max_model_len', 4096),
            trust_remote_code=True,
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.model_config.get('temperature', 0.7),
            top_p=self.model_config.get('top_p', 0.9),
            max_tokens=self.model_config.get('max_tokens', 2048),
        )
        
        logger.info("vLLM model loaded successfully")
    
    def _load_fsdp(self):
        """Load model using FSDP for distributed inference"""
        logger.info("Loading model with FSDP")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.model.eval()
        logger.info("FSDP model loaded successfully")
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        if self.use_vllm:
            return self._generate_vllm(prompts)
        else:
            return self._generate_fsdp(prompts)
    
    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """Generate using vLLM"""
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def _generate_fsdp(self, prompts: List[str]) -> List[str]:
        """Generate using FSDP model"""
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.get('max_model_len', 4096)
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.get('max_tokens', 2048),
                temperature=self.model_config.get('temperature', 0.7),
                top_p=self.model_config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input prompt from output
            input_len = inputs['input_ids'][i].shape[0]
            generated = output[input_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts