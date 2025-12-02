"""
Language Model Loading Module
Provides functions to load and configure language models for text generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import configuration
from config import CACHE_DIR, DEFAULT_LLM_MODEL, DEVICE

def load_llm(model_name=DEFAULT_LLM_MODEL, cache_dir=None):
    """
    Load a language model and its tokenizer from HuggingFace.
    
    Args:
        model_name: Name or path of the model to load (default: from config)
        cache_dir: Directory to cache downloaded models (default: from config)
    
    Returns:
        tuple: (model, tokenizer)
            - model: Loaded and configured language model (moved to device)
            - tokenizer: Corresponding tokenizer with padding configured
    """
    # Use default cache directory from config if not provided
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer
