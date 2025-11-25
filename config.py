"""
Configuration Module for CyberRAG
Centralized configuration management for all paths and hyperparameters.
"""

import torch

# User Configuration: Please modify these according to your environment
USERNAME = 'czhao93'
BASE_PATH = f'/scratch/{USERNAME}'


# Model Cache Directories
# HuggingFace cache directory for all models (LLM, retriever, etc.)
# This is the standard location where HuggingFace stores downloaded models
CACHE_DIR = f'{BASE_PATH}/huggingface/hub/'


# Dataset Paths
# Knowledge base path
QA_KB_PATH = f'{BASE_PATH}/CyberBot/dataset/kb/'
# Query dataset path
QUERY_PATH = f'{BASE_PATH}/CyberBot/dataset/qa/split/'
# Ontology path (relative to project root)
ONTOLOGY_PATH = 'dataset/ontology/'
# Output directory for results
SAVE_PATH = f'{BASE_PATH}/CyberBot/result/'


# Model Configuration
# Default language model
DEFAULT_LLM_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
# Default retriever model
DEFAULT_RETRIEVER_MODEL = 'facebook/contriever'


# Hyperparameters
# Batch size for RAG generation
BATCH_SIZE = 16
# Number of times to repeat the experiment for statistical significance
REPEAT = 10
# Batch size for ontology validation (recommended: 1)
VALIDATION_BATCH_SIZE = 1
# Number of retrieved documents for RAG
RETRIEVE_K = 1


# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Utility Functions
def get_cache_dir():
    """
    Get the cache directory for HuggingFace models.

    Returns:
        str: Path to the cache directory
    """
    return CACHE_DIR


def get_device():
    """
    Get the computing device (cuda or cpu).

    Returns:
        torch.device: The device to use for computation
    """
    return DEVICE


def print_config():
    """
    Print current configuration for debugging.
    """
    print("=" * 80)
    print("CyberRAG Configuration")
    print("=" * 80)
    print(f"Username: {USERNAME}")
    print(f"Base Path: {BASE_PATH}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"KB Path: {QA_KB_PATH}")
    print(f"Query Path: {QUERY_PATH}")
    print(f"Ontology Path: {ONTOLOGY_PATH}")
    print(f"Save Path: {SAVE_PATH}")
    print(f"Device: {DEVICE}")
    print(f"LLM Model: {DEFAULT_LLM_MODEL}")
    print(f"Retriever Model: {DEFAULT_RETRIEVER_MODEL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Repeat: {REPEAT}")
    print("=" * 80)

