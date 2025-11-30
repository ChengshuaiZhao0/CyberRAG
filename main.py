"""
Main script for CyberRAG: Retrieval-Augmented Generation with Ontology Validation
"""

import os
import gc
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from llm import load_llm
from answer_retriever import load_qa_retriever, load_qa_kb, retrieve_qa_context
from utils import QADataset, remove_prompt, evaluate_answer
from validation import validate_batch

# Import configuration from centralized config module
from config import (
    CACHE_DIR, QA_KB_PATH, QUERY_PATH, SAVE_PATH, ONTOLOGY_PATH,
    BATCH_SIZE, REPEAT, VALIDATION_BATCH_SIZE, DEVICE
)

# Helper Functions
def process_single_repeat(query_file_path, model, tokenizer, kb, qa_retriever, qa_tokenizer,
                          batch_size, device):
    """
    Process a single repeat iteration: load queries, retrieve context, generate predictions, and evaluate.

    Args:
        query_file_path: Path to the CSV file containing questions and answers
        model: Loaded language model
        tokenizer: Model tokenizer
        kb: Knowledge base for retrieval
        qa_retriever: QA retriever model
        qa_tokenizer: QA tokenizer
        batch_size: Batch size for generation
        device: Computing device

    Returns:
        tuple: (questions, answers, predictions, metric)
            - questions: List of questions
            - answers: Ground truth answers
            - predictions: Generated predictions
            - metric: List of evaluation metrics [bertscore_f1, meteor_score, rouge1, rouge2]
    """
    # Load queries from CSV file
    df = pd.read_csv(query_file_path)
    questions = df['Question']
    answers = df['Answer']

    # Retrieve relevant context from knowledge base
    prompts = retrieve_qa_context(questions.tolist(), kb, qa_retriever, qa_tokenizer, k=1)

    # Create dataset and dataloader for batch processing
    dataset = QADataset(prompts, answers.tolist(), tokenizer, max_len=256)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Generate predictions
    raw_predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                num_beams=4
            )
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            raw_predictions.extend(decoded_preds)

            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

    # Remove prompt from predictions to get clean answers
    predictions = remove_prompt(prompts, raw_predictions)

    # Evaluate predictions against ground truth
    references = [[ref] for ref in answers]
    bertscore_f1, meteor_score, rouge1, rouge2 = evaluate_answer(predictions, references)

    metric = [bertscore_f1, meteor_score, rouge1, rouge2]

    return questions, answers, predictions, metric


def run_ontology_validation(questions, predictions, model, tokenizer, ontology_path,
                           device, save_path, query_name):
    """
    Run ontology validation on predictions and save results.

    Args:
        questions: List of questions
        predictions: Generated predictions to validate
        model: Loaded language model
        tokenizer: Model tokenizer
        ontology_path: Path to ontology directory
        device: Computing device
        save_path: Directory to save validation results
        query_name: Name of the query file (for output naming)

    Returns:
        validation_data: List of dictionaries containing validation results
    """
    print(f"\nRunning ontology validation for {query_name}...")

    # Run batch validation
    validation_results = validate_batch(
        questions=questions.tolist(),
        answers=predictions,
        model=model,
        tokenizer=tokenizer,
        ontology_path=ontology_path,
        device=device,
        batch_size=VALIDATION_BATCH_SIZE
    )

    # Create validation results dictionary
    validation_data = [
        {
            "Question": q,
            "Prediction": p,
            "Validation": v
        }
        for q, p, v in zip(questions, predictions, validation_results)
    ]

    # Print validation summary
    print(f"Validation completed for {len(validation_results)} samples")

    # Save validation results to JSON
    validation_filename = f'validation_{query_name.replace(".csv", "")}.json'
    validation_path = os.path.join(save_path, validation_filename)
    with open(validation_path, 'w') as f:
        json.dump(validation_data, f, indent=4)
    print(f"Validation results saved to {validation_path}")

    return validation_data


def process_query_file(query_file_name, query_path, model, tokenizer, kb, qa_retriever,
                      qa_tokenizer, ontology_path, device, save_path, repeat, batch_size):
    """
    Process a single query file: run multiple repeats, collect metrics, and run validation.

    Args:
        query_file_name: Name of the query CSV file
        query_path: Directory containing query files
        model: Loaded language model
        tokenizer: Model tokenizer
        kb: Knowledge base for retrieval
        qa_retriever: QA retriever model
        qa_tokenizer: QA tokenizer
        ontology_path: Path to ontology directory
        device: Computing device
        save_path: Directory to save results
        repeat: Number of times to repeat the experiment
        batch_size: Batch size for generation
    """
    query_file_path = os.path.join(query_path, query_file_name)
    metrics = []

    # Run multiple repeats for statistical significance
    for repeat_idx in range(repeat):
        print(f"\nProcessing {query_file_name} - Repeat {repeat_idx + 1}/{repeat}")
        questions, answers, predictions, metric = process_single_repeat(
            query_file_path, model, tokenizer, kb, qa_retriever, qa_tokenizer,
            batch_size, device
        )
        metrics.append(metric)

    # Run ontology validation on the final predictions
    run_ontology_validation(
        questions, predictions, model, tokenizer, ontology_path,
        device, save_path, query_file_name
    )

    # Save metrics
    metrics_array = np.array(metrics)
    metric_filename = f'metric_{query_file_name.replace(".csv", "")}.npy'
    metric_path = os.path.join(save_path, metric_filename)
    np.save(metric_path, metrics_array)
    print(f"Metrics saved to {metric_path}")


# Main Execution
if __name__ == '__main__':
    # Load retrieval model and knowledge base
    print("Loading retrieval model and knowledge base...")
    qa_retriever, qa_tokenizer = load_qa_retriever(cache_dir=CACHE_DIR)
    kb = load_qa_kb(path=QA_KB_PATH)
    print(f"Knowledge base loaded with {len(kb)} entries")
    
    # Load language model
    print("Loading language model...")
    model, tokenizer = load_llm(cache_dir=CACHE_DIR)
    print("Model loaded successfully")
    
    # Get list of query files
    query_files = [f for f in os.listdir(QUERY_PATH) if f.endswith('.csv')]
    if not query_files:
        print(f"Warning: No CSV files found in {QUERY_PATH}")
    
    # Process each query file
    for query_file_name in query_files:
        print(f"\n{'='*80}")
        print(f"Processing query file: {query_file_name}")
        print(f"{'='*80}")
        
        process_query_file(
            query_file_name, QUERY_PATH, model, tokenizer, kb, 
            qa_retriever, qa_tokenizer, ONTOLOGY_PATH, DEVICE, 
            SAVE_PATH, REPEAT, BATCH_SIZE
        )

    print(f"\n{'='*80}")
    print("All query files processed successfully!")
    print(f"{'='*80}")
