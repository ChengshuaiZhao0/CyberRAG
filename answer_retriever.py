"""
Answer Retriever Module
Provides functions for loading retriever models and knowledge base, 
and retrieving relevant context for questions.
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Import configuration
from config import CACHE_DIR, DEFAULT_RETRIEVER_MODEL, DEVICE, QA_KB_PATH

def load_qa_retriever(model_name=DEFAULT_RETRIEVER_MODEL, cache_dir=None):
    """
    Load the QA retriever model and tokenizer.
    
    Args:
        model_name: Name or path of the retriever model (default: from config)
        cache_dir: Directory to cache downloaded models (default: from config)
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Use default cache directory from config if not provided
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
    return model, tokenizer

def load_qa_kb(name='kb.csv', path=None):
    """
    Load the knowledge base from CSV file.
    
    Args:
        name: Name of the KB file (default: 'kb.csv')
        path: Path to the KB directory (default: from config)
    
    Returns:
        pd.DataFrame: Knowledge base as a pandas DataFrame
    """
    # Use default path from config if not provided
    if path is None:
        path = QA_KB_PATH
    
    kb = pd.read_csv(os.path.join(path, name))
    return kb

def compute_embedding(texts, model, tokenizer):
    """
    Compute embeddings for given texts using the retriever model.
    
    Args:
        texts: List of text strings to embed
        model: Retriever model
        tokenizer: Model tokenizer
    
    Returns:
        torch.Tensor: Mean pooled embeddings
    """
    inputs = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def normalize(matrix):
    """
    Normalize a matrix by dividing each row by its L2 norm.
    
    Args:
        matrix: torch.Tensor of shape (n, d) where n is the number of vectors and d is the dimension
    
    Returns:
        torch.Tensor: Normalized matrix with unit-length rows
    """
    return matrix / matrix.norm(dim=1, keepdim=True)

def retrieve_question(query_embedding, doc_embedding, k=3):
    """
    Retrieve top-k most similar documents for each query using cosine similarity.
    
    Args:
        query_embedding: torch.Tensor of shape (n_queries, embedding_dim) - query embeddings
        doc_embedding: torch.Tensor of shape (n_docs, embedding_dim) - document embeddings
        k: int, number of top documents to retrieve for each query (default: 3)
    
    Returns:
        torch.Tensor: Indices of top-k documents for each query, shape (n_queries, k)
    """
    similarities = torch.matmul(query_embedding, doc_embedding.T)
    top_k_indices = torch.topk(similarities, k).indices.cpu()
    return top_k_indices

def apply_prompt(query, document, usr_prompt=None):
    """
    Format a prompt for RAG by combining document, question, and instructions.
    
    Args:
        query: str, the user's question
        document: str, the retrieved document context
        usr_prompt: str, optional custom instruction prompt. If None, uses default instruction
    
    Returns:
        str: Formatted prompt string ready for language model input
    """
    if not usr_prompt:
        usr_prompt = "Answer the users QUESTION using the DOCUMENT text above.\nKeep your answer ground in the facts of the DOCUMENT.\nIf the DOCUMENT doesn't contain the facts to answer the QUESTION give a response based on you knowledge."
    result = 'DOCUMENT:\n{0}\n\nQUESTION:\n{1}\n\nINSTRUCTIONS:\n{2}'.format(document,query,usr_prompt)

    return result

def retrieve_qa_context(queries, kb, model, tokenizer, k=3):
    """
    Retrieve relevant context from knowledge base for given queries and format as prompts.
    
    This function performs the following steps:
    1. Compute embeddings for queries and knowledge base questions
    2. Find top-k most similar KB entries for each query
    3. Format retrieved context into prompts for language model
    
    Args:
        queries: list of str, user questions to retrieve context for
        kb: pd.DataFrame, knowledge base with 'Question' and 'Answer' columns
        model: Retriever model for computing embeddings
        tokenizer: Model tokenizer
        k: int, number of top documents to retrieve per query (default: 3)
    
    Returns:
        list of str: Formatted prompts ready for language model, one per query
    """
    kb_question = kb['Question']
    kb_answer = kb['Answer']

    query_embedding = compute_embedding(queries, model, tokenizer)
    doc_embedding = compute_embedding(kb_question.tolist(), model, tokenizer)
    query_embedding = normalize(query_embedding)
    doc_embedding = normalize(doc_embedding)
   
    doc_mat = retrieve_question(query_embedding, doc_embedding, k=k)
    prompt_list = []
    for doc_id, query in zip(doc_mat, queries):
        # doc_lis = ['Example {0}: {1}\nAnswer: {2}\n'.format(i, kb_question[idx], kb_answer[idx]) for i, idx in enumerate(doc_id.tolist(), 1)]
        doc_lis = ['{0} {1}'.format(kb_question[idx], kb_answer[idx]) for idx in doc_id.tolist()]
        document = ''.join(doc_lis)
        prompt = apply_prompt(query, document)
        prompt_list.append(prompt)

    return prompt_list

if __name__ == '__main__':
    model, tokenizer = load_qa_retriever()
    queries = ['What criteria are used to determine the severity level of a vulnerability?', 'What methods can a system administrator employ to assign access privileges to a user?',  'What options are typically available to a system administrator for granting access privileges to a user?']
    kb = load_qa_kb()
    prompt_list = retrieve_qa_context(queries, kb, model, tokenizer, k=3)
    for prompt in prompt_list:
        print(prompt)
        print('-----------------------------------')