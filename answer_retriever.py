import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_qa_retriever(model_name='facebook/contriever', cache_dir='/scratch/czhao93/llms/'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    return model, tokenizer

def load_qa_kb(name='kb.csv', path='/scratch/czhao93/aisecrag/dataset/kb'):
    kb = pd.read_csv(os.path.join(path, name))
    return kb

def compute_embedding(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def normalize(matrix):
    return matrix / matrix.norm(dim=1, keepdim=True)

def retrieve_question(query_embedding, doc_embedding, k=3):
    similarities = torch.matmul(query_embedding, doc_embedding.T)
    top_k_indices = torch.topk(similarities, k).indices.cpu()
    return top_k_indices

def apply_prompt(query, document, usr_prompt=None):
    # if not usr_prompt:
    #     usr_prompt = 'Please answer my question using the following examples. If the answer is in the examples, select the most relevant one. Otherwise, generate a new response.'
    # result = '{0}\n{1}My question: {2}\nAnswer: '.format(usr_prompt, document, query)
    if not usr_prompt:
        usr_prompt = 'Answer the users QUESTION using the DOCUMENT text above.\nKeep your answer ground in the facts of the DOCUMENT.\nIf the DOCUMENT doesn’t contain the facts to answer the QUESTION give a response based on you knowledge.'
    result = 'DOCUMENT:\n{0}\n\nQUESTION:\n{1}\n\nINSTRUCTIONS:\n{2}'.format(document,query,usr_prompt)

    return result

def retrieve_qa_context(queries, kb, model, tokenizer, k=3):
    '''Retrieve context from the knowledge base for the given queries.'''
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