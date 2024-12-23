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

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

cache_dir = '/scratch/czhao93/huggingface/hub/'
qa_kb_path='/scratch/czhao93/CyberBot/dataset/kb/'
query_path = '/scratch/czhao93/CyberBot/dataset/qa/split/'
save_path = '/scratch/czhao93/CyberBot/result/'

batch_size = 16
repeat = 10

if __name__ == '__main__':

    # load retrieval model
    qa_retriever, qa_tokenizer = load_qa_retriever(cache_dir=cache_dir)
    kb = load_qa_kb(path=qa_kb_path)

    # load llm
    model, tokenizer = load_llm(cache_dir=cache_dir)

    query_name = os.listdir(query_path)
    for name in query_name:
        metrics = []
        for _ in range(repeat):
            
            # load queries
            current_path = os.path.join(query_path, name)
            df = pd.read_csv(current_path)
            questions = df['Question']
            answers = df['Answer']
    
            # retrieve context
            prompts = retrieve_qa_context(questions.tolist(), kb, qa_retriever, qa_tokenizer, k=1)
    
            # load dataset
            dataset = QADataset(prompts, answers.tolist(), tokenizer, max_len=256)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
            raw_predictions = []
            with torch.no_grad():
                for batch in tqdm(data_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256, num_beams=4)
                    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    raw_predictions.extend(decoded_preds)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
    
            predictions = remove_prompt(prompts, raw_predictions)
    
            results = [{"Question": q, 'Retrieval': r, 'Answer': a, "Prediction": p} for q, r, a, p in zip(questions, prompts, answers, predictions)]
            # save_name = "rag_{0}.json".format(name)
            # with open(os.path.join(save_path,save_name), 'w') as f:
            #     json.dump(results, f, indent=4)
    
            references = [[ref] for ref in answers]
            bertscore_f1, meteor_score, rouge1, rouge2 = evaluate_answer(predictions, references)
            
            metric = [bertscore_f1, meteor_score, rouge1, rouge2]
            metrics.append(metric)
        
        metrics = np.array(metrics)
        metric_path = os.path.join(save_path,'metric_{0}.npy'.format(name))
        np.save(metric_path, metrics)
    