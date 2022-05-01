
from datasets import DatasetDict, load_from_disk, load_metric
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import os
import pandas as pd
import torch
import torch.nn.functional as F
from dense_retrieval_package import *
import json
import pickle
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

def main():
    dataset = load_data('/opt/ml/input/data/train.csv')
    val_dataset = dataset[3952:]

    MODEL_NAME = 'klue/bert-base'
    #MODEL_NAME = 'bert-base-multilingual-cased'
    

    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    p_encoder = Dense_Retrieval_Model.from_pretrained(MODEL_NAME)
    q_encoder = Dense_Retrieval_Model.from_pretrained(MODEL_NAME)

    p_encoder.load_state_dict(torch.load('/opt/ml/input/code/dense_model/p_encoder_epoch7.pth'))
    q_encoder.load_state_dict(torch.load('/opt/ml/input/code/dense_model/q_encoder_epoch7.pth'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p_encoder.to(device)
    q_encoder.to(device)

    print('opening wiki passage...')
    with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print('wiki loaded!!!')

    query = list(val_dataset['query'])
    ground_truth = list(val_dataset['context'])


    p_embs=[]
    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()

        q_seqs_val = tokenizer(query, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        q_emb = q_encoder(**q_seqs_val).to('cpu')
        for p in tqdm(context):
            p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            p_emb = p_encoder(**p).to('cpu').numpy()
            p_embs.append(p_emb)

        
    p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
    #torch.save(p_embs,'/opt/ml/input/code/dense_embeds/dense_embedding.pth')
    p_embs = torch.load('/opt/ml/input/code/dense_embeds/dense_embedding.pth')  # (num_passage, emb_dim)
    print(p_embs.size(), q_emb.size())

    length = len(val_dataset['context'])
        
    dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    torch.save(rank,'/opt/ml/input/code/rank.pth')
    print(rank.size())
    

    k = 100
    score=0

    for idx in range(length):
        print(dot_prod_scores[idx])
        print(rank[idx])
        print()
        for i in range(k):
            if ground_truth[idx] == context[rank[idx][i]]:
                print("[Search query]\n", query[idx], "\n")
                print("[Ground truth passage]")
                print(ground_truth[idx][:50], "\n")
                score+=1
                print("Top-%d passage with score %.4f" % (i+1, dot_prod_scores[idx][rank[idx][i]]))
                print(context[rank[idx][i]][:50])

    print(f'{score} over {length} context found!!')
    print(f'final score is {score/length}')

    #print(context[54668])
    #print(len(context[54668]))
    #print(context[30499])
    #print(len(context[30499]))
    

if __name__=='__main__':
    main()