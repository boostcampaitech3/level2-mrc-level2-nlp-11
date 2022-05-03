
import json
import torch.nn.functional as F
from model import *
from tokenizer import *
import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer



def main():
    epoch=12
    MODEL_NAME = 'klue/bert-base'

    dataset = load_data('/opt/ml/input/data/train.csv')
    val_dataset = dataset[3952:]



    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    
    model.load_state_dict(torch.load(f'/opt/ml/input/code/colbert/best_model/colbert_epoch{epoch}.pth'))


    print('opening wiki passage...')
    with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print('wiki loaded!!!')
    
    query = list(val_dataset['query'])
    ground_truth = list(val_dataset['context'])


    batched_p_embs = []
    with torch.no_grad():

        model.eval()

        #토크나이저 수정필요
        q_seqs_val = tokenize_colbert(query,tokenizer,corpus='query').to('cuda')
        q_emb = model.query(**q_seqs_val).to('cpu')

        print(q_emb.size())

        print('Start passage embedding......')
        p_embs=[]
        for step,p in enumerate(tqdm(context)):
            p = tokenize_colbert(p,tokenizer,corpus='doc').to('cuda')
            p_emb = model.doc(**p).to('cpu').numpy()
            p_embs.append(p_emb)
            if (step+1)%200 ==0:
                batched_p_embs.append(p_embs)
                p_embs=[]
        batched_p_embs.append(p_embs)



    print('passage tokenizing done!!!!')
    length = len(val_dataset['context'])


    dot_prod_scores = model.get_score(q_emb,batched_p_embs,eval=True)
        
    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    print(rank.size())
    torch.save(rank,f'/opt/ml/input/code/colbert/rank/rank_epoch{epoch}.pth')
    
    k = 100
    score=0

    for idx in range(length):
        print(dot_prod_scores[idx])
        print(rank[idx])
        print()
        for i in range(k):
            if ground_truth[idx] == context[rank[idx][i]]:
                score+=1


    print(f'{score} over {length} context found!!')
    print(f'final score is {score/length}')

if __name__ =='__main__':
    main()