import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import AutoModel, BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup



class Dense_Retrieval_Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Dense_Retrieval_Model, self).__init__(config)

        #모델 수정 가능-현재는 기존 BertModel 사용중
        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(self, input_ids, 
                attention_mask=None, token_type_ids=None): 
    
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        
        pooled_output = outputs[1]

        return pooled_output

class ColbertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(ColbertModel, self).__init__(config)

        #모델 수정 가능-현재는 기존 BertModel 사용중
        self.similarity_metric = 'cosine'
        self.dim = 128
        self.batch = 8
        self.bert = BertModel(config)
        self.init_weights()
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)  


    def forward(self, p_inputs,q_inputs):
        Q = self.query(**q_inputs)
        D = self.doc(**p_inputs)
        return self.get_score(Q,D)


    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)


    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        D = self.linear(D)
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def get_score(self,Q,D,eval=False):
        if eval:
            if self.similarity_metric == 'cosine':
                final_score=torch.tensor([])
                for D_batch in tqdm(D):
                    D_batch = torch.Tensor(D_batch).squeeze()
                    p_seqeunce_output=D_batch.transpose(1,2) #(batch_size,hidden_size,p_sequence_length)
                    q_sequence_output=Q.view(300,1,-1,self.dim) #(batch_size, 1, q_sequence_length, hidden_size)
                    dot_prod = torch.matmul(q_sequence_output,p_seqeunce_output) #(batch_size,batch_size, q_sequence_length, p_seqence_length)
                    max_dot_prod_score =torch.max(dot_prod, dim=3)[0] #(batch_size,batch_size,q_sequnce_length)
                    score = torch.sum(max_dot_prod_score,dim=2)#(batch_size,batch_size)
                    final_score = torch.cat([final_score,score],dim=1)
                print(final_score.size())
                return final_score

        else:
            if self.similarity_metric == 'cosine':

                p_seqeunce_output=D.transpose(1,2) #(batch_size,hidden_size,p_sequence_length)
                q_sequence_output=Q.view(self.batch,1,-1,self.dim) #(batch_size, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(q_sequence_output,p_seqeunce_output) #(batch_size,batch_size, q_sequence_length, p_seqence_length)
                max_dot_prod_score =torch.max(dot_prod, dim=3)[0] #(batch_size,batch_size,q_sequnce_length)
                final_score = torch.sum(max_dot_prod_score,dim=2)#(batch_size,batch_size)
            
                return final_score

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)


    #def mask(self, input_ids):
    #    mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    #    return mask






def preprocess(dataset):
    return dataset

def load_data(datadir):
    dataset = pd.read_csv(datadir)
    dataset = preprocess(dataset)
    dataset = pd.DataFrame({'context':dataset['context'],'query':dataset['question'],'title':dataset['title']})
    return dataset

def add_negative_sample(dataset,num_neg):
    corpus = list(set(dataset['context']))
    corpus = np.array(corpus)
    p_with_neg = []
    for c in dataset['context']:
        while True:
            neg_idxs = np.random.randint(len(corpus), size=num_neg)

            if not c in corpus[neg_idxs]:
                p_neg = corpus[neg_idxs]

                p_with_neg.append(c)
                p_with_neg.extend(p_neg)
                break
    return p_with_neg



def tokenized_data(dataset,tokenizer,train,num_neg):
    tokenized_query = tokenizer(
        list(dataset['query']),
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        )
    tokenized_title = tokenizer(
        list(dataset['title']),
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        )

    if train == True and num_neg != 'in_batch':
        p_with_neg = add_negative_sample(dataset,num_neg)
        tokenized_context = tokenizer(
            p_with_neg,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            )
        max_len = tokenized_context['input_ids'].size(-1)
        tokenized_context['input_ids'] = tokenized_context['input_ids'].view(-1, num_neg+1, max_len)
        tokenized_context['attention_mask'] = tokenized_context['attention_mask'].view(-1, num_neg+1, max_len)
        tokenized_context['token_type_ids'] = tokenized_context['token_type_ids'].view(-1, num_neg+1, max_len)
    
    else:
        tokenized_context = tokenizer(
            list(dataset['context']),
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            )



    return tokenized_context,tokenized_query,tokenized_title


def tokenize_colbert(dataset,tokenizer,corpus):

    #for inference
    if corpus == 'query':
        preprocessed_data=[]
        for query in dataset:
            preprocessed_data.append('[Q] '+query)

        tokenized_query = tokenizer(
        preprocessed_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
        )
        return tokenized_query

    elif corpus == 'doc':
        preprocessed_data = '[D] '+dataset
        tokenized_context = tokenizer(
            preprocessed_data,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            )

        return tokenized_context

    #for train
    else:
        preprocessed_query=[]
        preprocessed_context=[]
        for query,context in zip(dataset['query'],dataset['context']):
            preprocessed_context.append('[D] '+context)
            preprocessed_query.append('[Q] '+query)
        tokenized_query = tokenizer(
        preprocessed_query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
        )

        tokenized_context = tokenizer(
        preprocessed_context,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        )
        return tokenized_context, tokenized_query