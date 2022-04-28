import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup



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

    if train == True:
        p_with_neg = add_negative_sample(dataset,num_neg)
        tokenized_context = tokenizer(
            list(p_with_neg),
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
