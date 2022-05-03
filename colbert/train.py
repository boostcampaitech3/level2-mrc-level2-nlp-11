from datasets import DatasetDict, load_from_disk, load_metric
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizer import *
from model import *
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


    set_seed(42)
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        num_train_epochs=12,
        weight_decay=0.01
    )

    MODEL_NAME = 'klue/bert-base'
    tokenizer = load_tokenizer(MODEL_NAME)


    dataset = load_data('/opt/ml/input/data/train.csv')
    additional_dataset = load_data('/opt/ml/input/data/masked_query_and_titled_passage.csv')


    print('csv loading.....')
    train_dataset = dataset[:3952].append(additional_dataset)
    train_dataset = train_dataset.reset_index(drop=True)
    print('dataset tokenizing.......')



    #토크나이저 수정필요
    train_context, train_query = tokenize_colbert(train_dataset, tokenizer,corpus='both')


    train_dataset = TensorDataset(train_context['input_ids'], train_context['attention_mask'], train_context['token_type_ids'], 
                        train_query['input_ids'], train_query['attention_mask'], train_query['token_type_ids'])


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    model.to(device)


    trained_model = train(args,train_dataset,model)
    torch.save(trained_model.state_dict(), '/opt/ml/input/code/dense_model/colbert.pth')



def train(args, dataset, model):
  
    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    

    ### 추가 부분 ###
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    ### 추가 부분 ###
    
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0
    
    model.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss=0
        steps=0

        for step, batch in enumerate(epoch_iterator):
            steps+=1
            model.train()
            
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                        }
            
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

            #outputs with similiarity score
            outputs = model(p_inputs,q_inputs)

            # target: position of positive samples = diagonal element 
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to('cuda')

            sim_scores = F.log_softmax(outputs, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            #print(loss)
            total_loss+=loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), f'/opt/ml/input/code/dense_model/colbert_epoch{epoch+1}.pth')
        print(total_loss/steps)

        
    return model


if __name__ == '__main__':
    main()
