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
    set_seed(42)
    num_neg=3
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01
    )
    MODEL_NAME = 'klue/bert-base'


    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    dataset = load_data('/opt/ml/input/data/train.csv')

    train_dataset = dataset[:3952]
    val_dataset = dataset[3952:]


    train_context, train_query, train_title = tokenized_data(train_dataset, tokenizer,train=True, num_neg=num_neg)
    val_context, val_query, val_title = tokenized_data(val_dataset, tokenizer,train=False,num_neg=num_neg)


    train_dataset = TensorDataset(train_context['input_ids'], train_context['attention_mask'], train_context['token_type_ids'], 
                        train_query['input_ids'], train_query['attention_mask'], train_query['token_type_ids'])

    #val dataset 필요

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    passage_model = Dense_Retrieval_Model.from_pretrained(MODEL_NAME)
    query_model = Dense_Retrieval_Model.from_pretrained(MODEL_NAME)
    
    passage_model.to(device)
    query_model.to(device)


    p_encoder,q_encoder = train(args,num_neg,train_dataset,passage_model,query_model)
    



    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()
        p_embs = []
        

        pickle_name = f"dense_embedding.bin"
        emd_path =f'/opt/ml/input/code/dense_embeds/{pickle_name}'

        #if os.path.isfile(emd_path):
        #    with open(emd_path, "rb") as file:
        #        p = pickle.load(file)

        #else:
        #    print('opening wiki passage...')
        #    with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        #        wiki = json.load(f)
        #    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        #    print('wiki loaded!!!')
        #
        #    print('Start passage tokenizing.....')
        #    for p in tqdm(context):
        #        p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        #        p_emb = p_encoder(**p).to('cpu').numpy()
        #        p_embs.append(p_emb)
        #
        #    with open(emd_path, "wb") as file:
        #        pickle.dump(p_embs, file)
        #    print("Embedding passage saved.")
        print('opening wiki passage...')
        with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
                wiki = json.load(f)
        context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print('wiki loaded!!!')

        print('Start passage tokenizing.....')
        for p in tqdm(context):
            p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            p_emb = p_encoder(**p).to('cpu').numpy()
            p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
        print('passage tokenizing done!!!!')

        score=0
        length = len(val_dataset['context'])
        for ground_truth,query in zip(val_dataset['context'],val_dataset['query']):
            q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            q_emb = q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

            #print(p_embs.size(), q_emb.size())

            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
            dot_prod_scores = F.softmax(dot_prod_scores)
            #print(dot_prod_scores.size())

            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
            print('-'*30)
            print(dot_prod_scores)
            print(rank)

            k = 80


            for i in range(k):
                if context[rank[i]] == ground_truth:
                    print("[Search query]\n", query, "\n")
                    print("[Ground truth passage]")
                    print(ground_truth[:50], "\n")
                    print("Top-%d passage with score %.4f" % (i+1, dot_prod_scores.squeeze()[rank[i]]))
                    score+=1
                    print(context[rank[i]][:50])
                    print()

        print(f'{score} over {length} context found!!')
        print(f'final score is {score/length}')

        torch.save(q_encoder.state_dict(), '/opt/ml/input/code/dense_embeds/q_encoder.pth')
        torch.save(p_encoder.state_dict(), '/opt/ml/input/code/dense_embeds/p_encoder.pth')



def train(args,num_neg,dataset,p_model,q_model):
    
  # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    print("Start Training!!!!")
    global_step = 0
    
    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    #target 수정, tqdm 바 수정, bestmodel 수정
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss=0
        steps=0
        for step, batch in enumerate(epoch_iterator):
            steps+=1
            q_model.train()
            p_model.train()
        
            targets = torch.zeros(args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                targets = targets.cuda()

            p_inputs = {'input_ids': batch[0].view(
                                            args.per_device_train_batch_size*(num_neg+1), -1),
                        'attention_mask': batch[1].view(
                                            args.per_device_train_batch_size*(num_neg+1), -1),
                        'token_type_ids': batch[2].view(
                                            args.per_device_train_batch_size*(num_neg+1), -1)
                        }
            
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}
            
            p_outputs = p_model(**p_inputs)  #(batch_size*(num_neg+1), emb_dim)
            q_outputs = q_model(**q_inputs)  #(batch_size*, emb_dim)

            # Calculate similarity score & loss
            p_outputs = p_outputs.view(args.per_device_train_batch_size, -1, num_neg+1)
            q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

            sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
            sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            total_loss+=loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1
            
            torch.cuda.empty_cache()
        print(total_loss/steps)

        
    return p_model, q_model


if __name__ == '__main__':
    main()

