import json
import argparse
from lib2to3.pgen2 import token
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from xml.etree.ElementPath import prepare_star
import faiss
import numpy as np
import pandas as pd
from sklearn import datasets
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.dataset import random_split
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from tqdm import trange
from utils_qa import preprocess_df


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class DenseRetrieval:
    def __init__(self, p_encoder, q_encoder, tokenizer, args, num_neg):
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.tokenizer = tokenizer
        self.args = args
        print(self.args)
        self.num_neg = num_neg
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self):
        # with open(
        #     os.path.join("../data/", "wikipedia_documents.json"), "r", encoding="utf-8"
        # ) as f:
        #     wiki = json.load(f)

        # contexts = list(
        #     dict.fromkeys([v["text"] for v in wiki.values()])
        # )  # set 은 매번 순서가 바뀌므로
        # print(f"Lengths of unique contexts : {len(contexts)}")
        # # ids = list(range(len(contexts)))

        org_dataset = load_from_disk("../data/train_dataset")
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
        full_ds = preprocess_df(full_ds)
        corpus = np.array(full_ds["context"])
        p_with_neg = []

        for c in full_ds["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = self.tokenizer(
            full_ds["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        p_seqs = self.tokenizer(
            p_with_neg, padding="max_length", truncation=True, return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, self.num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, self.num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, self.num_neg + 1, max_len
        )

        dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        print("Dataset is prepared...")
        return dataset

    def train(self):
        print("training start")
        train_dataloader = DataLoader(
            self.dataset,
            sampler=RandomSampler(self.dataset),
            batch_size=self.args.per_device_train_batch_size,
        )

        ### 추가 부분 ###
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        ### 추가 부분 ###

        t_total = (
            len(train_dataloader)
            // self.args.gradient_accumulation_steps
            * self.args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Start training!
        global_step = 0
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):

                self.q_encoder.train()
                self.p_encoder.train()
                targets = torch.zeros(self.args.per_device_train_batch_size).long()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                    targets = targets.cuda()

                p_inputs = {
                    "input_ids": batch[0].view(
                        self.args.per_device_train_batch_size * (self.num_neg + 1), -1
                    ),
                    "attention_mask": batch[1].view(
                        self.args.per_device_train_batch_size * (self.num_neg + 1), -1
                    ),
                    "token_type_ids": batch[2].view(
                        self.args.per_device_train_batch_size * (self.num_neg + 1), -1
                    ),
                }

                q_inputs = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                }

                p_outputs = self.p_encoder(
                    **p_inputs
                )  # (batch_size*(num_neg+1), emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                # Calculate similarity score & loss
                p_outputs = p_outputs.view(
                    self.args.per_device_train_batch_size, -1, self.num_neg + 1
                )
                q_outputs = q_outputs.view(self.args.per_device_train_batch_size, 1, -1)

                sim_scores = torch.bmm(
                    q_outputs, p_outputs
                ).squeeze()  # (batch_size, num_neg+1)
                sim_scores = sim_scores.view(self.args.per_device_train_batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()

        print("training is done")

    def retrieve(
        self,
    ):
        pass

    def get_relevant_doc(
        self,
    ):
        pass

    def get_relevant_doc_bulk(
        self,
    ):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        metavar="./data/train_dataset",
        default="./data/train_dataset",
        type=str,
        help="",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-cased",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--num_neg", default=3, metavar=3, type=int, help="")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    p_encoder = BertEncoder.from_pretrained(args.model_name_or_path)
    q_encoder = BertEncoder.from_pretrained(args.model_name_or_path)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    T_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    retriever = DenseRetrieval(
        p_encoder, q_encoder, tokenizer, T_args, num_neg=args.num_neg
    )
    retriever.train()

    # query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retrieve(full_ds)
    #     df["correct"] = df["original_context"] == df["context"]
    #     print(
    #         "correct retrieval result by exhaustive search",
    #         df["correct"].sum() / len(df),
    #     )

    # with timer("single query by exhaustive search"):
    #     scores, indices = retriever.retrieve(query)
