import json
from model import *
from tokenizer import *
import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_from_disk


def main():
    dataset = load_from_disk("/opt/ml/input/data/test_dataset")
    test_dataset = dataset["validation"]
    MODEL_NAME = "klue/bert-base"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    special_tokens = {"additional_special_tokens": ["[Q]", "[D]"]}
    ret_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ret_tokenizer.add_special_tokens(special_tokens)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(ret_tokenizer.vocab_size + 2)

    model.to(device)

    model.load_state_dict(torch.load("/opt/ml/input/code/colbert/best_model/colbert_epoch12.pth"))

    print("opening wiki passage...")
    with open("/opt/ml/input/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print("wiki loaded!!!")

    query = list(test_dataset["question"])
    batched_p_embs = []

    with torch.no_grad():
        model.eval()

        q_seqs_val = tokenize_colbert(query, ret_tokenizer, corpus="query").to("cuda")
        q_emb = model.query(**q_seqs_val).to("cpu")

        print("Start passage embedding......")
        p_embs = []
        for step, p in enumerate(tqdm(context)):
            p = tokenize_colbert(p, ret_tokenizer, corpus="doc").to("cuda")
            p_emb = model.doc(**p).to("cpu").numpy()
            p_embs.append(p_emb)
            if (step + 1) % 200 == 0:
                batched_p_embs.append(p_embs)
                p_embs = []
        batched_p_embs.append(p_embs)

    dot_prod_scores = model.get_score(q_emb, batched_p_embs, eval=True)
    torch.save(dot_prod_scores, f"/opt/ml/input/code/colbert/score/score_test.pth")
    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    torch.save(rank, "/opt/ml/input/code/colbert/rank/test_rank.pth")
    print(rank.size())


if __name__ == "__main__":
    main()
