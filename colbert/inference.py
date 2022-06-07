def run_colbert_retrieval(datasets):
    test_dataset = datasets["validation"].flatten_indices().to_pandas()
    test_dataset1=test_dataset[:300]
    test_dataset2=test_dataset[300:]
    MODEL_NAME = 'klue/bert-base'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    special_tokens={'additional_special_tokens' :['[Q]','[D]']}
    ret_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ret_tokenizer.add_special_tokens(special_tokens)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(ret_tokenizer.vocab_size + 2)


    model.to(device)


    model.load_state_dict(torch.load('/opt/ml/input/code/dense_model/colbert_epoch12.pth'))

    print('opening wiki passage...')
    with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print('wiki loaded!!!')

    query= list(test_dataset['question'])
    query1 = list(test_dataset1['question'])
    query2 = list(test_dataset2['question'])
    mrc_ids =test_dataset['id']
    length = len(test_dataset)


    batched_p_embs = []
    with torch.no_grad():
        model.eval
        #q_seqs_val1 = tokenize_colbert(query1,ret_tokenizer,corpus='query').to('cuda')
        #q_seqs_val2 = tokenize_colbert(query2,ret_tokenizer,corpus='query').to('cuda')
        #q_emb1 = model.query(**q_seqs_val1).to('cpu')
        #q_emb2 = model.query(**q_seqs_val2).to('cpu')
        #q_emb = torch.cat([q_emb1,q_emb2], dim=0)

        q_seqs_val = tokenize_colbert(query,ret_tokenizer,corpus='query').to('cuda')
        q_emb = model.query(**q_seqs_val).to('cpu')
        print(q_emb.size())

        print(q_emb.size())

        print('Start passage embedding......')
        p_embs=[]
        for step,p in enumerate(tqdm(context)):
            p = tokenize_colbert(p,ret_tokenizer,corpus='doc').to('cuda')
            p_emb = model.doc(**p).to('cpu').numpy()
            p_embs.append(p_emb)
            if (step+1)%200 ==0:
                batched_p_embs.append(p_embs)
                p_embs=[]
        batched_p_embs.append(p_embs)
    
    #q_emb = torch.cat([q_emb1,q_emb2], dim=1)


    dot_prod_scores = model.get_score(q_emb,batched_p_embs,eval=True)
    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    torch.save(rank,'/opt/ml/input/code/inferecne_colbert_rank.pth')
    print(rank.size())
    

    k = 100
    passages=[]

    for idx in range(length):
        passage=''
        for i in range(k):
            passage += context[rank[idx][i]]
            passage += ' '
        passages.append(passage)

    df = pd.DataFrame({'question':query,'id':mrc_ids,'context':passages})
    f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    complete_datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return complete_datasets