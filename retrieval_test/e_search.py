from elasticsearch import Elasticsearch

import json


def insert(es: Elasticsearch, index: str, doc_type: str, data: str):
    doc = {"context": data}
    
    es.index(index=index, doc_type=doc_type, body=doc)


def search(es: Elasticsearch, index: str, data: str):
    if data is None:
        doc = {"match_all": {}}
    else:
        doc = {"match_all": data}
    body = {"query": doc}

    return es.search(index=index, body=body)


def delete(es: Elasticsearch, index: str, data: str):
    if data is None:
        doc = {"match_all": {}}
    else:
        doc = {"match": data}
    body = {"query": doc}

    return es.delete_by_query(index, body=body)


if __name__ == "__main__":
    # Load wikipedia json
    with open('../../data/wikipedia_documents.json', 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    wiki_contents = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    # Elasticsearch Configurations
    host = 'http://127.0.0.1' # https 로 하면 SSL Version Error 발생
    port = 30001              # 서버에서 허용한 (elasticsearch.yml 에 지정한) 포트 사용
    index = 'news'
    doc_type = 'wiki'

    es = Elasticsearch(f'{host}:{port}')

    # Test elasticsearch
    # 1. Create
    insert(es, index, doc_type, wiki_contents[0])
