# Open-Domain-Question-Answering

### 네이버 부스트캠프 AI Tech 3기 (최종2위)

<br>

<br>

# 1. 프로젝트 개요

본 프로젝트에서는 검색엔진과 유사한 형태의 시스템을 만들어보는 것이 목표입니다.

Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 다양한 QA 시스템 중, **Open-Domain Question Answering (ODQA)** 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가됩니다.

![Untitled](Open-Domain-Question-Answering%2076c1043df3d544bc974b75b38bb6d2be/Untitled.png)

<br>


## **파일 구성**

```yaml

                         # dense retriever 모듈
colbert/train.py         # colbert 모델 학습
colbert/inference.py     # colbert 모델 평가

                         # sparse retreiver 모듈
e_search.ipynb           # Elastic Search 모듈

arguments.py             # 실행되는 모든 argument가 dataclass 의 형태로 저장되어있음
trainer_qa.py            # MRC 모델 학습에 필요한 trainer 제공.
utils_qa.py              # 기타 유틸 함수 제공 

train.py                 # MRC, Retrieval 모델 학습 및 평가 
inference.py	    	 # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
```

<br>

## **훈련, 평가, 추론**

### **train**

```yaml
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train

python train.py --output_dir ./models/train_dataset --do_train --do_eval --overwrite_cache --overwrite_output_dir
```

### **eval**

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.

위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```yaml
# mrc 모델 평가 (train_dataset 사용)
python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval
```

### **inference**

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

- 학습한 모델의 test_dataset에 대한 결과를 도출하기 위해선 추론(`-do_predict`)만 진행하면 됩니다.
- 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`-do_eval`)를 진행하면 됩니다.

```yaml
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict

python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict --overwrite_output_di
```

1. 모델의 경우 `-overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다.
2. `./outputs/` 폴더 또한 `-overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.

<br>

<br>

# 2. 프로젝트 팀 구성 및 역할

| 이름 | 역할 |
| --- | --- |
| 김덕래 | Reader Model Architecture |
| 김용희 | Sparse Retrieval |
| 이두호 | Sparse Retrieval, Elastic Search |
| 이승환 | Dense Retrieval |
| 조혁준 | pre-processing, inference |
| 최진혁 | Data Augmentation |

<br>

<br>

# 3. 프로젝트 수행 결과

## Retriever

### Elasticsearch

- 내부적으로 Okapi BM25 로 동작하는 Java 오픈소스 분산 검색 엔진인 Elasticsearch 를 이용하여 실험한 결과, 특별한 조건 설정 없는 경우 Top 1 35.42%, Top 20 65.00% 성능을 보였습니다.
- Konlpy 의 Mecab.nouns 를 활용하여, 순서가 고려된 명사들을 저장하고 question 도 명사로 전처리 한 뒤 사용한 결과, Top 1 63.75%, Top 20 **90.42%** 성능을 보였습니다.

![Elasticsearch.png](Open-Domain-Question-Answering%2076c1043df3d544bc974b75b38bb6d2be/Elasticsearch.png)

### ColBERT

![Untitled](Open-Domain-Question-Answering%2076c1043df3d544bc974b75b38bb6d2be/Untitled%201.png)

- ColBERT 는 question 과 passage 에 속한 모든 토큰들에 대하여 유사도를 계산하기 때문에 더 미세한 문맥적 유사도를 계산할 수 있습니다.
- ColBERT 의 Retrieval 성능은 다음과 같습니다.

![Untitled](Open-Domain-Question-Answering%2076c1043df3d544bc974b75b38bb6d2be/Untitled%202.png)

|  | Context Match |
| --- | --- |
| Top 1 | 85.00% |
| Top 5 | 90.80% |
| Top 10 | 95.83% |
| Top 20 | 97.08% |

## Reader

### **RoBERTa-large**

- RoBERTa-large 를 Backbone 모델로 이용하였습니다.

<br>

<br>

# 4. 최종 결과

![Untitled](Open-Domain-Question-Answering%2076c1043df3d544bc974b75b38bb6d2be/Untitled%203.png)

<br>

<br>

# 5. Reference

**Retrieval**

- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT - [https://github.com/stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
- Elasticsearch - [https://github.com/elastic/elasticsearch](https://github.com/elastic/elasticsearch)
- BM25 [https://github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)

**Reader**

- CNN-Layer - SAMSUNG SDS
- • [klue/roberta-large](https://huggingface.co/klue/roberta-large)
