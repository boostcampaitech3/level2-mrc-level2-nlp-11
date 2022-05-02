import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25:
    def __init__(
        self,
        b=0.75,
        k1=1.2,
        tokenizer=None,
        ngram_range=(1, 2),
        max_features=50000,
        sparse_embedding_path=None,
        tfidfv_path=None):
        self.b = b # 0.5, 0.75
        self.k1 = k1 # 1.2, 1.6, 2.0
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sparse_embedding_path=sparse_embedding_path
        self.tfidfv_path=tfidfv_path
        
        if self.sparse_embedding_path:
            with open(self.sparse_embedding_path, 'rb') as file:
                self.p_embedding = pickle.load(file)
            print('passage embedding data found.')
        
        if self.tfidfv_path:
            with open(self.tfidfv_path, 'rb') as file:
                self.tfidf = pickle.load(file)
            print('tfidfvectorizer data found.')
        else:
            self.tfidf = TfidfVectorizer(
                tokenizer=self.tokenizer.tokenize,
                ngram_range=self.ngram_range,
                max_features=self.max_features
            )
        
    def fit(self, X=None):
        """ Fit IDF to documents X """
        if X:
            self.p_embedding = self.tfidf.fit_transform(X)
        self.avdl = self.p_embedding.sum(1).mean()

    def transform(self, query: str):
        """ Calculate BM25 between query q and documents X """
        assert self.avdl != None, 'You should fit TfidVectorizer object at first'
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        len_X = self.p_embedding.sum(1).A1
        query = self.tfidf.transform([query])
        assert sparse.isspmatrix_csr(query)

        # convert to csc for better column slicing
        X = self.p_embedding.tocsc()[:, query.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.tfidf._tfidf.idf_[None, query.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1