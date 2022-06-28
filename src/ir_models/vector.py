from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm


class Vectorial:
    def __init__(self, documents) -> None:
        self._documents = documents
        self._vectorizer: CountVectorizer | None = None
        self._idf: np.ndarray | None = None
        self._rank: np.ndarray | None = None

    @property
    def vectorizer(self) -> CountVectorizer:
        if self._vectorizer is None:
            self.process_tf_idf()

        assert self._vectorizer is not None
        return self._vectorizer

    @property
    def idf(self) -> np.ndarray:
        if self._idf is None:
            self.process_tf_idf()

        assert self._idf is not None
        return self._idf

    @property
    def rank(self) -> np.ndarray:
        if self._rank is None:
            self.process_tf_idf()

        assert self._rank is not None
        return self._rank

    def process_tf_idf(self):
        vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(self._documents)
        x = counts.toarray()
        tf = np.zeros(x.shape)
        for i in range(x.shape[0]):
            max_term_freq = x[i].max() + 1
            tf[i] = 0.5 + 0.5 * x[i] / max_term_freq

        idf = np.zeros(x.shape[1])
        N = x.shape[1]
        for i in range(x.shape[1]):
            df = sum(1 for term_freq in x[:, i] if term_freq != 0)
            idf[i] = np.log(N + 1 / (df + 1))  # avoid zero division

        tfidf = np.zeros(x.shape)
        for i in range(x.shape[1]):
            tfidf[:, i] = tf[:, i] * idf[i]

        for i in range(x.shape[0]):
            tfidf[i] = tfidf[i] / norm(tfidf[i])

        self._vectorizer = vectorizer
        self._rank = tfidf
        self._idf = idf

    def process_query_tf_idf(self, query):
        query_tfidf = self.vectorizer.transform([query]).toarray()[0]
        max_term_freq = query_tfidf.max() + 1  # avoid zero division
        query_tfidf = 0.5 + 0.5 * query_tfidf / max_term_freq
        query_tfidf = query_tfidf * self.idf

        return query_tfidf

    def get_relevan_documents(self, query_vector):
        documents_vectors = self.rank
        results = {}
        for i in range(documents_vectors.shape[0]):
            results[i] = np.dot(query_vector, documents_vectors[i])
            results[i] = results[i] / (norm(query_vector) * norm(documents_vectors[i]))
        return sorted(results.items(), key=lambda kv: -1 * kv[1])

    def __call__(self, query: str) -> List[Tuple[str, int]]:
        query_vector = self.process_query_tf_idf(query)
        return self.get_relevan_documents(query_vector)
