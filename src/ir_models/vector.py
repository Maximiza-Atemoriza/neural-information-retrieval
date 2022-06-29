from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm
import dill


class VectorModel:
    def __init__(self) -> None:
        self.count_vectorizer: CountVectorizer | None = None
        self.documents_tfidf: np.ndarray | None = None
        self.idf: np.ndarray | None = None

    def save(self, path):
        file = open(path, "wb")
        dill.dump(self, file)
        file.close()

    @staticmethod
    def load(path):
        file = open(path, "rb")
        model = dill.load(file)
        file.close()
        return model

    def index(self, dataset):
        docs = [doc.text for doc in dataset.docs_iter()]
        self._documents_tfidf(docs)

    def _documents_tfidf(self, documents, k=0.5):
        self.count_vectorizer = CountVectorizer()
        counts = self.count_vectorizer.fit_transform(documents)

        x = counts.toarray()
        tf = np.zeros(x.shape)
        for i in range(x.shape[0]):
            max_term_freq = x[i].max()
            tf[i] = x[i] / max_term_freq

        self.idf = np.zeros(x.shape[1])
        N = x.shape[0]
        for i in range(x.shape[1]):
            df = sum(1 for term_freq in x[:, i] if term_freq != 0)
            self.idf[i] = np.log(N / df)

        self.documents_tfidf = np.zeros(x.shape)
        for i in range(x.shape[1]):
            self.documents_tfidf[:, i] = tf[:, i] * self.idf[i]

    def _query_tfidf(self, query, a=0.01):
        if self.count_vectorizer is None:
            raise Exception("No indexed dataset!")

        query_tfidf = self.count_vectorizer.transform([query]).toarray()
        max_term_freq = query_tfidf.max()
        if max_term_freq > 0:
            query_tfidf = (a + (1 - a) * (query_tfidf / max_term_freq)) * self.idf
        return query_tfidf

    def relevant_docs(self, query, n=10):
        if self.documents_tfidf is None:
            raise Exception("No indexed dataset!")

        results = {}
        query_vector = self._query_tfidf(query)

        zeros = query_vector == 0
        if zeros.all():
            return []

        for i in range(self.documents_tfidf.shape[0]):
            results[i] = np.dot(query_vector, self.documents_tfidf[i])
            results[i] = results[i] / (
                norm(query_vector) * norm(self.documents_tfidf[i])
            )
        doc_score = sorted(results.items(), key=lambda kv: -1 * kv[1])

        return doc_score[:n]
