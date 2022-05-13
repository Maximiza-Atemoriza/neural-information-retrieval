from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm


def documents_tfidf(documents):
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(documents)
    x = counts.toarray()
    tf = np.zeros(x.shape)
    for i in range(x.shape[0]):
        max_term_freq = x[i].max() + 1
        tf[i] = 0.5 +  0.5 * x[i] / max_term_freq

    idf = np.zeros(x.shape[1])
    N = x.shape[1] + 1
    for i in range(x.shape[1]):
        df = sum(1 for term_freq in x[:, i] if term_freq != 0)
        idf[i] = np.log(N + 1 / (df + 1))  # avoid zero division

    tfidf = np.zeros(x.shape)
    for i in range(x.shape[1]):
        tfidf[:, i] = tf[:, i] * idf[i]

    for i in range(x.shape[0]):
        tfidf[i] = tfidf[i]/norm(tfidf[i])
    

    return vectorizer, idf, tfidf


def query_tfidf(query, vectorizer, idf):
    query_tfidf = vectorizer.transform([query]).toarray()[0]
    max_term_freq = query_tfidf.max() + 1  # avoid zero division
    query_tfidf = 0.5 + 0.5 * query_tfidf / max_term_freq
    query_tfidf = query_tfidf * idf
    return query_tfidf


def relevant_documents(query_vector, documents_vectors):
    results = {}
    for i in range(documents_vectors.shape[0]):
        results[i] = np.dot(query_vector, documents_vectors[i])
        results[i] = results[i] / (norm(query_vector) * norm(documents_vectors[i]))
    return sorted(results.items(), key=lambda kv: -1*kv[1])
