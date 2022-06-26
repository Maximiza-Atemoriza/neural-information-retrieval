from keras.preprocessing.text import Tokenizer
import numpy as np

from keras.layers import TextVectorization


def vocabulary_count(text: list[str]):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return len(tokenizer.word_counts)


def get_vectorizer(dataset):
    docs = [doc.text for doc in dataset.docs_iter()]
    queries = [query.text for query in dataset.queries_iter()]
    vectorizer = TextVectorization()
    vectorizer.adapt(docs + queries)
    return vectorizer


def get_training_dataset(dataset):
    docs = {int(doc.doc_id): doc.text for doc in dataset.docs_iter()}
    queries = {int(query.query_id): query.text for query in dataset.queries_iter()}
    X = []
    Y = []
    for qrel in dataset.qrels_iter():
        try:
            X.append([docs[int(qrel.doc_id)], queries[int(qrel.query_id)]])
            Y.append(int(qrel.relevance))
        except KeyError: # missing data
            pass

    return X, Y
