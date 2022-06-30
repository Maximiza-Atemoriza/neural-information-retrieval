from typing import Dict
from keras.layers import TextVectorization
from keras.preprocessing.text import Tokenizer
import numpy as np
import spacy


def remove_stopwords(text, lemma=False):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    new_doc = []
    for token in doc:
        if not token.is_stop:
            if lemma:
                new_doc.append(token.lemma_)
            else:
                new_doc.append(token.text)
    new_text = " ".join(new_doc)

    return new_text


def vocabulary_count(text: list[str]):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return len(tokenizer.word_counts)


def get_vectorizer(dataset, max_tokens=20000, output_sequence_length=200):
    docs = [doc.text for doc in dataset.docs_iter()]
    queries = [query.text for query in dataset.queries_iter()]
    vectorizer = TextVectorization(
        max_tokens=max_tokens, output_sequence_length=output_sequence_length
    )
    vectorizer.adapt(docs + queries)
    return vectorizer


def get_training_dataset(dataset):
    docs = {int(doc.doc_id): doc.text for doc in dataset.docs_iter()}
    queries = {int(query.query_id): query.text for query in dataset.queries_iter()}
    X = []
    Y = []
    for qrel in dataset.qrels_iter():
        try:
            doc = docs[int(qrel.doc_id)]
            query = queries[int(qrel.query_id)]
            X.append([doc, query])
            Y.append(int(qrel.relevance))
            if Y[-1] == -1:  # TODO: remove the patch :D
                Y[-1] = 0
        #
        except KeyError:  # missing data
            pass

    return X, Y


def get_test_set(dataset, sample_size):
    docs = {int(doc.doc_id): doc.text for doc in dataset.docs_iter()}
    queries = {int(query.query_id): query.text for query in dataset.queries_iter()}
    qrels = {
        (qrel.query_id, qrel.doc_id): qrel.relevance for qrel in dataset.qrels_iter()
    }

    sample_size = min(len(docs), len(queries), sample_size)

    queries_sample = np.array([query for query in dataset.queries_iter()])
    queries_sample = queries_sample[
        np.random.choice(len(queries_sample), size=sample_size, replace=False)
    ]

    docs_sample = np.array([doc for doc in dataset.queries_iter()])
    docs_sample = docs_sample[
        np.random.choice(len(docs_sample), size=sample_size, replace=False)
    ]

    output = []
    for (q, d) in zip(queries_sample, docs_sample):
        print(q, d)
        try:
            relevance = qrels[int(q[0]), int(d[0])]
        except KeyError:
            relevance = 0
        output.append((q[1], d[1],0 if relevance < 0 else  relevance))

    return output


def get_word_index(vectorizer) -> Dict[str, int]:
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    return word_index
