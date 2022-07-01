from typing import Dict
from keras.layers.preprocessing.text_vectorization import TextVectorization
import numpy as np
from tensorflow import keras
from ..embeddings.glove import load_glove
from ..embeddings.utils import get_embedding_matrix
from keras.layers import Embedding
from keras import layers
from keras import Model
from ..utils import get_test_set, get_vectorizer, get_word_index, get_training_dataset
import dill
from keras.models import load_model
import matplotlib.pyplot as plt


class NetRank:
    def __init__(self) -> None:
        self._vectorizer: TextVectorization | None = None
        self._word_index: Dict[str, int] | None = None
        self._model: Model | None = None

    @property
    def vectorizer(self) -> TextVectorization:
        assert self._vectorizer is not None
        return self._vectorizer

    @property
    def model(self) -> Model:
        assert self._model is not None
        return self._model

    @property
    def word_index(self) -> Dict[str, int]:
        assert self._word_index is not None
        return self._word_index

    def save(self, path, model_name) -> None:
        real_path = path + "/" + "net_rank" + model_name
        self.model.save(real_path)

        file = open(real_path + "_vectorizer", "wb")
        dill.dumps(self.vectorizer, file)
        file.close()

        file = open(real_path + "_word_index", "wb")
        dill.dumps(self.word_index, file)
        file.close()

    def load(self, file_path, model_name) -> None:
        self._model = load_model(file_path + model_name)

        file = open(file_path + model_name + "_vectorizer")
        self._vectorizer = dill.load(file)
        file.close()

        file = open(file_path + model_name + "_word_index")
        self._word_index = dill.load(file)
        file.close()

    def train(self, dataset, cat):
        self._vectorizer = get_vectorizer(dataset)
        self._word_index = get_word_index(self.vectorizer)

        embeddings_index = load_glove()
        embedding_dim = 300

        num_tokens = len(self._word_index) + 2
        embedding_matrix = get_embedding_matrix(
            self.word_index, embeddings_index, num_tokens, embedding_dim
        )

        embedding_layer = Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )

        input_doc = keras.Input(shape=(None,), dtype="int64")
        input_query = keras.Input(shape=(None,), dtype="int64")

        embedded_sequences = embedding_layer(input_doc)
        x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        # x = layers.MaxPooling1D(5)(x)
        # x = layers.Conv1D(128, 5, activation="relu")(x)
        # x = layers.MaxPooling1D(5)(x)
        # x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(4, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        embedded_sequences = embedding_layer(input_query)
        y = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        # y = layers.MaxPooling1D(5)(y)
        # y = layers.Conv1D(128, 5, activation="relu")(y)
        # y = layers.MaxPooling1D(5)(y)
        # y = layers.Conv1D(128, 5, activation="relu")(y)
        y = layers.GlobalMaxPooling1D()(y)
        y = layers.Dense(4, activation="relu")(y)
        y = layers.Dropout(0.5)(y)

        combined = layers.Concatenate()([x, y])

        # z = layers.Dense(128, activation="relu")(combined)
        # z = layers.Dense(64, activation="relu")(z)
        # z = layers.Dense(cat, activation="softmax")(z)
        z = layers.Dense(cat, activation="softmax")(combined)
        self._model = Model(inputs=[input_doc, input_query], outputs=z)
        self._model.summary()

        X, Y = get_training_dataset(dataset)

        docs = []
        queries = []
        for doc, query in X:
            docs.append(self.vectorizer(doc).numpy())
            queries.append(self.vectorizer(query).numpy())

        docs = np.array(docs)
        queries = np.array(queries)

        score = np.array(Y)
        print("*******************************************************")
        print(f"score len {len(score)}")
        print(f"docs len {len(docs)} and queries len {len(queries)}")
        print("*******************************************************")

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(score))
        train_docs = docs[:-num_validation_samples]
        train_queries = queries[:-num_validation_samples]
        train_score = score[:-num_validation_samples]

        val_docs = docs[-num_validation_samples:]
        val_queries = queries[-num_validation_samples:]
        val_score = score[-num_validation_samples:]

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        history = self.model.fit(
            [train_docs, train_queries],
            train_score,
            batch_size=128,
            epochs=30,
            validation_data=([val_docs, val_queries], val_score),
        )
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        self.plot_graphs(history, "accuracy")
        plt.ylim(None, 1)
        plt.subplot(1, 2, 2)
        self.plot_graphs(history, "loss")
        plt.ylim(0, None)
        # plt.show()

    def predict_score(self, doc: str, query: str):
        doc_vec = self.vectorizer(doc).numpy()
        query_vec = self.vectorizer(query).numpy()
        d = np.array([doc_vec])
        q = np.array([query_vec])
        return int(np.argmax(self.model([d, q])))

    def get_relevance(self, dataset, amount: int):
        test_set = get_test_set(dataset, amount)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        total_relevant = 0
        for doc, query, expected_relevance in test_set:
            relevance = self.predict_score(doc, query)
            if expected_relevance > 0:
                total_relevant += 1

            if relevance > 0 and expected_relevance > 0:
                val = min(expected_relevance, relevance) / max(
                    expected_relevance, relevance
                )
                tp += 1
                # tp += val
                # fp += 1 - val
            elif relevance == 0 and expected_relevance == 0:
                tn += 1
            elif relevance > 0 and expected_relevance == 0:
                fp += 1
            elif relevance == 0 and expected_relevance > 0:
                fn += 1

        print(f"Target size: {amount} Real size: {len(test_set)}")
        print(f"tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
        precission = 1 if tp + fp == 0 else tp / (tp + fp)
        recall = 1 if tp + fn == 0 else tp / (tp + fn)
        fscore = 1 if (2 * tp + fp + fn == 0) else 2 * tp / (2 * tp + fp + fn)

        return (precission, recall, fscore)

    def plot_graphs(self, history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric], "")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, "val_" + metric])
