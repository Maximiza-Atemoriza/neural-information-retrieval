import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..embeddings.glove import load_glove
from ..embeddings.utils import get_embedding_matrix
from keras.layers import Embedding
from keras import layers
from keras import Model
from ..utils import get_vectorizer, get_word_index, get_training_dataset


class NetRank:
    def __init__(self) -> None:
        self.vectorizer = None
        self.word_index = None
        self.model = None

    def train(self, dataset):
        self.vectorizer = get_vectorizer(dataset)
        self.word_index = get_word_index(self.vectorizer)

        embeddings_index = load_glove()
        embedding_dim = 50

        num_tokens = len(self.word_index) + 2
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
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(4, activation="relu")(x)

        embedded_sequences = embedding_layer(input_query)
        y = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        y = layers.MaxPooling1D(5)(y)
        y = layers.Conv1D(128, 5, activation="relu")(y)
        y = layers.MaxPooling1D(5)(y)
        y = layers.Conv1D(128, 5, activation="relu")(y)
        y = layers.GlobalMaxPooling1D()(y)
        y = layers.Dense(4, activation="relu")(y)

        combined = layers.Concatenate()([x, y])

        number_of_relevance_levels = 5
        z = layers.Dense(128, activation="relu")(combined)
        z = layers.Dense(64, activation="relu")(z)
        z = layers.Dense(number_of_relevance_levels, activation="softmax")(z)
        self.model = Model(inputs=[input_doc, input_query], outputs=z)
        self.model.summary()

        X, Y = get_training_dataset(dataset)

        docs = []
        queries = []
        for doc, query in X:
            docs.append(self.vectorizer(doc).numpy())
            queries.append(self.vectorizer(query).numpy())

        docs = np.array(docs)
        queries = np.array(queries)

        score = np.array(Y)

        self.model.compile(
            # loss=tf.keras.losses.MeanSquaredError(),
            loss="sparse_categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["acc"],
        )
        self.model.fit([docs, queries], score, batch_size=128, epochs=50)
