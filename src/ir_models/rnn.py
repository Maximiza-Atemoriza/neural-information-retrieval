from src.ir_models.neural_network import NetRank
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ..utils import get_training_dataset
from tensorflow import keras


class RNN(NetRank):
    def __init__(self) -> None:
        self._vectorizer = None
        self._word_index = None
        self._model = None

    def train(self, dataset, cat):
        train_dataset, test_dataset = self._processs_input(dataset)
        self._build_model(cat)
        self._compile_model()
        self._train_model(train_dataset, test_dataset)

    def _processs_input(self, dataset):
        X, Y = get_training_dataset(dataset)

        validation_split = 0.2
        num_validation_samples = int(validation_split * len(Y))

        def make_dataset(x, y):
            docs = []
            queries = []
            for d, q in x:
                docs.append(d)
                queries.append(q)
            dataset = tf.data.Dataset.from_tensor_slices(
                ({"input_1": docs, "input_2": queries}, y)
            )
            return dataset

        train_X, val_X = X[:-num_validation_samples], X[-num_validation_samples:]
        train_Y, val_Y = Y[:-num_validation_samples], Y[-num_validation_samples:]
        # train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
        # test_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_Y))
        train_dataset = make_dataset(train_X, train_Y)
        test_dataset = make_dataset(val_X, val_Y)

        # Creating the text encoder
        VOCAB_SIZE = 20000
        encoder = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE, output_sequence_length=200
        )
        encoder.adapt(
            train_dataset.map(
                lambda doc_query, label: doc_query["input_1"]
                + " "
                + doc_query["input_2"]
            )
        )
        self._vectorizer = encoder

        def encode_input(x, y):
            doc, query = x["input_1"], x["input_2"]
            doc = encoder(doc)
            query = encoder(query)

            return {"input_1": doc, "input_2": query}, y

        train_dataset = train_dataset.map(encode_input)
        test_dataset = test_dataset.map(encode_input)

        BUFFER_SIZE = 10000
        BATCH_SIZE = 64
        train_dataset = (
            train_dataset.shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return train_dataset, test_dataset

    def _build_model(self, cat):
        # Model is: Inputs -> TextVectorization -> Embedding -> Concatenate ->
        #  Bidirectional -> Dense -> Clasification(Regression)
        encoder = self._vectorizer
        embedding = tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True,
        )

        # Input

        # input_doc = keras.Input(shape=(1,), dtype="string")
        # input_query = keras.Input(shape=(1,), dtype="string")
        input_doc = keras.Input(shape=(None,), dtype="int64")
        input_query = keras.Input(shape=(None,), dtype="int64")

        # TextVectorization
        # doc_encoded = encoder(input_doc)
        # query_encoded = encoder(input_query)

        # Embedding
        # doc_embedding = embedding(doc_encoded)
        # query_embedding = embedding(query_encoded)
        doc_embedding = embedding(input_doc)
        query_embedding = embedding(input_query)

        # Bidirectional
        doc_bid = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(doc_embedding)
        doc_bid = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(doc_bid)

        query_bid = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(query_embedding)
        query_bid = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(query_bid)

        # Concatenate
        x = keras.layers.Concatenate()([doc_bid, query_bid])

        # Dense
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # #Clasification(Regression)
        # output = keras.layers.Dense(cat, "softmax")(x)
        output = keras.layers.Dense(cat)(x)

        model = keras.models.Model(inputs=[input_doc, input_query], outputs=output)
        self._model = model

    def _compile_model(self):
        self._model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=["accuracy"],
        )

    def _train_model(self, train_dataset, test_dataset):
        history = self.model.fit(train_dataset, epochs=10, validation_data=test_dataset)
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        self.plot_graphs(history, "accuracy")
        plt.ylim(None, 1)
        plt.subplot(1, 2, 2)
        self.plot_graphs(history, "loss")
        plt.ylim(0, None)
        # plt.show()

    def plot_graphs(self, history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric], "")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, "val_" + metric])

    def predict_score(self, doc: str, query: str):
        doc_vec = self.vectorizer(doc).numpy()
        query_vec = self.vectorizer(query).numpy()
        d = np.array([doc_vec])
        q = np.array([query_vec])
        return int(np.argmax(self.model([d, q])))
