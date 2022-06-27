import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..embeddings.glove import load_glove
from ..embeddings.utils import get_embedding_matrix
from keras.layers import Embedding
from ..utils import get_vectorizer, get_word_index


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

        # TODO: define nn model 
