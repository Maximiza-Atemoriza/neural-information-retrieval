import os
import sys
import numpy as np

path_to_glove_file = "./glove.6B.50d.txt"


def load_glove():
    # change cwd to import modules
    sys.path.insert(0, "")
    actual_wd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    # parse embedding
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    # restore working directory
    os.chdir(actual_wd)

    return embeddings_index
