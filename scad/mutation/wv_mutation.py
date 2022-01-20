import fasttext
import os

import math
import numpy as np
from numpy.linalg import norm

MODEL_INSTANCE = None

def _load_model():
    global MODEL_INSTANCE

    if MODEL_INSTANCE is None:
        base_dir = os.path.dirname(__file__)
        MODEL_INSTANCE = fasttext.load_model(os.path.join(base_dir, "java_cbow.bin"))
    
    return MODEL_INSTANCE


def compute_cosine(query, vocab_keys):
    vocab_matrix = np.stack(vocab_keys)
    
    dot_prod = np.dot(query.reshape((1, -1)), vocab_matrix.transpose())
    cos_sim  = dot_prod / (norm(query) * norm(vocab_matrix, axis = 1))
    
    return cos_sim[0]


def dict_softmax(D):

    max_value = max(D.values())
    output = {k: math.exp(v - max_value) for k, v in D.items()}
    norm   = sum(output.values())
    
    return {k: v / norm for k, v in output.items()}
    


def wv_cbow_mutation(token_batch, location_batch, vocab_batch, type_batch = None):
    model = _load_model()

    word_batch = []
    for i, tokens in enumerate(token_batch):
        word_batch.append(tokens[location_batch[i]])

    word_vectors = list(map(lambda x: model[x], word_batch))

    mutation_dists = []

    for i, word_vector in enumerate(word_vectors):
        word_token = word_batch[i]
        vocab = vocab_batch[i]

        if len(vocab) == 0: mutation_dists.append({}); continue

        vocab_vectors = list(map(lambda x: model[x], vocab))

        wv_cosine = compute_cosine(word_vector, vocab_vectors)
        token_dist = {vocab[i]: sim for i, sim in enumerate(wv_cosine)}
        
        try:
            del token_dist[word_token]
        except KeyError:
            pass

        if len(token_dist) == 0: mutation_dists.append(token_dist); continue

        token_dist = dict_softmax(token_dist)

        # To speed up sampling, we remove everything with prob lesser 0.1%
        token_dist = {k: v for k, v in token_dist.items() if v >= 0.001}

        mutation_dists.append(token_dist)
    
    return mutation_dists
