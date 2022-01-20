import random
import numpy as np


class MutationSampler:

    def __init__(self, mutation_op):
        self.mutation_op = mutation_op

    def __call__(self, tokens, location, types = None):
        mutation_dist = self.mutation_op(tokens, location, types)
        
        operators = list(mutation_dist.keys())
        probs     = [mutation_dist[operator] for operator in operators]

        cum_probs = np.cumsum(probs)
        selection = np.searchsorted(cum_probs, np.random.rand())

        return operators[selection]



        