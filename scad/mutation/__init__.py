
from .java_mutations import weak_binary_mutation as java_wbm
from .java_mutations import strong_binary_mutation as java_sbm
from .java_mutations import binary_location_mask as java_blm

from .lm_mutation import mlm_codebert_rerank, lm_codegpt_mutation
from .wv_mutation import wv_cbow_mutation


# Mutation mask ----------------------------------------------------------------

def _typed_location_mask(types, target_types):
    return [i for i, t in enumerate(types) if t in target_types]


class TypeMutationMask:

    def __init__(self, targets):
        self.targets = targets
    
    def __call__(self, token, types, lang = "java"):
        return _typed_location_mask(types, self.targets)


def binary_location_mask(tokens, types, lang = "java"):

    if lang == "java": return java_blm(tokens, types)
    
    return _typed_location_mask(types, [7])


def varmisuse_location_mask(tokens, types, lang = "java"):
    
    return _typed_location_mask(types, [5])


def funcmisuse_location_mask(tokens, types, lang = "java"):
    
    return _typed_location_mask(types, [3])


# Binary mutation --------------------------------------------------------------

def weak_binary_mutation(tokens, location, types = None, lang = "java"):

    if lang == "java": return java_wbm(tokens, location, types)

    raise ValueError("Unknown language: %s" % lang)


def strong_binary_mutation(tokens, location, types = None, lang = "java"):

    if lang == "java": return java_sbm(tokens, location, types)

    raise ValueError("Unknown language: %s" % lang)    


# Varmisuse mutation - This operator is language independent ---------------------

def weak_varmisuse_mutation(tokens, location, types, lang = "java"):
    assert types is not None

    current_token = tokens[location]

    index = [i for i, t in enumerate(types) if t == 4]
    variables = set(tokens[i] for i in index if tokens[i] != current_token)

    return {v: 1 / len(variables) for v in variables}


# Function misuse -------------------------------------------------------------

class WeakFunctionMisuseMutation:

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens, location, types, lang = "java"):
        assert types is not None

        current_token = tokens[location]
        index = [i for i, t in enumerate(types) if t in [2, 3]]
        functions = set(tokens[i] for i in index)
        functions = set.union(functions, set(self.vocab))

        functions.discard(current_token)

        return {f: 1 / len(functions) for f in functions}



# Vocab mutation --------------------------------------------------------------

class WeakVocabularyMutation:

    def __init__(self, targets):
        self.targets = targets
        self.probs   = {t: 1 / len(self.targets) for t in self.targets}

    def __call__(self, tokens, location, types, lang = "java"):
        return self.probs


# Batch mutation -------------------------------------------------------------------

class SequentialBatchMutation:

    def __init__(self, base_mutator, lang = "java"):
        self.base_mutator = base_mutator
        self.lang = lang

    def __call__(self, candidates):
        mutation_probs = []

        for candidate in candidates:
            mutation_prob = self.base_mutator(candidate.tokens, candidate.location, candidate.types, self.lang)
            mutation_probs.append(mutation_prob)

        return mutation_probs

# Contextual Mutation --------------------------------------------------------------

class ContextualBatchMutation:

    def __init__(self, base_mutator, rerank_fn, lang = "java"):
        self.base_mutator = base_mutator
        self.rerank_fn    = rerank_fn
        self.lang = lang

    def __call__(self, candidates):
        token_batch, location_batch, vocab_batch, type_batch = [], [], [], []

        for candidate in candidates:
            mutation_prob = self.base_mutator(candidate.tokens, candidate.location, candidate.types, self.lang)
            token_batch.append(candidate.tokens)
            location_batch.append(candidate.location)
            vocab_batch.append(list(mutation_prob.keys()))
            type_batch.append(candidate.types)

        return self.rerank_fn(token_batch, location_batch, vocab_batch, type_batch)
        

# Mask mutation ----------------------------------------------------------------------------

def mask_mutation(token, location, types, lang = "java"):
    return {"[MASK]": 1.0}

# Utils ------------------------------------------------------------------------------------

def epsilon_greedy(mutation_dist, eps = 0.1):
    max_key, _ = max(mutation_dist.items(), key=lambda x: x[1])
    norm = len(mutation_dist) - 1

    return {k: 1.0 - eps if k == max_key else eps / norm for k in mutation_dist.keys()}


def adjust_mutation(source_dist, target_dist):

    output_dist = {k: 0.0 for k in source_dist.keys()}

    for k, score in target_dist.items():
        if k in output_dist: output_dist[k] = score
    
    norm = sum(output_dist.values())

    return {k: v / (1e-9 + norm) for k, v in output_dist.items()}
