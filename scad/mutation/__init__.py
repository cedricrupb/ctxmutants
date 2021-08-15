
from .java_mutations import weak_binary_mutation as java_wbm
from .java_mutations import strong_binary_mutation as java_sbm


def weak_binary_mutation(tokens, locations, types = None, lang = "java"):

    if lang == "java": return java_wbm(tokens, locations, types)

    raise ValueError("Unknown language: %s" % lang)



def strong_binary_mutation(tokens, locations, types = None, lang = "java"):

    if lang == "java": return java_sbm(tokens, locations, types)

    raise ValueError("Unknown language: %s" % lang)


# Weak Varmisuse operator - This operator is language independent ----------------------------------------------------------------

def weak_varmisuse_mutation(tokens, location, types = None, lang = "java"):
    assert types is not None

    current_token = tokens[location]

    index = [i for i, t in enumerate(types) if t == 4]
    variables = set(tokens[i] for i in index if tokens[i] != current_token)

    return {v: 1 / len(variables) for v in variables}


def strong_varmisuse_mutation(tokens, location, types = None, lang = "java"):

    mutations = weak_varmisuse_mutation(tokens, location, types)

    index = {m: min(i for i, t in enumerate(tokens) if t == m and types[i] == 4)
                 for m in mutations.keys()}

    index = {m: 1 / (location - v + 1) for m, v in index.items() if v <= location}
    norm = sum(index.values())
    index = {m: v / norm for m, v in index.items()}

    return index