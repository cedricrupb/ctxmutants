
## Weak binary operator mutation ###########################################################################

BINARY_OPS = {">", "<", "==", ">=", "<=", "!=",
             "&&", "||",
             "+", "-", "*", "/", 
             "&", "|", "%", 
             "<<", ">>" , ">>>", "^"}

def weak_binary_mutation(tokens, location, types = None):
    replace_token = tokens[location]

    assert replace_token in BINARY_MUTATION_RULES, "Token %s cannot be a binary operation" % replace_token

    return {b: 1 / len(BINARY_OPS) for b in BINARY_OPS if b != replace_token}


## Strong binary operator mutation ###########################################################################

# Operator classes

ARITHM     = {"+", "-", "*", "/", "%"}
COMPARATOR = {"<", ">", "<=", ">=", "==", "!="}
LOGICAL    = {"&&", "||"}
BIT_OR_LOG = {"&", "|"} 
BIT        = {"<<", ">>", ">>>"}

OP_CLASSES = [ARITHM, LOGICAL, COMPARATOR, BIT_OR_LOG, BIT]

SPECIAL_RULES = {
    "^" : {"&", "|"}
}

def _compute_sbm_rules():
    rules = {b: set() for b in BINARY_OPS}

    for CLAZZ in OP_CLASSES:
        for op in CLAZZ:
            for rop in CLAZZ:
                if op != rop: rules[op].add(rop)

    for k, rule in SPECIAL_RULES.items():
        for r in rule:
            rules[k].add(r)

    return rules    

BINARY_MUTATION_RULES = _compute_sbm_rules()

assert all(len(R) > 0 for R in BINARY_MUTATION_RULES.values())

def strong_binary_mutation(tokens, location, types = None):
    replace_token = tokens[location]

    assert replace_token in BINARY_MUTATION_RULES, "Token %s cannot be a binary operation" % replace_token

    replacements = BINARY_MUTATION_RULES[replace_token]

    return {r: 1 / len(replacements) for r in replacements}


# Java location mask ----------------------------------------------------------------

def _is_string_concat(types, pos):
    return types[pos - 1] == 10 or types[pos + 1] == 10

def binary_location_mask(tokens, types):
    candidates = [i for i, t in enumerate(types) if t == 7]

    # instanceof is a binary operator but there is no replacement
    candidates = filter(lambda x: tokens[x] != "instanceof", candidates)

    # There is no real replacement for string concats
    candidates = filter(lambda x: not _is_string_concat(types, x), candidates)

    return list(candidates)


