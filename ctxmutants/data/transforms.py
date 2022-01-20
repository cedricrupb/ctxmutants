
import torch
import json
import collections
import random

from .util import Data

# Load utis ----------------------------------------------------------------

def json_load(line):
    return json.loads(line)

AnnotatedCode = collections.namedtuple('AnnotatedCode', ["tokens", "annotations"])

def to_annotated(D):
    return AnnotatedCode(D["tokens"], {"types": D["types"]})

def load_tokens(tokens):
    return AnnotatedCode(tokens, {})


def load_varmisuse_example(D):

    tokens = D["tokens"]

    error_locations = D["location"]
    if len(error_locations) == 0: error_locations = [0]
    
    repair_targets = D["target"]
    repair_mask    = D["mask"]
    repair_mask.append(0)

    error_index    = [1 if i in error_locations else 0 for i in range(len(tokens))]
    repair_index   = [1 if t in repair_targets and i in repair_mask else 0 for i, t in enumerate(tokens)]
    mask_index     = [1 if i in repair_mask else 0 for i in range(len(tokens))]

    return AnnotatedCode(
        tokens, {
            "location": error_index,
            "repair": repair_index,
            "mask": mask_index,
        }
    )


def load_annotated_code(D):

    tokens = D["tokens"]
    types  = D["types"]

    return AnnotatedCode(
        tokens, {
            "types": types
        }
    ) 


# Locate Repair tasks + Vocab ----------------------------------------------------------------

class BugExampleLoader:

    def __init__(self, vocab = None):
        self.vocab = vocab

    def __call__(self, D):
        tokens = D["tokens"]

        error_locations = D["location"]
        if len(error_locations) == 0: error_locations = [0]
    
        repair_mask    = D["mask"]
        repair_mask.append(0)

        repair_targets = D["target"]

        # Create general indexes
        token_mask = [1 if i in repair_mask else 0 for i in range(len(tokens))]
        error_index = [1 if i in error_locations else 0 for i in range(len(tokens))]
        repair_index = [1 if t in repair_targets and token_mask[i] == 1 else 0
                            for i, t in enumerate(tokens)]

        annotations = {
            "location": error_index,
            "repair"  : repair_index,
            "mask"    : token_mask,
        }

        if self.vocab is not None:
            if len(repair_targets) == 0: repair_targets.append("[UNK]")

            labels = list(tokens)
            for i, l in enumerate(error_locations): labels[l] = repair_targets[i]

            labels = [self.vocab[t] if t in self.vocab else 0 for t in labels]
            annotations["labels"] = labels
            
        return AnnotatedCode(tokens, annotations)



# Data helpers -------------------------------------------------------------

class RandomTruncate:

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, data):
        length = len(data.tokens)
        if length <= self.max_length: return data

        offset = random.randrange(length - self.max_length)

        return AnnotatedCode(
            data.tokens[offset: offset + self.max_length],
            {
                k: v[offset: offset + self.max_length] for k, v in data.annotations.items()
            }
        )



# To token ids --------------------------------------------------------------------


class TokenEncode:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def _encode_token(self, token):
        self.vocabulary.close()
        try:
            return self.vocabulary.index(token)
        except KeyError:
            return self.vocabulary.index("[UNK]")

    def __call__(self, tokens):
        return list(map(self._encode_token, tokens))


class SubwordEncode:

    def __init__(self, bpe_encode, max_length=10):
        self.bpe_encode = bpe_encode
        self.max_length = max_length

    def _subword_encode(self, token):
        subtokens = self.bpe_encode(token)[:self.max_length]
        subtokens = subtokens + (0,)*(self.max_length - len(subtokens))
        return list(subtokens)

    def __call__(self, tokens):
        return list(map(self._subword_encode, tokens))



# Annotated to data ----------------------------------------------------------------


class AnnotatedCodeToData:

    def __init__(self, vocab_encoder):
        self.vocab_encoder = vocab_encoder

    def __call__(self, ancode):

        tokens        = ancode.tokens
        subtokens     = self.vocab_encoder(tokens)
        token_mask    = [1]*len(tokens)

        annotations = ancode.annotations


        assert all([len(v) == len(tokens) for v in annotations.values()]), str([len(v) for v in annotations.values()])

        kwargs = {
            "input_ids": subtokens,
            "token_mask": token_mask,
        }
        kwargs.update(annotations)

        return Data(**{k: torch.LongTensor(v) for k, v in kwargs.items()})


# Utils --------------------------------

class SequentialTransform:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data): 
        for transform in self.transforms:
            data = transform(data)
        
        return data