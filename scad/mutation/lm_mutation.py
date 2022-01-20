import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

import math

  
MODEL_INSTANCE = None

def _load_model():
    global MODEL_INSTANCE

    if MODEL_INSTANCE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
        model = model.to(device)
        model.eval()

        MODEL_INSTANCE = (tokenizer, model, device)


    return MODEL_INSTANCE

def _mlm_filter(tokens):
    return [t for t in tokens if not (t.startswith("#") and t.endswith("#"))]


def _mlm_preprocess(tokenizer, token_batch, location_batch):

    # Get forbidden
    input_batch, forbidden = [], []

    for i, tokens in enumerate(token_batch):
        forbidden.append(tokens[location_batch[i]])
        input_ = list(tokens)
        input_[location_batch[i]] = tokenizer.mask_token
        input_ = _mlm_filter(input_)
        input_batch.append(" ".join(input_))

    input_batch = tokenizer(input_batch)

    # Now we have to truncate
    for i, tokens in enumerate(input_batch["input_ids"]):
        if len(tokens) <= tokenizer.model_max_length: continue

        mask_position = min(i for i, t in enumerate(tokens) if t == tokenizer.mask_token_id)

        max_length = tokenizer.model_max_length
        right = min(len(tokens), mask_position + max_length // 2)
        left  = max(0, right - max_length)

        for k in input_batch.keys():
            input_batch[k][i] = input_batch[k][i][left:right]

    # Padding
    longest = max(len(t) for t in input_batch["input_ids"])
    for i, tokens in enumerate(input_batch["input_ids"]):
        attention_mask = input_batch["attention_mask"][i]

        tokens[0] = tokenizer.bos_token_id
        tokens[-1] = tokenizer.eos_token_id
        tokens += [tokenizer.pad_token_id] * (longest - len(tokens))
        
        attention_mask += [0] * (longest - len(attention_mask))
        input_batch["input_ids"][i] = tokens
        input_batch["attention_mask"][i] = attention_mask

    return input_batch, forbidden


# Handling subword prediction ------------------------------------

SUBWORD_CACHE = {}
BPE_WORKER = None

class BPEEncoder:

    def __init__(self, vocab):
        self.vocab = vocab

        bpe_index = {}
        for subtoken in vocab.keys():
            token_index = subtoken[:2]
            if token_index not in bpe_index:
                bpe_index[token_index] = set()
            bpe_index[token_index].add(subtoken)

        self.bpe_index = bpe_index

    def encode(self, token):
        token = "Ġ" + token
        tokens = []

        ix = 0
        token_len = len(token)

        while ix < token_len:
            candiates = self.bpe_index.get(token[ix:ix+2], [])
            candiates = [t for t in candiates
                            if t == token[ix:ix+len(t)] and not len(token) == ix + len(t) + 1]

            if not candiates:
                top_candidate = token[ix]
            else:
                top_candidate = max(candiates, key=lambda x: len(x))

            tokens.append(top_candidate)
            ix += len(top_candidate)

        return [self.vocab[token] for token in tokens]
        


def _bpe_encoder(tokenizer):
    global BPE_WORKER

    if BPE_WORKER is None:
        BPE_WORKER = BPEEncoder(tokenizer.get_vocab())

    return BPE_WORKER


def subword_ids(tokenizer, token):
    try:
        return SUBWORD_CACHE[token]
    except KeyError:
        SUBWORD_CACHE[token] = _bpe_encoder(tokenizer).encode(token)
        return SUBWORD_CACHE[token]


def compute_token_dist(tokenizer, vocab, logits):
    
    def to_subword_ids(token):
        return (token, subword_ids(tokenizer, token))

    def to_logits(token_sb):
        subword_ids = token_sb[1]
        slogits = sum(logits[i] for i in subword_ids) / len(subword_ids)
        return token_sb[0], slogits.item()
    
    subtoken_map = map(to_subword_ids, vocab)
    logit_map    = map(to_logits, subtoken_map)

    return dict(logit_map)
    

def dict_softmax(D):

    max_value = max(D.values())
    output = {k: math.exp(v - max_value) for k, v in D.items()}
    norm   = sum(output.values())
    
    return {k: v / norm for k, v in output.items()}


# ----------------------------------------------------------------

def mlm_codebert_rerank(token_batch, location_batch, vocab_batch, type_batch = None):

    tokenizer, model, device = _load_model()
    input_batch, forbid = _mlm_preprocess(tokenizer, token_batch, location_batch)
    input_batch = {k: torch.LongTensor(t).to(device) for k, t in input_batch.items()}

    location_mask = input_batch["input_ids"] == tokenizer.mask_token_id

    with torch.no_grad():
        output = model(**input_batch)
        mask_logits = output.logits[location_mask].cpu()

    mutation_dists = []

    for i, vocab in enumerate(vocab_batch):
        subtoken_logits = mask_logits[i]
        token_dist = compute_token_dist(tokenizer, vocab, subtoken_logits)

        try:
            forbidden_token = forbid[i]
            del token_dist[forbidden_token]
        except KeyError:
            pass

        if len(token_dist) == 0: mutation_dists.append(token_dist); continue

        token_dist = dict_softmax(token_dist)

        # To speed up sampling, we remove everything with prob lesser 0.1%
        # token_dist = {k: v for k, v in token_dist.items() if v >= 0.001}

        mutation_dists.append(token_dist)

    return mutation_dists



# CodeGPT-J mutations --------------------------------------------------------------

CODEGPT_INSTANCE = None

def _load_codegpt():
    global CODEGPT_INSTANCE

    if CODEGPT_INSTANCE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
        tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
        model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
        model = model.to(device)
        model.eval()

        CODEGPT_INSTANCE = (tokenizer, model, device)


    return CODEGPT_INSTANCE


def _lm_preprocess(tokenizer, token_batch, location_batch, max_length=1024):

    # Get forbidden
    input_batch, forbidden = [], []

    for i, tokens in enumerate(token_batch):
        forbidden.append(tokens[location_batch[i]])
        input_ = list(tokens)[:location_batch[i]]
        input_batch.append(" ".join(input_))

    input_batch = tokenizer(input_batch)

    # Now we have to truncate
    for i, tokens in enumerate(input_batch["input_ids"]):
        if len(tokens) <= max_length: continue

        left  = max(0, len(tokens) - max_length)

        for k in input_batch.keys():
            input_batch[k][i] = input_batch[k][i][left:]

    # Padding
    locations = []
    longest = max(len(t) for t in input_batch["input_ids"])
    for i, tokens in enumerate(input_batch["input_ids"]):
        attention_mask = input_batch["attention_mask"][i]

        locations.append(len(tokens) - 1)
        tokens += [tokenizer.pad_token_id] * (longest - len(tokens))
        
        attention_mask += [0] * (longest - len(attention_mask))
        input_batch["input_ids"][i] = tokens
        input_batch["attention_mask"][i] = attention_mask

    return input_batch, forbidden, locations


def lm_codegpt_mutation(token_batch, location_batch, vocab_batch, type_batch = None):
    tokenizer, model, device = _load_codegpt()
    input_batch, forbid, prediction_locations = _lm_preprocess(tokenizer, token_batch, location_batch)
    input_batch = {k: torch.LongTensor(t).to(device) for k, t in input_batch.items()}

    prediction_mask = torch.zeros(input_batch["input_ids"].shape)
    for i, pos in enumerate(prediction_locations):
        prediction_mask[i, pos] = 1
    prediction_mask = prediction_mask.bool().to(device)

    with torch.no_grad():
        output = model(**input_batch)
        pred_logits = output.logits[prediction_mask].cpu()

    mutation_dists = []

    for i, vocab in enumerate(vocab_batch):
        subtoken_logits = pred_logits[i]
        token_dist = compute_token_dist(tokenizer, vocab, subtoken_logits)

        try:
            forbidden_token = forbid[i]
            del token_dist[forbidden_token]
        except KeyError:
            pass

        if len(token_dist) == 0: mutation_dists.append(token_dist); continue

        token_dist = dict_softmax(token_dist)

        # To speed up sampling, we remove everything with prob lesser 0.1%
        token_dist = {k: v for k, v in token_dist.items() if v >= 0.001}

        mutation_dists.append(token_dist)

    return mutation_dists