import argparse
import os
import json
import random

from tqdm import tqdm
from glob import glob
from collections import namedtuple

from scad import mutation as M

# Location mask ---------------------------------------

MutationCandidate = namedtuple('MutationCandidate', ["tokens", "types", "mask", "location"])

class Localizer:

    def __init__(self, mask_fn, lang = "java", repitition = 1):
        self.mask_fn = mask_fn
        self.lang    = lang
        self.repitition = repitition

    def __call__(self, example):
        tokens, types = example["tokens"], example["types"]
        mask = self.mask_fn(tokens, types, lang=self.lang)
        
        if len(mask) < 2: return None # Ignore example

        locations = random.sample(mask, min(self.repitition, len(mask)))

        return [MutationCandidate(tokens, types, mask, location) for location in locations]


class AddRepairMask:

    def __init__(self, mutation_type):
        self.mutation_type = mutation_type
    
    def __call__(self, examples):
        if self.mutation_type != "varmisuse": return examples

        for example in examples:
            repair_mask = [i for i, t in enumerate(example.types) if t == 4]
            example.mask.extend(repair_mask)

        return examples


def _load_mask_fn(mutation_type):

    if mutation_type == "binary": return M.binary_location_mask
    if mutation_type == "varmisuse": return M.varmisuse_location_mask
    if mutation_type == "funcmisuse": return M.funcmisuse_location_mask

    raise ValueError("Unknown mutation type: %s" % mutation_type)


def load_localizer(args):
    
    mutation_type = args.mutation_type
    mask_fn = _load_mask_fn(mutation_type)

    return Localizer(mask_fn, args.lang, args.repitition)


# Mutation --------------------------------------------

def sample_mutation(mutation_dist):
    random_selection = random.random()
    cumsum = 0.0

    for key, prob in mutation_dist.items():
        if cumsum + prob >= random_selection: return key
        cumsum += prob
    
    return key # Because of numerical instabilities sum(probs) smaller 1.0


def perform_mutation(candidate, mutation_token):
    
    tokens = list(candidate.tokens)
    tokens[candidate.location] = mutation_token

    return {
        "bug_type": "error",
        "location": [candidate.location],
        "target":   [candidate.tokens[candidate.location]],
        "tokens":   tokens,
        "mask"  :   candidate.mask,
        "types" :   candidate.types,
    }


def load_mutation(args):
    
    mutation_type = args.mutation_type
    mutation_strength = args.mutation_strength

    base_mutator, rerank_fn = None, None

    if args.mutation_vocab: 
        with open(args.mutation_vocab, "r") as line_stream:
            vocab = [line.strip() for line in line_stream]

        if mutation_type == "funcmisuse":
            base_mutator = M.WeakFunctionMisuseMutation(vocab)
        else:
            base_mutator = M.WeakVocabularyMutation(vocab)

    elif mutation_strength == "masked":
        base_mutator = M.mask_mutation
    else:
        mutator_selection = {
            ("weak", "binary"): M.weak_binary_mutation,
            ("strong", "binary"): M.strong_binary_mutation,
            ("weak", "varmisuse"): M.weak_varmisuse_mutation,
            ("weak", "funcmisuse"): M.WeakFunctionMisuseMutation([])
        }

        mut_key = (mutation_strength, mutation_type)
        if mut_key in mutator_selection: base_mutator = mutator_selection[mut_key]

    if base_mutator is None: 
        if mutation_type == "binary": base_mutator = M.weak_binary_mutation
        if mutation_type == "varmisuse": base_mutator = M.weak_varmisuse_mutation
        if mutation_type == "funcmisuse": base_mutator = M.WeakFunctionMisuseMutation([])

    if args.repair_model:
        rerank_fn = load_mutator(args.repair_model, mutation_type, args.lang)

    # Contextual mutations
    elif mutation_strength.startswith("contextual"):

        if "mlm"  in mutation_strength: rerank_fn = M.mlm_codebert_rerank
        elif "lm" in mutation_strength: rerank_fn = M.lm_codegpt_mutation

        else: 
            raise ValueError("Unknown contextual mutator: %s, %s" % (mutation_strength, mutation_type))

    elif mutation_strength.startswith("wv"):
      
        if "cbow"  in mutation_strength: rerank_fn = M.wv_cbow_mutation
        else: 
            raise ValueError("Unknown contextual mutator: %s, %s" % (mutation_strength, mutation_type))
    
    assert base_mutator is not None, "Unknown mutation type for (%s, %s)" % (mutation_strength, mutation_type)

    if rerank_fn is not None:
         return M.ContextualBatchMutation(base_mutator, rerank_fn, args.lang)
    else:
        return M.SequentialBatchMutation(base_mutator, args.lang)


# Utils -----------------------------------------------

def load_batches(iterator, batch_size):

    buffer = []

    for element in iterator:
        buffer.append(element)

        if len(buffer) >= batch_size:
            yield buffer
            buffer = []

    if len(buffer) > 0: yield buffer


def flatten(object):
    if isinstance(object, str): yield object; return

    try:
        for item in object: yield item
    except TypeError: 
        yield object


class Preprocessor:

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, iterator):

        for item in iterator:

            for transform in self.transforms:
                if item is None: break
                item = transform(item)
            
            if item is not None:
                for _item in flatten(item): yield _item


def serialize_example(candidate):
    return {
        "bug_type": "None",
        "location": [],
        "target":   [],
        "tokens":   candidate.tokens,
        "mask"  :   candidate.mask,
        "types" :   candidate.types,
    }


# Store results in rolling files ----------------------------------------------------------------

class RollingJsonl:

    def __init__(self, output_dir, max_saves = 10_000):
        self.output_dir = output_dir
        self.max_saves = max_saves

        self._current_saves = max_saves
        self._current_file_index = 0
        self._current_io = None

    def _current_file_io(self):
        if self._current_saves >= self.max_saves:
            if self._current_io is not None: self._current_io.close()

            self._current_file_index += 1
            self._current_io = open(os.path.join(self.output_dir, "file-%d.jsonl" % self._current_file_index), "w")
            self._current_saves = 0

        return self._current_io
    
    def save(self, object):
        self._current_file_io().write(json.dumps(object) + "\n")
        self._current_saves += 1
    
    def close(self):
        if self._current_io is not None: self._current_io.close()


# Custom contextual mutator --------------------------------

import model_utils as mu

class MaskedRepairMutator:

    def __init__(self, model):
        self.model = model

    def _truncate(self, tokens, mask):
        max_length = self.model.config.max_test_length
        if len(tokens) <= max_length: return tokens, mask

        mask_position = min(i for i, t in enumerate(tokens) if t == "[M]")

        right = min(len(tokens), mask_position + max_length // 2)
        left  = max(0, right - max_length)

        return tokens[left:right], [l - left for l in mask if l - left >= 0]


    def __call__(self, token_batch, location_batch, vocab_batch, type_batch):
    
        mutation_dists = []

        for i, tokens in enumerate(token_batch):
            location = location_batch[i]
            vocab    = vocab_batch[i]
            types    = type_batch[i]

            mask = _load_mask_fn(self.model.config.mutator_type)(tokens, types, lang=self.model.config.lang)
            mask.append(location)

            input_tokens = list(tokens)
            input_tokens[location] = "[M]"

            input_tokens, mask = self._truncate(input_tokens, mask)

            result = self.model.inference(input_tokens, mask, mask_repair = True, temp = 1.5)
            result = result["repair"]

            print(input_tokens[location - 5: location + 5])
            print("Real: ", tokens[location])
            print({k: v for k, v in result.items() if v > 0.01})

            token_dist = {}
            for v in vocab: token_dist[v] = result[v]

            # Reweighing
            token_dist.pop(tokens[location], None)
            dist_norm = sum(token_dist.values())

            mutation_dists.append({k: v / dist_norm for k, v in token_dist.items()})

        return mutation_dists




def load_mutator(model_dir, mutation_type, lang = "java"):
    model = mu.load_model_from_dir(model_dir, "repair_%s" % mutation_type, lang)
    return MaskedRepairMutator(model)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder")
    parser.add_argument("mutant_folder")

    parser.add_argument("--lang", default="java")
    parser.add_argument("--batch_size", type = int, default=1)
    parser.add_argument("--repitition", type = int, default = 1)
    
    parser.add_argument("--mutation_type", default = "binary")
    parser.add_argument("--mutation_strength", default = "weak")
    parser.add_argument("--mutation_vocab")
    parser.add_argument("--add_repair", action = "store_true")

    parser.add_argument("--max_saves", type = int, default = 10_000)
    parser.add_argument("--no_correct", action = "store_true")
    parser.add_argument("--repair_model")

    args = parser.parse_args()

    def iterate_examples():
        input_path = args.input_folder
        files = glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True)

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    yield json.loads(line)

    # Calculate total batches ---
    total = sum(1 for _ in iterate_examples())
    total = (total * args.repitition) // args.batch_size

    # Preprocessor

    example_iterator = iterate_examples()

    preprocessors = [load_localizer(args)]

    if args.add_repair: preprocessors.append(AddRepairMask(args.mutation_type))
    
    preprocessor = Preprocessor(preprocessors)
    example_iterator = preprocessor(example_iterator)

    # Mutation
    batch_mutation = load_mutation(args)

    # Store
    saver = RollingJsonl(args.mutant_folder, args.max_saves)

    try:

        for batch in tqdm(load_batches(example_iterator, batch_size = args.batch_size), total = total):
            mutants = batch_mutation(batch)

            for n, mutation_dist in enumerate(mutants):
                if len(mutation_dist) == 0: continue # Can happen if localization candidates are different from mutants

                candidate = batch[n]

                # Build mutant
                mutant_token  = sample_mutation(mutation_dist)
                mutant        = perform_mutation(candidate, mutant_token)
                mutant["bug_type"] = args.mutation_type
                correct            = serialize_example(candidate)

                saver.save(mutant)

                if not args.no_correct: saver.save(correct)
    
    finally:
        saver.close()



if __name__ == '__main__':
    main()