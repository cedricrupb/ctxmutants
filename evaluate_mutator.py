import os
import json
import argparse
import math

import numpy as np

from tqdm import tqdm

from apply_mutation import MutationCandidate, load_mutation

# Scorings -----------------------------

def inverse_brier_score(resample_prob):
    return 1.0 - brier_score(resample_prob)

def brier_score(resample_prob):
    return (resample_prob - 1)**2

def cross_entropy(resample_prob):
    return -math.log(resample_prob)


NAME2SCORE = {
    "inverse_brier": inverse_brier_score,
    "brier": brier_score,
    "nll": cross_entropy
}

# -------------------------------------------------

def _repair_example(example):
    tokens = example["tokens"]
    location = example["location"][0]
    target = example["target"][0]

    mutation_target = tokens[location]
    tokens[location] = target

    return tokens, location, mutation_target


def resampling_prob(mutator, tokens, location, mutation_target, type = None):
    candidate = MutationCandidate(tokens, type, None, location)
    mutation_dist = mutator([candidate])[0]

    try:
        return mutation_dist[mutation_target]
    except KeyError:
        print("%s -> %s" % (tokens[location], mutation_target))
        return 0.0


def evaluate_mutation(mutation, example, scoring_fn = inverse_brier_score):
    tokens, location, mutation_target = _repair_example(example) 
    resample_prob = resampling_prob(mutation, tokens, location, 
                                        mutation_target, example["types"])

    return scoring_fn(resample_prob)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("evaluation_path")

    parser.add_argument("--result_path")

    parser.add_argument("--score", default = "inverse_brier")
    parser.add_argument("--lang", default = "java")

    # For the mutator
    parser.add_argument("--mutation_type", default = "binary")
    parser.add_argument("--mutation_strength", default = "weak")
    parser.add_argument("--mutation_vocab")
    parser.add_argument("--repair_model")

    args = parser.parse_args()

    mutator = load_mutation(args)
    scoring = NAME2SCORE[args.score]

    if args.repair_model: args.mutation_strength = "contextual"

    print("Run test for mutator (%s, %s) on test set %s" % (args.mutation_strength, args.mutation_type, args.evaluation_path))

    results = []

    base_name = os.path.basename(args.evaluation_path)
    base_name, _ = os.path.splitext(base_name)

    total = sum(1 for _ in open(args.evaluation_path, "r"))

    with open(args.evaluation_path, "r") as line_stream:
        T = tqdm(line_stream, total = total)
        for line in T:
            task = json.loads(line)
            results.append( evaluate_mutation(mutator, task, scoring) )
            T.set_description("Exp. Score: %f" % np.mean(results))
    
    print("Evaluation result for: %s" % args.evaluation_path)
    print("Evaluated examples: %d" % len(results))
    print("Average score (%s): %4f (+- %4f)" % (args.score, np.mean(results), np.std(results)))

if __name__ == '__main__':
    main()