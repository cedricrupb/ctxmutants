import torch
from torch import nn

import logging

from termcolor import colored

from config import inference_config_from_args
from run_train import setup_model, load_checkpoint

from scad.data import Vocabulary, BPEEncoder
from scad.tokenize import func_tokenize
from scad.data import transforms, min_collate

logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)



def init_config():
    config = inference_config_from_args()

     # Vocabulary ---
    vocabulary = Vocabulary()
    vocabulary.load(config.vocab_path)
    vocabulary.close()

    config.vocabulary = vocabulary
    config.vocab_size = len(vocabulary)

    logger.info("Load vocabulary with %d tokens..." % len(vocabulary))

    config.encoder = BPEEncoder(vocabulary)
    config.target_size = 0

    # Targets
    if len(config.target_path) > 0:
        targets = Vocabulary()
        targets.load(config.target_path)
        targets.close()

        config.targets = targets
        config.target_size = len(targets)

        logger.info("Load targets vocab with %d targets..." % config.target_size)

    return config

def _comply(tokens, types):
    # Merge func def
    tokens = ["%s %s%s" % tuple(tokens[:3])] + tokens[3:]
    types  = [types[1]] + types[3:]

    # NEWLINE at start
    tokens = ["#NEWLINE#"] + tokens
    types  = [0] + types

    # REMOVE NEWLINE/DEDENT FROM END
    while tokens[-1].endswith("#"):
        tokens = tokens[:-1]
        types  = types[:-1]

    return tokens, types


def _to_varmisuse_example(tokens, types):
    tokens, types = _comply(tokens, types)
    return {"tokens": tokens, "mask": [i for i, t in enumerate(types) if t in [4, 5]]}


def tokenize_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    functions = func_tokenize(content, lang = "python")
    
    return {k: _to_varmisuse_example(V["tokens"], V["types"]) for k, V in functions.items()}


def batch_funcs(config, functions):

    pipeline = transforms.SequentialTransform([
        transforms.load_varmisuse_example,
        transforms.AnnotatedCodeToData(
            transforms.SubwordEncode(config.encoder, max_length=config.bpe_cutoff)
        )
    ])

    batch = []

    for function in functions:
        function["location"] = []
        function["target"]  = []
        batch.append(pipeline(function))

    return min_collate(batch)


def format_tokens(tokens):
    tokens = [t for t in tokens if not t.endswith("#")]
    return " ".join(tokens)


def format_prediction(tokens, prediction):
    error_prob = prediction[0, 0]

    new_lines = [i for i, t in enumerate(tokens) if t == "#NEWLINE#"]
    new_lines.append(len(tokens))

    first_line = " ".join(tokens[new_lines[0] + 1: new_lines[1]])

    (error_conf, repair_conf), (error_ix, repair_ix) = prediction.max(dim = 1)

    if error_prob >= 0.5 or error_ix == 0:
        text = "%s [...] [Conf: %.2f%%]" % (first_line, error_prob * 100)
        print(colored(text, "green"))
        return
    
    print(colored("%s ..." % first_line, "red"))

    repair_token  = tokens[repair_ix]
    repair_tokens = [repair_token if i == error_ix else t for i, t in enumerate(tokens)]

    error_range = None
    for i, n in enumerate(new_lines):
        if n >= error_ix: error_range = (new_lines[i - 1], n); break
    
    error_line = tokens[error_range[0]: error_range[1]]
    error_line = format_tokens(error_line)
    repair_line = repair_tokens[error_range[0]: error_range[1]]
    repair_line = format_tokens(repair_line)

    print(colored("- %s [%.2f%%]" % (error_line, 100 * error_conf), "red"))
    print(colored("+ %s [%.2f%%]" % (repair_line, 100 * repair_conf), "green"))


def main():
    config = init_config()

    # Setup Inference environment
    enable_gpu = not config.no_cuda and torch.cuda.is_available()
    config.n_gpu = torch.cuda.device_count() if enable_gpu else 0
    config.device = torch.device("cuda" if enable_gpu else "cpu")

    model = setup_model(config)
    model = load_checkpoint(config, model, config.model_path)
    model.eval()

    examples = tokenize_file(config.example_path)

    if len(examples) == 0:
        raise ValueError("Could not identify any python function in %s" % config.example_path)
 
    example_order = list(examples.keys())
    example_batch = batch_funcs(config, [examples[k] for k in example_order])

    prediction = model(example_batch.input_ids,
                        token_mask = example_batch.mask)
    
    prediction = nn.Softmax(dim = 2)(prediction)

    for i, example_name in enumerate(example_order):
        example_prediction = prediction[i]
        example_tokens     = examples[example_name]
        format_prediction(example_tokens["tokens"], example_prediction)    
        print("\n--------------------------------")

if __name__ == '__main__':
    main()