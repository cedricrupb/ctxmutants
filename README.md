# Learning Realistic Bugs: Bug Creation for Neural Bug Detectors
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cedricrupb/ctxmutants/blob/main/demo.ipynb) 
[[**PAPER**]() | [**DATASETS**](https://zenodo.org/record/5547824) | [**ARTIFACT**](https://zenodo.org/record/5547853)]
> Generate more realistics mutations with contextual mutants

A core problem of neural bug detection is the requirement for realistic training examples. While it is easy to obtain likely correct code (e.g. from public realeases of open source projects), obtaining realistic bugs to learn from is much harder. Therefore, neural bug detection commonly relies on plain random mutations to produce buggy code.

This repository provides a simple alternative to plain random mutants by producing more realistic mutants
with contextual mutations.

**Key idea:** Employ contextual mutators that select mutants based on the likelihood of appearing in realistic code 
by using a masked language model.

Paper will be available soon.

## Quickstart
To quickly start with contextual mutations, we provide a [**demo**](https://colab.research.google.com/github/cedricrupb/ctxmutants/blob/main/demo.ipynb) notebook. This also highlights the key functionalities of this library. If you want to employ this repository as a library for your own project, please read on.

**Note:** While it is not necessary, it is recommended to run
the notebook on a machine with a GPU.

## Installation
To run the code contained in this repository, Python 3.7 and PyTorch 1.8 is required. Further dependencies can be easily installed via Pip
```bash
$ pip install requirements.txt
```

## Usage
This repository contains all code necessary to replicate our paper experiments. Therefore,
this repository contains code for tokenization, different mutation types and also
training scripts for neural bug detection and repair. In the following we present the structure of this repository:
```
ctxmutants
    |`-- data # Utils for data preprocessing
    |`-- modelling # Neural models for bug detection and repair
    |`-- mutation # Collects all compared mutation types
    |       |`-- __init__.py # Root: contains all language independent mutations
    |       |`-- java_mutations.py # Java specific mutations
    |       |`-- lm_mutations.py # Rerank functions for contextual mutations
    |        `-- wv_mutations.py # Rerank functions based on word2vec
    |
     `-- tokenizer # Helper libraries for tokenization and syntactic tagging
            |`-- __init__.py # Wrapper function for easy tokenization
            |`-- jstokenizer.py # Tokenizer for JavaScript (based on esprima)
            |`-- jvtokenizer.py # Tokenizer for Java (based on javalang)
             `-- pytokeniter.py # Tokenizer for Python (based on Python AST)
apply_mutations.py # Create mutants on an unmutated dataset
evaluate_mutators.py # Based on real bug fixes, evaluate the quality of a mutator
run_train.py # Run training on dataset preprocessed by apply_mutations.py
```
All scripts list config options together with a description. A full description how to preprocess a dataset or how to run the train script will be coming soon.

## Project info

Cedric Richter - [@cedricrupb](https://twitter.com/cedricrupb) - cedric.richter@uol.de

Distributed under the MIT license. See `LICENSE` for more information.

Feel free to open an issue if anything unexpected happens.