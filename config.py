import os
import argparse
import inspect


class Configuration:

    def __init__(self, **kwargs):
        self.update(kwargs)
    
    def update(self, config_options):
        for k, v in config_options.items():
            setattr(self, k, v)

    def __repr__(self):
        content = []

        for k, v in self.__dict__.items():
            if v is not None:
                content.append((k, v))
        
        return "Config(%s)" % (", ".join(["%s=%s" % (str(k), str(v)) for k, v in content]))


# Configuration ------------------------------------------------------------------

class TrainConfig(Configuration):

    def __init__(self, model_name, data_dir, **kwargs):

        self.model_name = model_name
        self.data_dir   = data_dir

        self.model_type = "loc_repair"
        self.model_size = "great"
        
        self.do_train = False
        self.do_test  = False

        # Train parameter
        self.learning_rate = 1e-4
        self.weight_decay  = 0.01

        self.max_batch_size = 12500
        self.max_sequence_length = 250
        self.max_test_length = 250

        self.do_warmup = False
        self.num_warmup_steps = 10_000
        self.num_train_steps = 1_000_000
        self.num_samples_validate = 250000

        self.num_validate_samples = 25000
        self.validate_batch_size = 12500

        self.train_dir = os.path.join(data_dir, "train")
        self.validate_dir = os.path.join(data_dir, "validate")
        self.test_dir = os.path.join(data_dir, "test")
        self.model_dir = os.path.join(data_dir, "checkpoint")
        self.vocab_path = os.path.join(data_dir, "vocab.txt")
        self.trained_model_path = ""
        self.max_checkpoints = -1
        self.multiple_eval_datasets = False

        self.overwrite_files = False

        # Custom model options

        self.target_path = ""
        self.random_offsets = False
        self.sinoid_pos = False
        self.annotation_mask = False


        self.bpe_cutoff = 10
        self.num_workers = 1
        self.seed  = 42
        self.wandb = False
        self.no_cuda = False

        super().__init__(**kwargs)


class InferenceConfig(Configuration):

    def __init__(self, model_path, data_dir, example_path, **kwargs):

        self.model_path = model_path
        self.data_dir = data_dir
        self.example_path = example_path

        self.model_type = "pointer"
        self.model_size = "great"

        self.max_test_length = 512
        self.vocab_path = os.path.join(data_dir, "vocab.txt")

        # Custom model options

        self.target_path = ""
        self.random_offsets = False
        self.sinoid_pos = False
        self.annotation_mask = False
        self.bpe_cutoff = 10

        self.no_cuda = False

        super().__init__(**kwargs)


# Argparse Utils ----------------------------------------------------------------

def _argparse_from_config(config, required=set()):
    parser = argparse.ArgumentParser()

    for key, v in config.__dict__.items():

        if type(v) == bool and not v:
            parser.add_argument("--%s" %key, action="store_true", required=key in required)
        else:
            parser.add_argument("--%s" % key, type=type(v), required=key in required)

    return parser


def _from_args(config, required=set()):
    parser = _argparse_from_config(config, required=required)
    args = parser.parse_args()

    kwargs = {k: v for k, v in args.__dict__.items() if v is not None}
    return config.__class__(**kwargs)

# Assumes every positional parameter is string
def _bootstrap_config(config_cls):
    signature = inspect.signature(config_cls)
    required  = set([p.name for p in signature.parameters.values()
                        if p.kind not in [p.VAR_POSITIONAL, p.VAR_KEYWORD]])

    config = config_cls(**{r: "" for r in required})
    return config, required


def train_config_from_args():
    return _from_args(*_bootstrap_config(TrainConfig))

def inference_config_from_args():
    return _from_args(*_bootstrap_config(InferenceConfig))

