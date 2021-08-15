import os
import sys
import random
import logging
import time
import math
import collections

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from glob import glob

from config import train_config_from_args

from scad.data import Vocabulary, BPEEncoder
from scad.data import BufferingDataset

from scad.data import VarMisuseDataset

from scad.modelling import TransformerConfig
from scad.modelling import TransformerEncoder

from scad.modelling import VarMisuseModel

from transformers import get_linear_schedule_with_warmup

logformat = "%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"
datefmt = "%m-%d %H:%M"

logger = logging.getLogger('lrp_trainer')
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))
logger.addHandler(stream_handler)

EPS = 1e-9

# Model setup ----------------------------------------------------------------

def create_transformer_config(config):

    if config.model_size == "great":
        logger.info("Use great setup for Transformer")
        return TransformerConfig(
            config.vocab_size,
            hidden_size=512,
            ffn_size=2048,
            max_length=config.max_test_length,
            sinoid=config.sinoid_pos,
        )

    if config.model_size == "debug":
        logger.info("Use debug setup for Transformer")
        return TransformerConfig(
            config.vocab_size,
            hidden_size=16,
            ffn_size=64,
            max_length=config.max_test_length,
            sinoid=config.sinoid_pos,
        )



# Loading utils --------------------------------------------------------------

def setup_directories(config):

    if (config.do_train
            and not os.path.exists(config.train_dir)):
        raise ValueError("Could not find training directory at %s" % config.train_dir)

    if (config.do_eval
            and not os.path.exists(config.eval_dir)):
        raise ValueError("No evalution data at %s" % config.eval_dir)

    if config.do_train:
        if (os.path.exists(config.model_dir)
             and os.listdir(config.model_dir)
             and not config.overwrite_files):
            raise ValueError("Model directory %s exists. To overwrite, set --overwrite_files")

        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)


def _no_collate(batch):
    return batch[0]


def init_train_loader(config, data_path):
    logger.info("Index train dataset...")
    files = glob(os.path.join(data_path, "*"))
    files = [f for f in files if os.path.isfile(f)]

    dataset = VarMisuseDataset(files, config.encoder)

    logger.info("Use token batching...")
    loading_set = BufferingDataset(
        dataset, max_buffer_size=4,
        batch_size=config.max_batch_size,
        max_sequence_length=config.max_sequence_length
    )
    loader = DataLoader(
        loading_set, collate_fn=_no_collate,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return loader


def init_test_loader(config, data_path, full_run = False):
    logger.info("Index dev dataset...")
    files = glob(os.path.join(data_path, "*"))
    files = [f for f in files if os.path.isfile(f)]
    dataset = VarMisuseDataset(files, config.encoder)

    num_samples = config.num_validate_samples if not full_run else -1

    loading_set = BufferingDataset(
        dataset, max_buffer_size=4,
        batch_size=config.validate_batch_size,
        num_samples=num_samples,
        max_sequence_length=config.max_test_length
    )
    loader = DataLoader(
        loading_set, collate_fn=_no_collate
    )

    return loader

# Model utils ----------------------------------------------------------------


def setup_model(config):
    t_config = create_transformer_config(config)
    t_encoder = TransformerEncoder(t_config)

    logger.info("Apply Transformer with the Locate Pointer head...")
    return VarMisuseModel(t_config, t_encoder)


def save_checkpoint(config, model, num_steps, quality=-1):

    model_dir = config.model_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if quality > 0:
        qval = int(quality * 1e+3)
        file_name = "%s_%d_%d.pt" % (config.model_name, qval, num_steps)
    else:
        file_name = "%s_%d.pt" % (config.model_name, num_steps)

    disc_path = os.path.join(model_dir, file_name)

    logger.debug("Save loc_repair to %s" % disc_path)
    torch.save(model.state_dict(), disc_path)


def load_checkpoint(config, model, model_path):
    logger.info("Load model checkpoint from %s" % model_path)

    if not os.path.exists(model_path):
        raise ValueError("Cannot load model since the checkpoint does not exists")
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    return model

# Training loop ----------------------------------------------------------------

def train_step(config, model, batch):
    model.train()

    location_labels = batch.location
    repair_labels   = batch.repair
    token_mask      = batch.mask

    labels = torch.stack([location_labels, repair_labels], dim=1)
    
    loss, logits = model(batch.input_ids,
                            token_mask = token_mask,
                            position_ids = batch.position_ids,
                            labels = labels)

    loss.backward()

    # Evaluate the prediction via model score
    scores = model.score(
        logits, labels
    )
    scores["loss"] = loss.item()
    config.train_report.update(scores)


def train(config, model):
    logger.info("Setup train loop...")
    
    train_loader = init_train_loader(config, config.train_dir)
    dev_loader = init_test_loader(config, config.validate_dir)

    model = model.to(config.device)

    # Define parameters with decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                    lr=config.learning_rate,
                                    eps=1e-6)
    scheduler = None

    if config.do_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=config.num_train_steps)

    num_samples = 0
    last_eval = 0

    def cycle_batch():
        while True:
            for batch in train_loader:
                add_position_ids(config, batch, config.random_offsets)
                yield batch

    start_time = time.time()
    cum_tokens = 0

    batch_iter = cycle_batch()
    T = trange(config.num_train_steps)

    batch = next(batch_iter)

    for ts in T:

        cum_tokens += batch.input_ids.numel()

        # Add position ids before transfering to GPU
        # add_position_ids(config, batch)
        batch = batch.to(config.device)

        # Zeroout all grads
        optimizer.zero_grad()

        # Execute one step
        train_step(config, model, batch)

        # Clip gradient to prohibit blowup
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler: scheduler.step()

        token_per_sec = format_num(cum_tokens / (time.time() - start_time))
        format_samples = format_num(num_samples)
        T.set_description("Samples %s Loss %f [%s tk/s]" % (format_samples, config.train_report["loss"], token_per_sec))

        num_samples += batch.input_ids.shape[0]

        if (num_samples - last_eval) >= config.num_samples_validate:
            validate(config, model, dev_loader)
            val_score = config.validate_report["localization_acc"]
            save_checkpoint(config, model, ts, val_score)
            config.train_report.step()
            last_eval = num_samples


# Testing loop -----------------------------------------------------------------

def validate_step(config, log, model, batch):
    model.eval()

    location_labels = batch.location
    repair_labels   = batch.repair
    token_mask      = batch.mask

    with torch.no_grad():

        labels = torch.stack([location_labels, repair_labels], dim=1)
    
        loss, logits = model(batch.input_ids,
                                token_mask = token_mask,
                                position_ids = batch.position_ids,
                                labels = labels)

        # Evaluate the prediction via model score
        scores = model.score(
            logits, labels
        )
        scores["loss"] = loss.item()
        log.update(scores)


def validate(config, model, loader):
    logger.info("Validate model...")

    with tqdm(total=config.num_validate_samples) as pbar:
        for batch in loader:
            add_position_ids(config, batch)
            batch = batch.to(config.device)
            validate_step(config, config.validate_report, model, batch)
            pbar.update(batch.input_ids.shape[0])


def test(config, model):
    logger.info("Setup test run...")

    dev_loader = init_test_loader(config, config.test_dir, full_run=True)
    model = model.to(config.device)
    
    T = tqdm(dev_loader)

    log = AvgBackend()

    for batch in T:
        add_position_ids(config, batch)
        batch = batch.to(config.device)
        validate_step(config, log, model, batch)
        T.set_description("Loss: %f" % log["loss"])

    return log.avg_scores()

# Main setup -------------------------------------------------------------------
def init_config():
    config = train_config_from_args()

    # Vocabulary ---
    vocabulary = Vocabulary()
    vocabulary.load(config.vocab_path)
    vocabulary.close()

    config.vocabulary = vocabulary
    config.vocab_size = len(vocabulary)

    logger.info("Load vocabulary with %d tokens..." % len(vocabulary))

    config.encoder = BPEEncoder(vocabulary)

    return config


def main():
    config = init_config()

    # Setup Train environment
    enable_gpu = not config.no_cuda and torch.cuda.is_available()
    config.n_gpu = torch.cuda.device_count() if enable_gpu else 0
    config.device = torch.device("cuda" if enable_gpu else "cpu")

    if config.wandb:
        import wandb
        cfg_desc = {
            "max_sequence_length": config.max_sequence_length,
            "batch_size": config.max_batch_size,
            "num_train_steps": config.num_train_steps,
            "num_warmup_steps": config.num_warmup_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gpu": enable_gpu,
        }
        # TODO: Use other / clean project
        wandb.init(project="selfrepair", config=cfg_desc)

    #Logging
    if config.wandb:
        log_backend = WandbBackend()
    else:
        log_backend = LogBackend()

    config.train_report = Report(ChildBackend("train", log_backend))
    config.validate_report = Report(ChildBackend("validate", log_backend))

    set_seed(config)

    model = setup_model(config)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if len(config.trained_model_path) > 0:
        model = load_checkpoint(config, model, config.trained_model_path)

    if config.do_train:
        logger.info(f"Training {config.model_name} ({format_num(num_parameters)}) | path: {config.data_dir} | device: {config.device} | num_gpus: {config.n_gpu}  ")
        logger.info(f"max_train: {format_num(config.num_train_steps)} | warmup: {format_num(config.num_warmup_steps)} | batch_size: {format_num(config.max_batch_size)} | max_seq_size: {config.max_sequence_length}")
        logger.info(f"model_directory: {config.model_dir}")
        logger.info(f"vocabulary size: {config.vocab_size}")
        train(config, model)

    if config.do_test:
        logger.info("Run test for trained model")
        results = test(config, model)

        logger.info("Report on test results...")
        for k, v in results.items():
            if config.wandb:
                wandb.run.summary[k] = v
            else:
                logger.info("%s: %f" % (k, v))


# Random position augmentation ---------------------------------------------------

def random_offset_augmentation(config, position_ids):
    if position_ids.shape[1] >= config.max_sequence_length: return position_ids

    max_length = config.max_sequence_length - position_ids.shape[1]
    random_offsets = torch.randint(max_length, (position_ids.shape[0],),
                                    device=position_ids.device)
    random_offsets = random_offsets.unsqueeze(1).expand_as(position_ids)
    return position_ids + random_offsets


def add_position_ids(config, batch, random_offsets = False):
    input_ids = batch.input_ids

    # Create position ids
    if hasattr(batch, 'position_ids'):
        position_ids = batch.position_ids
        assert position_ids.max() <= input_ids.shape[1], "%d: %d" % (position_ids.max(), input_ids.shape[1])
    else:
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

    if random_offsets:
        position_ids = random_offset_augmentation(config, position_ids)
   
    batch.position_ids = position_ids

# Train utils -------------------------------------------------------------

def mlm_loss(logits, label_ids):
    vocab_size = logits.shape[-1]
    loss_fct = nn.CrossEntropyLoss()
    logits = logits.view(-1, vocab_size)
    label_ids = label_ids.view(-1)
    loss = loss_fct(logits, label_ids)
    return loss


# Validate ----------------------------------------------------------------
class Report:

    def __init__(self, backend):
        self.backend = backend

    def __getitem__(self, key):
        return self.backend[key]

    def update(self, D):
        self.backend.update(D)

    def step(self):
        self.backend.step()


class AvgBackend:

    def __init__(self):
        self._store = collections.defaultdict(float)
        self._num_evals = collections.defaultdict(int)

    def __getitem__(self, key):
        score = self._store[key]
        norm  = self._num_evals[key]

        if norm == 0: return 0.0

        return score / norm


    def update(self, scores):
        for key, score in scores.items():
            self._store[key] += score
            self._num_evals[key] += 1

    def avg_scores(self):
        return {
            k: v / max(1, self._num_evals[k]) for k, v in self._store.items()
        }

    def step(self):
        self._store = collections.defaultdict(float)
        self._num_evals = collections.defaultdict(int)


class LogBackend(AvgBackend):

    def step(self):

        scores = self.avg_scores()

        for key, score in scores.items():
            logger.info("%s: %f" % (key, score))

        super().step()

class WandbBackend(AvgBackend):

    def step(self):
        wandb.log(self.avg_scores())
        super().step()


class ChildBackend:

    def __init__(self, prefix, parent):
        self.prefix = prefix
        self.parent = parent

    def __getitem__(self, key):
        key = "%s_%s" % (self.prefix, key)
        return self.parent[key]

    def update(self, scores):
        return self.parent.update({
            "%s_%s" % (self.prefix, k): v
            for k, v in scores.items()
        })

    def step(self):
        return self.parent.step()

# Utils ---------------

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def format_num(number):
    if number == 0: return "0"
    magnitude = int(math.log10(number)) // 3
    number /= 10**(magnitude * 3)
    return "%.2f%s" % (number, ["", "K", "M", "G", "T", "P"][magnitude])


if __name__ == '__main__':
    main()