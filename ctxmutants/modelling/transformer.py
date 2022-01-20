import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig

import math

from .utils import init_weights


def detect_to_hf_config(config):
    return BertConfig(
        vocab_size=1, #We delete the word embeddings (Never used)
        hidden_size=config.hidden_size,
        intermediate_size=config.ffn_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        max_position_embeddings=config.max_length,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        pad_token_id=config.pad_token_id
    )

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe.requires_grad = False

    def forward(self, position_ids):
        return F.embedding(position_ids, self.pe)
       

class SubtokenEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                                config.embedding_size,
                                                padding_idx=config.pad_token_id)

    def forward(self, subtokens):
        assert subtokens.max() < self.config.vocab_size, "Found token %d not in vocabulary" % subtokens.max()
        subtoken_embeddings = self.word_embeddings(subtokens)

        with torch.no_grad():
            token_weights = subtokens.ne(0).float()
            token_norm    = token_weights.sum(dim=2).unsqueeze(-1)
            token_norm    = token_norm.expand_as(token_weights)
            token_weights /= (token_norm + 1e-9)

        subtoken_embeddings *= token_weights.unsqueeze(-1)
        subtoken_embeddings = subtoken_embeddings.sum(dim=2)

        return subtoken_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embeddings = SubtokenEmbeddings(config)

        if config.hidden_size != config.embedding_size:
            self.embedding_proj = nn.Linear(config.embedding_size, config.hidden_size)

        # Bert encoder
        hf_config = detect_to_hf_config(config)
        model = BertModel(hf_config, False)
        model.embeddings.word_embeddings = None

        if config.sinoid:
            encoding = PositionalEncoding(config.hidden_size)
            model.embeddings.position_embeddings = encoding

        self.encoder = model

        self.init_weights()

    def init_weights(self):
        self.embeddings.apply(init_weights)

        if hasattr(self, "embedding_proj"):
            self.embedding_proj.apply(init_weights)

    def get_embedding(self):
        return self.embeddings

    def forward(self, tokens,
                    attention_mask=None,
                    position_ids=None,
                    token_type_ids=None):

        subtoken = self.embeddings(tokens)

        if hasattr(self, "embedding_proj"):
            subtoken = self.embedding_proj(subtoken)

        token_encoding = self.encoder(
            inputs_embeds = subtoken,
            attention_mask = attention_mask,
            position_ids = position_ids,
            token_type_ids = token_type_ids
        )

        return token_encoding[0], subtoken


