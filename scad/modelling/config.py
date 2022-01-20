class DetectionConfig:

    def __init__(self,
                    vocab_size,
                    embedding_size=128,
                    hidden_size=128,
                    num_layers=6,
                    max_length=128,
                    dropout=0.1,
                    encoder_type="transformer",
                    pad_token_id=0,
                    decoder_vocab_size=0):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.pad_token_id = pad_token_id
        self.decoder_vocab_size = decoder_vocab_size

    def update(self, kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

class TransformerConfig(DetectionConfig):

    def __init__(self,
                    vocab_size,
                    embedding_size=128,
                    hidden_size=128,
                    ffn_size=512,
                    num_heads=8,
                    num_layers=6,
                    max_length=128,
                    dropout=0.1,
                    sinoid=False,
                    pad_token_id=0,
                    decoder_vocab_size = 0,
                    token_annotate = False,):
        super().__init__(
            vocab_size, embedding_size,
            hidden_size, num_layers,
            max_length, dropout, "transformer",
            pad_token_id,
            decoder_vocab_size
        )

        self.ffn_size = ffn_size
        self.num_heads = num_heads
        self.sinoid = sinoid
        self.token_annotate = token_annotate