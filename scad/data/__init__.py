from .vocabulary import (Vocabulary, BPEEncoder)
from .dataset import (
    LineDataset, BufferingDataset
)

from .dataset import min_collate

# Predefine datasets ----------------------------------------------------------------
from . import transforms as T


class VarMisuseDataset(LineDataset):

    def __init__(self, train_files, bpe_encoder, cutoff = 10):

        transform = T.SequentialTransform([
            T.json_load,
            T.load_varmisuse_example,
            T.AnnotatedCodeToData(
                T.SubwordEncode(bpe_encoder, max_length=cutoff)
            )
        ])

        super().__init__(train_files, transform = transform)



class VarMisuseVocabDataset(LineDataset):

    def __init__(self, train_files, bpe_encoder, targets, cutoff = 10):

        transform = T.SequentialTransform([
            T.json_load,
            T.load_varmisuse_example,
            T.VarMisuseWithVocab(targets),
            T.AnnotatedCodeToData(
                T.SubwordEncode(bpe_encoder, max_length=cutoff)
            )
        ])

        super().__init__(train_files, transform = transform)