from .vocabulary import (Vocabulary, BPEEncoder)
from .dataset import (
    LineDataset, BufferingDataset
)

# Predefine datasets ----------------------------------------------------------------
from . import transforms as T


class VarMisuseDataset(LineDataset):

    def __init__(self, train_files, bpe_encoder):

        transform = T.SequentialTransform([
            T.json_load,
            T.load_varmisuse_example,
            T.AnnotatedCodeToData(
                T.SubwordEncode(bpe_encoder)
            )
        ])

        super().__init__(train_files, transform = transform)