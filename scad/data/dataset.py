import torch
import math
import random

from torch.utils.data import Dataset, IterableDataset

from .util import pad, stack

#- Line indexing -----------------------------------------

def _index_dataset(file_index, train_file):

    assert train_file is not None, "Expect train file path but got None."
    offsets = []

    with open(train_file, "r", encoding="utf-8") as f:
            #Initial offset is always 0
            #Size can only be determined after discovering the next element
        offsets.append([file_index, 0, -1])

        while f.readline():

            train_offset = f.tell()
            size = train_offset - offsets[-1][1]
            offsets[-1][-1] = size
            offsets.append([file_index, train_offset, -1])

        # The last offset is an EOF
        offsets.pop(-1)

    offsets = [tuple(x) for x in offsets]

    return offsets


def index_dataset(train_files):

    offsets = []

    for file_index, train_file in enumerate(train_files):

        offsets.extend(
            _index_dataset(file_index, train_file)
        )


    return offsets



class LineDataset(Dataset):

    def __init__(self, train_files, sort=False, transform=None):
        self.train_files = train_files
        self.transform   = transform
        self.is_sorted   = sort
        self._offsets = index_dataset(self.train_files)

        if sort:
            len_fn = lambda x: x[2] - x[1] + 1
            self._offsets = sorted(self._offsets, key=len_fn)


    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        file_index, train_offset, _ = self._offsets[idx]

        with open(self.train_files[file_index], "r", encoding="utf-8") as f:
            f.seek(train_offset)
            train_line = f.readline().rstrip()

        if self.transform:
            return self.transform(train_line)

        return train_line



# Collate --------------------------------------------

def random_pad(x, max_length):
    """
    Randomizes the padding length from left and right.
    Useful if we work with position embeddings
    """
    length = x.shape[0]

    if length >= max_length: return x

    pad_size = max_length - length
    pad_left = random.randint(pad_size)

    return pad(x, (pad_left, pad_size - pad_left))


def right_pad(x, max_length):
    """
    Standard padding at the right end of the sequence.
    Have to be used for testing (Otherwise, not reliable results).
    """

    length = x.shape[0]

    if length >= max_length: return x

    pad_size = max_length - length

    return pad(x, (0, pad_size))


def min_collate(batch, pad_fn=random_pad):
    """
    Pads all sequences to a common length
    """
    max_length = max(d.shape[0] for d in batch)

    for i, x in enumerate(batch):
        batch[i] = pad_fn(x, max_length)

    return stack(batch)


# To replicate: https://github.com/VHellendoorn/ICLR20-Great/blob/master/data/data_loader.py
class BufferingDataset(IterableDataset):

    def __init__(self, dataset,
                    batch_size=64,
                    max_buffer_size=4,
                    max_sequence_length=1024,
                    num_samples=-1,
                    pad_fn=right_pad):
        self.dataset = dataset
        self.max_batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.max_sequence_length = max_sequence_length
        self.num_samples = num_samples
        self.pad_fn = pad_fn

    def _compute_load_span(self):
        worker_info = torch.utils.data.get_worker_info()
        length = len(self.dataset)
        if worker_info is None: return (0, length)

        per_worker = int(math.ceil(length / worker_info.num_workers))
        worker_id = worker_info.id

        iter_start = per_worker * worker_id
        iter_end = min(iter_start + per_worker, length)
        return (iter_start, iter_end)

    def to_batches(self):

        def create_batch(buffer):
            pivot = len(random.choice(buffer))
            buffer = sorted(buffer, key=lambda x: abs(len(x) - pivot))
            batch = []

            max_seq_len = 0
            for sample in buffer:
                max_seq_len = max(max_seq_len, len(sample))
                if max_seq_len*(len(batch) + 1) > self.max_batch_size:
                    break
                batch.append(sample)

            buffer = buffer[len(batch):]

            if len(batch) == 1:
                data = batch[0]
                return buffer, data.unsqueeze(0)

            return buffer, min_collate(batch, pad_fn=self.pad_fn)

        start, end = self._compute_load_span()

        buffer = []
        num_samples = 0

        for ix in range(start, end):
            data = self.dataset[ix]

            if len(data) > self.max_sequence_length:
                continue
            buffer.append(data)
            num_samples += 1

            if self.num_samples > 0 and num_samples >= self.num_samples:
                break

            if sum(len(d) for d in buffer) > self.max_batch_size * self.max_buffer_size:
                buffer, batch = create_batch(buffer)
                yield batch

        while len(buffer) > 0:
            buffer, batch = create_batch(buffer)
            yield batch


    def __iter__(self):
        return iter(self.to_batches())

