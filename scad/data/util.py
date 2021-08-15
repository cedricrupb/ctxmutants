import torch


# Dict like object for Tensors --------------------------------

class Data(object):

    def __init__(self, seq_dim=0, **kwargs):
        self._seq_dim = seq_dim
        self._seq_shape = None

        for key, value in kwargs.items():
            self[key] = value

    def _validate_shape(self, value):
        size = value.shape[self._seq_dim]
        if self._seq_shape is None:
            self._seq_shape = [-1,]*len(value.shape)
            self._seq_shape[self._seq_dim] = size
            return

        assert size == self._seq_shape[self._seq_dim], "All elements have to share the sequence dimension ( %d )" % self._seq_dim

    def apply(self, fn):
        kwargs = {k: fn(v) for k, v in self.items()}
        return Data(self._seq_dim, **kwargs)

    @property
    def shape(self):
        if self._seq_shape is None: raise AttributeError()
        return torch.Size(self._seq_shape)

    @property
    def seq_dim(self):
        return self._seq_dim

    # Standard dict methods ----
    def __setitem__(self, key, value):
        self._validate_shape(value)
        setattr(self, key, value)

    def __getitem__(self, key):

        if isinstance(key, slice):
            return self._slice(key)

        if type(key) == int:
            return self.apply(lambda x: x[key])

        return getattr(self, key, None)

    def _slice(self, slice_obj):
        fn = lambda x: x[slice_obj]
        return self.apply(fn)

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return self.shape[0]

    def keys(self):
        return set(k for k in self.__dict__.keys() if not k.startswith("_"))

    def items(self):
        return ((k, self[k]) for k in self.keys())

    # Tensor operations -----
    def to(self, device):
        fn = lambda x: x.to(device)
        return self.apply(fn)

    def squeeze(self, dim):
        fn = lambda x: x.squeeze(dim)
        return self.apply(fn)

    def unsqueeze(self, dim):
        fn = lambda x: x.unsqueeze(dim)
        return self.apply(fn)

    def clone(self):
        return self.apply(lambda x: x.clone())

    def __str__(self):
        inner = ", ".join(["%s=%s" % (k, str(v.shape)) for k, v in self.items()])
        return "Data(%s)" % inner

# -------------------------------------------------------------

# Util functions to handle Data as Tensor object ------

def _torch_pad(x, pad_offsets, dim=0):
    length = x.shape[dim] + sum(pad_offsets)

    new_shape = list(x.shape)
    new_shape[dim] = length

    output = torch.zeros(new_shape, dtype=x.dtype)

    new_index = [slice(0, s) for s in new_shape]
    new_index[dim] = slice(pad_offsets[0], pad_offsets[0]+x.shape[dim])

    output[new_index] = x
    return output

def _data_pad(x, pad_offsets, dim=-1):
    seq_dim = dim if dim >= 0 else x.seq_dim
    fn = lambda t: _torch_pad(t, pad_offsets, seq_dim)
    return x.apply(fn)

def pad(x, pad_offsets, dim=-1):

    if hasattr(x, "apply"):
        return _data_pad(x, pad_offsets, dim)

    return _torch_pad(x, pad_offsets, dim=max(0, dim))


def stack(batch):

    # Assume that all elements of the same type
    if torch.is_tensor(batch[0]):
        return torch.stack(batch)

    seq_dims = set([d.seq_dim for d in batch])
    common_keys = set.intersection(*[set(data.keys()) for data in batch])

    assert len(seq_dims) == 1
    assert all(len(data.keys()) == len(common_keys) for data in batch), str([len(data.keys()) for data in batch])

    seq_dim = next(iter(seq_dims))
    stacks = {k: [] for k in common_keys}

    for data in batch:
        for key, value in data.items():
            stacks[key].append(value)

    batch_obj = Data(seq_dim)

    try:
        for k, v in stacks.items():
            batch_obj[k] = torch.stack(v)
    except Exception as e:
        raise e

    return batch_obj

# -----------------------------------------------------
