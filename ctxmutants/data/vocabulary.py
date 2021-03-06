import collections

class Vocabulary:

    def __init__(self, tokens=[]):
        self._index = {}
        self._rv_index = []
        self._close = False

        for token in tokens:
            self.index(token)

    def __getitem__(self, x):
        try:
            return self._rv_index[int(x)]
        except Exception:
            return self._index[x]

    def __len__(self):
        return len(self._rv_index)

    def __contains__(self, key):
        return key in self._index

    def __iter__(self):
        return iter(self._rv_index)

    def index(self, x):
        assert type(x) is str, "Vocabulary only indexes strings, but got %s" % str(x)

        try:
            return self._index[x]
        except KeyError as e:
            if self._close: raise e

            result = len(self._index)
            self._index[x] = len(self._index)
            self._rv_index.append(x)

            assert len(self._index) == len(self._rv_index)

            return result

    def close(self):
        self._close = True

    def open(self):
        self._close = False

    def load(self, file_path, encoding="utf-8"):

        self._index = {}
        self._rv_index = []

        with open(file_path, "r", encoding=encoding) as file_io:
            position = 0
            line = file_io.readline()

            while line is not None and len(line) > 0:
                line = line[:-1]
                if line in self._index: # Crash: Something is defined twice in vocabulary
                    line = file_io.readline()
                    continue

                self._index[line] = position
                self._rv_index.append(line)
                position += 1
                line = file_io.readline()


# Based on https://github.com/VHellendoorn/ICLR20-Great/blob/master/data/vocabulary.py
class BPEEncoder:

    def __init__(self, bpe_vocabulary, eot="#"):
        self.vocab = bpe_vocabulary
        self.bpe_cache = {}
        self.eot   = eot

        self.bpe_index = {}

        for token in bpe_vocabulary:
            if token.startswith("[") and token.endswith("]"):
                continue
            token_index = token[:2]
            if token_index not in self.bpe_index:
                self.bpe_index[token_index] = set()
            self.bpe_index[token_index].add(token)


    def __call__(self, token):
        return self.encode(token)

    def _lookup_subtoken(self, subtoken):
        self.vocab.close()
        try:
            return self.vocab.index(subtoken)
        except KeyError:
            return self.vocab["[UNK]"]


    def _encode_subtoken_string(self, token):
        tokens = []

        ix = 0
        token_len = len(token)

        while ix < token_len:
            candiates = self.bpe_index.get(token[ix:ix+2], [])
            candiates = [t for t in candiates
                            if t == token[ix:ix+len(t)] and not len(token) == ix + len(t) + 1]

            if not candiates:
                top_candidate = token[ix]
            else:
                top_candidate = max(candiates, key=lambda x: len(x))

            tokens.append(top_candidate)
            ix += len(top_candidate)

        return tokens

    def encode(self, token):
        token += self.eot

        if token in self.vocab:
            return (self._lookup_subtoken(token),)

        if token in self.bpe_cache:
            return self.bpe_cache[token]

        subtokens = self._encode_subtoken_string(token)
        subtoken_ids = tuple(map(self._lookup_subtoken, subtokens))
        self.bpe_cache[token] = subtoken_ids
        return subtoken_ids



class LFUCacheVocabulary:

    def __init__(self, max_size = 10_000):
        self.max_size = max_size

        self.vocabulary       = {}
        self.frequency_bins   = collections.defaultdict(set)
        self.lowest_frequency = 0

    def items(self):
        return self.vocabulary.keys()

    def put(self, token):

        token_frequency = 0
        
        try:
            token_frequency = self.vocabulary[token]
            self.frequency_bins[token_frequency].remove(token)
            token_frequency += 1
        except KeyError:
            self.lowest_frequency = 0

        self.vocabulary[token] = token_frequency
        self.frequency_bins[token_frequency].add(token)

        # Adjust lowest frequency
        for frequency in range(self.lowest_frequency, token_frequency):
            if len(self.frequency_bins[frequency]) == 0:
                self.lowest_frequency += 1
            else:
                break

        self.purge()


    def extend(self, tokens):
        for token in tokens: self.put(token)

    def purge(self):

        if len(self.vocabulary) < self.max_size: return

        def iter_lfu_tokens():
            while True:
                current_bin = self.frequency_bins[self.lowest_frequency]

                for token in list(current_bin):
                    yield token
                    current_bin.remove(token)

                self.lowest_frequency += 1

        lfu_tokens = iter(iter_lfu_tokens())
        
        while len(self.vocabulary) >= self.max_size:
            lfu_token = next(lfu_tokens)
            del self.vocabulary[lfu_token]








