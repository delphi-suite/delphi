import torch
from torch.utils.data import Dataset

from delphi.train.shuffle import shuffle_list


class TokenizedChunksDataset(Dataset):
    def __init__(self, tokenized_docs, max_seq_len, device):
        self.device = device
        self.tokenized_docs = tokenized_docs
        self.max_len = max_seq_len
        self.batched_tokens = (
            torch.Tensor()
        )  # will be initialized in initialize_samples

    def initialize_samples(self):
        # self.tokenized_docs is an (X, 1) tensor of dicts. Each entry is just {"tokens": [int]}
        # where [int] is doc_len long
        # we want to turn this into a (num_batches, max_len + 1) tensor of ints
        # the +1 is for the last Y token prediction, and implies an overlap of 1 token between batches
        # this is because each batch will be broken into X [:-1] and Y [1:]
        tensor_tokens = torch.stack(
            [torch.tensor(doc["tokens"]) for doc in self.tokenized_docs]
        ).to(self.device)
        self.batched_tokens = tensor_tokens.flatten().unfold(
            0, self.max_len + 1, self.max_len
        )
        self.indices = self._default_indices()

    def _default_indices(self):
        return list(range(len(self.batched_tokens)))

    def shuffle(self, epoch: int):
        """this is inefficient, but tinyevals are tiny, so nbd probably"""
        # reset for idempotent determinism
        self.indices = self._default_indices()
        shuffle_list(self.indices, seed=epoch)

    def __len__(self):
        return len(self.batched_tokens)

    def get_sample_window(self, idx):
        return self.batched_tokens[idx % len(self.batched_tokens), :]

    def __getitem__(self, idx):
        sample = self.get_sample_window(idx)
        X = sample[:-1]
        Y = sample[1:]
        return X, Y

    def __iter__(self):
        while True:
            for idx in self.indices:
                X, Y = self[idx]
                yield X, Y
