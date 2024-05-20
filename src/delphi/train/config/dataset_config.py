from dataclasses import dataclass, field
from typing import cast

import datasets
from beartype import beartype
from datasets import Dataset, load_dataset


@beartype
@dataclass(frozen=True)
class DatasetConfig:
    name: str = field(
        metadata={"help": "tokenized dataset on huggingface to use for train"},
    )
    feature: str = field(
        default="tokens",
        metadata={
            "help": "feature in the train dataset to use for train; should be a list of max_seq_len+1 token ints"
        },
    )
    train_split: str = field(
        default="train",
        metadata={"help": "split of the dataset to use for training"},
    )
    validation_split: str = field(
        default="validation",
        metadata={"help": "split of the dataset to use for validation"},
    )

    def _load(self, split) -> Dataset:
        ds = load_dataset(
            self.name,
            split=split,
            features=datasets.Features(
                {self.feature: datasets.Sequence(datasets.Value("int32"))}
            ),
        )
        ds = cast(Dataset, ds)
        ds.set_format("torch")
        return ds

    def load_train(self) -> Dataset:
        return self._load(self.train_split)

    def load_validation(self) -> Dataset:
        return self._load(self.validation_split)
