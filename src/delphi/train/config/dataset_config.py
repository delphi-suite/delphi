from dataclasses import dataclass, field

from beartype import beartype
from datasets import Dataset

from delphi import utils


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
        ds = utils.load_dataset_split_sequence_int32_feature(
            self.name, split, self.feature
        )
        ds.set_format("torch")
        return ds

    def load_train(self) -> Dataset:
        return self._load(self.train_split)

    def load_validation(self) -> Dataset:
        return self._load(self.validation_split)
