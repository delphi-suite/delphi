from dataclasses import dataclass, field

from beartype import beartype
from datasets import Dataset

from delphi import utils


@beartype
@dataclass(frozen=True)
class DatasetConfig:
    # tokenized dataset; HF repo id or local directory
    path: str

    # feature in the dataset; should be a list of <= max_seq_len token ints
    feature: str = "tokens"

    # split of the dataset to use for training
    train_split: str = "train"

    # split of the dataset to use for validation
    validation_split: str = "validation"

    def _load(self, split) -> Dataset:
        ds = utils.load_dataset_split_sequence_int32_feature(
            self.path, split, self.feature
        )
        ds.set_format("torch")
        return ds

    def load_train(self) -> Dataset:
        return self._load(self.train_split)

    def load_validation(self) -> Dataset:
        return self._load(self.validation_split)
