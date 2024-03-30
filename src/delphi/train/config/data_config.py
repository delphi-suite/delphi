from dataclasses import dataclass, field
from typing import Optional

from beartype import beartype

from delphi import constants


@beartype
@dataclass(frozen=True)
class DataConfig:
    train_dataset: str = field(
        # TODO: remove default after updating configs to include this field
        default=constants.TINYSTORIES_TOKENIZED_HF_DATASET,
        metadata={"help": "tokenized dataset on huggingface to use for train"},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "split of the train dataset to use for train"},
    )
    train_feature: str = field(
        default="tokens",
        metadata={
            "help": "feature in the train dataset to use for train; should be a list of max_seq_len+1 token ints"
        },
    )
    train_sample_limit: Optional[int] = field(
        default=None,
        metadata={"help": "limit the number of train samples to use"},
    )

    validation_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "tokenized dataset on huggingface to use for validation. "
                "If not set, validation defaults to using train_dataset"
            )
        },
    )
    validation_split: str = field(
        default="validation",
        metadata={"help": "split of the validation dataset to use for validation"},
    )
    validation_feature: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "feature in the validation dataset to use for validation; "
                "should be a list of max_seq_len+1 token ints. "
                "If not set, validation defaults to using train_feature."
            )
        },
    )
    validation_sample_limit: Optional[int] = field(
        default=None,
        metadata={"help": "limit the number of validation samples to use"},
    )
