import os
from dataclasses import dataclass, field
from typing import Optional

from beartype import beartype


def get_hf_token():
    token = os.getenv("HF_TOKEN", "")
    assert token, "HF_TOKEN env variable must be set or specified manually"
    return token


@beartype
@dataclass(frozen=True)
class HuggingfaceConfig:
    repo_id: Optional[str] = None
    push_checkpoints_to_hub: bool = False
    token: str = field(default_factory=get_hf_token)
