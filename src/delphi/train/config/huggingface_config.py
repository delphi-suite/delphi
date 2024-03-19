from dataclasses import dataclass
from typing import Optional

from beartype import beartype


@beartype
@dataclass(frozen=True)
class HuggingfaceConfig:
    repo_id: Optional[str] = None
    push_checkpoints_to_hub: bool = False
