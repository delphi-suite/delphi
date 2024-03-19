from dataclasses import dataclass
from typing import Optional

from beartype import beartype


@beartype
@dataclass(frozen=True)
class HuggingfaceConfig:
    repo_id: Optional[str] = None
    save_checkpoints: bool = False
