from dataclasses import dataclass

from beartype import beartype


@beartype
@dataclass
class WandbConfig:
    project: str
    entity: str
    silence: bool = False
