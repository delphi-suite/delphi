from dataclasses import dataclass


@dataclass
class WandbConfig:
    project: str
    entity: str
    silence: bool = False
