from dataclasses import dataclass
from typing import Optional

from beartype import beartype

from .model_types import ModelTypes
from .typed_llama_config import TypedLlamaConfig
from .typed_mamba_config import TypedMambaConfig
from .typed_model_config import TypedModelConfig


@beartype
@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    mamba: Optional[TypedMambaConfig] = None
    llama2: Optional[TypedLlamaConfig] = None

    def __post_init__(self):
        if get_delphi_config(self) is None:
            raise ValueError(
                f"Model config specifies model_type = {self.model_type} "
                "but doesn't provide a corresponding config."
            )


def get_delphi_config(config: ModelConfig) -> TypedModelConfig:
    # get delphi config corresponding to model_type in model config
    # e.g. {model_type: "llama2", llama2: my_delphi_llama_config} ->
    #           my_delphi_llama_config
    delphi_config = getattr(config, config.model_type)
    return delphi_config
