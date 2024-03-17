from dataclasses import dataclass
from typing import Optional

from beartype import beartype

from .delphi_llama_config import DelphiLlamaConfig
from .delphi_mamba_config import DelphiMambaConfig
from .delphi_model_config import DelphiModelConfig
from .model_types import ModelTypes


@beartype
@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    mamba: Optional[DelphiMambaConfig] = None
    llama: Optional[DelphiLlamaConfig] = None


def get_delphi_config(config: ModelConfig) -> DelphiModelConfig:
    # get delphi config corresponding to model_type in model config
    # e.g. {model_type: "llama", llama: my_delphi_llama_config} -> my_delphi_llama_config
    delphi_config = getattr(config, config.model_type)
    return delphi_config
