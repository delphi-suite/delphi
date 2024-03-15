from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    PreTrainedModel,
)

from delphi.constants import ModelTypes
from delphi.train.config.model.delphi_llama_config import DelphiLlamaConfig
from delphi.train.config.model.delphi_mamba_config import DelphiMambaConfig


@dataclass
class ModelConfig:
    model_type: str = ModelTypes.LLAMA
    mamba: DelphiMambaConfig = DelphiMambaConfig()
    llama: DelphiLlamaConfig = DelphiLlamaConfig()


def config_to_model(config: ModelConfig) -> PreTrainedModel:
    if config.model_type == ModelTypes.LLAMA and config.llama is not None:
        return LlamaForCausalLM(LlamaConfig(**asdict(config.llama)))
    elif config.model_type == ModelTypes.MAMBA and config.mamba is not None:
        return MambaForCausalLM(MambaConfig(**asdict(config.mamba)))
    else:
        raise ValueError(f"Unknown config type: {config.model_type}")
