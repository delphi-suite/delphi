from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Type, cast

import transformers
from beartype import beartype
from transformers import PreTrainedModel

from .model_types import ModelType, ModelTypes
from .typed_llama_config import TypedLlamaConfig
from .typed_mamba_config import TypedMambaConfig


@beartype
@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    huggingface_config: dict[str, Any] = field(default_factory=dict)
    mamba: Optional[TypedMambaConfig] = None
    llama2: Optional[TypedLlamaConfig] = None

    def is_predefined_type(self):
        return hasattr(self, self.model_type)

    def delphi_config(self) -> dict[str, Any]:
        if self.is_predefined_type():
            return asdict(getattr(self, self.model_type))
        else:
            return dict()


def config_to_model(config: ModelConfig) -> PreTrainedModel:
    # try to get one of our pre-defined types
    # if it is one of our pre-defined typed, grab the model class and typed config
    if config.is_predefined_type():
        model_type = cast(ModelType, ModelTypes.get(config.model_type))
        model_class = model_type.model
        delphi_config = config.delphi_config()
    # otherwise, we're dealing with a generic huggingface model config
    else:
        model_class = getattr(transformers, config.model_type)
        delphi_config = config.huggingface_config
    # force cast class and config types
    model_class = cast(Type[transformers.PreTrainedModel], model_class)
    config_class = cast(Type[transformers.PretrainedConfig], model_class.config_class)
    return model_class(config_class(**(delphi_config)))
