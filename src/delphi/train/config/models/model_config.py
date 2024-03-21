from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Type, cast

import transformers
from beartype import beartype
from beartype.typing import Type
from transformers import PreTrainedModel

from .model_types import ModelType, ModelTypes
from .typed_llama_config import TypedLlamaConfig
from .typed_mamba_config import TypedMambaConfig


@beartype
@dataclass(frozen=True)
class ModelConfig:
    model_type: str = field(
        metadata={
            "help": (
                "The model type to train. May be either a predefined "
                "type (delphi, mamba) or any CausalLM Model from the transformers "
                "library (e.g. BartForCausalLM). Predefined types should "
                "specify their respective configs in this model config; "
                "transformer library models should specify their model "
                "config arguments in transformers_config."
            )
        }
    )
    transformers_config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "config for the transformers model specified by model_type"},
    )
    mamba: Optional[TypedMambaConfig] = field(
        default=None,
        metadata={"help": "config for Delphi mamba model. See TypedMambaConfig"},
    )
    llama2: Optional[TypedLlamaConfig] = field(
        default=None,
        metadata={"help": "config for Delphi llama2 model. See TypedLlamaConfig"},
    )

    def is_predefined_type(self):
        return hasattr(self, self.model_type)

    def get_config_args(self) -> dict[str, Any]:
        if self.is_predefined_type():
            return asdict(getattr(self, self.model_type))
        else:
            return self.transformers_config

    def get_model_class(self) -> type[PreTrainedModel]:
        if self.is_predefined_type():
            model_type = cast(ModelType, ModelTypes.get(self.model_type))
            return model_type.model
        else:
            model_class = getattr(transformers, self.model_type)
            return model_class

    def get_model(self):
        model_class = self.get_model_class()
        config_class = cast(
            Type[transformers.PretrainedConfig], model_class.config_class
        )
        return model_class(config_class(**(self.get_config_args())))
