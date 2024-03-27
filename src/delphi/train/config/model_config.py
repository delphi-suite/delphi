from dataclasses import dataclass, field
from typing import Any, cast

import transformers
from beartype import beartype
from beartype.typing import Type


@beartype
@dataclass(frozen=True)
class ModelConfig:
    model_type: str = field(
        metadata={
            "help": (
                "Name of any CausalLM Model from the transformers "
                "library (e.g. 'BartForCausalLM'). Model configuration arguments, "
                "e.g. hidden size, should be specified in model_params"
            )
        }
    )
    model_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "config for the transformers model specified by model_type. "
                "e.g. {'hidden_size': 128, ...}"
            )
        },
    )

    def get_model(self):
        model_class = getattr(transformers, self.model_type)
        config_class = cast(
            Type[transformers.PretrainedConfig], model_class.config_class
        )
        return model_class(config_class(**(self.model_params)))
