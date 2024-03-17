"""
For any given model we use, there are three associated types:
- TypedModelConfig: a typed dataclass that defines the arguments to the model.
    We use this to enforce some semblance of type safety in configs and code in general.
- PretrainedConfig: a transformers config that defines the model architecture.
    The arguments for this are defined in TypedModelConfig.
- PreTrainedModel: a transformers model that implements the model architecture.
    Configured by PretrainedConfig.

    This file defines a ModelType dataclass that associated these three types for a given model,
    and a ModelTypes container class that defines all the models we use in Delphi along with a
    helpful ModelTypes.get() method for getting ModelType from a string.
"""
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Type
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from .typed_llama_config import TypedLlamaConfig
from .typed_mamba_config import TypedMambaConfig
from .typed_model_config import TypedModelConfig


@beartype
@dataclass(frozen=True)
class ModelType:
    name: str
    delphi_config: type[TypedModelConfig]
    config: type[PretrainedConfig]
    model: type[PreTrainedModel]

    # Allow for ModelType == 'llama'
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        else:
            return super().__eq__(other)

    def __post_init__(self):
        # register the ModelType so ModelTypes.get(model_type_name) works
        _model_name_to_model_type[self.name.lower()] = self


_model_name_to_model_type: dict[str, ModelType] = {}


# define new model types here
class ModelTypes:
    MAMBA = ModelType(
        name="mamba",
        delphi_config=TypedMambaConfig,
        config=MambaConfig,
        model=MambaForCausalLM,
    )
    LLAMA = ModelType(
        name="llama",
        delphi_config=TypedLlamaConfig,
        config=LlamaConfig,
        model=LlamaForCausalLM,
    )

    # NEWMODEL = ModelType(  # var name should match name
    #    name="newmodel",  # string that will be associated with model in configs, etc
    #    typed_config=TypedNewModelConfig,  # typed dataclass for args to config
    #    config=NewModelConfig,  # transformers config
    #    model=NewModelForCausalLM,  # transformers model
    # )

    @classmethod
    def get(cls: Type["ModelTypes"], name: str) -> ModelType:
        return _model_name_to_model_type[name.lower()]
