from dataclasses import dataclass

from llama2.model import ModelArgs, Transformer


@dataclass
class LLaMA2Args(ModelArgs):
    pass


class LLaMA2(Transformer):
    
    def __init__(self, params) -> None:
        super().__init__(self, params)
