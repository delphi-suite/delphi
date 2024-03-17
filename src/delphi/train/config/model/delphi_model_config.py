from dataclasses import dataclass


@dataclass(frozen=True)
class DelphiModelConfig:
    """
    This is a dummy class for typing purposes. We could make a Union class that we update
    every time we add a ModelConfig class, but that would mean remembering to go update
    another thing when adding a new ModelConfig.
    """

    def __init__(self):
        raise NotImplementedError(
            "DelphiModelConfig is a dummy class to provide typing for actual ModelConfig classes. It shouldn't ever be instantiated."
        )
