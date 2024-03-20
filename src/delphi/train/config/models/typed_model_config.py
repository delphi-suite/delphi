from dataclasses import dataclass


@dataclass(frozen=True)
class TypedModelConfig:
    """
    This is a dummy class for typing purposes. We could make a Union class that we update
    every time we add a TypedModelConfig class, but that would mean remembering to go update
    another thing when adding a new TypedModelConfig.
    """

    def __init__(self):
        raise NotImplementedError(
            "TypedModelConfig is a dummy class to provide typing for actual ModelConfig classes. It shouldn't ever be instantiated."
        )
