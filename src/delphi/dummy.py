import torch
from jaxtyping import Float, Int

Type1 = Float[torch.Tensor, "dim"]
Type2 = Int[torch.Tensor, "batch dim"]


def dummy(arg: Type1 | Type2) -> Type1:
    if isinstance(arg, Type1):
        return arg + 1
    elif isinstance(arg, Type2):
        return arg[0] - 0.1
