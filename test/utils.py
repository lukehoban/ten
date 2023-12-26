# Copyright 2023 Luke Hoban

from ten import tenast
from typing import Union, Sequence


def op(op: str) -> tenast.Token:
    return tenast.Token("OP", op, 0, 0)


def var(name: str) -> tenast.Token:
    return tenast.Token("IDENT", name, 0, 0)


def tensor_type(dims: Sequence[Union[int, str]]) -> tenast.TensorType:
    ret_dims: list[tenast.Token] = []
    for d in dims:
        if d == "...":
            ret_dims.append(tenast.Token("OP", "...", 0, 0))
        elif isinstance(d, str):
            ret_dims.append(tenast.Token("IDENT", d, 0, 0))
        elif isinstance(d, int):
            ret_dims.append(tenast.Token("NUMBER", str(d), 0, 0))
        else:
            raise ValueError(f"Invalid dimension: {d}")
    return tenast.TensorType(ret_dims)
