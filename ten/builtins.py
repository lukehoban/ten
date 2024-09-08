# Copyright 2023 Luke Hoban

from ten import tenast, interpreter_numpy as interp
from typing import Sequence, Union
import numpy as np


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


built_ins = {
    "Exp": tenast.FunctionDeclaration(
        var("Exp"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Sqrt": tenast.FunctionDeclaration(
        var("Sqrt"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Max": tenast.FunctionDeclaration(
        var("Max"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Sum": tenast.FunctionDeclaration(
        var("Sum"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Tanh": tenast.FunctionDeclaration(
        var("Tanh"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Tri": tenast.FunctionDeclaration(
        var("Tri"),
        [var("N")],
        [],
        [],
        tensor_type(["N", "N"]),
        None,
    ),
    "Transpose": tenast.FunctionDeclaration(
        var("Transpose"),
        [var("N"), var("M")],
        [],
        [(var("x"), tensor_type(["...", "N", "M"]))],
        tensor_type(["...", "M", "N"]),
        None,
    ),
    "Mean": tenast.FunctionDeclaration(
        var("Mean"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Var": tenast.FunctionDeclaration(
        var("Var"),
        [],
        [],
        [(var("x"), tensor_type(["..."]))],
        tensor_type(["..."]),
        None,
    ),
    "Range": tenast.FunctionDeclaration(
        var("Range"),
        [var("N")],
        [],
        [],
        tenast.TensorType([var("N")]),
        None,
    ),
}

built_in_impls = {
    "Exp": interp.Func(lambda *static_args: lambda *args: np.exp(args[0])),
    "Sqrt": interp.Func(lambda *static_args: lambda *args: np.sqrt(args[0])),
    "Max": interp.Func(
        lambda *static_args: lambda *args: np.max(args[0], axis=-1, keepdims=True)
    ),
    "Sum": interp.Func(
        lambda *static_args: lambda *args: np.sum(args[0], axis=-1, keepdims=True)
    ),
    "Tanh": interp.Func(lambda *static_args: lambda *args: np.tanh(args[0])),
    "Tri": interp.Func(lambda *static_args: lambda *args: np.tri(static_args[0])),
    "Tri_2": interp.Func(lambda *static_args: lambda *args: np.tri(static_args[0])),
    "Tri_3": interp.Func(lambda *static_args: lambda *args: np.tri(static_args[0])),
    "Transpose": interp.Func(
        lambda *static_args: lambda *args: np.transpose(
            args[0],
            list(range(len(args[0].shape) - 2))
            + [len(args[0].shape) - 1, len(args[0].shape) - 2],
        )
    ),
    "Mean": interp.Func(
        lambda *static_args: lambda *args: np.mean(args[0], axis=-1, keepdims=True)
    ),
    "Var": interp.Func(
        lambda *static_args: lambda *args: np.var(args[0], axis=-1, keepdims=True)
    ),
    "Range": interp.Func(lambda *static_args: lambda *args: np.arange(static_args[0])),
    "Range_1": interp.Func(
        lambda *static_args: lambda *args: np.arange(static_args[0])
    ),
}
