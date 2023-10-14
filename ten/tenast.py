# Copyright 2023 Luke Hoban

from dataclasses import dataclass
from typing import Mapping, Sequence, Optional, Union, Tuple


@dataclass
class Token:
    kind: str
    text: str
    pos: int
    indentation_level: int


Expr = Union[
    "FloatExpr",
    "VariableExpr",
    "CallExpr",
    "ReshapeExpr",
    "BinaryExpr",
    "ForExpr",
    "IndexExpr",
    "LetExpr",  # Currently only used as compilation target for ForExpr
]


@dataclass
class FloatExpr:
    value: float


@dataclass
class VariableExpr:
    name: Token


@dataclass
class CallExpr:
    f: Expr
    static_args: Sequence[Expr]
    param_args: Sequence[Expr]
    args: Sequence[Expr]


@dataclass
class ReshapeExpr:
    expr: Expr
    reshape_from: "ReshapeTensorShape"
    reshape_to: "ReshapeTensorShape"
    constraints: Mapping[str, float]


@dataclass
class BinaryExpr:
    op: Token
    left: Expr
    right: Expr


@dataclass
class ForExpr:
    index: Token
    start: Expr
    end: Expr
    init: Expr
    var: Token
    loop: Expr


@dataclass
class LetExpr:
    initializers: Sequence[Tuple[Token, Expr]]
    body: Expr


@dataclass
class IndexExpr:
    expr: Expr
    index: Expr


Statement = Union["LetStatement", "ReturnStatement"]


@dataclass
class LetStatement:
    variables: Sequence[Token]
    expr: Expr


@dataclass
class ReturnStatement:
    expr: Expr


@dataclass
class TensorType:
    dims: Sequence[Token]


@dataclass
class ReshapeTensorShape:
    dims: Sequence[Union[Token, "ReshapeTensorShape"]]


@dataclass
class FunctionDeclaration:
    name: Token
    static_args: Sequence[Token]
    params: Sequence[Tuple[Token, TensorType]]
    args: Sequence[Tuple[Token, TensorType]]
    ret: TensorType
    body: Optional[Sequence[Statement]]
