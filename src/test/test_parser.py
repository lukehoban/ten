# Copyright 2023 Luke Hoban

import unittest
from src.ten import ast, parse, compiler
import numpy as np
from typing import Union, Sequence, Optional, Any
import baseline

gpt2_example = """
Gelu(x: {...}) -> {...}:
    return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))

SoftMax[N](x: {...,N}) -> {...,N}:
    exp_x = Exp(x - Max(x))
    return exp_x / Sum(exp_x)

LayerNorm[S,E]|g:{E},b:{E}|(x:{S,E}) -> {S,E}:
    mean = Mean(x)
    variance = Var(x)
    return g * (x - mean) / Sqrt(variance + 1e-5) + b

Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...,K}:
    return x@w + b

FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
    a = Gelu(Linear[E,E*4]|c_fc|(x))
    return Linear[E*4,E]|c_proj|(a)

Attention[Q,K,N,V](q:{...,Q,K}, k:{...,N,K}, v:{...,N,V}, mask:{Q,N}) -> {...,Q,V}:
    return Softmax[N](q @ Transpose[N,K](k) / Sqrt(K) + mask) @ v

MHA[H,S,E,K]|c_attn, c_proj|(x:{S,E}) -> {S,E}:
    q, k, v = Linear[E,E*3]|c_attn|(x) {S,(3,H,K) -> 3,H,S,K}
    causal_mask = (Tri[S]() - 1) * 1e10
    out = Attention[S,K,S,K](q, k, v, causal_mask) {H,S,K -> S,(H,K)}   
    return Linear[E,E]|c_proj|(out)

Transformer[H,S,E]|mlp, attn, ln_1, ln_2|(x:{S,E}) -> {S, E}:
    y = x + MHA[H,S,E,E/H]|attn|(LayerNorm[S,E]|ln_1|(x))
    return y + FFN[S,E]|mlp|(LayerNorm[S,E]|ln_2|(y))

GPT2[H,S,E,B,V]|wte, wpe, blocks|(inputs:{S}) -> {S,V}:
    x = wte.[inputs] + wpe.[Range[S]()]
    z = for i in 0...B: x, y -> Transformer[H,S,E]|blocks.[i]|(y)
    return LayerNorm[S,E]|ln_f|(z) @ Transpose[V,E](wte)     

"""


class ParserTestCase(unittest.TestCase):
    def test_tokens_simple(self):
        parser = parse.Parser("")
        tok = parser.read_token()
        self.assertEqual(tok.kind, "EOF")

    def test_tokens(self):
        parser = parse.Parser("Hello World >= x,")
        tok = parser.read_token()
        self.assertEqual(tok.kind, "IDENT")
        self.assertEqual(tok.text, "Hello")
        tok = parser.read_token()
        self.assertEqual(tok.kind, "IDENT")
        self.assertEqual(tok.text, "World")
        tok = parser.read_token()
        self.assertEqual(tok.kind, "OP")
        self.assertEqual(tok.text, ">=")
        tok = parser.read_token()
        self.assertEqual(tok.kind, "IDENT")
        self.assertEqual(tok.text, "x")
        tok = parser.read_token()
        self.assertEqual(tok.kind, "OP")
        self.assertEqual(tok.text, ",")

    def test_tokens_full(self):
        str = """
        CausalSelfAttention[Embed, Heads, dropout](x : {B T Embed}) -> {B T Embed}:
            q,k,v = Linear[Embed, Embed*3](x) {B T (3 Heads K) -> 3 B Heads T K}
        """
        parser = parse.Parser(str)
        while True:
            tok = parser.read_token()
            print(tok)
            if tok.kind == "EOF":
                break

    def test_parse_csa(self):
        str = """
        CausalSelfAttention[Embed, Heads, dropout](x : {B, T, Embed}) -> {B, T, Embed}:
            q,k,v = Linear[Embed, Embed*3](x) {B, T, (3,Heads,K) -> 3, B, Heads, T, K}
            return q
        """
        parser = parse.Parser(str)
        funcs = parser.parse_program()
        print(funcs)
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name.text, "CausalSelfAttention")
        self.assertEqual(len(funcs[0].static_args), 3)
        self.assertEqual(len(funcs[0].args), 1)
        self.assertEqual(len(funcs[0].args[0][1].dims), 3)
        self.assertEqual(len(funcs[0].ret.dims), 3)
        self.assertEqual(len(funcs[0].body or []), 2)

    def test_parse_gpt2(self):
        parser = parse.Parser(gpt2_example)
        funcs = parser.parse_program()
        print(funcs)
        self.assertEqual(len(funcs), 9)


def op(op: str) -> ast.Token:
    return ast.Token("OP", op, 0, 0)


def var(name: str) -> ast.Token:
    return ast.Token("IDENT", name, 0, 0)


def tensor_type(dims: Sequence[Union[int, str]]) -> ast.TensorType:
    ret_dims: list[ast.Token] = []
    for d in dims:
        if d == "...":
            ret_dims.append(ast.Token("OP", "...", 0, 0))
        elif isinstance(d, str):
            ret_dims.append(ast.Token("IDENT", d, 0, 0))
        elif isinstance(d, int):
            ret_dims.append(ast.Token("NUMBER", str(d), 0, 0))
        else:
            raise ValueError(f"Invalid dimension: {d}")
    return ast.TensorType(ret_dims)


class CompilerTestCase(unittest.TestCase):
    def test_applied_return_type(self):
        comp = compiler.Compiler()
        test_cases: list[
            tuple[
                tuple[list[list[int | str]], list[int | str]],
                list[list[int | str]],
                Optional[list[int | str]],
            ]
        ] = [
            (([["..."]], ["..."]), [[]], []),  # Tanh
            (([["..."]], ["..."]), [["...", 3]], ["...", 3]),  # Exp
            (([["...", 3]], ["...", 1]), [["...", 4, 3]], ["...", 4, 1]),  # Max
            (([["...", 3]], ["...", 3]), [["...", 3]], ["...", 3]),  # SoftMax
        ]
        for ((a, b), c, tr) in test_cases:
            a = [tensor_type(x) for x in a]
            b = tensor_type(b)
            c = [tensor_type(x) for x in c]
            if tr is None:
                self.assertRaises(Exception, comp.applied_return_type, a, b, c)
            else:
                tr = tensor_type(tr)
                f = ast.FunctionDeclaration(
                    var("f"),
                    [],
                    [],
                    [(var(f"x_{i}"), at) for i, at in enumerate(a)],
                    b,
                    None,
                )
                self.assertEqual(comp.applied_return_type(f, c), tr)

    def test_check_broadcastable(self):
        comp = compiler.Compiler()
        test_cases: list[
            tuple[tuple[list[int | str], list[int | str]], Optional[list[int | str]]]
        ] = [
            (([], [3]), [3]),
            (([3], [3]), [3]),
            (([3], [3, 4]), None),
            (([1, 4], [3, 1]), [3, 4]),
            (([1, 4], [1, 3, 1]), [1, 3, 4]),
            (([], ["..."]), ["..."]),
            ((["...", 3], ["..."]), None),
            ((["...", 4], ["...", 3, 1]), None),
            # ((["...", 4], [1, 1]), ["...", 4]), # TODO: Is this legit?  The zero-d expansion of ... is invalid
        ]
        for ((a, b), tr) in test_cases:
            a = tensor_type(a)
            b = tensor_type(b)
            if tr is None:
                self.assertRaises(Exception, comp.check_broadcastable, a, b)
            else:
                tr = tensor_type(tr)
                self.assertEqual(comp.check_broadcastable(a, b), tr)


class InterpreterTestCase(unittest.TestCase):
    gelu_expr = ast.BinaryExpr(
        op("*"),
        ast.FloatExpr(0.5),
        ast.BinaryExpr(
            op("*"),
            ast.VariableExpr(var("x")),
            ast.BinaryExpr(
                op("+"),
                ast.FloatExpr(1.0),
                ast.CallExpr(
                    ast.VariableExpr(var("Tanh")),
                    [],
                    [],
                    [
                        ast.BinaryExpr(
                            op("*"),
                            ast.FloatExpr(0.7978845608),
                            ast.BinaryExpr(
                                op("+"),
                                ast.VariableExpr(var("x")),
                                ast.BinaryExpr(
                                    op("*"),
                                    ast.FloatExpr(0.044715),
                                    ast.BinaryExpr(
                                        op("**"),
                                        ast.VariableExpr(var("x")),
                                        ast.FloatExpr(3.0),
                                    ),
                                ),
                            ),
                        )
                    ],
                ),
            ),
        ),
    )
    gelu_decl = ast.FunctionDeclaration(
        var("Gelu"),
        [],
        [],
        [(var("x"), ast.TensorType([op("...")]))],
        ast.TensorType([op("...")]),
        [ast.ReturnStatement(gelu_expr)],
    )
    softmax_decl = ast.FunctionDeclaration(
        var("Softmax"),
        [var("N")],
        [],
        [(var("x"), ast.TensorType([op("..."), var("N")]))],
        ast.TensorType([op("..."), var("N")]),
        [
            ast.LetStatement(
                [var("exp_x")],
                ast.CallExpr(
                    ast.VariableExpr(var("Exp")),
                    [],
                    [],
                    [
                        ast.BinaryExpr(
                            op("-"),
                            ast.VariableExpr(var("x")),
                            ast.CallExpr(
                                ast.VariableExpr(var("Max")),
                                # TODO:[ast.VariableExpr(var("N"))],
                                [],
                                [],
                                [ast.VariableExpr(var("x"))],
                            ),
                        ),
                    ],
                ),
            ),
            ast.ReturnStatement(
                ast.BinaryExpr(
                    op("/"),
                    ast.VariableExpr(var("exp_x")),
                    ast.CallExpr(
                        ast.VariableExpr(var("Sum")),
                        # TODO: [ast.VariableExpr(var("N"))],
                        [],
                        [],
                        [ast.VariableExpr(var("exp_x"))],
                    ),
                ),
            ),
        ],
    )

    """
    Linear[N,K]|w:{N,K},b:{K}|(x: {...N}) -> {...K}:
        return @{...K}(x{...N}, w{N,K}) + b
    """
    linear_decl = ast.FunctionDeclaration(
        var("Linear"),
        [var("N"), var("K")],
        [(var("w"), tensor_type(["N", "K"])), (var("b"), tensor_type(["K"]))],
        [(var("x"), tensor_type(["...", "N"]))],
        tensor_type(["...", "K"]),
        [
            ast.ReturnStatement(
                ast.BinaryExpr(
                    op("+"),
                    ast.BinaryExpr(
                        op("@"),
                        ast.VariableExpr(var("x")),
                        ast.VariableExpr(var("w")),
                    ),
                    ast.VariableExpr(var("b")),
                )
            )
        ],
    )

    """
    FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
        let a = Gelu(Linear[E,E*4]|c_fc...|(x))
        return Linear[E*4,E]|c_proj...|(a)
    """
    ffn_decl = ast.FunctionDeclaration(
        var("FFN"),
        [var("S"), var("E")],
        [(var("c_fc"), tensor_type([])), (var("c_proj"), tensor_type([]))],
        [(var("x"), ast.TensorType([var("S"), var("E")]))],
        ast.TensorType([var("S"), var("E")]),
        [
            ast.LetStatement(
                [var("a")],
                ast.CallExpr(
                    ast.VariableExpr(var("Gelu")),
                    [],
                    [],
                    [
                        ast.CallExpr(
                            ast.VariableExpr(var("Linear")),
                            [
                                ast.VariableExpr(var("E")),
                                ast.BinaryExpr(
                                    op("*"),
                                    ast.VariableExpr(var("E")),
                                    ast.FloatExpr(4.0),
                                ),
                            ],
                            [ast.VariableExpr(var("c_fc"))],
                            [ast.VariableExpr(var("x"))],
                        ),
                    ],
                ),
            ),
            ast.ReturnStatement(
                ast.CallExpr(
                    ast.VariableExpr(var("Linear")),
                    [
                        ast.BinaryExpr(
                            op("*"),
                            ast.VariableExpr(var("E")),
                            ast.FloatExpr(4.0),
                        ),
                        ast.VariableExpr(var("E")),
                    ],
                    [ast.VariableExpr(var("c_proj"))],
                    [ast.VariableExpr(var("a"))],
                )
            ),
        ],
    )

    """
    Attention[Q, K, N, V](q:{...,Q,K}, k:{...,N,K}, v:{...,N,V}, mask:{Q,N}) -> {...,Q,V}:
        return @(Softmax[N]((@(q, Transpose(k)) / Sqrt(K)) + mask), v)
    """
    attention_decl = ast.FunctionDeclaration(
        var("Attention"),
        [var("Q"), var("K"), var("N"), var("V")],
        [],
        [
            (var("q"), tensor_type(["...", "Q", "K"])),
            (var("k"), tensor_type(["...", "N", "K"])),
            (var("v"), tensor_type(["...", "N", "V"])),
            (var("mask"), tensor_type(["Q", "N"])),
        ],
        tensor_type(["...", "Q", "V"]),
        [
            ast.ReturnStatement(
                ast.BinaryExpr(
                    op("@"),
                    ast.CallExpr(
                        ast.VariableExpr(var("Softmax")),
                        [ast.VariableExpr(var("N"))],
                        [],
                        [
                            ast.BinaryExpr(
                                op("+"),
                                ast.BinaryExpr(
                                    op("/"),
                                    ast.BinaryExpr(
                                        op("@"),
                                        ast.VariableExpr(var("q")),
                                        ast.CallExpr(
                                            ast.VariableExpr(var("Transpose")),
                                            [
                                                ast.VariableExpr(var("N")),
                                                ast.VariableExpr(var("K")),
                                            ],
                                            [],
                                            [ast.VariableExpr(var("k"))],
                                        ),
                                    ),
                                    ast.CallExpr(
                                        ast.VariableExpr(var("Sqrt")),
                                        [],
                                        [],
                                        [ast.VariableExpr(var("K"))],
                                    ),
                                ),
                                ast.VariableExpr(var("mask")),
                            ),
                        ],
                    ),
                    ast.VariableExpr(var("v")),
                )
            ),
        ],
    )

    """
    MHA[H,S,E,K=E/H]|c_attn, c_proj|(x:{S,E}) -> {S,E}:
        let q, k, v = Linear[E, E*H*3]|c_attn|(x) {S,(3,H,K) -> 3,H,S,K}
        let causal_mask = (1 - Tri[S]()) * -1e10
        let out: {H,E} = Attention[S,K,S,K](q, k, v, causal_mask) {H,S,K -> S,(H,K)}   
        return Linear[S,S]|c_proj|(out)
    """
    mha_decl = ast.FunctionDeclaration(
        var("MHA"),
        [var("H"), var("S"), var("E"), var("K")],
        [(var("c_attn"), tensor_type([])), (var("c_proj"), tensor_type([]))],
        [(var("x"), ast.TensorType([var("S"), var("E")]))],
        ast.TensorType([var("S"), var("E")]),
        [
            ast.LetStatement(
                [var("q"), var("k"), var("v")],
                ast.ReshapeExpr(
                    ast.CallExpr(
                        ast.VariableExpr(var("Linear")),
                        [
                            ast.VariableExpr(var("E")),
                            ast.BinaryExpr(
                                op("*"),
                                ast.VariableExpr(var("E")),
                                ast.FloatExpr(3.0),
                            ),
                        ],
                        [ast.VariableExpr(var("c_attn"))],
                        [ast.VariableExpr(var("x"))],
                    ),
                    ast.ReshapeTensorShape(
                        [
                            var("S"),
                            ast.ReshapeTensorShape(
                                [
                                    ast.Token("NUMBER", "3", 0, 0),
                                    var("H"),
                                    var("K"),
                                ]
                            ),
                        ]
                    ),
                    ast.ReshapeTensorShape(
                        [
                            ast.Token("NUMBER", "3", 0, 0),
                            var("H"),
                            var("S"),
                            var("K"),
                        ]
                    ),
                    {},
                ),
            ),
            ast.LetStatement(
                [var("causal_mask")],
                ast.BinaryExpr(
                    op("*"),
                    ast.BinaryExpr(
                        op("-"),
                        ast.FloatExpr(1.0),
                        ast.CallExpr(
                            ast.VariableExpr(var("Tri")),
                            [ast.VariableExpr(var("S"))],
                            [],
                            [],
                        ),
                    ),
                    ast.FloatExpr(-1e10),
                ),
            ),
            ast.LetStatement(
                [var("out")],
                ast.ReshapeExpr(
                    ast.CallExpr(
                        ast.VariableExpr(var("Attention")),
                        [
                            ast.VariableExpr(var("S")),
                            ast.VariableExpr(var("K")),
                            ast.VariableExpr(var("S")),
                            ast.VariableExpr(var("K")),
                        ],
                        [],
                        [
                            ast.VariableExpr(var("q")),
                            ast.VariableExpr(var("k")),
                            ast.VariableExpr(var("v")),
                            ast.VariableExpr(var("causal_mask")),
                        ],
                    ),
                    ast.ReshapeTensorShape([var("H"), var("S"), var("K")]),
                    ast.ReshapeTensorShape(
                        [var("S"), ast.ReshapeTensorShape([var("H"), var("K")])]
                    ),
                    {},
                ),
            ),
            ast.ReturnStatement(
                ast.CallExpr(
                    ast.VariableExpr(var("Linear")),
                    [ast.VariableExpr(var("E")), ast.VariableExpr(var("E"))],
                    [ast.VariableExpr(var("c_proj"))],
                    [ast.VariableExpr(var("out"))],
                )
            ),
        ],
    )

    """
    LayerNorm[S,E]|g:{E},b:{E}|(x: {S,E}) -> {S,E}:
        let mean = Mean(x)
        let variance = Var(x)
        return g * (x - mean) / Sqrt(variance + 1e-5) + b
    """
    layernorm_decl = ast.FunctionDeclaration(
        var("LayerNorm"),
        [var("S"), var("E")],
        [(var("g"), tensor_type(["E"])), (var("b"), tensor_type(["E"]))],
        [(var("x"), ast.TensorType([var("S"), var("E")]))],
        ast.TensorType([var("S"), var("E")]),
        [
            ast.LetStatement(
                [var("mean")],
                ast.CallExpr(
                    ast.VariableExpr(var("Mean")),
                    [],
                    [],
                    [ast.VariableExpr(var("x"))],
                ),
            ),
            ast.LetStatement(
                [var("variance")],
                ast.CallExpr(
                    ast.VariableExpr(var("Var")),
                    [],
                    [],
                    [ast.VariableExpr(var("x"))],
                ),
            ),
            ast.ReturnStatement(
                ast.BinaryExpr(
                    op("+"),
                    ast.BinaryExpr(
                        op("*"),
                        ast.VariableExpr(var("g")),
                        ast.BinaryExpr(
                            op("/"),
                            ast.BinaryExpr(
                                op("-"),
                                ast.VariableExpr(var("x")),
                                ast.VariableExpr(var("mean")),
                            ),
                            ast.CallExpr(
                                ast.VariableExpr(var("Sqrt")),
                                [],
                                [],
                                [
                                    ast.BinaryExpr(
                                        op("+"),
                                        ast.VariableExpr(var("variance")),
                                        ast.FloatExpr(1e-5),
                                    )
                                ],
                            ),
                        ),
                    ),
                    ast.VariableExpr(var("b")),
                )
            ),
        ],
    )

    """
    Transformer[H,S,E]|mlp, attn, ln_1, ln_2|(x: {S,E}) -> {S, E}:
        let y = x + MHA[H,S,E,E/H]|attn|(LayerNorm[]|ln_1|(x))
        return y + FFN[S,E]|mlp|(LayerNorm[]|ln_2|(y))    
    """
    transformer_decl = ast.FunctionDeclaration(
        var("Transformer"),
        [var("H"), var("S"), var("E")],
        [
            (var("mlp"), tensor_type([])),
            (var("attn"), tensor_type([])),
            (var("ln_1"), tensor_type([])),
            (var("ln_2"), tensor_type([])),
        ],
        [(var("x"), ast.TensorType([var("S"), var("E")]))],
        ast.TensorType([var("S"), var("E")]),
        [
            ast.LetStatement(
                [var("y")],
                ast.BinaryExpr(
                    op("+"),
                    ast.VariableExpr(var("x")),
                    ast.CallExpr(
                        ast.VariableExpr(var("MHA")),
                        [
                            ast.VariableExpr(var("H")),
                            ast.VariableExpr(var("S")),
                            ast.VariableExpr(var("E")),
                            ast.BinaryExpr(
                                op("/"),
                                ast.VariableExpr(var("E")),
                                ast.VariableExpr(var("H")),
                            ),
                        ],
                        [ast.VariableExpr(var("attn"))],
                        [
                            ast.CallExpr(
                                ast.VariableExpr(var("LayerNorm")),
                                [
                                    ast.VariableExpr(var("S")),
                                    ast.VariableExpr(var("E")),
                                ],
                                [ast.VariableExpr(var("ln_1"))],
                                [ast.VariableExpr(var("x"))],
                            )
                        ],
                    ),
                ),
            ),
            ast.ReturnStatement(
                ast.BinaryExpr(
                    op("+"),
                    ast.VariableExpr(var("y")),
                    ast.CallExpr(
                        ast.VariableExpr(var("FFN")),
                        [ast.VariableExpr(var("S")), ast.VariableExpr(var("E"))],
                        [ast.VariableExpr(var("mlp"))],
                        [
                            ast.CallExpr(
                                ast.VariableExpr(var("LayerNorm")),
                                [
                                    ast.VariableExpr(var("S")),
                                    ast.VariableExpr(var("E")),
                                ],
                                [ast.VariableExpr(var("ln_2"))],
                                [ast.VariableExpr(var("y"))],
                            )
                        ],
                    ),
                )
            ),
        ],
    )

    """
    GPT2[N,H,B,S,E]|wte, wpe:{}, blocks|(inputs: {S}) -> {S,E}:
        let x = wte[inputs] + wpe[Range(1,S)]
        let z = for i in 0..B: x, y => Transformer[H]|blocks[i]|(y)
        return @(LayerNorm[S,E]|ln_f|(x), Transpose(wte))
    """
    gpt2_decl = ast.FunctionDeclaration(
        var("GPT2"),
        [var("H"), var("S"), var("E"), var("B"), var("V")],
        [
            (var("wte"), tensor_type(["V", "E"])),
            (var("wpe"), tensor_type(["S", "E"])),
            (var("blocks"), tensor_type([])),
            (var("ln_f"), tensor_type([])),
        ],
        [(var("inputs"), ast.TensorType([var("S")]))],
        ast.TensorType([var("S"), var("V")]),
        [
            ast.LetStatement(
                [var("x")],
                ast.BinaryExpr(
                    op("+"),
                    ast.IndexExpr(
                        ast.VariableExpr(var("wte")),
                        ast.VariableExpr(var("inputs")),
                    ),
                    ast.IndexExpr(
                        ast.VariableExpr(var("wpe")),
                        ast.CallExpr(
                            ast.VariableExpr(var("Range")),
                            [ast.VariableExpr(var("S"))],
                            [],
                            [],
                        ),
                    ),
                ),
            ),
            ast.LetStatement(
                [var("z")],
                ast.ForExpr(
                    var("i"),
                    ast.FloatExpr(0),
                    ast.VariableExpr(var("B")),
                    ast.VariableExpr(var("x")),
                    var("y"),
                    ast.CallExpr(
                        ast.VariableExpr(var("Transformer")),
                        [
                            ast.VariableExpr(var("H")),
                            ast.VariableExpr(var("S")),
                            ast.VariableExpr(var("E")),
                        ],
                        [
                            ast.IndexExpr(
                                ast.VariableExpr(var("blocks")),
                                ast.VariableExpr(var("i")),
                            )
                        ],
                        [ast.VariableExpr(var("y"))],
                    ),
                ),
            ),
            ast.ReturnStatement(
                ast.BinaryExpr(
                    op("@"),
                    ast.CallExpr(
                        ast.VariableExpr(var("LayerNorm")),
                        [ast.VariableExpr(var("S")), ast.VariableExpr(var("E"))],
                        [ast.VariableExpr(var("ln_f"))],
                        [
                            ast.VariableExpr(var("z")),
                        ],
                    ),
                    ast.CallExpr(
                        ast.VariableExpr(var("Transpose")),
                        [
                            ast.VariableExpr(var("V")),
                            ast.VariableExpr(var("E")),
                        ],
                        [],
                        [ast.VariableExpr(var("wte"))],
                    ),
                ),
            ),
        ],
    )

    built_ins = {
        "Exp": ast.FunctionDeclaration(
            var("Exp"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Sqrt": ast.FunctionDeclaration(
            var("Sqrt"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Max": ast.FunctionDeclaration(
            var("Max"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Sum": ast.FunctionDeclaration(
            var("Sum"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Tanh": ast.FunctionDeclaration(
            var("Tanh"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Tri": ast.FunctionDeclaration(
            var("Tri"),
            [var("N")],
            [],
            [],
            tensor_type(["N", "N"]),
            None,
        ),
        "Transpose": ast.FunctionDeclaration(
            var("Transpose"),
            [var("N"), var("M")],
            [],
            [(var("x"), tensor_type(["...", "N", "M"]))],
            tensor_type(["...", "M", "N"]),
            None,
        ),
        "Mean": ast.FunctionDeclaration(
            var("Mean"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Var": ast.FunctionDeclaration(
            var("Var"),
            [],
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Range": ast.FunctionDeclaration(
            var("Range"),
            [var("N")],
            [],
            [],
            ast.TensorType([var("N")]),
            None,
        ),
    }

    built_in_impls = {
        "Exp": compiler.Func(lambda *static_args: lambda *args: np.exp(args[0])),
        "Sqrt": compiler.Func(lambda *static_args: lambda *args: np.sqrt(args[0])),
        "Max": compiler.Func(
            lambda *static_args: lambda *args: np.max(args[0], axis=-1, keepdims=True)
        ),
        "Sum": compiler.Func(
            lambda *static_args: lambda *args: np.sum(args[0], axis=-1, keepdims=True)
        ),
        "Tanh": compiler.Func(lambda *static_args: lambda *args: np.tanh(args[0])),
        "Tri": compiler.Func(lambda *static_args: lambda *args: np.tri(static_args[0])),
        "Tri_2": compiler.Func(lambda *static_args: lambda *args: np.tri(static_args[0])),
        "Tri_3": compiler.Func(lambda *static_args: lambda *args: np.tri(static_args[0])),
        "Transpose": compiler.Func(
            lambda *static_args: lambda *args: np.transpose(
                args[0],
                list(range(len(args[0].shape) - 2))
                + [len(args[0].shape) - 1, len(args[0].shape) - 2],
            )
        ),
        "Mean": compiler.Func(
            lambda *static_args: lambda *args: np.mean(args[0], axis=-1, keepdims=True)
        ),
        "Var": compiler.Func(
            lambda *static_args: lambda *args: np.var(args[0], axis=-1, keepdims=True)
        ),
        "Range": compiler.Func(
            lambda *static_args: lambda *args: np.arange(static_args[0])
        ),
        "Range_1": compiler.Func(
            lambda *static_args: lambda *args: np.arange(static_args[0])
        ),
    }

    def gelu(self, x: Union[np.ndarray, float]):
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3.0)))

    def test_eval_simple_expr(self):
        c = compiler.Compiler()
        i = compiler.Interpreter()
        tanh = lambda *static_args: lambda *args: np.tanh(args[0])
        for x in [-1.0, 0.0, 1.0]:
            expr, _ = c.compile_expr(
                self.gelu_expr,
                compiler.TypeEnv(None, {}, {"x": tensor_type([])}, self.built_ins),
            )
            ret = i.eval_expr(
                expr,
                compiler.Env(
                    None,
                    {
                        "x": x,
                        "Tanh_1": compiler.Func(tanh),
                        "Tanh_2": compiler.Func(tanh),
                        "Tanh_3": compiler.Func(tanh),
                    },
                    {},
                ),
            )
            self.assertEqual(ret, self.gelu(x))

    def test_eval_call_expr(self):
        i = compiler.Interpreter()
        c = compiler.Compiler()
        tanh = lambda *static_args: lambda *args: np.tanh(args[0])
        expr = lambda x: ast.CallExpr(
            ast.VariableExpr(var("Gelu")), [], [], [ast.FloatExpr(x)]
        )
        for x in [-1.0, 0.0, 1.0]:
            exp, _ = c.compile_expr(
                expr(x),
                compiler.TypeEnv(
                    None,
                    {},
                    {},
                    {
                        "Gelu": ast.FunctionDeclaration(
                            var("Gelu"),
                            [],
                            [],
                            [(var("x"), ast.TensorType([op("...")]))],
                            ast.TensorType([op("...")]),
                            [ast.ReturnStatement(self.gelu_expr)],
                        ),
                        **self.built_ins,
                    },
                ),
            )
            ret = i.eval_expr(
                exp,
                compiler.Env(
                    None,
                    {
                        "x": x,
                        "Tanh": compiler.Func(tanh),
                        "Gelu": compiler.Func(self.gelu_decl),
                        "Gelu_2": compiler.Func(self.gelu_decl),
                        "Gelu_4": compiler.Func(self.gelu_decl),
                        "Gelu_6": compiler.Func(self.gelu_decl),
                    },
                    {},
                ),
            )
            self.assertEqual(ret, self.gelu(x))

    def test_eval_call_linear(self):
        i = compiler.Interpreter()
        c = compiler.Compiler()
        linear_decl = c.compile_function(
            self.linear_decl, [3.0, 4.0], compiler.TypeEnv(None, {}, {}, self.built_ins)
        )
        w = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        )
        b = np.array([3.0, 4.0, 5.0, 6.0])
        x = np.array([2.0, 3.0, 4.0])
        expected = x @ w + b
        ret = i.eval_call_expr(linear_decl, [w, b], [x], compiler.Env(None, {}, {}))
        if isinstance(ret, np.ndarray):
            np.testing.assert_array_almost_equal(ret, expected)
        else:
            self.assertIsInstance(ret, np.ndarray)

    def test_eval_call_ffn(self):
        i = compiler.Interpreter()
        c = compiler.Compiler()

        ffn_decl = c.compile_function(
            self.ffn_decl,
            [3.0, 1.0],
            compiler.TypeEnv(
                None,
                {},
                {},
                {"Gelu": self.gelu_decl, "Linear": self.linear_decl, **self.built_ins},
            ),
        )
        # N=1, K=4
        w1 = np.array([[1.0, 2.0, 3.0, 4.0]])
        b1 = np.array([3.0, 4.0, 5.0, 6.0])
        # N=4, K=1
        w2 = np.array([[1.0], [2.0], [3.0], [4.0]])
        b2 = np.array([3.0])
        x = np.array([[2.0], [3.0], [4.0]])
        ret = i.eval_call_expr(
            ffn_decl,
            [[w1, b1], [w2, b2]],
            [x],
            compiler.Env(
                None,
                {
                    "Gelu_2": compiler.Func(self.gelu_decl),
                    "Linear_3": compiler.Func(self.linear_decl),
                    "Linear_4": compiler.Func(self.linear_decl),
                    **self.built_in_impls,
                },
                {},
            ),
        )
        expected = expected = self.gelu(x @ w1 + b1) @ w2 + b2
        if isinstance(ret, np.ndarray):
            np.testing.assert_array_almost_equal(ret, expected)
        else:
            self.assertIsInstance(ret, np.ndarray)

    def test_eval_gpt2(self):
        i = compiler.Interpreter()
        c = compiler.Compiler()

        V = 50257
        C = 1024
        E = 768
        H = 12
        B = 12
        S = 10
        params: Any = baseline.load_gpt2_params_from_tf_ckpt(
            "src/test/model/model.ckpt",
            {"n_vocab": V, "n_ctx": C, "n_embd": E, "n_head": H, "n_layer": B},
        )
        params_arr = [
            params["wte"],
            params["wpe"],
            [
                [
                    [
                        [block["mlp"]["c_fc"]["w"], block["mlp"]["c_fc"]["b"]],
                        [block["mlp"]["c_proj"]["w"], block["mlp"]["c_proj"]["b"]],
                    ],
                    [
                        [block["attn"]["c_attn"]["w"], block["attn"]["c_attn"]["b"]],
                        [block["attn"]["c_proj"]["w"], block["attn"]["c_proj"]["b"]],
                    ],
                    [block["ln_1"]["g"], block["ln_1"]["b"]],
                    [block["ln_2"]["g"], block["ln_2"]["b"]],
                ]
                for block in params["blocks"]
            ],
            [params["ln_f"]["g"], params["ln_f"]["b"]],
        ]

        gpt2_decl = c.compile_function(
            self.gpt2_decl,
            [12.0, 10.0, 768.0, 12.0, 50257.0],  # H, S, E, B, V
            compiler.TypeEnv(
                None,
                {},
                {},
                {
                    "Gelu": self.gelu_decl,
                    "Linear": self.linear_decl,
                    "FFN": self.ffn_decl,
                    "Attention": self.attention_decl,
                    "Softmax": self.softmax_decl,
                    "Transformer": self.transformer_decl,
                    "LayerNorm": self.layernorm_decl,
                    "MHA": self.mha_decl,
                    **self.built_ins,
                },
            ),
        )
        print(f"compiled funcs: {c.funcs.keys()}")

        # Alan Turing theorized that computers would one day become
        x = np.array([36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716])

        ret = i.eval_call_expr(
            gpt2_decl,
            params_arr,
            [x],
            compiler.Env(
                None,
                {
                    **{k: compiler.Func(v) for k, v in c.funcs.items()},
                    **self.built_in_impls,
                },
                {k: v.decl() for k, v in self.built_in_impls.items()},
            ),
        )
        if not isinstance(ret, np.ndarray):
            self.assertIsInstance(ret, np.ndarray)
            return
        self.assertEqual(ret.shape, (10, 50257))
        next_tok = np.argmax(ret[-1])
        self.assertEqual(next_tok, 262)  # ' the'

    def test_eval_call_mha(self):
        i = compiler.Interpreter()
        c = compiler.Compiler()

        """
        MHA[H=2,S=3,E=4,K=2]||([[2],[3],[4]])
            Linear[N=4,K=12]
            Attention[Q=3,K=2,N=3,V=2]
            Linear[3,3]
        """

        mha_decl = c.compile_function(
            self.mha_decl,
            [2.0, 3.0, 4.0, 2.0],
            compiler.TypeEnv(
                None,
                {},
                {},
                {
                    "Gelu": self.gelu_decl,
                    "Linear": self.linear_decl,
                    "FFN": self.ffn_decl,
                    "Attention": self.attention_decl,
                    "Softmax": self.softmax_decl,
                    **self.built_ins,
                },
            ),
        )
        print(f"compiled funcs: {c.funcs.keys()}")
        w1 = np.ones([4, 12])
        b1 = np.ones([12])
        w2 = np.ones([4, 4])
        b2 = np.ones([4])
        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        )
        ret = i.eval_call_expr(
            mha_decl,
            [[w1, b1], [w2, b2]],
            [x],
            compiler.Env(
                None,
                {
                    # "Gelu_2": parse.Func(self.gelu_decl),
                    # "Linear_1": parse.Func(self.linear_decl),
                    # "Attention_9": parse.Func(self.attention_decl),
                    # "Softmax": parse.Func(self.softmax_decl),
                    # "Linear_3": parse.Func(self.linear_decl),
                    # "Linear_4": parse.Func(self.linear_decl),
                    # "Gelu_4": parse.Func(self.gelu_decl),
                    # "Gelu_6": parse.Func(self.gelu_decl),
                    **{k: compiler.Func(v) for k, v in c.funcs.items()},
                    **self.built_in_impls,
                },
                {k: v.decl() for k, v in self.built_in_impls.items()},
            ),
        )
        # expected = expected = self.gelu(x @ w1 + b1) @ w2 + b2
        expected = baseline.mha(x, {"w": w1, "b": b1}, {"w": w2, "b": b2}, 2)
        if isinstance(ret, np.ndarray):
            np.testing.assert_array_almost_equal(ret, expected)
        else:
            self.assertIsInstance(ret, np.ndarray)

    def test_eval_call_softmax(self):
        i = compiler.Interpreter()
        c = compiler.Compiler()
        exp = lambda *static_args: lambda *args: np.exp(args[0])
        max = lambda *static_args: lambda *args: np.max(args[0], axis=-1, keepdims=True)
        sum = lambda *static_args: lambda *args: np.sum(args[0], axis=-1, keepdims=True)
        for (arr, expected) in [
            (
                np.array([-1.0, 0.0, 1.0]),
                np.array([0.09003057, 0.24472847, 0.66524096]),
            ),
            (
                np.array([[0.0, 1.0], [1.0, 2.0]]),
                np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]]),
            ),
        ]:
            softmax_decl = c.compile_function(
                self.softmax_decl,
                [np.shape(arr)[-1]],
                compiler.TypeEnv(
                    None,
                    {},
                    {},
                    self.built_ins,
                ),
            )
            vars: dict[str, compiler.Value] = dict(
                {
                    "Exp": compiler.Func(exp),
                    "Max": compiler.Func(max),
                    "Sum": compiler.Func(sum),
                }
            )
            for (k, f) in c.funcs.items():
                vars[k] = compiler.Func(f)
            ret = i.eval_call_expr(
                softmax_decl,
                [],  #  TODO: Remove this - it's part of compilation
                [arr],
                compiler.Env(
                    None,
                    vars,
                    {
                        "Exp": exp(),
                        "Max": max(),
                        "Sum": sum(),
                    },
                ),
            )
            if not isinstance(ret, np.ndarray):
                self.assertIsInstance(ret, np.ndarray)
            else:
                print(ret)
                print(expected)
                self.assertTrue(np.allclose(ret, expected))


if __name__ == "__main__":
    unittest.main()
