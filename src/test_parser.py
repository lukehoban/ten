import unittest
import parse
import numpy as np
from typing import Union, Sequence


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
            let q,k,v = Linear[Embed, Embed*3](x) {B T (3 Heads K) -> 3 B Heads T K}
        """
        parser = parse.Parser(str)
        while True:
            tok = parser.read_token()
            print(tok)
            if tok.kind == "EOF":
                break

    def test_parse_csa(self):
        str = """
        CausalSelfAttention[Embed, Heads, dropout](x : {B T Embed}) -> {B T Embed}: 
            let q,k,v = Linear[Embed, Embed*3](x) {B T (3 Heads K) -> 3 B Heads T K}
            return q
        """
        parser = parse.Parser(str)
        funcs = parser.parse()
        print(funcs)
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name.text, "CausalSelfAttention")
        self.assertEqual(len(funcs[0].static_args), 3)
        self.assertEqual(len(funcs[0].args), 1)
        self.assertEqual(len(funcs[0].args[0][1].dims), 3)
        self.assertEqual(len(funcs[0].ret.dims), 3)
        self.assertEqual(len(funcs[0].body or []), 1)


def op(op: str) -> parse.Token:
    return parse.Token("OP", op, 0, 0)


def var(name: str) -> parse.Token:
    return parse.Token("IDENT", name, 0, 0)


def tensor_type(dims: Sequence[Union[int, str]]) -> parse.TensorType:
    ret_dims: list[parse.Token] = []
    for d in dims:
        if d == "...":
            ret_dims.append(parse.Token("OP", "...", 0, 0))
        elif isinstance(d, int):
            ret_dims.append(parse.Token("NUMBER", str(d), 0, 0))
        else:
            raise ValueError(f"Invalid dimension: {d}")
    return parse.TensorType(ret_dims)


class InterpreterTestCase(unittest.TestCase):
    gelu_expr = parse.BinaryExpr(
        op("*"),
        parse.FloatExpr(0.5),
        parse.BinaryExpr(
            op("*"),
            parse.VariableExpr(var("x")),
            parse.BinaryExpr(
                op("+"),
                parse.FloatExpr(1.0),
                parse.CallExpr(
                    parse.VariableExpr(var("Tanh")),
                    [],
                    [
                        parse.BinaryExpr(
                            op("*"),
                            parse.FloatExpr(0.7978845608),
                            parse.BinaryExpr(
                                op("+"),
                                parse.VariableExpr(var("x")),
                                parse.BinaryExpr(
                                    op("*"),
                                    parse.FloatExpr(0.044715),
                                    parse.BinaryExpr(
                                        op("**"),
                                        parse.VariableExpr(var("x")),
                                        parse.FloatExpr(3.0),
                                    ),
                                ),
                            ),
                        )
                    ],
                ),
            ),
        ),
    )
    gelu_decl = parse.FunctionDeclaration(
        var("Gelu"),
        [],
        [(var("x"), parse.TensorType([op("...")]))],
        parse.TensorType([op("...")]),
        [parse.ReturnStatement(gelu_expr)],
    )
    softmax_decl = parse.FunctionDeclaration(
        var("Softmax"),
        [var("N")],
        [(var("x"), parse.TensorType([op("..."), var("N")]))],
        parse.TensorType([op("..."), var("N")]),
        [
            parse.LetStatement(
                [var("exp_x")],
                parse.CallExpr(
                    parse.VariableExpr(var("Exp")),
                    [],
                    [
                        parse.BinaryExpr(
                            op("-"),
                            parse.VariableExpr(var("x")),
                            parse.CallExpr(
                                parse.VariableExpr(var("Max")),
                                # TODO:[parse.VariableExpr(var("N"))],
                                [],
                                [parse.VariableExpr(var("x"))],
                            ),
                        ),
                    ],
                ),
            ),
            parse.ReturnStatement(
                parse.BinaryExpr(
                    op("/"),
                    parse.VariableExpr(var("exp_x")),
                    parse.CallExpr(
                        parse.VariableExpr(var("Sum")),
                        # TODO: [parse.VariableExpr(var("N"))],
                        [],
                        [parse.VariableExpr(var("exp_x"))],
                    ),
                ),
            ),
        ],
    )

    built_ins = {
        "Exp": parse.FunctionDeclaration(
            var("Exp"),
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Max": parse.FunctionDeclaration(
            var("Max"),
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Sum": parse.FunctionDeclaration(
            var("Sum"),
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
        "Tanh": parse.FunctionDeclaration(
            var("Tanh"),
            [],
            [(var("x"), tensor_type(["..."]))],
            tensor_type(["..."]),
            None,
        ),
    }

    built_in_impls = {
        "Exp": lambda *static_args: lambda *args: np.exp(args[0]),
        "Max": lambda *static_args: lambda *args: np.max(
            args[0], axis=-1, keepdims=True
        ),
        "Sum": lambda *static_args: lambda *args: np.sum(
            args[0], axis=-1, keepdims=True
        ),
        "Tanh": lambda *static_args: lambda *args: np.tanh(args[0]),
    }

    def test_eval_simple_expr(self):
        c = parse.Compiler()
        i = parse.Interpreter()
        tanh = lambda *static_args: lambda *args: np.tanh(args[0])
        for x in [-1.0, 0.0, 1.0]:
            expr, _ = c.compile_expr(
                self.gelu_expr,
                parse.TypeEnv(None, {}, {"x": tensor_type([])}, self.built_ins),
            )
            ret = i.eval_expr(
                expr,
                parse.Env(
                    None,
                    {},
                    {
                        "x": x,
                        "Tanh_1": parse.Func(tanh),
                        "Tanh_2": parse.Func(tanh),
                        "Tanh_3": parse.Func(tanh),
                    },
                    {},
                ),
            )
            self.assertEqual(
                ret, 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3.0)))
            )

    def test_eval_call_expr(self):
        i = parse.Interpreter()
        c = parse.Compiler()
        tanh = lambda *static_args: lambda *args: np.tanh(args[0])
        expr = lambda x: parse.CallExpr(
            parse.VariableExpr(var("Gelu")), [], [parse.FloatExpr(x)]
        )
        for x in [-1.0, 0.0, 1.0]:
            exp, _ = c.compile_expr(
                expr(x),
                parse.TypeEnv(
                    None,
                    {},
                    {},
                    {
                        "Gelu": parse.FunctionDeclaration(
                            var("Gelu"),
                            [],
                            [(var("x"), parse.TensorType([op("...")]))],
                            parse.TensorType([op("...")]),
                            [parse.ReturnStatement(self.gelu_expr)],
                        ),
                        **self.built_ins,
                    },
                ),
            )
            ret = i.eval_expr(
                exp,
                parse.Env(
                    None,
                    {},
                    {
                        "x": x,
                        "Tanh": parse.Func(tanh),
                        "Gelu": parse.Func(self.gelu_decl),
                        "Gelu_2": parse.Func(self.gelu_decl),
                        "Gelu_4": parse.Func(self.gelu_decl),
                        "Gelu_6": parse.Func(self.gelu_decl),
                    },
                    {},
                ),
            )
            self.assertEqual(
                ret, 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3.0)))
            )

    def test_eval_call_softmax(self):
        i = parse.Interpreter()
        c = parse.Compiler()
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
            softmax_decl, compiledfuncs = c.compile_function(
                self.softmax_decl,
                [np.shape(arr)[-1]],
                parse.TypeEnv(
                    None,
                    {},
                    {},
                    {
                        "Exp": parse.FunctionDeclaration(
                            var("Exp"),
                            [],
                            [(var("x"), tensor_type(["..."]))],
                            tensor_type(["..."]),
                            None,
                        ),
                        "Max": parse.FunctionDeclaration(
                            var("Max"),
                            [],
                            [(var("x"), tensor_type(["..."]))],
                            tensor_type(["..."]),
                            None,
                        ),
                        "Sum": parse.FunctionDeclaration(
                            var("Sum"),
                            [],
                            [(var("x"), tensor_type(["..."]))],
                            tensor_type(["..."]),
                            None,
                        ),
                    },
                ),
            )
            vars: dict[str, parse.Value] = dict(
                {
                    "Exp": parse.Func(exp),
                    "Max": parse.Func(max),
                    "Sum": parse.Func(sum),
                }
            )
            for (k, f) in compiledfuncs.items():
                vars[k] = parse.Func(f)
            ret = i.eval_call_expr(
                softmax_decl,
                [],  #  TODO: Remove this - it's part of compilation
                [arr],
                parse.Env(
                    None,
                    {},
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
