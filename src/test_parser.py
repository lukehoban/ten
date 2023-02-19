import unittest
import parse
import numpy as np


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
        self.assertEqual(len(funcs[0].body), 1)


def op(op: str) -> parse.Token:
    return parse.Token("OP", op, 0, 0)


def var(name: str) -> parse.Token:
    return parse.Token("IDENT", name, 0, 0)


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
                                [parse.VariableExpr(var("N"))],
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
                        [parse.VariableExpr(var("N"))],
                        [parse.VariableExpr(var("exp_x"))],
                    ),
                ),
            ),
        ],
    )

    def test_eval_simple_expr(self):
        i = parse.Interpreter()
        tanh = lambda *static_args: lambda *args: np.tanh(args[0])
        for x in [-1.0, 0.0, 1.0]:
            ret = i.eval_expr(
                self.gelu_expr, parse.Env(None, {}, {"x": x, "Tanh": parse.Func(tanh)})
            )
            self.assertEqual(
                ret, 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3.0)))
            )

    def test_eval_call_expr(self):
        i = parse.Interpreter()
        tanh = lambda *static_args: lambda *args: np.tanh(args[0])
        expr = lambda x: parse.CallExpr(
            parse.VariableExpr(var("Gelu")), [], [parse.FloatExpr(x)]
        )
        for x in [-1.0, 0.0, 1.0]:
            ret = i.eval_expr(
                expr(x),
                parse.Env(
                    None,
                    {},
                    {
                        "x": x,
                        "Tanh": parse.Func(tanh),
                        "Gelu": parse.Func(self.gelu_decl),
                    },
                ),
            )
            self.assertEqual(
                ret, 0.5 * x * (1.0 + np.tanh(0.7978845608 * x + 0.044715 * x**3.0))
            )

    def test_eval_call_softmax(self):
        i = parse.Interpreter()
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
            ret = i.eval_call_expr(
                self.softmax_decl,
                [np.shape(arr)[-1]],
                [arr],
                parse.Env(
                    None,
                    {},
                    {
                        "Exp": parse.Func(exp),
                        "Max": parse.Func(max),
                        "Sum": parse.Func(sum),
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
