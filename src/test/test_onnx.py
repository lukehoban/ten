# Copyright 2023 Luke Hoban

import unittest
from src.ten import parse, onnx_wip
import numpy as np
import onnxruntime as ort
import onnx.printer as onnx_printer


class OnnxCompilerTestCase(unittest.TestCase):
    def test_linear(self):
        p = parse.Parser(
            """
            Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...,K}:
                return x@w + b
            """
        )
        decls = p.parse_program()
        linear_decl = decls[0]
        compiler = onnx_wip.Compiler()
        static_args = {"N": 10, "K": 7}
        params = {"w": np.ones([10, 7]), "b": np.ones([7])}
        model = compiler.compile_program(decls, linear_decl, static_args, params)
        print(onnx_printer.to_text(model))
        ort_sess = ort.InferenceSession(model.SerializeToString())
        outputs = ort_sess.run(
            None,
            {"x": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]},
        )
        np.testing.assert_array_almost_equal(
            outputs[0],
            [
                [56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0],
                [56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0],
            ],
        )

    def test_gelu(self):
        p = parse.Parser(
            """
            Gelu(x: {...}) -> {...}:
                return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))
            """
        )
        compiler = onnx_wip.Compiler()
        decls = p.parse_program()
        gelu_decl = decls[0]
        static_args = {}
        params = {}
        model = compiler.compile_program(decls, gelu_decl, static_args, params)
        print(onnx_printer.to_text(model))
        ort_sess = ort.InferenceSession(model.SerializeToString())
        for (x, y) in [(-1, -0.156408), (0.0, 0.0), (1.0, 0.843592), (2.0, 1.96059)]:
            outputs = ort_sess.run(
                None,
                {"x": np.full([], x).astype(np.float32)},
            )
            np.testing.assert_array_almost_equal(
                outputs[0],
                np.float32(y),
            )


if __name__ == "__main__":
    unittest.main()
