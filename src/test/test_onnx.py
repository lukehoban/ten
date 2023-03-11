# Copyright 2023 Luke Hoban

import unittest
from src.ten import parse, onnx_wip
import numpy as np
import onnxruntime as ort


class OnnxCompilerTestCase(unittest.TestCase):
    def test_tokens_simple(self):
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


if __name__ == "__main__":
    unittest.main()
