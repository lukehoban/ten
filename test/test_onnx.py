# Copyright 2023 Luke Hoban

import unittest
from ten import parse, onnx_wip
import numpy as np
import onnxruntime as ort
import onnx.printer as onnx_printer
import onnx.shape_inference as onnx_shape_inference
import baseline


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
        expected = [
            [56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0],
            [56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0],
        ]
        np.testing.assert_array_almost_equal(outputs[0], np.array(expected))

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
        for x, y in [(-1, -0.156408), (0.0, 0.0), (1.0, 0.843592), (2.0, 1.96059)]:
            outputs = ort_sess.run(
                None,
                {"x": np.full([], x).astype(np.float32)},
            )
            np.testing.assert_array_almost_equal(
                outputs[0],
                np.float32(y),
            )

    def test_ffn(self):
        p = parse.Parser(
            """
            Gelu(x: {...}) -> {...}:
                return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))

            Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...,K}:
                return x@w + b

            FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
                return Linear[E*4,E]|c_proj|(Gelu(Linear[E,E*4]|c_fc|(x)))
            """
        )
        decls = p.parse_program()
        ffn_decl = decls[2]
        compiler = onnx_wip.Compiler()
        static_args = {"S": 3, "E": 2}
        params = {
            "c_fc": {"w": np.ones([2, 8]), "b": np.ones([8])},
            "c_proj": {"w": np.ones([8, 2]), "b": np.ones([2])},
        }
        model = compiler.compile_program(decls, ffn_decl, static_args, params)
        print(model.graph.initializer)
        print(onnx_printer.to_text(model.graph))
        ort_sess = ort.InferenceSession(model.SerializeToString())
        outputs = ort_sess.run(
            None,
            {
                "x": [
                    [1.1, 1.2],
                    [1.3, 1.4],
                    [1.5, 1.6],
                ]
            },
        )
        expected = [[27.39452, 27.39452], [30.599133, 30.599133], [33.7999, 33.7999]]
        np.testing.assert_array_almost_equal(outputs[0], np.array(expected), decimal=3)

    def test_mha(self):
        p = parse.Parser(
            """
            SoftMax[N](x: {...,N}) -> {...,N}:
                exp_x = Exp(x - Max(x))
                return exp_x / Sum(exp_x)

            Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...,K}:
                return x@w + b

            Attention[Q,K,N,V](q:{...,Q,K}, k:{...,N,K}, v:{...,N,V}, mask:{Q,N}) -> {...,Q,V}:
                return SoftMax[N](q @ Transpose[N,K](k) / Sqrt(K) + mask) @ v
            
            MHA[H,S,E,K]|c_attn, c_proj|(x:{S,E}) -> {S,E}:
                q, k, v = Linear[E,E*3]|c_attn|(x) {S,(3,H,K) -> 3,H,S,K}
                causal_mask = (Tri[S]() - 1) * 1e10
                out = Attention[S,K,S,K](q, k, v, causal_mask) {H,S,K -> S,(H,K)}   
                return Linear[E,E]|c_proj|(out)
            """
        )
        decls = p.parse_program()
        decl = decls[-1]
        compiler = onnx_wip.Compiler()
        static_args = {"H": 2, "S": 3, "E": 4, "K": 2}
        params = {
            "c_attn": {"w": np.ones([4, 12]), "b": np.ones([12])},
            "c_proj": {"w": np.ones([4, 4]), "b": np.ones([4])},
        }
        model = compiler.compile_program(decls, decl, static_args, params)
        model = onnx_shape_inference.infer_shapes(
            model, check_type=True, data_prop=True
        )
        print(model.graph.initializer)
        print(onnx_printer.to_text(model.graph))
        ort_sess = ort.InferenceSession(model.SerializeToString())
        x = [
            [1.1, 1.2, 1.3, 1.4],
            [1.3, 1.4, 1.5, 1.6],
            [1.5, 1.6, 1.7, 1.8],
        ]
        outputs = ort_sess.run(None, {"x": x})
        expected = baseline.mha(x, params["c_attn"], params["c_proj"], 2)
        print(expected)
        print(outputs)
        np.testing.assert_array_almost_equal(outputs[0], expected, decimal=3)

    def test_transformer(self):
        p = parse.Parser(
            """
            Gelu(x: {...}) -> {...}:
                return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))

            SoftMax[N](x: {...,N}) -> {...,N}:
                exp_x = Exp(x - Max(x))
                return exp_x / Sum(exp_x)

            LayerNorm[S,E]|g:{E},b:{E}|(x:{S,E}) -> {S,E}:
                mean = Mean(x)
                variance = Mean((x - mean)**2)
                return g * (x - mean) / Sqrt(variance + 1e-5) + b

            Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...,K}:
                return x@w + b

            FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
                a = Gelu(Linear[E,E*4]|c_fc|(x))
                return Linear[E*4,E]|c_proj|(a)

            Attention[Q,K,N,V](q:{...,Q,K}, k:{...,N,K}, v:{...,N,V}, mask:{Q,N}) -> {...,Q,V}:
                return SoftMax[N](q @ Transpose[N,K](k) / Sqrt(K) + mask) @ v

            MHA[H,S,E,K]|c_attn, c_proj|(x:{S,E}) -> {S,E}:
                q, k, v = Linear[E,E*3]|c_attn|(x) {S,(3,H,K) -> 3,H,S,K}
                causal_mask = (Tri[S]() - 1) * 1e10
                out = Attention[S,K,S,K](q, k, v, causal_mask) {H,S,K -> S,(H,K)}   
                return Linear[E,E]|c_proj|(out)

            Transformer[H,S,E]|mlp, attn, ln_1, ln_2|(x:{S,E}) -> {S, E}:
                y = x + MHA[H,S,E,E/H]|attn|(LayerNorm[S,E]|ln_1|(x))
                return y + FFN[S,E]|mlp|(LayerNorm[S,E]|ln_2|(y))
            """
        )
        decls = p.parse_program()
        decl = decls[-1]
        compiler = onnx_wip.Compiler()
        H, S, E = 2, 3, 4
        static_args = {"H": H, "S": S, "E": E}
        params = {
            "mlp": {
                "c_fc": {"w": np.ones([E, 4 * E]), "b": np.ones([4 * E])},
                "c_proj": {"w": np.ones([4 * E, E]), "b": np.ones([E])},
            },
            "attn": {
                "c_attn": {"w": np.ones([E, 3 * E]), "b": np.ones([3 * E])},
                "c_proj": {"w": np.ones([E, E]), "b": np.ones([E])},
            },
            "ln_1": {
                "g": np.ones([E]),
                "b": np.ones([E]),
            },
            "ln_2": {
                "g": np.ones([E]),
                "b": np.ones([E]),
            },
        }
        model = compiler.compile_program(decls, decl, static_args, params)
        model = onnx_shape_inference.infer_shapes(
            model, check_type=True, data_prop=True
        )
        print(model.graph.initializer)
        print(onnx_printer.to_text(model.graph))
        ort_sess = ort.InferenceSession(model.SerializeToString())
        x = [
            [1.1, 1.2, 1.3, 1.4],
            [1.3, 1.4, 1.5, 1.6],
            [1.5, 1.6, 1.7, 1.8],
        ]
        outputs = ort_sess.run(None, {"x": x})
        expected = baseline.transformer_block(
            x, params["mlp"], params["attn"], params["ln_1"], params["ln_2"], 2
        )
        print(expected)
        print(outputs)
        np.testing.assert_array_almost_equal(outputs[0], expected, decimal=3)


if __name__ == "__main__":
    unittest.main()
