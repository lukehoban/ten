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
        p = """
            Gelu(x: {...}) -> {...}:
                return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))

            Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...,K}:
                return x@w + b

            FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
                return Linear[E*4,E]|c_proj|(Gelu(Linear[E,E*4]|c_fc|(x)))
            """
        static_args = {"S": 3, "E": 2}
        params = {
            "c_fc": {"w": np.ones([2, 8]), "b": np.ones([8])},
            "c_proj": {"w": np.ones([8, 2]), "b": np.ones([2])},
        }
        outputs = self.compile_and_run(
            p,
            static_args,
            params,
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
        p = """
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
        static_args = {"H": 2, "S": 3, "E": 4, "K": 2}
        params = {
            "c_attn": {"w": np.ones([4, 12]), "b": np.ones([12])},
            "c_proj": {"w": np.ones([4, 4]), "b": np.ones([4])},
        }
        x = [
            [1.1, 1.2, 1.3, 1.4],
            [1.3, 1.4, 1.5, 1.6],
            [1.5, 1.6, 1.7, 1.8],
        ]
        outputs = self.compile_and_run(p, static_args, params, {"x": x})
        expected = baseline.mha(x, params["c_attn"], params["c_proj"], 2)
        print(expected)
        print(outputs)
        np.testing.assert_array_almost_equal(outputs[0], expected, decimal=3)

    def test_transformer(self):
        p = """
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
        x = [
            [1.1, 1.2, 1.3, 1.4],
            [1.3, 1.4, 1.5, 1.6],
            [1.5, 1.6, 1.7, 1.8],
        ]
        outputs = self.compile_and_run(p, static_args, params, {"x": x})

        expected = baseline.transformer_block(
            x, params["mlp"], params["attn"], params["ln_1"], params["ln_2"], 2
        )
        print(expected)
        print(outputs)
        np.testing.assert_array_almost_equal(outputs[0], expected, decimal=3)

    def test_gpt2(self):
        raise unittest.SkipTest("not fully implemented")
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

            GPT2[H,S,E,B,V]|wte:{V,E}, wpe:{S,E}, blocks, ln_f|(inputs:{S}) -> {S,V}:
                x = wte.[inputs] + wpe.[Range[S]()]
                z = for i in 0...B: x, y -> Transformer[H,S,E]|blocks.[i]|(y)
                return LayerNorm[S,E]|ln_f|(z) @ Transpose[V,E](wte)
            """
        )
        decls = p.parse_program()
        decl = decls[-1]
        compiler = onnx_wip.Compiler()
        H, S, E, B, V = 2, 3, 4, 1, 6
        static_args = {"H": H, "S": S, "E": E, "B": B, "V": V}
        params = {
            "wte": np.ones([V, E]),
            "wpe": np.ones([S, E]),
            "blocks": [
                {
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
            ],
            "ln_f": {
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
            1,
            2,
            3,
        ]
        outputs = ort_sess.run(None, {"x": x})
        expected = baseline.gpt2(
            x, params["wte"], params["wpe"], params["blocks"], params["ln_f"], 2
        )
        print(expected)
        print(outputs)
        np.testing.assert_array_almost_equal(outputs[0], expected, decimal=3)

    def test_index(self):
        p = """
        Index[A,B,C](x:{A,B}, i:{C}) -> {C,B}:
            return x.[i]
        """
        static_args = {"A": 2, "B": 3, "C": 4}
        params = {}
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)
        i = np.array([0.0, 0.0, 1.0, 1.0]).astype(np.float32)
        inputs = {"x": x, "i": i}
        outputs = self.compile_and_run(p, static_args, params, inputs)
        expected = x[i.astype(np.int64)]
        print(expected)
        print(outputs)
        np.testing.assert_array_almost_equal(outputs[0], expected, decimal=3)

    def compile_and_run(
        self, program: str, static_args: dict, params: dict, inputs: dict
    ):
        p = parse.Parser(program)
        decls = p.parse_program()
        decl = decls[-1]
        compiler = onnx_wip.Compiler()
        model = compiler.compile_program(decls, decl, static_args, params)
        model = onnx_shape_inference.infer_shapes(
            model, check_type=True, data_prop=True
        )
        print(model.graph.initializer)
        print(onnx_printer.to_text(model.graph))
        for info in model.graph.value_info:
            print(f"{info.name}: {info.type}")
        ort_sess = ort.InferenceSession(model.SerializeToString())
        run_options = ort.RunOptions()
        run_options.log_severity_level = 0
        outputs = ort_sess.run(None, inputs, run_options)
        return outputs


if __name__ == "__main__":
    unittest.main()
