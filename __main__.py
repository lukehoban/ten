# Copyright 2023 Luke Hoban

import fire
from ten import parse, compiler, tenast, interpreter_numpy as interp
from ten.util import encoder
from test import baseline
from typing import Optional, Any, Sequence, Union, Dict
import numpy as np


def main(
    prompt: str,
    model: Optional[str] = "GPT2",
    run: Optional[str] = None,
    tokens: int = 20,
) -> None:
    if model is None or model == "GPT2":
        enc = encoder.get_encoder("gpt2", "./test/model/")

        with open("./examples/gpt2.ten", "r") as f:
            src = f.read()
        p = parse.Parser(src)
        program = p.parse_program()
        funcs = {func.name.text: func for func in program}
        type_env = compiler.TypeEnv(None, {}, {}, {**built_ins, **funcs})
        gpt2 = funcs["GPT2"]

        V = 50257
        C = 1024
        E = 768
        H = 12
        B = 12
        S = 10
        params: Any = baseline.load_gpt2_params_from_tf_ckpt(
            "test/model/gpt2/model.ckpt",
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

        x = np.array(enc.encode(prompt))
        print(x)
        print(prompt)

        for t in range(0, tokens):
            S = len(x)
            c = compiler.Compiler()
            gpt2_compiled = c.compile_function(
                gpt2,
                [float(x) for x in [H, S, E, B, V]],
                type_env,
            )
            i = interp.Interpreter()
            ret = i.eval_call_expr(
                gpt2_compiled,
                params_arr,
                [x],
                interp.Env(
                    None,
                    {
                        **{k: interp.Func(v) for k, v in c.funcs.items()},
                        **built_in_impls,
                    },
                    {k: v.decl() for k, v in built_in_impls.items()},  # type: ignore
                ),
            )
            if not isinstance(ret, np.ndarray):
                raise RuntimeError("invalid return value")
            next_tok = np.argmax(ret[-1])
            x = np.append(x, next_tok)
            next_str = enc.decode([next_tok])
            print(next_str)


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


if __name__ == "__main__":
    fire.Fire(main)
