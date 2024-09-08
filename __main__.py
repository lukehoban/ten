# Copyright 2023 Luke Hoban

import fire
from ten import parse, compiler, builtins, interpreter_numpy as interp
from ten.util import encoder
from test import baseline
from typing import Optional, Any
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
        type_env = compiler.TypeEnv(None, {}, {}, {**builtins.built_ins, **funcs})
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
                        **builtins.built_in_impls,
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


if __name__ == "__main__":
    fire.Fire(main)
