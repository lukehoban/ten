# Copyright 2023 Luke Hoban

import fire
from ten import parse, compiler
from typing import Optional


def main(filename: str, run: Optional[str] = None) -> None:
    with open(filename, "r") as f:
        src = f.read()
    p = parse.Parser(src)
    c = compiler.Compiler()
    i = compiler.Interpreter()
    program = p.parse_program()
    type_env = compiler.TypeEnv(None, {}, {}, {})
    for f in program:
        decl = c.compile_function(f, [], type_env)
        type_env.funcs[decl.name.text] = decl
    # How to load weights?
    # How to load inputs?
    # How to present outputs?


if __name__ == "__main__":
    fire.Fire(main)
