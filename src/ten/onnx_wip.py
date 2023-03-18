# Copyright 2023 Luke Hoban

import onnx
from onnx import helper as onnxmod
import numpy as np
from dataclasses import dataclass
from typing import Mapping, Sequence, Optional, Dict, Union

from .tenast import (
    CallExpr,
    FloatExpr,
    FunctionDeclaration,
    TensorType,
    Token,
    ReturnStatement,
    LetStatement,
    Expr,
    BinaryExpr,
    VariableExpr,
)


@dataclass
class Env:
    parent: Optional["Env"]
    vars: Mapping[str, str]

    def lookup(self, name: str) -> Optional[str]:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None


ParamsValue = Union[Mapping[str, "ParamsValue"], np.ndarray]


class Compiler:
    graphs: Dict[str, onnx.GraphProto] = {}
    # TODO: Should this be in env?
    funcs: Dict[str, FunctionDeclaration] = {}
    i = 0

    def make_temp(self, name="t") -> str:
        self.i += 1
        return f"{name}{self.i}"

    def compile_program(
        self,
        program: Sequence[FunctionDeclaration],
        entry: FunctionDeclaration,
        static_args: Mapping[str, float],
        params: Mapping[str, ParamsValue],
    ) -> onnx.ModelProto:
        for func in program:
            self.funcs[func.name.text] = func
        graph = self.compile_entry_function(entry, static_args, params)
        model = onnxmod.make_model(graph, producer_name="ten")
        model.opset_import[0].version = 13
        return model

    def compile_entry_function(
        self,
        function: FunctionDeclaration,
        static_args: Mapping[str, float],
        params: Mapping[str, ParamsValue],
    ) -> onnx.GraphProto:
        # Args and Ret -> Inputs and Output(s)
        inputs: list[onnx.ValueInfoProto] = []
        # TODO: We can't be sure the rank if there are ... in the type - so for now we just don't try :-)
        env = {}
        for tok, typ in function.args:
            inputs.append(
                onnxmod.make_tensor_value_info(tok.text, onnx.TensorProto.FLOAT, None)
            )
            env[tok.text] = tok.text
        outputs = [onnxmod.make_tensor_value_info("ret", onnx.TensorProto.FLOAT, None)]

        # Body -> Nodes
        initializers: list[onnx.TensorProto] = []
        nodes: list[onnx.NodeProto] = []

        self.compile_call(
            function, static_args, params, Env(None, env), nodes, initializers, "ret"
        )
        ret = onnxmod.make_graph(
            name=function.name.text,
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        self.graphs[function.name.text] = ret
        return ret

    def compile_call(
        self,
        function: FunctionDeclaration,
        static_args: Mapping[str, float],
        params: Mapping[str, ParamsValue],
        env: Env,
        nodes: list[onnx.NodeProto],
        initializers: list[onnx.TensorProto],
        output: Optional[str] = None,
    ) -> str:
        if output is None:
            output = self.make_temp()
        if function.body is None:
            # TODO: Should builtins move here?
            raise NotImplementedError("No body")
        call_env_bindings = {}
        for tok, typ in function.params:
            shape = self.compile_tensor_type(typ, static_args)
            param = params[tok.text]
            if isinstance(param, np.ndarray):
                name = self.make_temp(tok.text)
                initializers.append(
                    onnxmod.make_tensor(
                        name=name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[d or 0 for d in shape],
                        vals=param.flatten(),
                    )
                )
                call_env_bindings[tok.text] = name
        for stmt in function.body:
            if isinstance(stmt, ReturnStatement):
                call_env = Env(env, call_env_bindings)
                self.compile_expr(
                    stmt.expr,
                    static_args,
                    params,
                    call_env,
                    nodes,
                    initializers,
                    output,
                )
            elif isinstance(stmt, LetStatement):
                raise NotImplementedError("Let statement")
            else:
                raise NotImplementedError("Unknown statement type")
        return output

    def compile_expr(
        self,
        expr: Expr,
        static_env: Mapping[str, float],
        param_env: Mapping[str, ParamsValue],
        env: Env,
        nodes: list[onnx.NodeProto],
        initializers: list[onnx.TensorProto],
        output: Optional[str] = None,
    ) -> str:
        if output is None:
            output = self.make_temp()
        if isinstance(expr, BinaryExpr):
            t1 = self.compile_expr(
                expr.left, static_env, param_env, env, nodes, initializers
            )
            t2 = self.compile_expr(
                expr.right, static_env, param_env, env, nodes, initializers
            )
            if expr.op.text == "@":
                nodes.append(
                    onnxmod.make_node(
                        "MatMul",
                        inputs=[t1, t2],
                        outputs=[output],
                    )
                )
            elif expr.op.text == "+":
                nodes.append(
                    onnxmod.make_node(
                        "Add",
                        inputs=[t1, t2],
                        outputs=[output],
                    )
                )
            elif expr.op.text == "*":
                nodes.append(
                    onnxmod.make_node(
                        "Mul",
                        inputs=[t1, t2],
                        outputs=[output],
                    )
                )
            elif expr.op.text == "**":
                nodes.append(
                    onnxmod.make_node(
                        "Pow",
                        inputs=[t1, t2],
                        outputs=[output],
                    )
                )
            else:
                raise NotImplementedError(f"BinaryExpr: {expr.op.text}")
        elif isinstance(expr, VariableExpr):
            res = env.lookup(expr.name.text)
            if res is None:
                raise NotImplementedError(f"Unbound variable: {expr.name.text}")
            return res
        elif isinstance(expr, FloatExpr):
            nodes.append(
                onnxmod.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[output],
                    value=onnxmod.make_tensor(
                        name=output,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[],
                        vals=[float(expr.value)],
                    ),
                )
            )
        elif isinstance(expr, CallExpr):
            static_args: list[float] = []
            for arg in expr.static_args:
                static_args.append(self.eval_static_expr(arg, static_env))
            param_args: list[ParamsValue] = []
            for arg in expr.param_args:
                if not isinstance(arg, VariableExpr):
                    raise NotImplementedError("non-variable param arg")
                param_args.append(param_env[arg.name.text])
            args: list[str] = []
            for arg in expr.args:
                args.append(
                    self.compile_expr(
                        arg, static_env, param_env, env, nodes, initializers
                    )
                )
            if not isinstance(expr.f, VariableExpr):
                raise NotImplementedError("non-variable function call")
            if expr.f.name.text in self.funcs:
                decl = self.funcs[expr.f.name.text]
                if len(decl.static_args) != len(static_args):
                    raise NotImplementedError("Wrong number of static args")
                static_env = {
                    tok.text: v for tok, v in zip(decl.static_args, static_args)
                }
                if len(param_args) == 1 and len(decl.params) != 1:
                    param_arg = param_args[0]
                    if isinstance(param_arg, np.ndarray):
                        raise NotImplementedError(
                            "Attempt to spread a tensor into a params list"
                        )
                    param_args = [param_arg[tok.text] for (tok, _) in decl.params]
                if len(param_args) != len(decl.params):
                    raise NotImplementedError("Wrong number of params")
                params_env = {
                    tok.text: v for (tok, _), v in zip(decl.params, param_args)
                }
                if len(args) != len(decl.args):
                    raise NotImplementedError("Wrong number of args")
                env_bindings = {tok.text: v for (tok, _), v in zip(decl.args, args)}
                self.compile_call(
                    decl,
                    static_env,
                    params_env,
                    Env(None, env_bindings),
                    nodes,
                    initializers,
                    output,
                )
            else:
                nodes.append(
                    onnxmod.make_node(
                        expr.f.name.text,
                        inputs=args,
                        # TODO - destructuring outputs
                        outputs=[output],
                    )
                )
        else:
            raise NotImplementedError(f"Unknown expr type: {type(expr)}")
        return output

    def eval_static_expr(self, expr: Expr, env: Mapping[str, float]) -> float:
        if isinstance(expr, BinaryExpr):
            t1 = self.eval_static_expr(expr.left, env)
            t2 = self.eval_static_expr(expr.right, env)
            if expr.op.text == "+":
                return t1 + t2
            elif expr.op.text == "*":
                return t1 * t2
            elif expr.op.text == "**":
                return t1**t2
            else:
                raise NotImplementedError(f"BinaryExpr: {expr.op.text}")
        elif isinstance(expr, VariableExpr):
            return env[expr.name.text]
        elif isinstance(expr, FloatExpr):
            return float(expr.value)
        else:
            raise NotImplementedError(f"Unknown expr type: {type(expr)}")

    def compile_tensor_type(
        self, tensor: TensorType, static_args: Mapping[str, float]
    ) -> Sequence[Optional[int]]:
        return [self.compile_dim(d, static_args) for d in tensor.dims]

    def compile_dim(
        self, dim: Token, static_args: Mapping[str, float]
    ) -> Optional[int]:
        if dim.text == "...":
            return None
        elif dim.kind == "NUMBER":
            return int(float(dim.text))
        elif dim.kind == "IDENT":
            return int(static_args[dim.text])
        else:
            raise NotImplementedError("Unknown dimension type")


def make_graph_linear() -> onnx.ModelProto:
    N = 10
    K = 7
    DotDotDot = ["A", "B"]

    # Inputs and Outputs
    x = onnxmod.make_tensor_value_info("X", onnx.TensorProto.FLOAT, DotDotDot + [N])
    ret = onnxmod.make_tensor_value_info("Ret", onnx.TensorProto.FLOAT, DotDotDot + [K])

    # Constants
    wnp = np.random.randn(N, K).astype(np.float32)
    w = onnxmod.make_tensor("W", onnx.TensorProto.FLOAT, [N, K], wnp)
    bnp = np.random.randn(K).astype(np.float32)
    b = onnxmod.make_tensor("B", onnx.TensorProto.FLOAT, [K], bnp)

    # Nodes
    t1 = onnxmod.make_node("MatMul", ["X", "W"], ["T1"])
    t2 = onnxmod.make_node("Add", ["T1", "B"], ["Ret"])

    graph = onnxmod.make_graph(
        name="test",
        inputs=[x],
        outputs=[ret],
        initializer=[w, b],
        nodes=[t1, t2],
    )

    model = onnxmod.make_model(graph, producer_name="tensorlang")
    return model
