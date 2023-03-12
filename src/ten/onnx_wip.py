# Copyright 2023 Luke Hoban

import onnx
from onnx import helper as onnxmod
import numpy as np
from dataclasses import dataclass
from typing import Mapping, Sequence, Optional, Dict, cast

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
    vars: Mapping[str, float]

    def lookup(self, name: str) -> float:
        return self.vars[name]


class Compiler:
    graphs: Dict[str, onnx.GraphProto] = {}
    i = 0

    def make_temp(self) -> str:
        self.i += 1
        return f"t{self.i}"

    def compile_program(
        self,
        program: Sequence[FunctionDeclaration],
        entry: FunctionDeclaration,
        static_args: Mapping[str, float],
        params: Mapping[str, np.ndarray],
    ) -> onnx.ModelProto:
        graph = self.compile_function(entry, static_args, params)
        model = onnxmod.make_model(graph, producer_name="ten")
        model.opset_import[0].version = 13
        return model

    def compile_function(
        self,
        function: FunctionDeclaration,
        static_args: Mapping[str, float],
        params: Mapping[str, np.ndarray],
    ) -> onnx.GraphProto:
        # Args and Ret -> Inputs and Output(s)
        inputs: list[onnx.ValueInfoProto] = []
        # TODO: We can't be sure the rank if there are ... in the type - so for now we just don't try :-)
        for tok, typ in function.args:
            inputs.append(
                onnxmod.make_tensor_value_info(tok.text, onnx.TensorProto.FLOAT, None)
            )
        outputs = [onnxmod.make_tensor_value_info("ret", onnx.TensorProto.FLOAT, None)]

        # Params -> Constants
        initializers: list[onnx.TensorProto] = []
        for tok, typ in function.params:
            shape = self.compile_tensor_type(typ, static_args)
            initializers.append(
                onnxmod.make_tensor(
                    name=tok.text,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[d or 0 for d in shape],
                    vals=params[tok.text].flatten(),
                )
            )

        # Body -> Nodes
        nodes: list[onnx.NodeProto] = []
        if function.body is None:
            raise NotImplementedError("No body")
        for stmt in function.body:
            if isinstance(stmt, ReturnStatement):
                self.compile_expr(stmt.expr, static_args, nodes, "ret")
            elif isinstance(stmt, LetStatement):
                raise NotImplementedError("Let statement")
            else:
                raise NotImplementedError("Unknown statement type")

        ret = onnxmod.make_graph(
            name=function.name.text,
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        self.graphs[function.name.text] = ret
        return ret

    def compile_expr(
        self,
        expr: Expr,
        env: Mapping[str, float],
        nodes: list[onnx.NodeProto],
        output: Optional[str] = None,
    ) -> str:
        if output is None:
            output = self.make_temp()
        if isinstance(expr, BinaryExpr):
            t1 = self.compile_expr(expr.left, env, nodes)
            t2 = self.compile_expr(expr.right, env, nodes)
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
            return expr.name.text
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
            args, static_args, param_args = cast(tuple[list[str], ...], ([], [], []))
            for arg in expr.static_args:
                raise NotImplementedError("call static args")
            for arg in expr.param_args:
                raise NotImplementedError("call param args")
            for arg in expr.args:
                args.append(self.compile_expr(arg, env, nodes))
            if not isinstance(expr.f, VariableExpr):
                raise NotImplementedError("non-variable function call")
            if expr.f.name.text in self.graphs:
                graph = self.graphs[expr.f.name.text]
                # TODO: Need to map args into params and returns into output
                # nodes.append(
                #     onnxmod.make_node(
                #         "Identity",
                #         inputs=[expr.f.name.text],
                #         outputs=[output],
                #     )
                # )
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
