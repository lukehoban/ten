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
    ReshapeExpr,
    IndexExpr,
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
        decls: Mapping[str, FunctionDeclaration],
        entry: FunctionDeclaration,
        static_args: Mapping[str, float],
        params: Mapping[str, ParamsValue],
    ) -> onnx.ModelProto:
        for name, func in decls.items():
            self.funcs[name] = func
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
        env_bindings = {}
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
                env_bindings[tok.text] = name
        env = Env(env, env_bindings)
        for stmt in function.body:
            if isinstance(stmt, ReturnStatement):
                self.compile_expr(
                    stmt.expr,
                    static_args,
                    params,
                    env,
                    nodes,
                    initializers,
                    output,
                )
            elif isinstance(stmt, LetStatement):
                out = self.compile_expr(
                    stmt.expr, static_args, params, env, nodes, initializers
                )
                if len(stmt.variables) == 1:
                    env_bindings[stmt.variables[0].text] = out
                else:
                    parts = [self.make_temp(v.text) for v in stmt.variables]
                    nodes.append(
                        onnxmod.make_node(
                            "Split",
                            inputs=[out],
                            outputs=parts,
                            axis=0,
                        )
                    )
                    parts_squeezed = [self.make_temp(v.text) for v in stmt.variables]
                    squeeze_zero_axis = self.make_temp("squeeze")
                    nodes.append(
                        onnxmod.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[squeeze_zero_axis],
                            value=onnxmod.make_tensor(
                                name=squeeze_zero_axis,
                                data_type=onnx.TensorProto.INT64,
                                dims=[1],
                                vals=[0],
                            ),
                        )
                    )
                    for part, part_squeezed in zip(parts, parts_squeezed):
                        nodes.append(
                            onnxmod.make_node(
                                "Squeeze",
                                inputs=[part, squeeze_zero_axis],
                                outputs=[part_squeezed],
                            )
                        )
                    for v, o in zip(stmt.variables, parts_squeezed):
                        env_bindings[v.text] = o
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
            op_mapping = {
                "@": "MatMul",
                "+": "Add",
                "-": "Sub",
                "*": "Mul",
                "/": "Div",
                "**": "Pow",
            }
            op = op_mapping.get(expr.op.text)
            if op is None:
                raise NotImplementedError(f"BinaryExpr: {expr.op.text}")
            nodes.append(
                onnxmod.make_node(
                    op,
                    inputs=[t1, t2],
                    outputs=[output],
                )
            )
        elif isinstance(expr, VariableExpr):
            res = env.lookup(expr.name.text)
            if res is not None:
                return res
            res = static_env.get(expr.name.text)
            if res is None:
                raise NotImplementedError(f"Unbound variable: {expr.name.text}")
            nodes.append(
                onnxmod.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[output],
                    value=onnxmod.make_tensor(
                        name=output,
                        data_type=onnx.TensorProto.FLOAT,  # TODO: This probably has to be an int?
                        dims=[],
                        vals=[float(res)],
                    ),
                )
            )
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
            if (
                expr.f.name.text in self.funcs
                and self.funcs[expr.f.name.text].body is not None
            ):
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
                builtin_name = expr.f.name.text.split("_")[0]
                # Call a built-in
                if builtin_name == "Tri":
                    if len(static_args) != 1:
                        raise RuntimeError(
                            "Expected a single static arg N for the dimension of the square lower triangular matrix for Tri"
                        )
                    N = int(static_args[0])
                    nodes.append(
                        onnxmod.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[output],
                            value=onnxmod.make_tensor(
                                name=output,
                                data_type=onnx.TensorProto.FLOAT,
                                dims=[N, N],
                                vals=np.tri(N),
                            ),
                        )
                    )
                elif builtin_name == "Transpose":
                    if len(static_args) != 2:
                        raise RuntimeError(
                            "Expected two static args N, K for Transpose"
                        )
                    nodes.append(
                        onnxmod.make_node(
                            "Transpose",
                            inputs=args,
                            outputs=[output],
                            # TODO: This is wrong - we must type check the argument, and then use N and K to select indeces to permute
                            perm=[0, 2, 1],
                        )
                    )
                elif builtin_name in ["Mean", "Max", "Min"]:
                    nodes.append(
                        onnxmod.make_node(
                            "Reduce" + builtin_name,
                            inputs=args,
                            outputs=[output],
                            axes=[-1],
                            keepdims=1,
                        )
                    )
                elif builtin_name in ["Sum"]:
                    last_dim = self.make_temp()
                    nodes.append(
                        onnxmod.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[last_dim],
                            value=onnxmod.make_tensor(
                                name=output,
                                data_type=onnx.TensorProto.INT64,
                                dims=[1],
                                vals=[-1],
                            ),
                        )
                    )
                    nodes.append(
                        onnxmod.make_node(
                            "Reduce" + builtin_name,
                            inputs=[args[0], last_dim],
                            outputs=[output],
                            keepdims=1,
                        )
                    )
                elif builtin_name in ["Sqrt", "Exp", "Tanh"]:
                    nodes.append(
                        onnxmod.make_node(
                            builtin_name,
                            inputs=args,
                            # TODO - destructuring outputs
                            outputs=[output],
                        )
                    )
                elif builtin_name in ["Range"]:
                    if len(static_args) != 1:
                        raise RuntimeError(
                            "Expected a single static arg N for the dimension of the range for Range"
                        )
                    zero = self.make_temp()
                    nodes.append(
                        onnxmod.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[zero],
                            value=onnxmod.make_tensor(
                                name=output + "_zero",
                                data_type=onnx.TensorProto.FLOAT,
                                dims=[],
                                vals=[0],
                            ),
                        )
                    )
                    one = self.make_temp()
                    nodes.append(
                        onnxmod.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[one],
                            value=onnxmod.make_tensor(
                                name=output + "_one",
                                data_type=onnx.TensorProto.FLOAT,
                                dims=[],
                                vals=[1],
                            ),
                        )
                    )
                    n = self.make_temp()
                    nodes.append(
                        onnxmod.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[n],
                            value=onnxmod.make_tensor(
                                name=output + "_n",
                                data_type=onnx.TensorProto.FLOAT,
                                dims=[],
                                vals=[int(static_args[0])],
                            ),
                        )
                    )
                    nodes.append(
                        onnxmod.make_node(
                            "Range",
                            inputs=[zero, n, one],
                            outputs=[output],
                        )
                    )
                else:
                    # Allow calling any built-in operator
                    # TODO: Move these into standard library functions with type signatures and
                    # wrappers over the ONNX operators
                    # TODO: Do we handle static_args as attributes (or params?)
                    raise NotImplementedError(f"Unknown function: {builtin_name}")
        elif isinstance(expr, ReshapeExpr):
            t1 = self.compile_expr(
                expr.expr, static_env, param_env, env, nodes, initializers
            )
            # (S, (3, H, K)) -> (3, H, S, K)
            # Steps to perform:
            # 1. reshape into the flat input shape - (S, 3, H, K)
            # 2. transpose to the flat output shape (3, H, S, K)
            # 3. transpose into the final output shape (3, H, S, K)

            # {H,S,K -> S,(H,K)}
            # 1. reshape into the flat input shape - (H, S, K)
            # 2. transpose to the output flat shape (S, H, K)
            # 3. reshape into the final output shape (S, (H, K))
            input_dims: list[int] = []
            input_positions: Dict[str, int] = {}
            position = 0
            for dim in expr.reshape_from.dims:
                if isinstance(dim, Token):
                    if dim.kind == "IDENT":
                        input_dims.append(int(static_env[dim.text]))
                        input_positions[dim.text] = position
                    elif dim.kind == "NUMBER":
                        input_dims.append(int(float(dim.text)))
                        input_positions[dim.text] = position
                    else:
                        raise RuntimeError("Only static dimensions allowed in reshape")
                else:
                    for inner_dim in dim.dims:
                        if not isinstance(inner_dim, Token):
                            raise RuntimeError(
                                "Only one level of nested reshape input types allowed"
                            )
                        else:
                            if inner_dim.kind == "IDENT":
                                input_dims.append(int(static_env[inner_dim.text]))
                                input_positions[inner_dim.text] = position
                            elif inner_dim.kind == "NUMBER":
                                input_dims.append(int(float(inner_dim.text)))
                                input_positions[inner_dim.text] = position
                            else:
                                raise RuntimeError(
                                    "Only static dimensions allowed in reshape"
                                )
                        position = position + 1
                position = position + 1
            input_dims_name = self.make_temp()
            nodes.append(
                onnxmod.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[input_dims_name],
                    value=onnxmod.make_tensor(
                        name=output,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(input_dims)],
                        vals=input_dims,
                    ),
                )
            )
            reshaped_name = self.make_temp()
            nodes.append(
                onnxmod.make_node(
                    "Reshape",
                    inputs=[t1, input_dims_name],
                    outputs=[reshaped_name],
                )
            )
            perm_dims: list[int] = []
            final_shape: list[int] = []
            for dim in expr.reshape_to.dims:
                if isinstance(dim, Token):
                    if dim.kind == "IDENT":
                        final_shape.append(int(static_env[dim.text]))
                    elif dim.kind == "NUMBER":
                        final_shape.append(int(float(dim.text)))
                    perm_dims.append(input_positions[dim.text])
                else:
                    out_dim = 1
                    for inner_dim in dim.dims:
                        if not isinstance(inner_dim, Token):
                            raise RuntimeError(
                                "Only one level of nested reshape output types allowed"
                            )
                        if inner_dim.kind == "IDENT":
                            out_dim = out_dim * int(static_env[inner_dim.text])
                        elif inner_dim.kind == "NUMBER":
                            out_dim = out_dim * int(float(inner_dim.text))
                        perm_dims.append(input_positions[inner_dim.text])
                    final_shape.append(out_dim)
            transposed_name = self.make_temp()
            nodes.append(
                onnxmod.make_node(
                    "Transpose",
                    inputs=[reshaped_name],
                    outputs=[transposed_name],
                    perm=perm_dims,
                )
            )
            output_dims_name = self.make_temp()
            nodes.append(
                onnxmod.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[output_dims_name],
                    value=onnxmod.make_tensor(
                        name=output,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(final_shape)],
                        vals=final_shape,
                    ),
                )
            )
            nodes.append(
                onnxmod.make_node(
                    "Reshape",
                    inputs=[transposed_name, output_dims_name],
                    outputs=[output],
                )
            )
        elif isinstance(expr, IndexExpr):
            receiver = self.compile_expr(
                expr.expr, static_env, param_env, env, nodes, initializers
            )
            index = self.compile_expr(
                expr.index, static_env, param_env, env, nodes, initializers
            )
            int_index = self.make_temp()
            nodes.append(
                onnxmod.make_node(
                    "Cast",
                    inputs=[index],
                    outputs=[int_index],
                    to=onnx.TensorProto.INT64,
                )
            )
            nodes.append(
                onnxmod.make_node(
                    "Gather",
                    inputs=[receiver, int_index],
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
            elif expr.op.text == "/":
                return t1 / t2
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
