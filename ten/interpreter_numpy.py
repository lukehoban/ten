# Copyright 2023 Luke Hoban

"""
An interpreter for the ten language that uses numpy for its tensor operations.
This is fairly slow, but it's useful for testing and debugging.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Optional, Union, Tuple, Dict, Callable
import numpy as np
import einops
from .tenast import (
    FunctionDeclaration,
    Token,
    Statement,
    LetStatement,
    ReturnStatement,
    Expr,
    FloatExpr,
    BinaryExpr,
    VariableExpr,
    CallExpr,
    LetExpr,
    IndexExpr,
    ReshapeExpr,
    ReshapeTensorShape,
)

Value = Union[float, np.ndarray, "Func", Sequence["Value"]]


@dataclass
class Func:
    decl: Union[FunctionDeclaration, Callable[..., Callable[..., Value]]]


@dataclass
class Env:
    parent: Optional["Env"]
    vars: Dict[str, Value]
    funcs: Dict[str, Callable[..., Value]]

    def lookup(self, name: str) -> Optional[Value]:
        ret = self.vars.get(name)
        if ret is not None:
            return ret
        if self.parent != None:
            return self.parent.lookup(name)
        return None

    def lookup_builtin(self, name: str) -> Optional[Callable[..., Value]]:
        ret = self.funcs.get(name)
        if ret != None:
            return ret
        if self.parent != None:
            return self.parent.lookup_builtin(name)
        return None


class Interpreter:
    def eval_call_expr(
        self,
        program: FunctionDeclaration,
        params: list[Value],
        args: list[Value],
        env: Env,
    ) -> Value:
        if program.body is None:
            # built-in function
            name = program.name.text.split("_")[0]
            built_in = env.lookup_builtin(name)
            if built_in is None:
                raise Exception(f"Could not find built-in function {name}")
            return built_in(*args)
        param_vars: Dict[str, Value] = {}
        for [(var, var_type), val] in zip(program.params, params):
            param_vars[var.text] = val
        vars: Dict[str, Value] = {}
        for [(var, typ), val] in zip(program.args, args):
            vars[var.text] = val
        env = Env(env, {**param_vars, **vars}, {})
        for stmt in program.body:
            res = self.eval_stmt(stmt, env)
            if res is not None:
                return res
        raise RuntimeError("No return statement in function.")

    def eval_stmt(self, stmt: Statement, env: Env) -> Optional[Value]:
        if isinstance(stmt, LetStatement):
            result = self.eval_expr(stmt.expr, env)
            if isinstance(result, list):
                for var, val in zip(stmt.variables, result):
                    env.vars[var.text] = val
            elif len(stmt.variables) == 1:
                env.vars[stmt.variables[0].text] = result
            else:
                if not isinstance(result, np.ndarray):
                    raise RuntimeError(
                        f"Cannot assign non-tensor {result} to multi-variable binding {stmt.variables}"
                    )
                items = np.split(result, len(stmt.variables))
                for var, val in zip(stmt.variables, items):
                    env.vars[var.text] = val[0]
            return None
        if isinstance(stmt, ReturnStatement):
            return self.eval_expr(stmt.expr, env)
        raise NotImplementedError("Unknown statement type.")

    def eval_expr(self, expr: Expr, env: Env) -> Value:
        if isinstance(expr, FloatExpr):
            return expr.value
        elif isinstance(expr, CallExpr):
            f = self.eval_expr(expr.f, env)
            if not isinstance(f, Func):
                raise RuntimeError(f"Cannot call non-function {expr}.")
            static_args = [self.eval_expr(arg, env) for arg in expr.static_args]
            args = [self.eval_expr(arg, env) for arg in expr.args]
            param_args = [self.eval_expr(arg, env) for arg in expr.param_args]
            # TODO: We currently treat passing a single list value to a param_args as
            # ... exapanding it into multiple arguments.
            if len(param_args) == 1 and isinstance(param_args[0], list):
                param_args = param_args[0]
            if isinstance(f.decl, Callable):
                a = f.decl(*static_args)
                b = a(*args)
                return b
            else:
                ret = self.eval_call_expr(f.decl, param_args, args, env)
                return ret
        elif isinstance(expr, ReshapeExpr):
            # (S, (3, H, K)) -> (3, H, S, K)
            i = 1
            m: Dict[str, str] = {}
            vals: Dict[str, int] = {}
            v = self.eval_expr(expr.expr, env)
            rearrange_str = ""
            # TODO: Use Transpose and Reshape instead of einops?
            for dim in expr.reshape_from.dims:
                if isinstance(dim, Token):
                    if dim.kind == "NUMBER":
                        s = f"x{i}"
                        i = i + 1
                        m[dim.text] = s
                        vals[s] = int(float(dim.text))
                    elif dim.kind == "IDENT":
                        s = dim.text
                    else:
                        raise RuntimeError("Unknown reshape dimension type.")
                    rearrange_str += s + " "
                elif isinstance(dim, ReshapeTensorShape):
                    rearrange_str += "("
                    for d in dim.dims:
                        if isinstance(d, Token):
                            if d.kind == "NUMBER":
                                s = f"x{i}"
                                i = i + 1
                                m[d.text] = s
                                vals[s] = int(float(d.text))
                            elif d.kind == "IDENT":
                                s = d.text
                            else:
                                raise RuntimeError("Unknown reshape dimension type.")
                            rearrange_str += s + " "
                        elif isinstance(d, ReshapeTensorShape):
                            raise RuntimeError(
                                "Cannot reshape tensors with nested shapes."
                            )
                        else:
                            raise RuntimeError("Unknown reshape dimension type.")
                    rearrange_str += ") "
                else:
                    raise RuntimeError("Unknown reshape dimension type.")
            rearrange_str += "-> "
            for dim in expr.reshape_to.dims:
                if isinstance(dim, Token):
                    if dim.kind == "NUMBER":
                        s = m[dim.text]
                    elif dim.kind == "IDENT":
                        s = dim.text
                    else:
                        raise RuntimeError("Unknown reshape dimension type.")
                    rearrange_str += s + " "
                elif isinstance(dim, ReshapeTensorShape):
                    rearrange_str += "("
                    for d in dim.dims:
                        if isinstance(d, Token):
                            if d.kind == "NUMBER":
                                s = m[d.text]
                            elif d.kind == "IDENT":
                                s = d.text
                            else:
                                raise RuntimeError("Unknown reshape dimension type.")
                            rearrange_str += s + " "
                        elif isinstance(d, ReshapeTensorShape):
                            raise RuntimeError(
                                "Cannot reshape tensors with nested shapes."
                            )
                        else:
                            raise RuntimeError("Unknown reshape dimension type.")
                    rearrange_str += ") "
                else:
                    raise RuntimeError("Unknown reshape dimension type.")
            return einops.rearrange(v, rearrange_str, **expr.constraints, **vals)
        elif isinstance(expr, BinaryExpr):
            left = self.eval_expr(expr.left, env)
            right = self.eval_expr(expr.right, env)
            if not isinstance(left, float) and not isinstance(left, np.ndarray):
                raise RuntimeError("Cannot +/*/** non-tensors.")
            if not isinstance(right, float) and not isinstance(right, np.ndarray):
                raise RuntimeError("Cannot add non-tensors.")
            if expr.op.text == "+":
                return left + right
            if expr.op.text == "-":
                return left - right
            if expr.op.text == "*":
                return left * right
            if expr.op.text == "/":
                return left / right
            if expr.op.text == "**":
                return left**right
            if expr.op.text == "@":
                return np.matmul(left, right)
            else:
                raise RuntimeError(f"Unknown binary operator: {expr.op.text}")
        elif isinstance(expr, VariableExpr):
            val = env.lookup(expr.name.text)
            if val is None:
                raise RuntimeError(f"Variable {expr.name.text} not found.")
            return val
        elif isinstance(expr, IndexExpr):
            e = self.eval_expr(expr.expr, env)
            idx = self.eval_expr(expr.index, env)
            if isinstance(e, np.ndarray):
                if isinstance(idx, np.ndarray):
                    return e[idx.astype(int)]
                elif isinstance(idx, float):
                    return e[int(idx)]
                else:
                    raise RuntimeError(f"Unknown index type: {type(idx)}")
            elif isinstance(e, list):
                if isinstance(idx, float) or isinstance(idx, int):
                    return e[int(idx)]
                else:
                    raise RuntimeError(f"Unknown index type: {type(idx)}")
            else:
                raise RuntimeError(f"Unknown array type: {type(e)}")
        elif isinstance(expr, LetExpr):
            new_vars = {k.text: self.eval_expr(v, env) for k, v in expr.initializers}
            env = Env(env, new_vars, {})
            return self.eval_expr(expr.body, env)
        else:
            raise NotImplementedError(f"Unknown expression type: {expr}")
