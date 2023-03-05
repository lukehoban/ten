# Copyright 2023 Luke Hoban

from dataclasses import dataclass
from typing import Mapping, Sequence, Optional, Union, Tuple, Dict, Callable
import numpy as np
import einops


@dataclass
class Token:
    kind: str
    text: str
    pos: int
    indentation_level: int


Expr = Union[
    "FloatExpr",
    "VariableExpr",
    "CallExpr",
    "ReshapeExpr",
    "BinaryExpr",
    "ForExpr",
    "IndexExpr",
    "LetExpr",  # Currently only used as compilation target for ForExpr
]


@dataclass
class FloatExpr:
    value: float


@dataclass
class VariableExpr:
    name: Token


@dataclass
class CallExpr:
    f: Expr
    static_args: Sequence[Expr]
    param_args: Sequence[Expr]
    args: Sequence[Expr]


@dataclass
class ReshapeExpr:
    expr: Expr
    reshape_from: "ReshapeTensorShape"
    reshape_to: "ReshapeTensorShape"
    constraints: Mapping[str, float]


@dataclass
class BinaryExpr:
    op: Token
    left: Expr
    right: Expr


@dataclass
class ForExpr:
    index: Token
    start: Expr
    end: Expr
    init: Expr
    var: Token
    loop: Expr


@dataclass
class LetExpr:
    initializers: Sequence[Tuple[Token, Expr]]
    body: Expr


@dataclass
class IndexExpr:
    expr: Expr
    index: Expr


Statement = Union["LetStatement", "ReturnStatement"]


@dataclass
class LetStatement:
    variables: Sequence[Token]
    expr: Expr


@dataclass
class ReturnStatement:
    expr: Expr


@dataclass
class TensorType:
    dims: Sequence[Token]


@dataclass
class ReshapeTensorShape:
    """
    Valid reshape tensor shapes are:
    * S,(3,H,K)
    * 3,H,S,K
    * H,S,K
    * S,(H,K)
    """

    dims: Sequence[Union[Token, "ReshapeTensorShape"]]


@dataclass
class FunctionDeclaration:
    name: Token
    static_args: Sequence[Token]
    params: Sequence[Tuple[Token, TensorType]]
    args: Sequence[Tuple[Token, TensorType]]
    ret: TensorType
    body: Optional[Sequence[Statement]]


Value = Union[float, np.ndarray, "Func", Sequence["Value"]]


@dataclass
class Func:
    decl: Union[FunctionDeclaration, Callable[..., Callable[..., Value]]]


@dataclass
class TypeEnv:
    parent: Optional["TypeEnv"]
    static_vars: Dict[str, Value]
    vars: Dict[str, TensorType]
    funcs: Dict[str, FunctionDeclaration]

    def lookup_func(self, name: str) -> Optional[FunctionDeclaration]:
        ret = self.funcs.get(name)
        if ret is not None:
            return ret
        if self.parent != None:
            return self.parent.lookup_func(name)
        return None

    def lookup(self, name: str) -> Optional[Tuple[TensorType, Optional[Value]]]:
        ret = self.static_vars.get(name)
        if ret is not None:
            if isinstance(ret, np.ndarray):
                return (
                    TensorType([Token("NUMBER", str(x), 0, 0) for x in ret.shape]),
                    ret,
                )
            elif isinstance(ret, float):
                # TODO: This seems wrong - probably should remove scalars from Value namespace entirely
                return TensorType([]), ret
            else:
                raise Exception(f"Unexpected static var type {type(ret)}")
        ret = self.vars.get(name)
        if ret is not None:
            return ret, None
        if self.parent != None:
            return self.parent.lookup(name)
        return None

    def lookup_static(self, name: str) -> Optional[Value]:
        ret = self.static_vars.get(name)
        if ret != None:
            return ret
        if self.parent != None:
            return self.parent.lookup_static(name)
        return None


class Compiler:
    i = 0
    funcs: Dict[str, FunctionDeclaration] = {}

    def compile_function(
        self, func: FunctionDeclaration, static_args: Sequence[Value], env: TypeEnv
    ) -> FunctionDeclaration:
        print(f"compiling {func.name.text}")
        if len(static_args) != len(func.static_args):
            raise Exception(
                f"Wrong number of static args for {func.name.text}, expected {func.static_args} got {static_args}"
            )
        static_vars: Dict[str, Value] = {}
        for [var, val] in zip(func.static_args, static_args):
            static_vars[var.text] = val
        env = TypeEnv(env, static_vars, {}, {})
        ret_type = self.eval_type(func.ret, env)
        args = [(var, self.eval_type(typ, env)) for (var, typ) in func.args]
        for (var, typ) in args:
            env.vars[var.text] = typ
        params = [(var, self.eval_type(typ, env)) for (var, typ) in func.params]
        for (var, typ) in params:
            env.vars[var.text] = typ
        body = (
            [self.compile_statement(stmt, env, ret_type) for stmt in func.body]
            if func.body is not None
            else None
        )
        return FunctionDeclaration(
            name=func.name,
            static_args=func.static_args,
            params=params,
            args=args,
            ret=ret_type,
            body=body,
        )

    def eval_type(self, typ: TensorType, env: TypeEnv) -> TensorType:
        return TensorType([self.eval_dim(dim, env) for dim in typ.dims])

    def eval_dim(self, dim: Token, env: TypeEnv) -> Token:
        if dim.kind == "NUMBER":
            return dim
        elif dim.kind == "OP" and dim.text == "...":
            return dim
        elif dim.kind == "IDENT":
            val = env.lookup_static(dim.text)
            if val is None:
                raise Exception(f"Variable {dim.text} not found")
            if not isinstance(val, float) and not isinstance(val, int):
                raise Exception(
                    f"Variable {dim.text} with value '{val}' is not a number"
                )
            return Token("NUMBER", str(val), dim.pos, dim.indentation_level)
        else:
            raise NotImplementedError(
                f"eval_dim not implemented for {dim.kind}:{dim.text}"
            )

    def compile_statement(
        self, statement: Statement, env: TypeEnv, ret_type: TensorType
    ) -> Statement:
        if isinstance(statement, LetStatement):
            expr, expr_type = self.compile_expr(statement.expr, env)
            if len(statement.variables) == 1:
                env.vars[statement.variables[0].text] = expr_type
            else:
                if len(expr_type.dims) < 1 or int(float(expr_type.dims[0].text)) != len(
                    statement.variables
                ):
                    raise Exception("Wrong number of variables in let statement")
                for var in statement.variables:
                    # We are splitting on the first dimension, so drop it from the type
                    env.vars[var.text] = TensorType(expr_type.dims[1:])
            return LetStatement(
                variables=statement.variables,
                expr=expr,
            )
        elif isinstance(statement, ReturnStatement):
            expr, expr_type = self.compile_expr(statement.expr, env)
            self.check_assignable_from_to(expr_type, ret_type, f"return statement")
            return ReturnStatement(expr=expr)

        else:
            raise NotImplementedError(
                f"compile_statement not implemented for {type(statement)}"
            )

    def check_assignable_from_to(
        self, from_type: TensorType, to_type: TensorType, context: str
    ):
        """
        ... -> ... IS OK
        ...N -> ... IS OK
        N -> N IS OK
        N,M -> N,M US OK

        ... -> ...N IS NOT OK
        N -> M IS NOT OK
        N,M -> M,M IS NOT OK
        """
        if len(to_type.dims) > 0 and to_type.dims[0].text == "...":
            # assert that there are no more elipses
            if any([d == ... for d in to_type.dims[1:]]):
                raise Exception(f"Cannot have multiple elipses in a type: {to_type}")
            # figure out how many more non-elipses N there are
            N = len(to_type.dims) - 1
            # ensure that last N dimensions of from are the same as the last N dimensions of to
            from_dims = from_type.dims[len(from_type.dims) - N :]
            to_dims = to_type.dims[len(to_type.dims) - N :]
        else:
            from_dims = from_type.dims
            to_dims = to_type.dims
        if len(from_dims) != len(to_dims):
            raise Exception(
                f"Cannot assign from dimensions {from_dims} to dimensions {to_dims} in {context}"
            )
        for from_dim, to_dim in zip(from_dims, to_dims):
            if from_dim.kind != to_dim.kind and from_dim.text == to_dim.text:
                raise Exception(
                    f"Cannot assign from dimensions {from_dims} to dimensions {to_dims} in {context}"
                )
        return

    def check_broadcastable(
        self, from_type: TensorType, to_type: TensorType
    ) -> TensorType:
        num_dims = max(len(from_type.dims), len(to_type.dims))
        ret_dims: list[Token] = []
        from_i = len(from_type.dims) - num_dims - 1
        to_i = len(to_type.dims) - num_dims - 1
        for i in range(num_dims):
            from_i = from_i + 1
            to_i = to_i + 1
            from_at_i = from_type.dims[from_i].text if from_i >= 0 else None
            to_at_i = to_type.dims[to_i].text if to_i >= 0 else None
            if from_at_i == "..." or to_at_i == "...":
                if from_at_i == "..." and to_at_i == "...":
                    continue
                if from_at_i == "..." and to_at_i is None:
                    ret_dims.append(Token("OP", "...", 0, 0))
                    continue
                if to_at_i == "..." and from_at_i is None:
                    ret_dims.append(Token("OP", "...", 0, 0))
                    continue
                raise Exception(f"invalid broadcast between {from_type} and {to_type}")
            from_at_i = int(float(from_at_i)) if from_at_i is not None else 1
            to_at_i = int(float(to_at_i)) if to_at_i is not None else 1
            if not (from_at_i == to_at_i or from_at_i == 1 or to_at_i == 1):
                raise Exception(f"invalid broadcast between {from_type} and {to_type}")
            ret_dims.append(Token("NUMBER", str(max(from_at_i, to_at_i)), 0, 0))
        return TensorType(ret_dims)

    def compile_expr(self, expr: Expr, env: TypeEnv) -> Tuple[Expr, TensorType]:
        if isinstance(expr, FloatExpr):
            return expr, TensorType([])
        elif isinstance(expr, VariableExpr):
            t = env.lookup(expr.name.text)
            if t is None:
                raise Exception(f"Variable {expr.name.text} not found")
            t, v = t
            if v is None:
                return expr, t
            else:
                return self.constant_value(v), t
        elif isinstance(expr, BinaryExpr):
            left, left_type = self.compile_expr(expr.left, env)
            right, right_type = self.compile_expr(expr.right, env)
            ret_type: TensorType
            if expr.op.text in ("+", "-", "*", "/", "**"):
                ret_type = self.check_broadcastable(left_type, right_type)
            elif expr.op.text in ("@"):
                if len(right_type.dims) == 0 or len(left_type.dims) == 0:
                    raise Exception(f"Cannot @ a scalar")
                if len(right_type.dims) == 1:
                    if right_type.dims[0].text == "...":
                        raise Exception(f"Cannot @ a ...")
                    right_type = TensorType(
                        [
                            *right_type.dims,
                            Token("NUMBER", "1", 0, 0),
                        ]
                    )
                elif len(left_type.dims) == 1:
                    if left_type.dims[0].text == "...":
                        raise Exception(f"Cannot @ a ...")
                    left_type = TensorType(
                        [Token("NUMBER", "1", 0, 0), *left_type.dims]
                    )
                # They now both have at least 2 non-... dimensions at the end
                stack_ret_type = self.check_broadcastable(
                    TensorType(left_type.dims[:-2]), TensorType(right_type.dims[:-2])
                )
                ret_type = TensorType(
                    [*stack_ret_type.dims, left_type.dims[-2], right_type.dims[-1]]
                )
            else:
                raise Exception(f"Unknown binary op {expr.op.text}")
            return BinaryExpr(expr.op, left, right), ret_type
        elif isinstance(expr, CallExpr):
            if not isinstance(expr.f, VariableExpr):
                raise Exception("Function call must be a variable")
            func = env.lookup_func(expr.f.name.text)
            if func is None:
                raise Exception(f"Could not find function {expr.f.name.text}")
            # TODO: Is it okay to ignore the transitively compiled functions?
            compiled_static_args = [
                self.eval_static_expr(e, env) for e in expr.static_args
            ]
            compiled_func = self.compile_function(func, compiled_static_args, env)
            self.i = self.i + 1
            func_name = f"{compiled_func.name.text}_{self.i}"
            env.funcs[func_name] = compiled_func
            self.funcs[func_name] = compiled_func
            compiled_args = [self.compile_expr(arg, env) for arg in expr.args]
            if len(compiled_args) != len(compiled_func.args):
                raise Exception(
                    f"Cannot call function {compiled_func.name.text} with {len(compiled_args)} args, expected {len(compiled_func.args)}"
                )
            for ((x, param_type), (y, arg_type)) in zip(
                compiled_func.args, compiled_args
            ):
                self.check_assignable_from_to(
                    arg_type,
                    param_type,
                    f"passing arg {y} to {x.text} of {compiled_func.name.text}",
                )
            ret_type = self.applied_return_type(
                compiled_func, [t for (_, t) in compiled_args]
            )
            return (
                CallExpr(
                    VariableExpr(Token("IDENT", func_name, 0, 0)),
                    [FloatExpr(a) for a in compiled_static_args],
                    expr.param_args,  # TODO: Compiled?
                    [e for (e, _) in compiled_args],
                ),
                ret_type,
            )
        elif isinstance(expr, ReshapeExpr):
            inner, inner_type = self.compile_expr(expr.expr, env)
            ret_type, reshape_from, reshape_to, contraints = self.reshape_type(
                expr.reshape_from, expr.reshape_to, inner_type, env
            )
            # TODO: Need to compile reshape_from and reshape_to?
            return (
                ReshapeExpr(inner, expr.reshape_from, expr.reshape_to, contraints),
                ret_type,
            )
        elif isinstance(expr, IndexExpr):
            e, e_type = self.compile_expr(expr.expr, env)
            idx, idx_type = self.compile_expr(expr.index, env)
            ret_type = TensorType([idx_type.dims[0], *e_type.dims[1:]])
            print(f"compiled IndexExpr with {e_type} and {idx_type} => {ret_type}")
            return IndexExpr(e, idx), ret_type
        elif isinstance(expr, ForExpr):
            start = self.eval_static_expr(expr.start, env)
            end = self.eval_static_expr(expr.end, env)
            init, init_type = self.compile_expr(expr.init, env)
            loop_env = TypeEnv(env, {}, {expr.var.text: init_type}, {})
            loop, loop_type = self.compile_expr(expr.loop, loop_env)
            self.check_assignable_from_to(init_type, loop_type, "")
            self.check_assignable_from_to(loop_type, init_type, "")
            if not isinstance(start, float) and not isinstance(start, int):
                raise Exception(f"Cannot compile for init {init} = {start}")
            if not isinstance(end, float) and not isinstance(end, int):
                raise Exception(f"Cannot compile for init {init} = {end}")
            ret = init
            for i in range(int(start), int(end)):
                ret = LetExpr(
                    [(expr.index, FloatExpr(i)), (expr.var, ret)],
                    loop,
                )
            return ret, init_type
        else:
            raise NotImplementedError(f"compile_expr not implemented for {type(expr)}")

    def constant_value(self, v: Value) -> Expr:
        if isinstance(v, float):
            return FloatExpr(v)
        else:
            raise NotImplementedError(f"constant_value not implemented for {type(v)}")

    def reshape_type(
        self,
        frm: ReshapeTensorShape,
        to: ReshapeTensorShape,
        t: TensorType,
        env: TypeEnv,
    ) -> Tuple[TensorType, ReshapeTensorShape, ReshapeTensorShape, Mapping[str, int]]:
        if len(frm.dims) != len(t.dims):
            raise Exception(f"Cannot reshape {t} from {frm}: dimensions don't match")
        constraints: Dict[str, int] = {}
        compiled_from_dims: Sequence[Union[Token, ReshapeTensorShape]] = []
        for (from_at_i, t_at_i) in zip(frm.dims, t.dims):
            if isinstance(from_at_i, Token):
                dim = self.eval_dim(from_at_i, env)
                if dim.text != t_at_i.text:
                    raise Exception(f"Cannot reshape {t} from {frm}: {dim} != {t_at_i}")
                compiled_from_dims.append(dim)
                if from_at_i.kind == "IDENT":
                    constraints[from_at_i.text] = int(float(dim.text))
            elif isinstance(from_at_i, ReshapeTensorShape):
                d = int(float(t_at_i.text))
                compiled_from_inner_dims: Sequence[
                    Union[Token, ReshapeTensorShape]
                ] = []
                for from_d in from_at_i.dims:
                    if isinstance(from_d, Token):
                        dim = self.eval_dim(from_d, env)
                        d = d / int(float(dim.text))
                        compiled_from_inner_dims.append(dim)
                        if from_d.kind == "IDENT":
                            constraints[from_d.text] = int(float(dim.text))
                    else:
                        raise NotImplementedError(
                            f"reshape_type not implemented: {frm}, {to}, {t}"
                        )
                if d != 1:
                    raise Exception(
                        f"Cannot reshape {t} from {frm}: dimensions don't divide evenly"
                    )
                compiled_from_dims.append(ReshapeTensorShape(compiled_from_inner_dims))
            else:
                raise NotImplementedError(
                    f"reshape_type not implemented: {frm}, {to}, {t}"
                )
        # TODO: Verify that the to shape is valid for the from shape
        ret_type_dims = []
        compiled_to_dims: Sequence[Union[Token, ReshapeTensorShape]] = []
        for to_d in to.dims:
            if isinstance(to_d, Token):
                dim = self.eval_dim(to_d, env)
                ret_type_dims.append(dim)
                compiled_to_dims.append(dim)
            elif isinstance(to_d, ReshapeTensorShape):
                d = 1
                compiled_to_inner_dims: Sequence[Union[Token, ReshapeTensorShape]] = []
                for to_d_inner in to_d.dims:
                    if isinstance(to_d_inner, Token):
                        dim = self.eval_dim(to_d_inner, env)
                        d = d * int(float(dim.text))
                        compiled_to_inner_dims.append(dim)
                    else:
                        raise NotImplementedError(
                            f"reshape_type not implemented: {frm}, {to}, {t}"
                        )
                ret_type_dims.append(Token("NUMBER", str(d), 0, 0))
                compiled_to_dims.append(ReshapeTensorShape(compiled_to_inner_dims))
            else:
                raise NotImplementedError(
                    f"reshape_type not implemented: {frm}, {to}, {t}"
                )
        return (
            TensorType(ret_type_dims),
            ReshapeTensorShape(compiled_from_dims),
            ReshapeTensorShape(compiled_to_dims),
            constraints,
        )

    def eval_static_expr(self, expr: Expr, env: TypeEnv) -> Value:
        if isinstance(expr, FloatExpr):
            return expr.value
        elif isinstance(expr, VariableExpr):
            v = env.lookup_static(expr.name.text)
            if v is None:
                raise Exception(f"Could not find {expr.name.text} in scope")
            return v
        elif isinstance(expr, BinaryExpr):
            left = self.eval_static_expr(expr.left, env)
            right = self.eval_static_expr(expr.right, env)
            if not isinstance(left, float) and not isinstance(left, np.ndarray):
                raise RuntimeError(f"Cannot '{expr.op.text}' non-tensor {left}.")
            if not isinstance(right, float) and not isinstance(right, np.ndarray):
                raise RuntimeError(f"Cannot '{expr.op.text}' non-tensor {right}.")
            if expr.op.text == "+":
                return left + right
            elif expr.op.text == "-":
                return left - right
            elif expr.op.text == "*":
                return left * right
            elif expr.op.text == "/":
                return left / right
            elif expr.op.text == "**":
                return left**right
            else:
                raise NotImplementedError(f"eval_static_expr: {expr} {env}")
        raise NotImplementedError(f"eval_static_expr: {expr} {env}")

    def applied_return_type(
        self, f: FunctionDeclaration, arg_types: list[TensorType]
    ) -> TensorType:
        if len(f.args) != len(arg_types):
            raise Exception(f"Cannot apply {f.name.text} to {arg_types}")
        dotdotdot_type = None
        for ((_, param_type), arg_type) in zip(f.args, arg_types):
            # substitute into ...
            if len(param_type.dims) > 0 and param_type.dims[0].text == "...":
                param_ending_dims = param_type.dims[1:]
                num_arg_ending_dims = len(arg_type.dims) - len(param_ending_dims)
                if num_arg_ending_dims < 0:
                    raise Exception(
                        f"Cannot apply {f.name.text} to {arg_types} - {arg_type} not compatible with {param_type}"
                    )
                arg_starting_dims = arg_type.dims[0:num_arg_ending_dims]
                if dotdotdot_type is None:
                    dotdotdot_type = TensorType(arg_starting_dims)
                if dotdotdot_type.dims != arg_starting_dims:
                    raise Exception(
                        f"Cannot apply {f.name.text} to {arg_types} - {arg_type} not compatible with {param_type}"
                    )
        if (
            dotdotdot_type is not None
            and len(f.ret.dims) > 0
            and f.ret.dims[0].text == "..."
        ):
            ret_dims = [*dotdotdot_type.dims, *f.ret.dims[1:]]
            return TensorType(ret_dims)
        return f.ret


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
                for (var, val) in zip(stmt.variables, result):
                    env.vars[var.text] = val
            elif len(stmt.variables) == 1:
                env.vars[stmt.variables[0].text] = result
            else:
                if not isinstance(result, np.ndarray):
                    raise RuntimeError(
                        f"Cannot assign non-tensor {result} to multi-variable binding {stmt.variables}"
                    )
                items = np.split(result, len(stmt.variables))
                for (var, val) in zip(stmt.variables, items):
                    env.vars[var.text] = val[0]
            return None
        if isinstance(stmt, ReturnStatement):
            return self.eval_expr(stmt.expr, env)
        raise NotImplementedError("Unknown statement type.")

    def eval_expr(self, expr: Expr, env: Env) -> Value:
        print(f"eval_expr({expr})")
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
                        # v = env.lookup(dim.text)
                        # if v is None:
                        #     raise RuntimeError(f"could not find {dim.text} in scope")
                        # if not isinstance(v, float):
                        #     raise RuntimeError(
                        #         f"variable {dim.text} of non-number type {type(v)} used in dimension"
                        #     )
                        # vals[dim.text] = v
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
                                # v = env.lookup(d.text)
                                # if v is None:
                                #     raise RuntimeError(
                                #         f"could not find {d.text} in scope"
                                #     )
                                # if not isinstance(v, float):
                                #     raise RuntimeError(
                                #         f"variable {d.text} of non-number type {type(v)} used in dimension"
                                #     )
                                # vals[d.text] = v
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
            print(f"rearrange -> {rearrange_str}")
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


class Parser:
    s: str
    pos: int
    peeked_token: Optional[Token] = None
    indentation_level: int = 0

    def __init__(self, s: str):
        self.s = s
        self.pos = 0

    def get_char(self) -> Optional[str]:
        if self.pos >= len(self.s):
            return None
        ch = self.s[self.pos]
        self.pos += 1
        return ch

    def peek_char(self, i=0) -> str:
        ch = self.s[self.pos + i]
        return ch

    ## LEXER

    def read_token(self) -> Token:
        if self.peeked_token != None:
            tok = self.peeked_token
            self.peeked_token = None
            return tok
        ch = self.get_char()
        while ch == " " or ch == "\t" or ch == "\n" or ch == "\r":
            if ch == "\n":
                self.indentation_level = 0
            elif ch == "\t":
                self.indentation_level += 4
            elif ch == " ":
                self.indentation_level += 1
            ch = self.get_char()
        p = self.pos
        i = self.indentation_level
        if ch == None:
            return Token("EOF", "", p, i)
        elif (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z"):
            s = ""
            while ch != None and (
                (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z") or (ch >= "0" and ch <= "9") or ch == "_"
            ):
                s += ch
                ch = self.get_char()
            self.pos -= 1
            return Token("IDENT", s, p, i)
        elif (ch >= "0" and ch <= "9"):
            s = ""
            seendot = False
            while ch != None and ((ch >= "0" and ch <= "9") or (ch == "." and not seendot)):
                if ch == ".":
                    seendot = True
                s += ch
                ch = self.get_char()
            if s[-1] == ".": # We can't allow a trailing .
                self.pos -= 1
                s = s[:-1]
            if ch == "e" or ch == "E":
                s += ch
                ch = self.get_char()
                if ch == "+" or ch == "-":
                    s += ch
                    ch = self.get_char()
                while ch != None and (ch >= "0" and ch <= "9"):
                    s += ch
                    ch = self.get_char()
            self.pos -= 1
            return Token("NUMBER", s, p, i)
        elif (
            ch == "("
            or ch == ")"
            or ch == "{"
            or ch == "}"
            or ch == "["
            or ch == "]"
            or ch == "|"
            or ch == ","
            or ch == "/"
            or ch == "?"
            or ch == "*"
            or ch == "+"
            or ch == "="
            or ch == ":"
            or ch == "-"
            or ch == ">"
            or ch == "<"
            or ch == "@"
            or ch == "."
        ):
            if ch == "-" and self.peek_char() == ">":
                self.get_char()
                return Token("OP", "->", p, i)
            if ch == "=" and self.peek_char() == ">":
                self.get_char()
                return Token("OP", "=>", p, i)
            if ch == "<" and self.peek_char() == "=":
                self.get_char()
                return Token("OP", "<=", p, i)
            if ch == ">" and self.peek_char() == "=":
                self.get_char()
                return Token("OP", ">=", p, i)
            if ch == "?" and self.peek_char() == "?":
                self.get_char()
                return Token("OP", "??", p, i)
            if ch == "*" and self.peek_char() == "*":
                self.get_char()
                return Token("OP", "**", p, i)
            if ch == "." and self.peek_char() == ".":
                self.get_char()
                if self.peek_char() == ".":
                    self.get_char()
                    return Token("OP", "...", p, i)
                raise NotImplementedError("Only ... is supported")
            return Token("OP", ch, p, i)
        else:
            # TODO:
            # * string
            # * comment
            raise Exception("Unexpected character: " + ch)

    def peek_token(self) -> Token:
        if self.peeked_token == None:
            self.peeked_token = self.read_token()
        return self.peeked_token

    def assert_ident(self, token: Token, id: Optional[str] = None):
        expected = "IDENT" if id is None else id
        if token.kind != "IDENT" or (id is not None and token.text != id):
            raise Exception(f"Expected {expected} found: {token}")

    def assert_op(self, token: Token, op: Optional[str] = None):
        expected = "OP" if op is None else op
        if token.kind != "OP" or (op is not None and token.text != op):
            raise Exception(f"Expected {expected} found: {token}")

    def assert_number(self, token: Token):
        if token.kind != "NUMBER":
            raise Exception("Expected NUMBER found: ", token)

    def read_ident(self, id: Optional[str] = None) -> Token:
        tok = self.read_token()
        self.assert_ident(tok, id)
        return tok

    def read_op(self, op: Optional[str] = None) -> Token:
        tok = self.read_token()
        self.assert_op(tok, op)
        return tok
    
    def read_number(self) -> Token:
        tok = self.read_token()
        self.assert_number(tok)
        return tok

    def peek_token_is_op(self, op: str) -> bool:
        tok = self.peek_token()
        return tok.text == op

    def peek_token_is_ident(self, id: Optional[str] = None) -> bool:
        tok = self.peek_token()
        return tok.kind == "IDENT" and (id is None or tok.text == id)

    ## PARSER

    def parse_program(self) -> Sequence[FunctionDeclaration]:
        functions = []
        while self.peek_token_is_ident():
            functions.append(self.parse_function())
        return functions

    def parse_function(self) -> FunctionDeclaration:
        name = self.read_ident()
        static_args: list[Token] = []
        if self.peek_token_is_op("["):
            self.read_op("[")
            if self.peek_token_is_ident():
                static_args = self.parse_ident_list()
            self.read_op("]")
        params: list[Tuple[Token, TensorType]] = []
        if self.peek_token_is_op("|"):
            self.read_op("|")
            if self.peek_token_is_ident():
                params = self.parse_param_list(True)
            self.read_op("|")
        args: list[Tuple[Token, TensorType]] = []
        self.read_op("(")
        if self.peek_token_is_ident():
            args = self.parse_param_list()
        self.read_op(")")
        self.read_op("->")
        ret = self.parse_tensor_type()
        body: Optional[Sequence[Statement]] = None
        if self.peek_token_is_op(":"):
            self.read_op(":")
            statements = []
            while self.peek_token().indentation_level > name.indentation_level:
                statements.append(self.parse_statement())
            body = statements
        return FunctionDeclaration(
            name, static_args, params, args, ret, body
        )

    def parse_ident_list(self) -> list[Token]:
        ret = [self.read_ident()]
        while self.peek_token_is_op(","):
            self.read_op(",")
            ret.append(self.read_ident())
        return ret

    def parse_param_list(self, allow_no_type = False) -> list[Tuple[Token, TensorType]]:
        ret = [self.parse_param(allow_no_type)]
        while self.peek_token_is_op(","):
            self.read_op(",")
            ret.append(self.parse_param(allow_no_type))
        return ret

    def parse_param(self, allow_no_type = False) -> Tuple[Token, TensorType]:
        tok = self.read_ident()
        if allow_no_type and not self.peek_token_is_op(":"):
            return (tok, TensorType([]))
        self.read_op(":")
        ty = self.parse_tensor_type()
        return (tok, ty)

    def parse_statement(self) -> Statement:
        if self.peek_token_is_ident("return"):
            return self.parse_return_statement()
        return self.parse_let_statement()

    def parse_return_statement(self) -> ReturnStatement:
        self.read_ident()
        expr = self.parse_expression()
        return ReturnStatement(expr)
    
    def parse_let_statement(self) -> LetStatement:
        idents = self.parse_ident_list()
        self.read_op("=")
        expr = self.parse_expression()
        return LetStatement(idents, expr)

    def parse_expression(self) -> Expr:
        return self.parse_maybe_sum()
    
    def parse_maybe_sum(self) -> Expr:
        lhs = self.parse_maybe_product()
        while self.peek_token_is_op("+") or self.peek_token_is_op("-"):
            op = self.read_op()
            rhs = self.parse_maybe_product()
            lhs = BinaryExpr(op, lhs,  rhs)
        return lhs
    
    def parse_maybe_product(self) -> Expr:
        lhs = self.parse_maybe_power()
        while self.peek_token_is_op("*") or self.peek_token_is_op("/"):
            op = self.read_op()
            rhs = self.parse_maybe_power()
            lhs = BinaryExpr(op, lhs,  rhs)
        return lhs
    
    def parse_maybe_power(self) -> Expr:
        lhs = self.parse_maybe_matmul()
        while self.peek_token_is_op("**"):
            op = self.read_op("**")
            rhs = self.parse_maybe_matmul()
            lhs = BinaryExpr(op, lhs,  rhs)
        return lhs
    
    def parse_maybe_matmul(self) -> Expr:
        lhs = self.parse_maybe_reshape()
        while self.peek_token_is_op("@"):
            op = self.read_op("@")
            rhs = self.parse_maybe_reshape()
            lhs = BinaryExpr(op, lhs,  rhs)
        return lhs
    
    def parse_maybe_reshape(self) -> Expr:
        lhs = self.parse_primitive_expr()
        while self.peek_token_is_op("{"):
            self.read_op("{")
            from_shape = self.parse_reshape_type()
            self.read_op("->")
            to_shape = self.parse_reshape_type()
            self.read_op("}")
            lhs = ReshapeExpr(lhs, from_shape, to_shape, {})
        return lhs
    
    def parse_primitive_expr(self) -> Expr:
        tok = self.peek_token()
        if tok.kind == "OP" and tok.text == "(":
            return self.parse_paren_expr()
        elif tok.kind == "IDENT" and tok.text == "for":
            return self.parse_for_expr()
        elif tok.kind == "NUMBER":
            return FloatExpr(float(self.read_number().text))
        elif tok.kind == "IDENT":
            # CallExpr / IndexExpr / Ident
            ident = self.read_ident()
            tok = self.peek_token()
            if tok.kind == "OP" and (tok.text == "["  or tok.text == "("): # TODO: What about | ?
                return self.parse_call_expr(ident)
            elif tok.kind == "OP" and tok.text == ".":
                return self.parse_index_expr(ident)
            else:
                return VariableExpr(ident)
        raise Exception("Expected expression, found " + tok.text)
    
    def parse_paren_expr(self) -> Expr:
        self.read_op("(")
        expr = self.parse_expression()
        self.read_op(")")
        return expr

    def parse_call_expr(self, ident: Token) -> Expr:
        static_args: list[Expr] = []
        if self.peek_token_is_op("["):
            self.read_op("[")
            if not self.peek_token_is_op("]"):
                static_args = self.parse_arg_list()
            self.read_op("]")
        params: list[Expr] = []
        if self.peek_token_is_op("|"):
            self.read_op("|")
            if not self.peek_token_is_op("|"):
                params = self.parse_arg_list()
            self.read_op("|")
        self.read_op("(")
        args: list[Expr] = []
        if not self.peek_token_is_op(")"):
            args = self.parse_arg_list()
        self.read_op(")")
        return CallExpr(VariableExpr(ident), static_args, params, args)

    def parse_arg_list(self) -> list[Expr]:
        ret = [self.parse_expression()]
        while self.peek_token_is_op(","):
            self.read_op(",")
            ret.append(self.parse_expression())
        return ret

    def parse_index_expr(self, ident: Token) -> Expr:
        self.read_op(".")
        self.read_op("[")
        index = self.parse_expression()
        self.read_op("]")
        return IndexExpr(VariableExpr(ident), index)

    def parse_for_expr(self) -> Expr:
        self.read_ident("for")
        index = self.read_ident()
        self.read_ident("in")
        start = self.parse_expression()
        self.read_op("...")
        end = self.parse_expression()
        self.read_op(":")
        init = self.parse_expression()
        self.read_op(",")
        var = self.read_token()
        self.read_op("->")
        loop = self.parse_expression()
        return ForExpr(index, start, end, init, var, loop)

    def parse_reshape_type(self) -> ReshapeTensorShape:
        tok = self.peek_token()
        dims: list[Union[Token, "ReshapeTensorShape"]] = []
        if tok.kind == "IDENT" or tok.kind == "NUMBER" or (tok.kind == "OP" and tok.text == "("):
            dims.append(self.parse_reshape_dimension())
            while self.peek_token_is_op(","):
                self.read_op(",")
                dims.append(self.parse_reshape_dimension())
        return ReshapeTensorShape(dims)
    
    def parse_reshape_dimension(self) -> Union[Token, ReshapeTensorShape]:
        tok = self.read_token()
        if tok.kind == "IDENT" or tok.kind == "NUMBER":
            return tok
        elif tok.kind == "OP" and tok.text == "(":
            sub = self.parse_reshape_type()
            self.read_op(")")
            return sub
        raise Exception("Expected tensor type dimension, got " + tok.text)

    def parse_tensor_type(self) -> TensorType:
        self.read_op("{")
        dims: list[Token] = []
        if not self.peek_token_is_op("}"):
            dims.append(self.parse_dimension())
            while self.peek_token_is_op(","):
                self.read_op(",")
                dims.append(self.parse_dimension())
        self.read_op("}")
        return TensorType(dims)
    
    def parse_dimension(self) -> Token:
        tok = self.read_token()
        if not (tok.kind == "IDENT" or tok.kind == "NUMBER" or (tok.kind == "OP" and tok.text == "...")): 
            raise Exception("Expected tensor type dimension, got " + tok.text)
        return tok
