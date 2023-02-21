import torch
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort
from dataclasses import dataclass
from typing import Sequence, Optional, Union, Tuple, Dict, Callable
import numpy as np


@dataclass
class Token:
    kind: str
    text: str
    pos: int
    indentation_level: int


Expr = Union[
    "FloatExpr", "VariableExpr", "CallExpr", "ReshapeExpr", "BinaryExpr", "MatMulExpr"
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
    args: Sequence[Expr]


@dataclass
class ReshapeExpr:
    expr: Expr
    reshape_from: "TensorType"
    reshape_to: "TensorType"


@dataclass
class BinaryExpr:
    op: Token
    left: Expr
    right: Expr


@dataclass
class MatMulExpr:
    pass


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
class FunctionDeclaration:
    name: Token
    static_args: Sequence[Token]
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

    def lookup(self, name: str) -> Optional[TensorType]:
        ret = self.static_vars.get(name)
        if ret is not None:
            raise NotImplementedError("static vars not implemented in typeenv lookup")
        ret = self.vars.get(name)
        if ret is not None:
            return ret
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

    def compile_function(
        self, func: FunctionDeclaration, static_args: Sequence[Value], env: TypeEnv
    ) -> Tuple[FunctionDeclaration, Dict[str, FunctionDeclaration]]:
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
        body = (
            [self.compile_statement(stmt, env, ret_type) for stmt in func.body]
            if func.body is not None
            else None
        )
        return (
            FunctionDeclaration(
                name=func.name,
                static_args=[],
                args=args,
                ret=ret_type,
                body=body,
            ),
            env.funcs,
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
                if len(expr_type.dims) < 1 or expr_type.dims[0] != len(
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
            self.check_assignable_from_to(expr_type, ret_type)
            return ReturnStatement(expr=expr)

        else:
            raise NotImplementedError(
                f"compile_statement not implemented for {type(statement)}"
            )

    def check_assignable_from_to(self, from_type: TensorType, to_type: TensorType):
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
                f"Cannot assign from dimensions {from_dims} to dimensions {to_dims}"
            )
        for from_dim, to_dim in zip(from_dims, to_dims):
            if from_dim.kind != to_dim.kind and from_dim.text == to_dim.text:
                raise Exception(
                    f"Cannot assign from dimensions {from_dims} to dimensions {to_dims}"
                )
        return

    def compile_expr(self, expr: Expr, env: TypeEnv) -> Tuple[Expr, TensorType]:
        if isinstance(expr, FloatExpr):
            return expr, TensorType([])
        elif isinstance(expr, VariableExpr):
            t = env.lookup(expr.name.text)
            if t is None:
                raise Exception(f"Variable {expr.name.text} not found")
            return expr, t
        elif isinstance(expr, BinaryExpr):
            left, left_type = self.compile_expr(expr.left, env)
            right, right_type = self.compile_expr(expr.right, env)
            # TODO: Need to actually compute the correct return type from left_type and right_type
            return BinaryExpr(expr.op, left, right), left_type
        elif isinstance(expr, CallExpr):
            if not isinstance(expr.f, VariableExpr):
                raise Exception("Function call must be a variable")
            func = env.lookup_func(expr.f.name.text)
            if func is None:
                raise Exception(f"Could not find function {expr.f.name.text}")
            # TODO: Is it okay to ignore the transitively compiled functions?
            compiled_func, _ = self.compile_function(
                func, [self.eval_static_expr(e, env) for e in expr.static_args], env
            )
            self.i = self.i + 1
            func_name = f"{compiled_func.name.text}_{self.i}"
            env.funcs[func_name] = compiled_func
            compiled_args = [self.compile_expr(arg, env) for arg in expr.args]
            if len(compiled_args) != len(compiled_func.args):
                raise Exception(
                    f"Cannot call function {compiled_func.name.text} with {len(compiled_args)} args, expected {len(compiled_func.args)}"
                )
            for ((x, param_type), (y, arg_type)) in zip(
                compiled_func.args, compiled_args
            ):
                print(
                    f"Calling {compiled_func.name.text} -- {x.text}: {param_type} with {y}: {arg_type}"
                )
                self.check_assignable_from_to(arg_type, param_type)
            return (
                CallExpr(
                    VariableExpr(Token("IDENT", func_name, 0, 0)),
                    [],
                    [e for (e, _) in compiled_args],
                ),
                compiled_func.ret,
            )
        else:
            raise NotImplementedError(f"compile_expr not implemented for {type(expr)}")

    def eval_static_expr(self, expr: Expr, env: TypeEnv) -> Value:
        if isinstance(expr, VariableExpr):
            v = env.lookup_static(expr.name.text)
            if v is None:
                raise Exception(f"Could not find {expr.name.text} in scope")
            return v
        else:
            raise NotImplementedError(f"eval_static_expr: {expr} {env}")


@dataclass
class Env:
    parent: Optional["Env"]
    static_vars: Dict[str, Value]
    vars: Dict[str, Value]
    funcs: Dict[str, Callable[..., Value]]

    def lookup(self, name: str) -> Optional[Value]:
        ret = self.static_vars.get(name)
        if ret is not None:
            return ret
        ret = self.vars.get(name)
        if ret is not None:
            return ret
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
        static_args: list[Value],
        args: list[Value],
        env: Env,
    ) -> Value:
        if program.body is None:
            # built-in function
            name = program.name.text.split("_")[0]
            built_in = env.lookup_builtin(name)
            if built_in is None:
                raise Exception(f"Could not find built-in function {name}")
            return built_in(args)
        static_vars: Dict[str, Value] = {}
        for [var, val] in zip(program.static_args, static_args):
            static_vars[var.text] = val
        vars: Dict[str, Value] = {}
        for [(var, typ), val] in zip(program.args, args):
            # TODO: type check
            vars[var.text] = val
        env = Env(env, static_vars, vars, {})
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
                raise RuntimeError("Too many variables for single value.")
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
            if isinstance(f.decl, Callable):
                a = f.decl(*static_args)
                b = a(*args)
                # print(f"call {expr.f}[{static_args}]({args}) => {b}")
                return b
            else:
                ret = self.eval_call_expr(f.decl, static_args, args, env)
                # print(f"call {f.decl.name.text}[{static_args}]({args}) => {ret}")
                return ret
        elif isinstance(expr, ReshapeExpr):
            raise NotImplementedError("ReshapeExpr not implemented yet.")
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
            else:
                raise RuntimeError(f"Unknown binary operator: {expr.op.text}")
        elif isinstance(expr, MatMulExpr):
            raise NotImplementedError("MatMulExpr not implemented yet.")
        elif isinstance(expr, VariableExpr):
            val = env.lookup(expr.name.text)
            if val is None:
                raise RuntimeError(f"Variable {expr.name.text} not found.")
            return val
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
                (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z")
            ):
                s += ch
                ch = self.get_char()
            self.pos -= 1
            return Token("IDENT", s, p, i)
        elif (ch >= "0" and ch <= "9") or ch == ".":
            s = ""
            while ch != None and ((ch >= "0" and ch <= "9") or ch == "."):
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
            return Token("OP", ch, p, i)
        else:
            # TODO:
            # * string
            # * number
            # * comment
            raise Exception("Unexpected character: " + ch)

    def peek_token(self) -> Optional[Token]:
        if self.peeked_token == None:
            self.peeked_token = self.read_token()
        return self.peeked_token

    def parse(self) -> Sequence[FunctionDeclaration]:
        tok = self.peek_token()
        if tok is None:
            raise Exception("Expected IDENT found: EOF")
        if tok.kind != "IDENT":
            raise Exception("Expected IDENT found: " + tok.kind)
        return self.parse_program()

    def parse_program(self) -> Sequence[FunctionDeclaration]:
        functions = []
        while True:
            tok = self.peek_token()
            if tok is None or tok.kind != "IDENT":
                break
            functions.append(self.parse_function())
        return functions

    def parse_function(self) -> FunctionDeclaration:
        name = self.read_token()
        tok = self.read_token()
        static_args = []
        while tok.text == "[" or tok.text == ",":
            tok = self.read_token()
            if tok == "]":
                break
            static_args.append(tok)
            tok = self.read_token()
        tok = self.read_token()
        args = []
        while tok.text == "(" or tok.text == ",":
            tok = self.read_token()
            if tok == ")":
                break
            arg_name = tok
            tok = self.read_token()
            if tok.text != ":":
                raise Exception("Expected : found: ", tok)
            arg_type = self.parse_tensor_type()
            tok = self.read_token()
            args.append((arg_name, arg_type))
        tok = self.read_token()
        if tok.text != "->":
            raise Exception("Expected -> found: ", tok)
        ret = self.parse_tensor_type()
        tok = self.read_token()
        if tok.text != ":":
            raise Exception("Expected : found: ", tok)

        statements = []
        next_tok = self.peek_token()
        print("next indent", next_tok, name)
        while (
            next_tok is not None and next_tok.indentation_level > name.indentation_level
        ):
            statements.append(self.parse_statement())
            next_tok = self.peek_token()
        return FunctionDeclaration(name, static_args, args, ret, statements)

    def parse_statement(self) -> Statement:
        next_tok = self.read_token()
        if next_tok.text == "return":
            raise Exception("NYI - return")
        elif next_tok.text == "let":
            lhs = []
            while True:
                tok = self.read_token()
                if tok.kind != "IDENT":
                    raise Exception("Expected IDENT found: ", next_tok)
                lhs.append(tok)
                tok = self.read_token()
                if tok.text == "=":
                    break
                elif tok.text == ",":
                    continue
                else:
                    raise Exception("Expected = or , found: ", next_tok)
            rhs = self.parse_expression()
            return LetStatement(lhs, rhs)
        else:
            raise Exception("NYI - STATEMENT")
        # elif next_tok.text == "[" or next_tok.text == "(":
        #     if len(lhs) != 1:
        #         raise Exception("Expected 1 expression in call found: ", len(lhs))
        #     f = lhs[0]
        #     tok = self.read_token()
        #     static_args = []
        #     while tok.text == "[" or tok.text == ",":
        #         tok = self.read_token()
        #         if tok == "]":
        #             break
        #         static_args.append(tok)
        #         tok = self.read_token()
        #     tok = self.read_token()
        #     args = []
        #     while tok.text == "(" or tok.text == ",":
        #         next_tok = self.peek_token()
        #         if next_tok == ")":
        #             self.read_token()
        #             break
        #         expr = self.parse_expression()
        #         args.append(expr)
        #     return CallExpr(f, static_args, args)

    def parse_expression(self) -> Expr:
        return self.parse_maybe_reshape_expression()

    def parse_maybe_reshape_expression(self) -> Expr:
        expr = self.parse_maybe_call_expression()
        tok = self.peek_token()
        if tok is not None and tok.text == "{":
            self.read_token()
            shape = self.parse_tensor_type()
            # TODO: from shape?
            return ReshapeExpr(expr, shape, shape)
        return expr

    def parse_maybe_call_expression(self) -> Expr:
        expr = self.parse_simple_expression()
        tok = self.peek_token()
        static_args = []
        if tok is not None and tok.text == "[":
            tok = self.read_token()
            while tok.text == "[" or tok.text == ",":
                static_arg = self.parse_expression()
                static_args.append(static_arg)
                tok = self.read_token()
            if tok.text != "]":
                raise Exception("Expected ] found: ", tok)
        args = []
        if tok is not None and tok.text == "(":
            tok = self.read_token()
            while tok.text == "[" or tok.text == ",":
                static_arg = self.parse_expression()
                static_args.append(static_arg)
                tok = self.read_token()
            if tok.text != "]":
                raise Exception("Expected ] found: ", tok)
            return CallExpr(expr, static_args, args)
        return expr

    def parse_simple_expression(self) -> Expr:
        tok = self.peek_token()
        if tok is None:
            raise Exception("Expected expression found: EOF")
        elif tok.kind == "IDENT":
            return self.parse_maybe_reshape_expression()
        elif tok.text == "[":
            raise Exception("NYI - mask index?")
        elif tok.text == "@":
            raise Exception("NYI - matmul")
        raise Exception("NYI - expression")

    def parse_tensor_type(self) -> TensorType:
        tok = self.read_token()
        if tok.text != "{":
            raise Exception("Expected { found: ", tok)
        dims = []
        while True:
            tok = self.read_token()
            if tok.text == "}":
                break
            if tok.kind != "IDENT":
                raise Exception("Expected IDENT found: ", tok)
            dims.append(tok)
        return TensorType(dims)
