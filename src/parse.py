import torch
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Mapping
import numpy as np


@dataclass
class Token:
    kind: str
    text: str
    pos: int
    indentation_level: int


Expr = Union["CallExpr", "ReshapeExpr", "BinaryExpr", "MatMulExpr"]


@dataclass
class VariableExpr:
    name: Token


@dataclass
class CallExpr:
    f: Expr
    static_args: List[Token]
    args: List[Expr]


@dataclass
class ReshapeExpr:
    expr: Expr
    reshape_from: "TensorType"
    reshape_to: "TensorType"


@dataclass
class BinaryExpr:
    pass


@dataclass
class MatMulExpr:
    pass


@dataclass
class Statement:
    variables: List[Token]
    expr: Expr


@dataclass
class TensorType:
    dims: List[Token]


@dataclass
class Function:
    name: Token
    static_args: List[Token]
    args: List[Tuple[Token, TensorType]]
    ret: TensorType
    body: List[Statement]


Value = Union[float, np.ndarray]


@dataclass
class Env:
    parent: Optional["Env"]
    static_vars: Mapping[str, Value]
    vars: Mapping[str, Value]

    def lookup(self, name: str) -> Optional[Value]:
        ret = self.static_vars.get(name)
        if ret != None:
            return ret
        ret = self.vars.get(name)
        if ret != None:
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


class Interpreter:
    def eval_call(self, program: Function, static_args: List[float], args: List[float]):
        static_vars: Mapping[str, Value] = {}
        for [var, val] in zip(program.static_args, static_args):
            static_vars[var.text] = val
        vars: Mapping[str, Value] = {}
        for [(var, typ), val] in zip(program.args, args):
            # TODO: check type
            vars[var.text] = val
        env = Env(None, static_vars, vars)
        pass


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

    def parse(self) -> List[Function]:
        tok = self.peek_token()
        if tok is None:
            raise Exception("Expected IDENT found: EOF")
        if tok.kind != "IDENT":
            raise Exception("Expected IDENT found: " + tok.kind)
        return self.parse_program()

    def parse_program(self) -> List[Function]:
        functions = []
        while True:
            tok = self.peek_token()
            if tok is None or tok.kind != "IDENT":
                break
            functions.append(self.parse_function())
        return functions

    def parse_function(self) -> Function:
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
        return Function(name, static_args, args, ret, statements)

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
            return Statement(lhs, rhs)
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
