import torch
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple


@dataclass
class Token:
    kind: str
    text: str
    pos: int


@dataclass
class CallExpr:
    pass


@dataclass
class ReshapeExpr:
    pass


@dataclass
class BinaryExpr:
    pass


@dataclass
class MatMulExpr:
    pass


Expr = Union[CallExpr, ReshapeExpr, BinaryExpr, MatMulExpr]


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


class Parser:
    s: str
    pos: int
    peeked_token: Optional[Token] = None

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
            ch = self.get_char()
        p = self.pos
        if ch == None:
            return Token("EOF", "", p)
        elif (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z"):
            s = ""
            while (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z"):
                s += ch
                ch = self.get_char()
            self.pos -= 1
            return Token("IDENT", s, p)
        elif (ch >= "0" and ch <= "9") or ch == ".":
            s = ""
            while (ch >= "0" and ch <= "9") or ch == ".":
                s += ch
                ch = self.get_char()
            self.pos -= 1
            return Token("NUMBER", s, p)
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
                return Token("OP", "->", p)
            if ch == "=" and self.peek_char() == ">":
                self.get_char()
                return Token("OP", "=>", p)
            if ch == "<" and self.peek_char() == "=":
                self.get_char()
                return Token("OP", "<=", p)
            if ch == ">" and self.peek_char() == "=":
                self.get_char()
                return Token("OP", ">=", p)
            if ch == "?" and self.peek_char() == "?":
                self.get_char()
                return Token("OP", "??", p)
            return Token("OP", ch, p)
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
        if tok.kind != "IDENT":
            raise Exception("Expected IDENT found: " + tok.kind)
        return self.parse_program()

    def parse_program(self) -> List[Function]:
        functions = []
        while True:
            tok = self.peek_token()
            if tok.kind != "IDENT":
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
        return Function(name, static_args, args, ret, [])

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
