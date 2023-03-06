# Copyright 2023 Luke Hoban

from typing import Sequence, Optional, Union, Tuple
from . import tenast


class Parser:
    s: str
    pos: int
    peeked_token: Optional[tenast.Token] = None
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

    def read_token(self) -> tenast.Token:
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
            return tenast.Token("EOF", "", p, i)
        elif (ch >= "a" and ch <= "z") or (ch >= "A" and ch <= "Z"):
            s = ""
            while ch != None and (
                (ch >= "a" and ch <= "z")
                or (ch >= "A" and ch <= "Z")
                or (ch >= "0" and ch <= "9")
                or ch == "_"
            ):
                s += ch
                ch = self.get_char()
            self.pos -= 1
            return tenast.Token("IDENT", s, p, i)
        elif ch >= "0" and ch <= "9":
            s = ""
            seendot = False
            while ch != None and (
                (ch >= "0" and ch <= "9") or (ch == "." and not seendot)
            ):
                if ch == ".":
                    seendot = True
                s += ch
                ch = self.get_char()
            if s[-1] == ".":  # We can't allow a trailing .
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
            return tenast.Token("NUMBER", s, p, i)
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
                return tenast.Token("OP", "->", p, i)
            if ch == "=" and self.peek_char() == ">":
                self.get_char()
                return tenast.Token("OP", "=>", p, i)
            if ch == "<" and self.peek_char() == "=":
                self.get_char()
                return tenast.Token("OP", "<=", p, i)
            if ch == ">" and self.peek_char() == "=":
                self.get_char()
                return tenast.Token("OP", ">=", p, i)
            if ch == "?" and self.peek_char() == "?":
                self.get_char()
                return tenast.Token("OP", "??", p, i)
            if ch == "*" and self.peek_char() == "*":
                self.get_char()
                return tenast.Token("OP", "**", p, i)
            if ch == "." and self.peek_char() == ".":
                self.get_char()
                if self.peek_char() == ".":
                    self.get_char()
                    return tenast.Token("OP", "...", p, i)
                raise NotImplementedError("Only ... is supported")
            return tenast.Token("OP", ch, p, i)
        else:
            # TODO:
            # * string
            # * comment
            raise Exception("Unexpected character: " + ch)

    def peek_token(self) -> tenast.Token:
        if self.peeked_token == None:
            self.peeked_token = self.read_token()
        return self.peeked_token

    def assert_ident(self, token: tenast.Token, id: Optional[str] = None):
        expected = "IDENT" if id is None else id
        if token.kind != "IDENT" or (id is not None and token.text != id):
            raise Exception(f"Expected {expected} found: {token}")

    def assert_op(self, token: tenast.Token, op: Optional[str] = None):
        expected = "OP" if op is None else op
        if token.kind != "OP" or (op is not None and token.text != op):
            raise Exception(f"Expected {expected} found: {token}")

    def assert_number(self, token: tenast.Token):
        if token.kind != "NUMBER":
            raise Exception("Expected NUMBER found: ", token)

    def read_ident(self, id: Optional[str] = None) -> tenast.Token:
        tok = self.read_token()
        self.assert_ident(tok, id)
        return tok

    def read_op(self, op: Optional[str] = None) -> tenast.Token:
        tok = self.read_token()
        self.assert_op(tok, op)
        return tok

    def read_number(self) -> tenast.Token:
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

    def parse_program(self) -> Sequence[tenast.FunctionDeclaration]:
        functions = []
        while self.peek_token_is_ident():
            functions.append(self.parse_function())
        return functions

    def parse_function(self) -> tenast.FunctionDeclaration:
        name = self.read_ident()
        static_args: list[tenast.Token] = []
        if self.peek_token_is_op("["):
            self.read_op("[")
            if self.peek_token_is_ident():
                static_args = self.parse_ident_list()
            self.read_op("]")
        params: list[Tuple[tenast.Token, tenast.TensorType]] = []
        if self.peek_token_is_op("|"):
            self.read_op("|")
            if self.peek_token_is_ident():
                params = self.parse_param_list(True)
            self.read_op("|")
        args: list[Tuple[tenast.Token, tenast.TensorType]] = []
        self.read_op("(")
        if self.peek_token_is_ident():
            args = self.parse_param_list()
        self.read_op(")")
        self.read_op("->")
        ret = self.parse_tensor_type()
        body: Optional[Sequence[tenast.Statement]] = None
        if self.peek_token_is_op(":"):
            self.read_op(":")
            statements = []
            while self.peek_token().indentation_level > name.indentation_level:
                statements.append(self.parse_statement())
            body = statements
        return tenast.FunctionDeclaration(name, static_args, params, args, ret, body)

    def parse_ident_list(self) -> list[tenast.Token]:
        ret = [self.read_ident()]
        while self.peek_token_is_op(","):
            self.read_op(",")
            ret.append(self.read_ident())
        return ret

    def parse_param_list(
        self, allow_no_type=False
    ) -> list[Tuple[tenast.Token, tenast.TensorType]]:
        ret = [self.parse_param(allow_no_type)]
        while self.peek_token_is_op(","):
            self.read_op(",")
            ret.append(self.parse_param(allow_no_type))
        return ret

    def parse_param(self, allow_no_type=False) -> Tuple[tenast.Token, tenast.TensorType]:
        tok = self.read_ident()
        if allow_no_type and not self.peek_token_is_op(":"):
            return (tok, tenast.TensorType([]))
        self.read_op(":")
        ty = self.parse_tensor_type()
        return (tok, ty)

    def parse_statement(self) -> tenast.Statement:
        if self.peek_token_is_ident("return"):
            return self.parse_return_statement()
        return self.parse_let_statement()

    def parse_return_statement(self) -> tenast.ReturnStatement:
        self.read_ident()
        expr = self.parse_expression()
        return tenast.ReturnStatement(expr)

    def parse_let_statement(self) -> tenast.LetStatement:
        idents = self.parse_ident_list()
        self.read_op("=")
        expr = self.parse_expression()
        return tenast.LetStatement(idents, expr)

    def parse_expression(self) -> tenast.Expr:
        return self.parse_maybe_sum()

    def parse_maybe_sum(self) -> tenast.Expr:
        lhs = self.parse_maybe_product()
        while self.peek_token_is_op("+") or self.peek_token_is_op("-"):
            op = self.read_op()
            rhs = self.parse_maybe_product()
            lhs = tenast.BinaryExpr(op, lhs, rhs)
        return lhs

    def parse_maybe_product(self) -> tenast.Expr:
        lhs = self.parse_maybe_power()
        while self.peek_token_is_op("*") or self.peek_token_is_op("/"):
            op = self.read_op()
            rhs = self.parse_maybe_power()
            lhs = tenast.BinaryExpr(op, lhs, rhs)
        return lhs

    def parse_maybe_power(self) -> tenast.Expr:
        lhs = self.parse_maybe_matmul()
        while self.peek_token_is_op("**"):
            op = self.read_op("**")
            rhs = self.parse_maybe_matmul()
            lhs = tenast.BinaryExpr(op, lhs, rhs)
        return lhs

    def parse_maybe_matmul(self) -> tenast.Expr:
        lhs = self.parse_maybe_reshape()
        while self.peek_token_is_op("@"):
            op = self.read_op("@")
            rhs = self.parse_maybe_reshape()
            lhs = tenast.BinaryExpr(op, lhs, rhs)
        return lhs

    def parse_maybe_reshape(self) -> tenast.Expr:
        lhs = self.parse_primitive_expr()
        while self.peek_token_is_op("{"):
            self.read_op("{")
            from_shape = self.parse_reshape_type()
            self.read_op("->")
            to_shape = self.parse_reshape_type()
            self.read_op("}")
            lhs = tenast.ReshapeExpr(lhs, from_shape, to_shape, {})
        return lhs

    def parse_primitive_expr(self) -> tenast.Expr:
        tok = self.peek_token()
        if tok.kind == "OP" and tok.text == "(":
            return self.parse_paren_expr()
        elif tok.kind == "IDENT" and tok.text == "for":
            return self.parse_for_expr()
        elif tok.kind == "NUMBER":
            return tenast.FloatExpr(float(self.read_number().text))
        elif tok.kind == "IDENT":
            # CallExpr / IndexExpr / Ident
            ident = self.read_ident()
            tok = self.peek_token()
            if tok.kind == "OP" and (
                tok.text == "[" or tok.text == "("
            ):  # TODO: What about | ?
                return self.parse_call_expr(ident)
            elif tok.kind == "OP" and tok.text == ".":
                return self.parse_index_expr(ident)
            else:
                return tenast.VariableExpr(ident)
        raise Exception("Expected expression, found " + tok.text)

    def parse_paren_expr(self) -> tenast.Expr:
        self.read_op("(")
        expr = self.parse_expression()
        self.read_op(")")
        return expr

    def parse_call_expr(self, ident: tenast.Token) -> tenast.Expr:
        static_args: list[tenast.Expr] = []
        if self.peek_token_is_op("["):
            self.read_op("[")
            if not self.peek_token_is_op("]"):
                static_args = self.parse_arg_list()
            self.read_op("]")
        params: list[tenast.Expr] = []
        if self.peek_token_is_op("|"):
            self.read_op("|")
            if not self.peek_token_is_op("|"):
                params = self.parse_arg_list()
            self.read_op("|")
        self.read_op("(")
        args: list[tenast.Expr] = []
        if not self.peek_token_is_op(")"):
            args = self.parse_arg_list()
        self.read_op(")")
        return tenast.CallExpr(tenast.VariableExpr(ident), static_args, params, args)

    def parse_arg_list(self) -> list[tenast.Expr]:
        ret = [self.parse_expression()]
        while self.peek_token_is_op(","):
            self.read_op(",")
            ret.append(self.parse_expression())
        return ret

    def parse_index_expr(self, ident: tenast.Token) -> tenast.Expr:
        self.read_op(".")
        self.read_op("[")
        index = self.parse_expression()
        self.read_op("]")
        return tenast.IndexExpr(tenast.VariableExpr(ident), index)

    def parse_for_expr(self) -> tenast.Expr:
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
        return tenast.ForExpr(index, start, end, init, var, loop)

    def parse_reshape_type(self) -> tenast.ReshapeTensorShape:
        tok = self.peek_token()
        dims: list[Union[tenast.Token, "tenast.ReshapeTensorShape"]] = []
        if (
            tok.kind == "IDENT"
            or tok.kind == "NUMBER"
            or (tok.kind == "OP" and tok.text == "(")
        ):
            dims.append(self.parse_reshape_dimension())
            while self.peek_token_is_op(","):
                self.read_op(",")
                dims.append(self.parse_reshape_dimension())
        return tenast.ReshapeTensorShape(dims)

    def parse_reshape_dimension(self) -> Union[tenast.Token, tenast.ReshapeTensorShape]:
        tok = self.read_token()
        if tok.kind == "IDENT" or tok.kind == "NUMBER":
            return tok
        elif tok.kind == "OP" and tok.text == "(":
            sub = self.parse_reshape_type()
            self.read_op(")")
            return sub
        raise Exception("Expected tensor type dimension, got " + tok.text)

    def parse_tensor_type(self) -> tenast.TensorType:
        self.read_op("{")
        dims: list[tenast.Token] = []
        if not self.peek_token_is_op("}"):
            dims.append(self.parse_dimension())
            while self.peek_token_is_op(","):
                self.read_op(",")
                dims.append(self.parse_dimension())
        self.read_op("}")
        return tenast.TensorType(dims)

    def parse_dimension(self) -> tenast.Token:
        tok = self.read_token()
        if not (
            tok.kind == "IDENT"
            or tok.kind == "NUMBER"
            or (tok.kind == "OP" and tok.text == "...")
        ):
            raise Exception("Expected tensor type dimension, got " + tok.text)
        return tok
