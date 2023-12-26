# Copyright 2023 Luke Hoban

import unittest
from ten import tenast, compiler
from typing import Optional
from utils import tensor_type, var


class CompilerTestCase(unittest.TestCase):
    def test_applied_return_type(self):
        comp = compiler.Compiler()
        test_cases: list[
            tuple[
                tuple[list[list[int | str]], list[int | str]],
                list[list[int | str]],
                Optional[list[int | str]],
            ]
        ] = [
            (([["..."]], ["..."]), [[]], []),  # Tanh
            (([["..."]], ["..."]), [["...", 3]], ["...", 3]),  # Exp
            (([["...", 3]], ["...", 1]), [["...", 4, 3]], ["...", 4, 1]),  # Max
            (([["...", 3]], ["...", 3]), [["...", 3]], ["...", 3]),  # SoftMax
        ]
        for (a, b), c, tr in test_cases:
            a = [tensor_type(x) for x in a]
            b = tensor_type(b)
            c = [tensor_type(x) for x in c]
            if tr is None:
                self.assertRaises(Exception, comp.applied_return_type, a, b, c)
            else:
                tr = tensor_type(tr)
                f = tenast.FunctionDeclaration(
                    var("f"),
                    [],
                    [],
                    [(var(f"x_{i}"), at) for i, at in enumerate(a)],
                    b,
                    None,
                )
                self.assertEqual(comp.applied_return_type(f, c), tr)

    def test_check_broadcastable(self):
        comp = compiler.Compiler()
        test_cases: list[
            tuple[tuple[list[int | str], list[int | str]], Optional[list[int | str]]]
        ] = [
            (([], [3]), [3]),
            (([3], [3]), [3]),
            (([3], [3, 4]), None),
            (([1, 4], [3, 1]), [3, 4]),
            (([1, 4], [1, 3, 1]), [1, 3, 4]),
            (([], ["..."]), ["..."]),
            ((["...", 3], ["..."]), None),
            ((["...", 4], ["...", 3, 1]), None),
            # ((["...", 4], [1, 1]), ["...", 4]), # TODO: Is this legit?  The zero-d expansion of ... is invalid
        ]
        for (a, b), tr in test_cases:
            a = tensor_type(a)
            b = tensor_type(b)
            if tr is None:
                self.assertRaises(Exception, comp.check_broadcastable, a, b)
            else:
                tr = tensor_type(tr)
                self.assertEqual(comp.check_broadcastable(a, b), tr)
