"""Optional Lark grammar for COBOL.

Real COBOL is too ambiguous for a small grammar, so this is intentionally
restricted: it parses a paragraph body composed of a few statement classes
useful for static-analysis sanity checks. The regex-based path remains the
primary route in :mod:`paragraphs`.
"""
from __future__ import annotations

GRAMMAR = r"""
start: statement+

statement: if_stmt
         | move_stmt
         | compute_stmt
         | perform_stmt
         | other_stmt

if_stmt: "IF"i condition statement+ ("ELSE"i statement+)? "END-IF"i?
condition: NAME OP NAME
         | NAME OP NUMBER

move_stmt: "MOVE"i (NAME | STRING | NUMBER) "TO"i NAME

compute_stmt: "COMPUTE"i NAME "=" expr
expr: NAME (OP NAME)*
    | NUMBER (OP NUMBER)*

perform_stmt: "PERFORM"i NAME ("THRU"i NAME)?

other_stmt: NAME ("." | NEWLINE)

OP: "<" | ">" | "=" | "<=" | ">=" | "+" | "-" | "*" | "/"
NAME: /[A-Za-z][A-Za-z0-9-]*/
NUMBER: /-?\d+(\.\d+)?/
STRING: /'[^']*'|"[^"]*"/
NEWLINE: "\n"

%import common.WS
%ignore WS
"""


def get_parser():  # pragma: no cover - optional path
    from lark import Lark

    return Lark(GRAMMAR, parser="lalr")
