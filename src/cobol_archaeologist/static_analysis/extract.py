"""Extract static facts (vars read/written, conditions, perform calls, file refs)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_IDENT = r"[A-Z][A-Z0-9-]*"
_LITERAL = r"(?:'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)"

_MOVE = re.compile(rf"\bMOVE\s+({_LITERAL}|{_IDENT})\s+TO\s+({_IDENT}(?:\s*,\s*{_IDENT})*)", re.I)
_COMPUTE = re.compile(rf"\bCOMPUTE\s+({_IDENT})\s*=\s*(.+?)(?=\.|$|\bEND-COMPUTE\b)", re.I | re.S)
_ARITH = re.compile(
    rf"\b(ADD|SUBTRACT|MULTIPLY|DIVIDE)\s+({_LITERAL}|{_IDENT})\s+(?:FROM|TO|BY|INTO)\s+({_IDENT})",
    re.I,
)
_IF = re.compile(r"\bIF\s+(.+?)(?=\bTHEN\b|\n|\bMOVE\b|\bPERFORM\b|\bGO\b|\bDISPLAY\b|\.)", re.I | re.S)
_WHEN = re.compile(r"\bWHEN\s+(.+?)(?=\n|\.)", re.I | re.S)
_PERFORM = re.compile(rf"\bPERFORM\s+({_IDENT})(?:\s+THRU\s+({_IDENT}))?", re.I)
_FILE_OP = re.compile(rf"\b(READ|WRITE|REWRITE|DELETE|OPEN|CLOSE)\s+(?:INPUT\s+|OUTPUT\s+|I-O\s+|EXTEND\s+)?({_IDENT})", re.I)
_SELECT = re.compile(rf"\bSELECT\s+({_IDENT})", re.I)
_FD = re.compile(rf"\bFD\s+({_IDENT})", re.I)


@dataclass
class StaticFacts:
    vars_read: list[str] = field(default_factory=list)
    vars_written: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    perform_calls: list[str] = field(default_factory=list)
    file_refs: list[str] = field(default_factory=list)


def _add(seq: list[str], item: str) -> None:
    item = item.strip().upper()
    if item and item not in seq:
        seq.append(item)


def _identifiers(text: str) -> list[str]:
    # Strip string literals first so words inside them aren't mistaken for identifiers.
    cleaned = re.sub(_LITERAL, " ", text)
    return [m.group(0).upper() for m in re.finditer(_IDENT, cleaned)]


# Tokens that look like identifiers but are reserved.
_RESERVED_TOKENS = {
    "AND", "OR", "NOT", "TRUE", "FALSE", "IS", "EQUAL", "TO", "FROM", "BY",
    "INTO", "GIVING", "ZERO", "ZEROS", "ZEROES", "SPACE", "SPACES",
    "HIGH-VALUE", "HIGH-VALUES", "LOW-VALUE", "LOW-VALUES", "ALL",
    "GREATER", "LESS", "THAN", "EQUALS", "OF", "IN",
}


def _filter_idents(idents: list[str]) -> list[str]:
    return [t for t in idents if t not in _RESERVED_TOKENS]


def extract_facts(code: str) -> StaticFacts:
    facts = StaticFacts()
    upper = code.upper()

    for m in _MOVE.finditer(upper):
        src, dst_group = m.group(1), m.group(2)
        if not re.match(rf"^{_LITERAL}$", src):
            _add(facts.vars_read, src)
        for dst in dst_group.split(","):
            _add(facts.vars_written, dst)

    for m in _COMPUTE.finditer(upper):
        target, expr = m.group(1), m.group(2)
        _add(facts.vars_written, target)
        for ident in _filter_idents(_identifiers(expr)):
            _add(facts.vars_read, ident)

    for m in _ARITH.finditer(upper):
        verb, src, dst = m.group(1).upper(), m.group(2), m.group(3)
        if not re.match(rf"^{_LITERAL}$", src):
            _add(facts.vars_read, src)
        _add(facts.vars_read, dst)
        _add(facts.vars_written, dst)

    for m in _IF.finditer(upper):
        cond = m.group(1).strip()
        if cond:
            _add(facts.conditions, cond)
            for ident in _filter_idents(_identifiers(cond)):
                _add(facts.vars_read, ident)

    for m in _WHEN.finditer(upper):
        cond = m.group(1).strip()
        if cond and cond not in {"OTHER"}:
            _add(facts.conditions, cond)
            for ident in _filter_idents(_identifiers(cond)):
                _add(facts.vars_read, ident)

    for m in _PERFORM.finditer(upper):
        _add(facts.perform_calls, m.group(1))
        if m.group(2):
            _add(facts.perform_calls, m.group(2))

    for m in _FILE_OP.finditer(upper):
        _add(facts.file_refs, m.group(2))
    for m in _SELECT.finditer(upper):
        _add(facts.file_refs, m.group(1))
    for m in _FD.finditer(upper):
        _add(facts.file_refs, m.group(1))

    return facts
