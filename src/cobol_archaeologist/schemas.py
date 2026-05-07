"""Pydantic data contracts shared across the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LogicBlock(BaseModel):
    """One unit of business logic extracted from a COBOL program."""

    id: str
    source_file: str
    paragraph: str
    code: str
    start_line: int = 0
    end_line: int = 0
    vars_read: list[str] = Field(default_factory=list)
    vars_written: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    perform_calls: list[str] = Field(default_factory=list)
    file_refs: list[str] = Field(default_factory=list)
    copybooks: list[str] = Field(default_factory=list)
    weak_label: Optional[str] = None
    weak_label_confidence: float = 0.0
    tags: list[str] = Field(default_factory=list)


class Confidence(BaseModel):
    level: Literal["High", "Medium", "Low"]
    justification: str = ""


class BusinessIntentCard(BaseModel):
    """Output produced by the model for a single LogicBlock."""

    logic_block_id: Optional[str] = None
    what: str
    why: str
    code_evidence: list[str] = Field(default_factory=list)
    regulation_link: Optional[str] = None
    regulation_sources: list[str] = Field(default_factory=list)
    confidence: Confidence


class RegulationChunk(BaseModel):
    id: str
    source: str
    section: Optional[str] = None
    page: Optional[int] = None
    text: str
