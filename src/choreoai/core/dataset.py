"""Core dataset entities independent from transport and storage adapters."""

from __future__ import annotations

from dataclasses import dataclass

from choreoai.core.errors import ValidationError


@dataclass(frozen=True)
class SequenceDescriptor:
    """Represents a sequence identity used in dataset operations."""

    seq_id: str

    def validate(self) -> None:
        cleaned = self.seq_id.strip()
        if not cleaned:
            raise ValidationError("seq_id cannot be empty")
        if "/" in cleaned or ".." in cleaned:
            # Why: avoid implicit directory traversal and accidental nested IDs.
            raise ValidationError("seq_id must be a simple identifier")
