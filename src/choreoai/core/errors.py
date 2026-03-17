"""Domain-level exceptions used across adapters and services."""


class ChoreoAIError(Exception):
    """Base application exception."""


class ValidationError(ChoreoAIError):
    """Raised when user-facing input fails validation."""
