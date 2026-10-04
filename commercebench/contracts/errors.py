"""Exceptions raised by core contract validation."""


class ContractValidationError(ValueError):
    """Raised when a core contract (case, manifest, trace, result) is invalid."""
