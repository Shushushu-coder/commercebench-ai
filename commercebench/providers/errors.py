"""Provider execution error taxonomy (Phase 3E).

These errors live strictly outside the benchmark failure taxonomy.
They describe failures of the provider boundary itself — missing
credential, rejected authentication, transport, rate limit, server,
invalid request, or a malformed provider payload — never task
outcomes. They are deliberately NOT ``ContractValidationError``: a
contract violation is a benchmark-authoring bug, while a provider
error is runtime infrastructure evidence. Inside a dialogue run the
harness records them as ``system_error`` terminations; they are
never re-mapped to ``TASK_INCOMPLETE`` / ``INVALID_TOOL_CALL`` /
``REQUIRED_TOOL_MISSING``.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for provider-boundary failures."""


class ProviderCredentialError(ProviderError):
    """The required provider credential was not configured.

    Raised before any transport attempt; the message never carries
    the credential value.
    """


class ProviderAuthenticationError(ProviderError):
    """The provider rejected the presented credential (401/403)."""


class ProviderTransportError(ProviderError):
    """Network-level failure reaching the provider."""


class ProviderTimeoutError(ProviderTransportError):
    """The provider request timed out."""


class ProviderRateLimitError(ProviderError):
    """The provider rate-limited the request (429)."""


class ProviderServerError(ProviderError):
    """The provider returned a 5xx status."""


class ProviderInvalidRequestError(ProviderError):
    """The provider rejected the request as invalid (4xx other than
    auth/rate-limit), including unsupported model or configuration."""


class ProviderResponseError(ProviderError):
    """The provider returned a failed status or a payload that does
    not match the expected protocol shape (malformed output)."""


__all__ = [
    "ProviderAuthenticationError",
    "ProviderCredentialError",
    "ProviderError",
    "ProviderInvalidRequestError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "ProviderTransportError",
]
