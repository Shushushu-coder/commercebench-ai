"""Provider transport seams (Phase 3E).

Only the OpenAI Responses surface exists so far: a narrow injectable
client protocol, the live SDK transport, and a deterministic scripted
transport for offline exercise. No provider-native session state is
expressible at this boundary.
"""

from .errors import (
    ProviderAuthenticationError,
    ProviderCredentialError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderTransportError,
)
from .openai_responses import (
    OPENAI_API_KEY_ENV,
    OPENAI_API_SURFACE,
    OPENAI_PROVIDER_ID,
    OpenAIResponsesClient,
    ResponsesClientProtocol,
    ScriptedResponsesClient,
    map_provider_error,
)

__all__ = [
    "OPENAI_API_KEY_ENV",
    "OPENAI_API_SURFACE",
    "OPENAI_PROVIDER_ID",
    "OpenAIResponsesClient",
    "ProviderAuthenticationError",
    "ProviderCredentialError",
    "ProviderError",
    "ProviderInvalidRequestError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "ProviderTransportError",
    "ResponsesClientProtocol",
    "ScriptedResponsesClient",
    "map_provider_error",
]
