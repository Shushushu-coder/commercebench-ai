"""OpenAI Responses API transport seam (Phase 3E).

A deliberately narrow boundary between the dialogue adapter and the
provider: the adapter renders a plain JSON-able ``Responses.create``
request payload, hands it to an injected client, and parses the
returned payload. Everything stays testable offline — a
``ScriptedResponsesClient`` can capture the exact outgoing request
and answer with a synthetic provider payload; no test needs
``OPENAI_API_KEY``, a network, an OpenAI account, or API credits.

``OpenAIResponsesClient`` is the only live path. The official
``openai`` SDK is imported lazily at the execution boundary and the
credential is resolved only inside ``create_response``, so importing
this module or constructing the client never requires a configured
key. There is no retry: the first real baseline keeps a single
attempt per request so provenance stays simple.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from commercebench.contracts.errors import ContractValidationError
from commercebench.dialogue.state import json_value_copy

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

OPENAI_PROVIDER_ID = "openai"
OPENAI_API_SURFACE = "responses"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


@runtime_checkable
class ResponsesClientProtocol(Protocol):
    """Minimal transport contract for the Responses API surface.

    ``request`` is the exact keyword-argument payload that a real
    ``openai`` client's ``responses.create(**request)`` would receive;
    implementations return the provider response as a plain JSON
    mapping. No provider-native session state exists at this seam —
    every request is independently constructed from CommerceBench
    state and history.
    """

    def create_response(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def _missing_credential_message() -> str:
    return (
        "provider credential missing: the OPENAI_API_KEY environment "
        "variable is required for live OpenAI requests"
    )


class OpenAIResponsesClient:
    """Live transport over the official ``openai`` SDK.

    The SDK and the credential are resolved lazily: ``api_key`` may be
    supplied explicitly (tests never do), otherwise the
    ``OPENAI_API_KEY`` environment variable is read inside
    ``create_response``. A missing credential raises
    :class:`ProviderCredentialError` before any network activity.

    Responses are normalized to plain JSON mappings via the SDK's
    ``model_dump(mode="json")`` so the adapter's parser stays
    SDK-shape-agnostic.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: Any = None

    # -- internals ---------------------------------------------------------

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ProviderCredentialError(_missing_credential_message())
        try:
            import openai  # noqa: WPS433 — lazy by design
        except ImportError as exc:  # pragma: no cover — dependency pinned
            raise ProviderError(
                "the 'openai' package is required for the live "
                "OpenAI transport"
            ) from exc
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        self._client = openai.OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _dump_response(response: Any) -> Mapping[str, Any]:
        dump = getattr(response, "model_dump", None)
        if callable(dump):
            payload = dump(mode="json")
        elif isinstance(response, Mapping):
            payload = dict(response)
        else:
            raise ProviderResponseError(
                "provider response object cannot be normalized to a "
                f"JSON mapping (type={type(response).__name__})"
            )
        if not isinstance(payload, Mapping):
            raise ProviderResponseError(
                "provider response did not normalize to a JSON mapping"
            )
        return payload

    # -- transport ----------------------------------------------------------

    def create_response(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        client = self._resolve_client()
        try:
            response = client.responses.create(**dict(request))
        except ProviderError:
            raise
        except Exception as exc:  # SDK error boundary
            raise map_provider_error(exc) from exc
        return self._dump_response(response)


def map_provider_error(exc: Exception) -> ProviderError:
    """Classify an ``openai`` SDK exception into the provider taxonomy.

    Fail-safe: anything unclassified becomes a plain
    :class:`ProviderError`. Only the exception message text is carried
    over — never request internals and never the credential.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover
        return ProviderError(f"provider error: {exc}")
    status = getattr(exc, "status_code", None)
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, openai.AuthenticationError) or isinstance(
        exc, openai.PermissionDeniedError
    ):
        return ProviderAuthenticationError(message)
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(message)
    if isinstance(exc, openai.APITimeoutError):
        return ProviderTimeoutError(message)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderTransportError(message)
    if isinstance(exc, openai.InternalServerError):
        return ProviderServerError(message)
    if isinstance(exc, openai.APIResponseValidationError):
        return ProviderResponseError(message)
    if isinstance(exc, openai.APIStatusError):
        if isinstance(status, int) and status >= 500:
            return ProviderServerError(message)
        return ProviderInvalidRequestError(message)
    if isinstance(exc, openai.OpenAIError):
        return ProviderError(message)
    return ProviderError(message)


class ScriptedResponsesClient:
    """Deterministic fake transport for offline adapter exercise.

    Two modes, exactly one required:

    - ``responses``: a FIFO queue of provider payload mappings popped
      per call — used when the expected request sequence is known.
    - ``handler``: a callable ``(request) -> Mapping`` producing the
      payload from request content (or raising a ``ProviderError``).

    Every request is deep-copied into ``self.requests`` before the
    response is produced, so tests can assert on the exact outgoing
    payloads the adapter rendered.
    """

    def __init__(
        self,
        *,
        responses: Optional[Sequence[Mapping[str, Any]]] = None,
        handler: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
    ) -> None:
        if (responses is None) == (handler is None):
            raise ContractValidationError(
                "ScriptedResponsesClient requires exactly one of "
                "'responses' or 'handler'"
            )
        self._queue: List[Mapping[str, Any]] = list(responses or ())
        self._handler = handler
        self.requests: List[Dict[str, Any]] = []

    def create_response(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        snapshot = json_value_copy(dict(request))
        self.requests.append(snapshot)
        if self._handler is not None:
            payload = self._handler(request)
        else:
            if not self._queue:
                raise ProviderResponseError(
                    "scripted responses exhausted: "
                    f"{len(self.requests)} requests issued"
                )
            payload = self._queue.pop(0)
        if not isinstance(payload, Mapping):
            raise ProviderResponseError(
                "scripted provider payload must be a JSON mapping"
            )
        return json_value_copy(dict(payload))


__all__ = [
    "OPENAI_API_KEY_ENV",
    "OPENAI_API_SURFACE",
    "OPENAI_PROVIDER_ID",
    "OpenAIResponsesClient",
    "ResponsesClientProtocol",
    "ScriptedResponsesClient",
    "map_provider_error",
]
