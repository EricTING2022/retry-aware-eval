"""Thin OpenRouter adapter. It performs one request and never retries."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .domain import CompletionRequest, CompletionResponse


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        response_body_text: str | None = None,
        raw_response: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.response_body_text = response_body_text
        self.raw_response = raw_response


class OpenRouterResponseError(OpenRouterError):
    """A syntactically valid API response that cannot be used as a completion."""

    def __init__(self, message: str, response: CompletionResponse) -> None:
        super().__init__(
            message,
            http_status=response.http_status,
            response_body_text=response.response_body_text,
            raw_response=dict(response.raw_response),
        )
        self.response = response


class OpenRouterClient:
    def __init__(self, api_key: str, *, timeout_seconds: float = 90.0) -> None:
        if not api_key:
            raise ValueError("An OpenRouter API key is required.")
        self._api_key = api_key
        self._timeout = timeout_seconds

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = request.as_payload()
        payload.setdefault("usage", {"include": True})
        encoded = json.dumps(payload).encode("utf-8")
        http_request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(http_request, timeout=self._timeout) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                http_status = getattr(response, "status", None)
                if type(http_status) is not int:
                    http_status = 200
                result = json.loads(response_body)
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                raw_response = json.loads(body)
            except json.JSONDecodeError:
                raw_response = None
            raise OpenRouterError(
                f"OpenRouter HTTP {error.code}: {body}",
                http_status=error.code,
                response_body_text=body,
                raw_response=raw_response if isinstance(raw_response, dict) else None,
            ) from error
        except json.JSONDecodeError as error:
            raise OpenRouterError(
                f"OpenRouter response was not valid JSON: {error}",
                http_status=http_status,
                response_body_text=response_body,
            ) from error
        except (URLError, TimeoutError) as error:
            raise OpenRouterError(f"OpenRouter request failed: {error}") from error

        if not isinstance(result, dict):
            raise OpenRouterError(
                "OpenRouter response JSON must be an object.",
                http_status=http_status,
                response_body_text=response_body,
            )
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        choices = result.get("choices")
        has_choice = isinstance(choices, list) and bool(choices)
        choice = choices[0] if has_choice and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        completion_details = usage.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}
        completion = CompletionResponse(
            content=content if isinstance(content, str) else "",
            input_tokens=_optional_nonnegative_int(
                usage.get("prompt_tokens")
            ),
            output_tokens=_optional_nonnegative_int(
                usage.get("completion_tokens")
            ),
            cost_usd=_nonnegative_float_or_none(usage.get("cost")),
            raw_response=result,
            returned_model=result.get("model") if isinstance(result.get("model"), str) else None,
            provider=result.get("provider") if isinstance(result.get("provider"), str) else None,
            response_id=result.get("id") if isinstance(result.get("id"), str) else None,
            http_status=http_status,
            finish_reason=(
                choice.get("finish_reason")
                if isinstance(choice.get("finish_reason"), str)
                else None
            ),
            reasoning_tokens=_optional_nonnegative_int(
                completion_details.get("reasoning_tokens")
            ),
            usage=usage,
            response_body_text=response_body,
        )
        if not has_choice:
            raise OpenRouterResponseError(
                "OpenRouter response has no choices.", completion
            )
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason")
            detail = (
                f" finish_reason={finish_reason!r}." if isinstance(finish_reason, str) else ""
            )
            raise OpenRouterResponseError(
                f"OpenRouter response has no usable text content.{detail}", completion
            )
        return completion


def _nonnegative_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        return None
    return value

