"""Model profiles for Qian Xie's proposed OpenRouter experiment."""

from __future__ import annotations

from .domain import ModelProfile


GPT_OSS_20B = "openai/gpt-oss-20b"
MINISTRAL_3_8B = "mistralai/ministral-8b-2512"
QWEN3 = "qwen/qwen3-32b"
GEMINI_FIXED_CRITIC = "google/gemini-2.5-flash-lite"

QIAN_MODEL_IDS = (GPT_OSS_20B, MINISTRAL_3_8B, QWEN3)

_SOLVER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "mathqa_solver_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "calculation": {"type": "string", "maxLength": 160},
                "option": {"type": "string", "enum": list("abcde")},
                "value": {"type": "string", "maxLength": 120},
            },
            "required": ["calculation", "option", "value"],
            "additionalProperties": False,
        },
    },
}
_VERIFIER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "mathqa_verifier_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"accepted": {"type": "boolean"}},
            "required": ["accepted"],
            "additionalProperties": False,
        },
    },
}
_BASE_SOLVER_PARAMETERS = {
    "temperature": 0,
    "stream": False,
    "usage": {"include": True},
    "provider": {"require_parameters": True},
    "response_format": _SOLVER_RESPONSE_FORMAT,
}
_BASE_VERIFIER_PARAMETERS = {
    "temperature": 0,
    "stream": False,
    "usage": {"include": True},
    "provider": {"require_parameters": True},
    "response_format": _VERIFIER_RESPONSE_FORMAT,
}
_GPT_OSS_PROVIDER = {
    "require_parameters": True,
    # Darkbloom advertised structured-output support but returned reasoning-only
    # HTTP 200 responses with message.content=null during the 2026-10-16 run.
    "ignore": ["darkbloom"],
}

QIAN_MODEL_PROFILES = {
    GPT_OSS_20B: ModelProfile(
        model=GPT_OSS_20B,
        solver_parameters={
            **_BASE_SOLVER_PARAMETERS,
            "provider": _GPT_OSS_PROVIDER,
            "reasoning": {"effort": "low"},
        },
        verifier_parameters={
            **_BASE_VERIFIER_PARAMETERS,
            "provider": _GPT_OSS_PROVIDER,
            "reasoning": {"effort": "low"},
        },
    ),
    MINISTRAL_3_8B: ModelProfile(
        model=MINISTRAL_3_8B,
        solver_parameters=dict(_BASE_SOLVER_PARAMETERS),
        verifier_parameters=dict(_BASE_VERIFIER_PARAMETERS),
    ),
    QWEN3: ModelProfile(
        model=QWEN3,
        solver_parameters={**_BASE_SOLVER_PARAMETERS, "reasoning": {"enabled": False}},
        verifier_parameters={
            **_BASE_VERIFIER_PARAMETERS,
            "reasoning": {"enabled": False},
        },
        solver_prompt_suffix="\n/no_think",
    ),
    GEMINI_FIXED_CRITIC: ModelProfile(
        model=GEMINI_FIXED_CRITIC,
        solver_parameters=dict(_BASE_SOLVER_PARAMETERS),
        verifier_parameters=dict(_BASE_VERIFIER_PARAMETERS),
    ),
}

