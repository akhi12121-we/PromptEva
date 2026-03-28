"""
config/pricing.py — Single source of truth for all model pricing.

All costs are in USD per million tokens (input / output).
Update this file when provider pricing changes — nowhere else.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model lists per provider (as specified)
# ---------------------------------------------------------------------------

OPENAI_MODELS: list[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

ANTHROPIC_MODELS: list[str] = [
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
]

GOOGLE_MODELS: list[str] = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

ALL_MODELS: list[str] = OPENAI_MODELS + ANTHROPIC_MODELS + GOOGLE_MODELS

# ---------------------------------------------------------------------------
# Pricing per million tokens (USD)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":                     {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":                {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":              {"input": 0.50,  "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-haiku-20240307":    {"input": 0.25,  "output": 1.25},
    "claude-3-opus-20240229":     {"input": 15.00, "output": 75.00},
    # Google
    "gemini-1.5-pro":             {"input": 1.25,  "output": 5.00},
    "gemini-1.5-flash":           {"input": 0.075, "output": 0.30},
}

# ---------------------------------------------------------------------------
# Human-readable display names (used in dropdowns)
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gpt-4o":                     "OpenAI — GPT-4o ($2.50/$10.00 per M)",
    "gpt-4o-mini":                "OpenAI — GPT-4o Mini ($0.15/$0.60 per M)",
    "gpt-3.5-turbo":              "OpenAI — GPT-3.5 Turbo ($0.50/$1.50 per M)",
    "claude-3-5-sonnet-20241022": "Anthropic — Claude 3.5 Sonnet ($3.00/$15.00 per M)",
    "claude-3-haiku-20240307":    "Anthropic — Claude 3 Haiku ($0.25/$1.25 per M)",
    "claude-3-opus-20240229":     "Anthropic — Claude 3 Opus ($15.00/$75.00 per M)",
    "gemini-1.5-pro":             "Google — Gemini 1.5 Pro ($1.25/$5.00 per M)",
    "gemini-1.5-flash":           "Google — Gemini 1.5 Flash ($0.075/$0.30 per M)",
}

# ---------------------------------------------------------------------------
# Provider lookup
# ---------------------------------------------------------------------------

PROVIDER_FOR_MODEL: dict[str, str] = {
    "gpt-4o":                     "OpenAI",
    "gpt-4o-mini":                "OpenAI",
    "gpt-3.5-turbo":              "OpenAI",
    "claude-3-5-sonnet-20241022": "Anthropic",
    "claude-3-haiku-20240307":    "Anthropic",
    "claude-3-opus-20240229":     "Anthropic",
    "gemini-1.5-pro":             "Google",
    "gemini-1.5-flash":           "Google",
}
