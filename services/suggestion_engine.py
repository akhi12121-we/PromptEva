"""
services/suggestion_engine.py — Generate cheaper alternative prompts.

Rewrites a prompt to be more concise while preserving intent,
then returns the compressed version + a one-line reason.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# System prompt for compression
# ---------------------------------------------------------------------------

_COMPRESSION_SYSTEM_PROMPT = """\
You are a prompt compression expert. Your job is to rewrite the user's prompt \
so it achieves the same goal with fewer tokens — without changing the intended task.

Rules:
1. Preserve all essential instructions, constraints, and output requirements.
2. Remove filler words, redundant phrases, repetition, and over-explanation.
3. Tighten language: prefer short direct sentences over long compound ones.
4. Keep all technical terms, format specifications, and key context.
5. Do NOT add new content or change the task.

Respond in EXACTLY this format (no extra text):

### Compressed Prompt
<the compressed prompt here — ready to use>

### Reason
<one sentence: what was trimmed or simplified and why it was safe to remove>
"""


def generate_cheaper_alternative(
    prompt_text: str, provider: str, model: str
) -> tuple[str, str]:
    """Rewrite prompt_text to be more concise using the given model.

    Args:
        prompt_text: The original prompt to compress.
        provider:    "OpenAI", "Anthropic", or "Google".
        model:       Specific model identifier to use.

    Returns:
        (compressed_prompt, reason) — both as plain strings.
    """
    if provider == "OpenAI":
        raw = _call_openai(prompt_text, model)
    elif provider == "Anthropic":
        raw = _call_anthropic(prompt_text, model)
    else:
        raw = _call_google(prompt_text, model)

    return _parse_response(raw, prompt_text)


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------


def _call_openai(prompt_text: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt_text: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        system=_COMPRESSION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.content[0].text


def _call_google(prompt_text: str, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.generate_content(
        model=model,
        contents=f"{_COMPRESSION_SYSTEM_PROMPT}\n\n{prompt_text}",
        config={"temperature": 0.3, "max_output_tokens": 2048},
    )
    return resp.text or ""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_response(raw: str, original: str) -> tuple[str, str]:
    """Extract compressed prompt and reason from the LLM response.

    Falls back to the original prompt + a generic reason on parse failure.
    """
    compressed = original
    reason = "Could not generate a compressed alternative."

    if "### Compressed Prompt" in raw:
        after_header = raw.split("### Compressed Prompt", 1)[1]
        if "### Reason" in after_header:
            compressed_part, reason_part = after_header.split("### Reason", 1)
            compressed = compressed_part.strip()
            reason = reason_part.strip()
        else:
            compressed = after_header.strip()

    return compressed, reason
