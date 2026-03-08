"""
evaluator.py — Prompt evaluation and refinement engine.

Contains:
  - Metric definitions (what we score and how)
  - Available model choices per provider
  - run_evaluation()  — scores a prompt via DeepEval G-Eval
  - refine_prompt()   — calls the LLM to produce an improved version

All DeepEval / OpenAI / Anthropic imports are deferred so the rest
of the app can load without requiring an API key.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Available models per provider
# ---------------------------------------------------------------------------

OPENAI_MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

ANTHROPIC_MODELS: list[str] = [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-latest",
]

GOOGLE_MODELS: list[str] = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
# Each dict drives both DeepEval evaluation and the UI display.

METRIC_DEFS: list[dict[str, Any]] = [
    {
        "name": "Clarity",
        "icon": "🔍",
        "color": "#6366f1",
        "short_def": "Is the prompt easy to understand?",
        "description": (
            "Measures whether the prompt uses clear, unambiguous language. "
            "A high score means the LLM can understand exactly what you're asking "
            "without guessing. A low score means the wording is vague or confusing."
        ),
        "criteria": (
            "Evaluate whether the prompt is written in clear, unambiguous "
            "language that an LLM can interpret without guessing the user's intent."
        ),
        "steps": [
            "Check if the prompt uses precise, direct language.",
            "Identify any vague or ambiguous phrases.",
            "Penalize prompts that require the model to make assumptions about what is being asked.",
        ],
    },
    {
        "name": "Specificity",
        "icon": "🎯",
        "color": "#f59e0b",
        "short_def": "Does the prompt include enough detail?",
        "description": (
            "Measures whether the prompt provides concrete constraints like format, "
            "length, audience, or scope. A high score means the prompt is focused and "
            "well-scoped. A low score means it's too open-ended or lacks direction."
        ),
        "criteria": (
            "Evaluate whether the prompt includes enough specific detail — such as "
            "constraints, format requirements, or context — to produce a focused response."
        ),
        "steps": [
            "Check for the presence of concrete constraints (length, format, audience).",
            "Penalize overly open-ended prompts that lack direction.",
            "Reward prompts that narrow scope without being restrictive.",
        ],
    },
    {
        "name": "Completeness",
        "icon": "📋",
        "color": "#10b981",
        "short_def": "Is all necessary context provided?",
        "description": (
            "Measures whether the prompt gives the LLM everything it needs to produce "
            "a useful answer without follow-up questions. A high score means no critical "
            "information is missing. A low score means key background or context is absent."
        ),
        "criteria": (
            "Evaluate whether the prompt provides all the information necessary "
            "for the LLM to produce a useful answer without additional follow-up."
        ),
        "steps": [
            "Check if the prompt contains all required context.",
            "Identify missing pieces of information that would force clarifying questions.",
            "Penalize prompts that leave out critical background.",
        ],
    },
    {
        "name": "Coherence",
        "icon": "🔗",
        "color": "#3b82f6",
        "short_def": "Is the prompt logically structured?",
        "description": (
            "Measures the logical flow and organization of the prompt. A high score means "
            "instructions are well-ordered and consistent. A low score means the prompt "
            "contains contradictions, jumps between topics, or is disorganized."
        ),
        "criteria": (
            "Evaluate the logical structure and flow of the prompt. A coherent prompt "
            "has a logical ordering of instructions and no contradictory statements."
        ),
        "steps": [
            "Check for logical ordering of instructions or questions.",
            "Identify contradictions between different parts of the prompt.",
            "Penalize disorganized or scattered requests.",
        ],
    },
    {
        "name": "Safety",
        "icon": "🛡️",
        "color": "#ef4444",
        "short_def": "Is the prompt free of harmful intent?",
        "description": (
            "Measures whether the prompt avoids requesting harmful, unethical, or dangerous "
            "content. A high score means the prompt is safe and responsible. A low score means "
            "it may contain jailbreak attempts, request illegal content, or lead to harm."
        ),
        "criteria": (
            "Evaluate whether the prompt avoids requesting harmful, unethical, or "
            "dangerous content and does not attempt to jailbreak the model."
        ),
        "steps": [
            "Check if the prompt requests harmful, illegal, or unethical content.",
            "Identify jailbreak patterns or prompt injection attempts.",
            "Penalize prompts that could lead to dangerous real-world actions.",
        ],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = """\
You are an expert prompt engineer. The user will give you a prompt they wrote \
for an LLM. Your job is to:

1. List 3-5 specific issues with the original prompt (bullet points).
2. Provide a refined, improved version of the prompt that fixes those issues.

Focus on: clarity, specificity, completeness, coherence, and safety.
Keep the user's original intent intact — do NOT change what they are asking for, \
only improve HOW they ask it.

Respond in this exact format (use the headings as shown):

### Issues Found
- issue 1
- issue 2
...

### Refined Prompt
<the improved prompt here>

### What Changed
- change 1
- change 2
...
"""


def _resolve_model(provider: str, model: str) -> str:
    """Return a valid model string, falling back to the first available model.

    Args:
        provider: "OpenAI", "Anthropic", or "Google".
        model:    User-selected model string (may be empty).

    Returns:
        A guaranteed-valid model identifier.
    """
    if provider == "OpenAI":
        return model if model in OPENAI_MODELS else OPENAI_MODELS[0]
    if provider == "Google":
        return model if model in GOOGLE_MODELS else GOOGLE_MODELS[0]
    return model if model in ANTHROPIC_MODELS else ANTHROPIC_MODELS[0]


def _check_api_key(provider: str) -> None:
    """Raise ValueError if the required API key is missing.

    Args:
        provider: "OpenAI", "Anthropic", or "Google".
    """
    if provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not set. Please save it in the Settings tab.")
    if provider == "Anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Anthropic API key not set. Please save it in the Settings tab.")
    if provider == "Google" and not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("Google API key not set. Please save it in the Settings tab.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _make_eval_model(provider: str, model_str: str):
    """Create the appropriate DeepEval model object for the given provider.

    Args:
        provider:   "OpenAI", "Anthropic", or "Google".
        model_str:  Model identifier (e.g. "gpt-4o-mini", "claude-3-5-haiku-20241022").

    Returns:
        A model string (OpenAI) or DeepEval model object (Anthropic/Google).
    """
    if provider == "Anthropic":
        from deepeval.models import AnthropicModel
        return AnthropicModel(model=model_str)
    if provider == "Google":
        from deepeval.models import GeminiModel
        return GeminiModel(model=model_str)
    # OpenAI — DeepEval uses string directly
    return model_str


def _build_metrics(provider: str, model_str: str):
    """Create one GEval instance per metric definition.

    Args:
        provider:  "OpenAI", "Anthropic", or "Google".
        model_str: Model identifier.

    Returns:
        List of configured GEval metric objects.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    eval_model = _make_eval_model(provider, model_str)

    metrics = []
    for mdef in METRIC_DEFS:
        metrics.append(
            GEval(
                name=mdef["name"],
                criteria=mdef["criteria"],
                evaluation_steps=mdef["steps"],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model=eval_model,
                threshold=0.5,
                async_mode=False,
            )
        )
    return metrics


def run_evaluation(
    prompt_text: str, provider: str, model: str = ""
) -> dict[str, dict[str, Any]]:
    """Score a prompt across all metrics using the chosen provider and model.

    Args:
        prompt_text: The raw prompt string to evaluate.
        provider:    "OpenAI", "Anthropic", or "Google".
        model:       Specific model to use (falls back to default if empty).

    Returns:
        Dict mapping metric name -> {"score": float, "reason": str, "passed": bool}.

    Raises:
        ValueError: If the required API key is not set.
    """
    from deepeval.test_case import LLMTestCase

    _check_api_key(provider)
    model_str = _resolve_model(provider, model)

    test_case = LLMTestCase(input=prompt_text, actual_output=prompt_text)
    metrics = _build_metrics(provider, model_str)

    results: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            results[metric.name] = {
                "score": round(metric.score, 4),
                "reason": metric.reason or "",
                "passed": metric.score >= metric.threshold,
            }
        except Exception as exc:
            results[metric.name] = {
                "score": 0.0,
                "reason": f"Error: {exc}",
                "passed": False,
            }

    return results


# ---------------------------------------------------------------------------
# Prompt refinement
# ---------------------------------------------------------------------------


def refine_prompt(prompt_text: str, provider: str, model: str = "") -> str:
    """Call the LLM to produce an improved version of the user's prompt.

    Args:
        prompt_text: The original prompt to improve.
        provider:    "OpenAI", "Anthropic", or "Google".
        model:       Specific model to use (falls back to default if empty).

    Returns:
        The LLM's response containing issues, refined prompt, and changes.

    Raises:
        ValueError: If the required API key is not set.
    """
    _check_api_key(provider)
    model_str = _resolve_model(provider, model)

    if provider == "OpenAI":
        return _refine_openai(prompt_text, model_str)
    if provider == "Google":
        return _refine_google(prompt_text, model_str)
    return _refine_anthropic(prompt_text, model_str)


def _refine_openai(prompt_text: str, model_str: str) -> str:
    """Call OpenAI chat completions to refine a prompt.

    Args:
        prompt_text: The original prompt.
        model_str:   OpenAI model identifier.

    Returns:
        The assistant's response text.
    """
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_str,
        messages=[
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.4,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def _refine_anthropic(prompt_text: str, model_str: str) -> str:
    """Call Anthropic messages API to refine a prompt.

    Args:
        prompt_text: The original prompt.
        model_str:   Anthropic model identifier.

    Returns:
        The assistant's response text.
    """
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=model_str,
        system=REFINEMENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.4,
        max_tokens=2048,
    )
    return response.content[0].text


def _refine_google(prompt_text: str, model_str: str) -> str:
    """Call Google Gemini to refine a prompt.

    Args:
        prompt_text: The original prompt.
        model_str:   Gemini model identifier.

    Returns:
        The model's response text.
    """
    from google import genai

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=model_str,
        contents=f"{REFINEMENT_SYSTEM_PROMPT}\n\n{prompt_text}",
        config={
            "temperature": 0.4,
            "max_output_tokens": 2048,
        },
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# Token pricing
# ---------------------------------------------------------------------------

# Pricing per million tokens (USD).  Keys = model identifier.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    "gpt-4.1-mini":             {"input": 0.40,  "output": 1.60},
    "gpt-4.1":                  {"input": 2.00,  "output": 8.00},
    # Anthropic
    "claude-3-5-haiku-20241022":   {"input": 0.80,  "output": 4.00},
    "claude-3-5-sonnet-20241022":  {"input": 3.00,  "output": 15.00},
    "claude-3-7-sonnet-latest":    {"input": 3.00,  "output": 15.00},
    # Google
    "gemini-2.0-flash-lite":    {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash":         {"input": 0.10,  "output": 0.40},
}


def _count_tokens_openai(text: str, model: str) -> int:
    """Count tokens using tiktoken (offline, no API call).

    Args:
        text:  The prompt string to tokenize.
        model: OpenAI model identifier.

    Returns:
        Approximate token count.
    """
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def _count_tokens_anthropic(text: str, model: str) -> int:
    """Count tokens via the free Anthropic count_tokens endpoint.

    Args:
        text:  The prompt string.
        model: Anthropic model identifier.

    Returns:
        Token count from the API.
    """
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def _count_tokens_google(text: str, model: str) -> int:
    """Count tokens via Google genai count_tokens.

    Args:
        text:  The prompt string.
        model: Gemini model identifier.

    Returns:
        Token count from the API.
    """
    from google import genai

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = client.models.count_tokens(model=model, contents=text)
    return resp.total_tokens


def calculate_token_pricing(
    prompt_text: str,
) -> list[dict[str, Any]]:
    """Calculate token counts and costs for ALL models across all providers.

    Token counting is always available for all models:
      - OpenAI: uses tiktoken (offline, free).
      - Anthropic / Google: uses their native count_tokens API when the key
        is configured; falls back to tiktoken approximation otherwise.

    Args:
        prompt_text: The prompt to analyse.

    Returns:
        List of dicts, each with keys:
            provider, model, tokens, input_cost, output_cost, source
    """
    results: list[dict[str, Any]] = []

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))

    # Shared tiktoken fallback for approximate counts.
    def _tiktoken_approx(text: str) -> int:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))

    # --- OpenAI models (always exact via tiktoken) ---
    for model in OPENAI_MODELS:
        try:
            tokens = _count_tokens_openai(prompt_text, model)
            pricing = MODEL_PRICING[model]
            results.append({
                "provider": "OpenAI",
                "model": model,
                "tokens": tokens,
                "input_cost": round(tokens * pricing["input"] / 1_000_000, 6),
                "output_cost": round(tokens * pricing["output"] / 1_000_000, 6),
            })
        except Exception as exc:
            results.append({
                "provider": "OpenAI", "model": model, "tokens": 0,
                "input_cost": 0.0, "output_cost": 0.0, "error": str(exc),
            })

    # --- Anthropic models ---
    for model in ANTHROPIC_MODELS:
        try:
            if has_anthropic:
                tokens = _count_tokens_anthropic(prompt_text, model)
            else:
                tokens = _tiktoken_approx(prompt_text)
            pricing = MODEL_PRICING[model]
            results.append({
                "provider": "Anthropic",
                "model": model,
                "tokens": tokens,
                "input_cost": round(tokens * pricing["input"] / 1_000_000, 6),
                "output_cost": round(tokens * pricing["output"] / 1_000_000, 6),
                **({"source": "approx"} if not has_anthropic else {}),
            })
        except Exception as exc:
            results.append({
                "provider": "Anthropic", "model": model, "tokens": 0,
                "input_cost": 0.0, "output_cost": 0.0, "error": str(exc),
            })

    # --- Google models ---
    for model in GOOGLE_MODELS:
        try:
            if has_google:
                tokens = _count_tokens_google(prompt_text, model)
            else:
                tokens = _tiktoken_approx(prompt_text)
            pricing = MODEL_PRICING[model]
            results.append({
                "provider": "Google",
                "model": model,
                "tokens": tokens,
                "input_cost": round(tokens * pricing["input"] / 1_000_000, 6),
                "output_cost": round(tokens * pricing["output"] / 1_000_000, 6),
                **({"source": "approx"} if not has_google else {}),
            })
        except Exception as exc:
            results.append({
                "provider": "Google", "model": model, "tokens": 0,
                "input_cost": 0.0, "output_cost": 0.0, "error": str(exc),
            })

    return results
