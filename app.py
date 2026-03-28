"""
PromptAnalyzer — Gradio app for evaluating LLM prompts using DeepEval.

Tabs:
  1. Analyze Prompt File  — upload/paste, token cost, INPUT metrics, OUTPUT metrics, cheaper alt
  2. Write New Prompt     — free-form editor with PROMPT/CROFT framework auto-detection
  3. Settings             — API key configuration

Evaluation logic      → evaluator.py
Prompt builder        → prompt_builder.py
Pricing config        → config/pricing.py
Suggestion engine     → services/suggestion_engine.py
CSS + HTML rendering  → styles.py

Run:
    uv run python app.py
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from config.pricing import (
    ALL_MODELS,
    ANTHROPIC_MODELS,
    GOOGLE_MODELS,
    MODEL_DISPLAY_NAMES,
    MODEL_PRICING,
    OPENAI_MODELS,
    PROVIDER_FOR_MODEL,
)
from evaluator import (
    generate_prompt_response,
    run_evaluation,
    run_output_evaluation,
)
from prompt_builder import (
    build_prompt,
    detect_framework,
    render_detection_badge,
    render_empty_badge,
)
from services.suggestion_engine import generate_cheaper_alternative
from styles import (
    APP_CSS,
    EMPTY_RESULTS_HTML,
    HEADER_HTML,
    build_comparison_html,
    build_cost_card_html,
    build_metric_cards_html,
    build_output_metric_cards_html,
    build_output_results_html,
    build_results_html,
)

ENV_FILE = Path(".env")

# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------


def _load_saved_keys() -> tuple[str, str, str]:
    load_dotenv(ENV_FILE, override=True)
    return (
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("ANTHROPIC_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
    )


def _persist_keys(openai_key: str, anthropic_key: str, google_key: str) -> str:
    openai_key = openai_key.strip()
    anthropic_key = anthropic_key.strip()
    google_key = google_key.strip()

    if not openai_key and not anthropic_key and not google_key:
        return "⚠ Please provide at least one API key."

    errors: list[str] = []
    if openai_key and not openai_key.startswith("sk-"):
        errors.append("OpenAI key should start with 'sk-'.")
    if anthropic_key and not anthropic_key.startswith("sk-ant-"):
        errors.append("Anthropic key should start with 'sk-ant-'.")
    if google_key and not google_key.startswith("AIza"):
        errors.append("Google key should start with 'AIza'.")
    if errors:
        return "⚠ Validation failed: " + " ".join(errors)

    lines: list[str] = []
    if openai_key:
        lines.append(f'OPENAI_API_KEY="{openai_key}"')
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        lines.append(f'ANTHROPIC_API_KEY="{anthropic_key}"')
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if google_key:
        lines.append(f'GOOGLE_API_KEY="{google_key}"')
        os.environ["GOOGLE_API_KEY"] = google_key

    ENV_FILE.write_text("\n".join(lines) + "\n")
    return "✓ API keys saved to .env and loaded into environment."


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".yaml", ".yml", ".json"}


def _load_prompt_text(uploaded_file: Any, pasted_text: str) -> tuple[str, str]:
    """Resolve prompt text from whichever input was used.

    Priority: uploaded file → pasted text.
    Returns: (text, error_message)
    """
    if uploaded_file is not None:
        try:
            path = Path(
                uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            )
            return path.read_text(encoding="utf-8", errors="replace"), ""
        except Exception as exc:
            return "", f"Could not read uploaded file: {exc}"

    if pasted_text and pasted_text.strip():
        return pasted_text.strip(), ""

    return "", "Please upload a file or paste content."


# ---------------------------------------------------------------------------
# Token pricing (inline, no separate tab)
# ---------------------------------------------------------------------------


def _count_tokens_single(text: str, model: str) -> int:
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def _pricing_for_model(text: str, model: str) -> dict[str, Any]:
    pricing = MODEL_PRICING[model]
    tokens = _count_tokens_single(text, model)
    return {
        "model": model,
        "provider": PROVIDER_FOR_MODEL[model],
        "tokens": tokens,
        "input_cost": round(tokens * pricing["input"] / 1_000_000, 6),
        "output_cost": round(tokens * pricing["output"] / 1_000_000, 6),
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyze_prompt(
    uploaded_file: Any,
    pasted_text: str,
    selected_model: str,
) -> tuple[str, str, str, str]:
    """Full analysis pipeline.

    Steps:
        1. Load prompt text
        2. Token count + pricing (cost card)
        3. INPUT evaluation  — score the prompt quality
        4. Generate LLM response
        5. OUTPUT evaluation — score the response quality
        6. Generate cheaper alternative + comparison

    Returns:
        (cost_html, input_quality_html, output_quality_html, comparison_html)
    """
    text, err = _load_prompt_text(uploaded_file, pasted_text)
    if err:
        error_html = f"<p style='color:#dc2626;font-weight:600;'>{err}</p>"
        return error_html, "", "", ""

    provider = PROVIDER_FOR_MODEL.get(selected_model, "OpenAI")

    # ── Step 1: Token cost card ────────────────────────────────────────────
    orig_pricing = _pricing_for_model(text, selected_model)
    cost_html = build_cost_card_html(
        model=selected_model,
        provider=provider,
        tokens=orig_pricing["tokens"],
        input_cost=orig_pricing["input_cost"],
        output_cost=orig_pricing["output_cost"],
    )

    # ── Step 2: INPUT evaluation (prompt quality) ──────────────────────────
    start = time.time()
    input_scores: dict = {}
    try:
        input_scores = run_evaluation(text, provider, selected_model)
    except Exception as exc:
        tb = traceback.format_exc()
        cost_html += (
            f"<pre style='color:#dc2626;font-size:12px;'>"
            f"Input evaluation failed:\n{exc}\n\n{tb}</pre>"
        )
    input_elapsed = time.time() - start

    input_quality_html = (
        build_results_html(input_scores, provider, input_elapsed, selected_model)
        if input_scores else ""
    )

    # ── Step 3: Generate actual LLM response ───────────────────────────────
    response_text = ""
    try:
        response_text = generate_prompt_response(text, provider, selected_model)
    except Exception as exc:
        response_text = f"[Could not generate response: {exc}]"

    # ── Step 4: OUTPUT evaluation (response quality) ───────────────────────
    start = time.time()
    output_scores: dict = {}
    try:
        output_scores = run_output_evaluation(text, response_text, provider, selected_model)
    except Exception as exc:
        tb = traceback.format_exc()
        output_scores = {}
        cost_html += (
            f"<pre style='color:#dc2626;font-size:12px;'>"
            f"Output evaluation failed:\n{exc}\n\n{tb}</pre>"
        )
    output_elapsed = time.time() - start

    output_quality_html = (
        build_output_results_html(
            output_scores,
            provider,
            output_elapsed,
            selected_model,
            actual_output_preview=response_text,
        )
        if output_scores else ""
    )

    # ── Step 5: Cheaper alternative + comparison ───────────────────────────
    try:
        alt_text, reason = generate_cheaper_alternative(text, provider, selected_model)
    except Exception as exc:
        alt_text, reason = text, f"Could not generate alternative: {exc}"

    alt_pricing = _pricing_for_model(alt_text, selected_model)
    alt_scores: dict = {}
    try:
        alt_scores = run_evaluation(alt_text, provider, selected_model)
    except Exception:
        pass

    comparison_html = build_comparison_html(
        original_text=text,
        original_tokens=orig_pricing["tokens"],
        original_input_cost=orig_pricing["input_cost"],
        original_output_cost=orig_pricing["output_cost"],
        original_scores=input_scores,
        alt_text=alt_text,
        alt_tokens=alt_pricing["tokens"],
        alt_input_cost=alt_pricing["input_cost"],
        alt_output_cost=alt_pricing["output_cost"],
        alt_scores=alt_scores,
        reason=reason,
    )

    return cost_html, input_quality_html, output_quality_html, comparison_html


# ---------------------------------------------------------------------------
# Write New Prompt handlers
# ---------------------------------------------------------------------------


def evaluate_new_prompt(
    title: str, text: str, category: str, provider: str, model: str,
) -> str:
    if not text.strip():
        return "<p style='color:#dc2626;font-weight:600;'>Please enter a prompt to evaluate.</p>"

    start = time.time()
    try:
        scores = run_evaluation(text.strip(), provider, model)
    except Exception as exc:
        tb = traceback.format_exc()
        return f"<pre style='color:#dc2626;'>Evaluation failed:\n{exc}\n\n{tb}</pre>"

    elapsed = time.time() - start
    return build_results_html(scores, provider, elapsed, model)


def save_prompt_as_txt(title: str, text: str) -> tuple[str | None, str]:
    """Write prompt to a .txt file and return (path, status_message)."""
    if not text.strip():
        return None, "⚠ Prompt text is empty."
    safe_name = "".join(
        c if c.isalnum() or c in "_ -" else "_"
        for c in (title.strip() or "prompt")
    )[:60] or "prompt"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix=f"{safe_name}_",
        delete=False, encoding="utf-8",
    )
    tmp.write(text.strip())
    tmp.close()
    return tmp.name, "✓ Ready to download."


def _update_model_choices(provider: str):
    if provider == "OpenAI":
        return gr.update(choices=OPENAI_MODELS, value=OPENAI_MODELS[0])
    if provider == "Google":
        return gr.update(choices=GOOGLE_MODELS, value=GOOGLE_MODELS[0])
    return gr.update(choices=ANTHROPIC_MODELS, value=ANTHROPIC_MODELS[0])


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_ANALYZER_CHOICES = [MODEL_DISPLAY_NAMES[m] for m in ALL_MODELS]
_DISPLAY_TO_MODEL = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}

# Highlighted section header HTML
_INPUT_HEADER = (
    '<div style="display:flex;align-items:center;gap:12px;padding:14px 20px;'
    'border-radius:12px;background:linear-gradient(135deg,#ede9fe,#dbeafe);'
    'border:1px solid #c4b5fd;margin-bottom:16px;">'
    '<span style="font-size:22px;">📥</span>'
    '<div>'
    '<div style="font-size:15px;font-weight:800;color:#4c1d95;">Input Evaluator</div>'
    '<div style="font-size:12px;color:#6b7280;margin-top:2px;">'
    'Scores the <strong>prompt itself</strong> — Clarity · Specificity · Completeness · Coherence · Safety'
    '</div></div></div>'
)
_OUTPUT_HEADER = (
    '<div style="display:flex;align-items:center;gap:12px;padding:14px 20px;'
    'border-radius:12px;background:linear-gradient(135deg,#ccfbf1,#cffafe);'
    'border:1px solid #67e8f9;margin-bottom:16px;">'
    '<span style="font-size:22px;">📤</span>'
    '<div>'
    '<div style="font-size:15px;font-weight:800;color:#164e63;">Output Evaluator</div>'
    '<div style="font-size:12px;color:#6b7280;margin-top:2px;">'
    'Scores the <strong>LLM\'s response</strong> — Answer Relevancy · Hallucination · Bias · Toxicity · Conciseness · Context Precision'
    '</div></div></div>'
)


def build_app() -> gr.Blocks:
    from prompt_library import CATEGORIES as PROMPT_CATEGORIES

    openai_key_init, anthropic_key_init, google_key_init = _load_saved_keys()

    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="PromptAnalyzer") as app:

        gr.HTML(HEADER_HTML)

        # ── Metric definitions (two accordions: input + output) ───────────
        with gr.Accordion("📥 Input Metrics — what do they mean?  (click to expand)", open=False):
            gr.HTML(build_metric_cards_html())

        with gr.Accordion("📤 Output Metrics — what do they mean?  (click to expand)", open=False):
            gr.HTML(build_output_metric_cards_html())

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════
            # TAB 1: Analyze Prompt File
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Analyze Prompt File", id="tab-analyze"):
                gr.Markdown(
                    "Upload a file or paste content, pick a model, and click **Analyze**. "
                    "You'll get token cost, prompt quality scores **(Input Evaluator)**, "
                    "response quality scores **(Output Evaluator)**, and a cheaper alternative."
                )

                with gr.Tabs():
                    with gr.Tab("Upload File"):
                        file_upload = gr.File(
                            label="Drag & drop or click to upload (.md .txt .py .yaml .json)",
                            file_types=[".md", ".txt", ".py", ".yaml", ".yml", ".json"],
                            file_count="single",
                        )
                    with gr.Tab("Paste Content"):
                        paste_input = gr.Textbox(
                            label="Paste prompt / agent / skill content here",
                            placeholder="Paste any text content…",
                            lines=10,
                        )

                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=_ANALYZER_CHOICES,
                        value=_ANALYZER_CHOICES[0],
                        label="Model — controls tokenizer, pricing, and which AI judge runs evaluations",
                        scale=3,
                        interactive=True,
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary", scale=1)

                # Cost card
                analyze_cost_html = gr.HTML(value="", elem_id="analyze-cost-panel")

                # Input evaluator section
                gr.HTML(_INPUT_HEADER)
                analyze_input_html = gr.HTML(value="", elem_id="analyze-input-panel")

                # Output evaluator section
                gr.HTML(_OUTPUT_HEADER)
                analyze_output_html = gr.HTML(value="", elem_id="analyze-output-panel")

                # Comparison panel
                analyze_comparison_html = gr.HTML(value="", elem_id="analyze-comparison-panel")

            # ═══════════════════════════════════════════════════════════════
            # TAB 2: Write New Prompt
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Write New Prompt", id="tab-new-prompt"):
                gr.Markdown(
                    "Describe your goal — the **PROMPT** or **CROFT** framework is "
                    "auto-detected. Click **Build Structured Prompt**, then evaluate it."
                )

                with gr.Group(elem_classes=["prompt-builder-section"]):
                    gr.HTML(
                        '<div class="section-title">'
                        "🧠 Prompt Builder — 4D Principles "
                        "(Delegation · Description · Discernment · Diligence)"
                        "</div>"
                    )
                    new_idea = gr.Textbox(
                        label="What do you want to accomplish?",
                        placeholder=(
                            "e.g. 'Write a Python function that validates email addresses' "
                            "or 'Explain the difference between REST and GraphQL APIs'"
                        ),
                        lines=3,
                        info="Framework is auto-detected — no LLM call needed.",
                    )
                    new_framework_badge = gr.HTML(value=render_empty_badge())
                    with gr.Row():
                        new_framework_radio = gr.Radio(
                            choices=["PROMPT", "CROFT"],
                            value="PROMPT",
                            label="Framework Override (auto-selected above)",
                            elem_classes=["framework-radio"],
                            info=(
                                "PROMPT — Persona · Request · Output · Method · Purpose · Task  |  "
                                "CROFT — Context · Role · Objective · Format · Tone"
                            ),
                        )
                        new_build_btn = gr.Button(
                            "✨ Build Structured Prompt", variant="primary", scale=0,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        new_title = gr.Textbox(
                            label="Prompt Title", placeholder="My awesome prompt…", max_lines=1,
                        )
                        new_text = gr.Textbox(
                            label="Prompt Text",
                            placeholder="Your structured prompt appears here…",
                            lines=12,
                        )
                        with gr.Row():
                            new_category = gr.Dropdown(
                                choices=PROMPT_CATEGORIES, value="cursor prompt", label="Category",
                            )
                            new_provider = gr.Radio(
                                choices=["OpenAI", "Anthropic", "Google"],
                                value="OpenAI", label="Provider",
                            )
                        new_model = gr.Dropdown(
                            choices=OPENAI_MODELS, value=OPENAI_MODELS[0], label="Model",
                        )
                        with gr.Row():
                            new_evaluate_btn = gr.Button("Evaluate Prompt", variant="primary")
                            new_save_btn = gr.Button("💾 Save as .txt", variant="secondary")
                        new_status = gr.Markdown("")
                        new_download = gr.File(
                            label="Download your prompt", visible=False, interactive=False,
                        )

                    with gr.Column(scale=1):
                        gr.HTML(_INPUT_HEADER)
                        new_results_html = gr.HTML(
                            value=EMPTY_RESULTS_HTML, elem_id="new-results-panel",
                        )

            # ═══════════════════════════════════════════════════════════════
            # TAB 3: Settings
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("Settings", id="tab-settings"):
                gr.Markdown("### API Key Configuration")
                gr.Markdown(
                    "Keys are validated, saved to `.env`, and loaded automatically on startup."
                )
                openai_key_input = gr.Textbox(
                    label="OpenAI API Key", placeholder="sk-…",
                    type="password", value=openai_key_init,
                )
                anthropic_key_input = gr.Textbox(
                    label="Anthropic API Key", placeholder="sk-ant-…",
                    type="password", value=anthropic_key_init,
                )
                google_key_input = gr.Textbox(
                    label="Google API Key", placeholder="AIza…",
                    type="password", value=google_key_init,
                )
                save_keys_btn = gr.Button("Save API Keys", variant="primary")
                keys_status = gr.Markdown("")

        # ══════════════════════════════════════════════════════════════════
        # Event wiring
        # ══════════════════════════════════════════════════════════════════

        # Settings
        save_keys_btn.click(
            fn=_persist_keys,
            inputs=[openai_key_input, anthropic_key_input, google_key_input],
            outputs=[keys_status],
        )

        # Analyze tab
        def _run_analyze(uploaded_file, pasted_text, display_name):
            model = (
                _DISPLAY_TO_MODEL.get(display_name)
                or (display_name if display_name in ALL_MODELS else ALL_MODELS[0])
            )
            return analyze_prompt(uploaded_file, pasted_text, model)

        analyze_btn.click(
            fn=_run_analyze,
            inputs=[file_upload, paste_input, model_selector],
            outputs=[
                analyze_cost_html,
                analyze_input_html,
                analyze_output_html,
                analyze_comparison_html,
            ],
            api_name="analyze_prompt_api",
        )

        # Write New Prompt: framework detection
        def _detect_and_update(idea: str) -> tuple[str, str]:
            fw, reason, confidence = detect_framework(idea)
            return render_detection_badge(fw, reason, confidence), fw

        new_idea.change(
            fn=_detect_and_update,
            inputs=[new_idea],
            outputs=[new_framework_badge, new_framework_radio],
        )

        new_build_btn.click(
            fn=lambda idea, fw: build_prompt(idea, fw),  # type: ignore[arg-type]
            inputs=[new_idea, new_framework_radio],
            outputs=[new_text],
        )

        new_provider.change(
            fn=_update_model_choices,
            inputs=[new_provider],
            outputs=[new_model],
        )

        new_evaluate_btn.click(
            fn=evaluate_new_prompt,
            inputs=[new_title, new_text, new_category, new_provider, new_model],
            outputs=[new_results_html],
        )

        # Save as .txt
        def _save_and_show(title: str, text: str):
            path, msg = save_prompt_as_txt(title, text)
            visible = path is not None
            return gr.update(visible=visible, value=path), msg

        new_save_btn.click(
            fn=_save_and_show,
            inputs=[new_title, new_text],
            outputs=[new_download, new_status],
        )

    app._theme = theme
    app._css = APP_CSS
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=app._theme,
        css=app._css,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
