"""
LLM abstraction — generate lesson phrases via OpenAI / Sarvam / fallback.

Usage:
    from llm import generate_phrase
    phrase = generate_phrase(prompt, target_language="mr")
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower().strip()


def _get_openai_client():
    """Lazy-import so we don't blow up if openai isn't installed."""
    try:
        import openai  # noqa: F811
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    return openai.OpenAI(api_key=api_key)


def _generate_openai(prompt: str) -> str:
    """Call OpenAI ChatCompletion (gpt-4o-mini by default, cheap & fast)."""
    client = _get_openai_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a multilingual language-teaching assistant. "
                    "Return ONLY the requested phrase in the target language. "
                    "No explanations, no transliterations, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    return response.choices[0].message.content.strip()


def _generate_sarvam(prompt: str) -> str:
    """Call Sarvam AI API."""
    import requests

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in .env")

    base_url = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
    model = os.getenv("SARVAM_MODEL", "sarvam-2b-v0.5")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a multilingual language-teaching assistant. "
                        "Return ONLY the requested phrase in the target language. "
                        "No explanations, no transliterations, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 256,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _generate_fallback(prompt: str) -> str:
    """
    Fallback: extract the 'Intent:' line from the prompt and return it as-is.
    Not ideal, but means the system still works without an API key.
    """
    logger.warning("LLM fallback active — returning template text as phrase")
    for line in prompt.splitlines():
        if line.startswith("Intent:"):
            return line.removeprefix("Intent:").strip()
    # Last resort: return the whole prompt
    return prompt.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_phrase(prompt: str, target_language: str | None = None) -> str:
    """
    Generate a lesson phrase using the configured LLM provider.

    Parameters
    ----------
    prompt : str
        The full prompt built by ``build_lesson_prompt``.
    target_language : str | None
        Optional language code (for logging / future routing).

    Returns
    -------
    str
        The generated phrase in the target language.
    """
    provider = LLM_PROVIDER
    logger.info("Generating phrase via LLM provider=%s lang=%s", provider, target_language)

    try:
        if provider == "openai":
            return _generate_openai(prompt)
        elif provider == "sarvam":
            return _generate_sarvam(prompt)
        else:
            logger.warning("Unknown LLM_PROVIDER=%s, using fallback", provider)
            return _generate_fallback(prompt)
    except Exception as exc:
        logger.exception("LLM call failed (provider=%s), falling back to template text: %s", provider, exc)
        return _generate_fallback(prompt)
