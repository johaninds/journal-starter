"""Task 4: Implement analyze_journal_entry using any OpenAI-compatible API.

This project mandates the OpenAI Python SDK, which works with:
  - GitHub Models (default, free, no credit card required)
  - OpenAI proper
  - Azure OpenAI
  - Groq, Together, OpenRouter, Fireworks, DeepInfra
  - Ollama, LM Studio, vLLM (local)
  - Anthropic via their OpenAI-compat endpoint

Set OPENAI_API_KEY, and optionally OPENAI_BASE_URL and OPENAI_MODEL
in your .env file. Settings are loaded by ``api.config.Settings``.
"""

import json

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from api.config import get_settings


def _default_client() -> AsyncOpenAI:
    """Construct the real OpenAI client from application settings.

    Called lazily from ``analyze_journal_entry`` so tests can inject a
    ``MockAsyncOpenAI`` without ever triggering this code path.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Analyze a journal entry using an OpenAI-compatible LLM.

    Args:
        entry_id: ID of the entry being analyzed (pass through to the result).
        entry_text: Combined work + struggle + intention text.
        client: OpenAI client. If None, a default one is constructed from
            application settings. Tests pass in a MockAsyncOpenAI here; production code
            in the router calls this with no ``client`` argument.

    Returns:
        A dict matching AnalysisResponse:
            {
                "entry_id":  str,
                "sentiment": str,   # "positive" | "negative" | "neutral"
                "summary":   str,
                "topics":    list[str],
            }

    Task 4 implementation:
      1. If ``client is None``, call ``_default_client()`` to construct one.
      2. Build a messages list that includes ``entry_text`` somewhere
         (the unit tests check that the entry text reaches the LLM).
      3. Call ``client.chat.completions.create(...)`` with a model name
         (use ``get_settings().openai_model`` — defaults to "gpt-4o-mini").
      4. Parse the assistant's JSON response with ``json.loads()``.
      5. Return a dict with ``entry_id``, ``sentiment``, ``summary``, ``topics``.
    """
    should_close_client = client is None
    if client is None:
        client = _default_client()

    try:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are a journal analysis assistant. Analyze the provided entry text "
                    "and return only valid JSON with keys: sentiment, summary, topics. "
                    "Sentiment must be one of positive, negative, or neutral. "
                    "Topics should be a list of short strings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the following journal entry and return JSON only:\n\n{entry_text}"
                ),
            },
        ]

        response = await client.chat.completions.create(
            model=get_settings().openai_model,
            messages=messages,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM response content is empty")
        analysis = json.loads(content)

        return {
            "entry_id": entry_id,
            "sentiment": analysis["sentiment"],
            "summary": analysis["summary"],
            "topics": analysis["topics"],
        }
    finally:
        if should_close_client:
            await client.close()
