"""LLM utilities module: together ai and others."""
from typing import Optional
from together import AsyncTogether, APIError

async def get_together_client():
    """Create and return an AsyncTogether client. Lazy-loaded to ensure env vars are available."""
    client = AsyncTogether()
    return client

async def ainvoke(
    messages: list[dict[str, str]],
    model: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """Send messages to a Together AI model and return the chat completion response."""
    if not messages:
        raise ValueError("Messages list cannot be empty.")
    client = await get_together_client()
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    try:
        response = await client.chat.completions.create(**kwargs)
    except APIError as e:
        print("Error during Together API call:", e)
        raise e
    return response



# TODO: add openai compatibile invoke for together ai