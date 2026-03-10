"""LLM utilities module: together ai and others."""
from typing import Optional
from together import AsyncTogether, APIError

async def get_together_client():
    """Create and return an AsyncTogether client. Lazy-loaded to ensure env vars are available."""
    client = AsyncTogether()
    return client

async def ainvoke(
    messages: list[dict[str, str]], model: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
) -> dict:
    """Send messages to a Together AI model and return the chat completion response."""
    if not messages:
        raise ValueError("Messages list cannot be empty.")
    client = await get_together_client()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
    except APIError as e:
        print("Error during Together API call:", e)
        raise e
    return response