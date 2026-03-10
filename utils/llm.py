"""LLM utilities module: together ai and others."""
from typing import Optional
from together import AsyncTogether

async def get_together_client():
    """Create and return an AsyncTogether client. Lazy-loaded to ensure env vars are available."""
    client = AsyncTogether()
    return client

async def ainvoke(
    messages: list[dict[str, str]], model: Optional[str] = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
) -> dict:
    """Send messages to a Together AI model and return the chat completion response."""
    client = await get_together_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response