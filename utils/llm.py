"""LLM utilities module: together ai and others."""
from together import AsyncTogether

async def get_together_client():
    client = AsyncTogether()
    return client

async def ainvoke(messages: list[dict[str, str]]):
    client = await get_together_client()
    response = await client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=messages
    )
    return response