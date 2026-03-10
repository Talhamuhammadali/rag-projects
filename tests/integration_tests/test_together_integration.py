"""Integration tests for together ai client — real API calls."""

import pytest
from dotenv import load_dotenv, find_dotenv

from utils.llm import ainvoke
from together._exceptions import APIError

load_dotenv(find_dotenv())


# --- Happy Path ---
@pytest.mark.asyncio
async def test_single_prompt_returns_valid_response():
    """Test that ainvoke returns a valid response for a simple prompt."""
    response = await ainvoke([{"role": "user", "content": "Say hello"}])
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0].message, "content")
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
    
    
@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Test that ainvoke can handle a multi-turn conversation."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! How can I assist you today?"},
        {"role": "user", "content": "What is my name?"}
    ]
    response = await ainvoke(messages)
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0].message, "content")
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
    assert "Alice".lower() in response.choices[0].message.content.lower()

@pytest.mark.asyncio
async def test_custom_model_selection():
    """Test that ainvoke can use a custom model."""
    response = await ainvoke(
        [{"role": "user", "content": "What is the capital of France?"}],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0].message, "content")
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
    assert "Paris".lower() in response.choices[0].message.content.lower()

# --- Edge Cases ---
@pytest.mark.asyncio
async def test_invalid_model_raises_error():
    """Test that ainvoke raises an API error for an invalid model."""
    with pytest.raises(APIError):
        await ainvoke(
            [{"role": "user", "content": "What is the capital of France?"}],
            model="fake-model/doesnt-exist"
        )

@pytest.mark.asyncio
async def test_empty_messages_list():
    """Test that ainvoke rejects an empty messages list."""
    with pytest.raises(ValueError, match="Messages list cannot be empty"):
        await ainvoke(messages=[], model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")