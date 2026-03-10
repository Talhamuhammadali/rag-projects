"""Unit tests for together ai client."""

import pytest
import httpx
from unittest.mock import patch, AsyncMock

from together._exceptions import NotFoundError, APIError
from utils.llm import ainvoke


@pytest.mark.asyncio
@patch("utils.llm.get_together_client")
async def test_ainvoke_raises_on_api_error(mock_get_client):
    """When the API returns a 404 (bad model), ainvoke should re-raise as APIError."""
    fake_client = AsyncMock()

    response = httpx.Response(
        status_code=404,
        request=httpx.Request("POST", "https://api.together.ai/v1/chat/completions"),
    )
    not_found_error = NotFoundError(message="Model not found", response=response, body={"error": "model_not_found"})
    
    fake_client.chat.completions.create.side_effect = not_found_error
    mock_get_client.return_value = fake_client

    # Assert ainvoke propagates the error
    with pytest.raises(APIError):
        await ainvoke([{"role": "user", "content": "hello"}])
