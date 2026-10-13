"""Unit tests for src/common/llm_client.py with mocking."""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.common.llm_client import (
    LLMClient,
    LLMJSONParseError,
    MaxLLMCallsExceededError,
)


@pytest.fixture
def mock_cache_client(tmp_path):
    """Fixture providing an LLMClient pointing to a temporary cache directory."""
    return LLMClient(
        api_key="test_mock_key",
        cache_dir=tmp_path / "cache",
        max_calls_per_run=3,
    )


def test_cache_hit_makes_zero_api_calls(mock_cache_client):
    # Manually populate cache file
    cache_path = mock_cache_client._get_json_cache_path(
        prompt="Test prompt",
        schema=None,
        prompt_version="test_v1",
    )
    test_data = {"status": "cached_success", "answer": 42}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    # Mock the internal raw client to verify it is NEVER called
    mock_raw_client = MagicMock()
    mock_cache_client._client = mock_raw_client

    result = mock_cache_client.generate_json(
        prompt="Test prompt",
        schema=None,
        prompt_version="test_v1",
        use_cache=True,
    )

    assert result == test_data
    assert mock_cache_client.call_count == 0
    mock_raw_client.models.generate_content.assert_not_called()


def test_bad_json_retries_and_raises_typed_error(mock_cache_client):
    mock_raw_client = MagicMock()
    # Return invalid JSON string
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON at all {"
    mock_raw_client.models.generate_content.return_value = mock_response
    mock_cache_client._client = mock_raw_client

    with pytest.raises(LLMJSONParseError) as exc_info:
        mock_cache_client.generate_json(
            prompt="Prompt expecting JSON",
            prompt_version="test_v2",
            use_cache=False,
            max_retries=1,
        )

    assert "Model returned invalid JSON" in str(exc_info.value)
    # 1 initial attempt + 1 retry = 2 calls
    assert mock_cache_client.call_count == 2
    assert mock_raw_client.models.generate_content.call_count == 2


def test_call_guard_raises_when_exceeded(mock_cache_client):
    mock_cache_client.call_count = 3  # Set to max limit (3)

    with pytest.raises(MaxLLMCallsExceededError) as exc_info:
        mock_cache_client.generate_json(
            prompt="Another prompt",
            prompt_version="test_v3",
            use_cache=False,
        )

    assert "Call guard exceeded" in str(exc_info.value)


def test_successful_generate_json_and_caching(mock_cache_client):
    mock_raw_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"remedy": "Tulsi tea", "safe": true}'
    mock_raw_client.models.generate_content.return_value = mock_response
    mock_cache_client._client = mock_raw_client

    result = mock_cache_client.generate_json(
        prompt="Describe Tulsi",
        prompt_version="test_v4",
        use_cache=True,
    )

    assert result == {"remedy": "Tulsi tea", "safe": True}
    assert mock_cache_client.call_count == 1

    # Second call should read from cache with zero API calls
    mock_cache_client.reset_call_count()
    result_cached = mock_cache_client.generate_json(
        prompt="Describe Tulsi",
        prompt_version="test_v4",
        use_cache=True,
    )
    assert result_cached == result
    assert mock_cache_client.call_count == 0
