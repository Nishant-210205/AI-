"""
Tests for Qdrant Cloud URL validation, API key sanitization, whitespace handling,
and sanitized connection error handling (Requirement 10).
"""

import pytest
import os
from unittest.mock import MagicMock, patch
from vector_store import (
    validate_and_sanitize_qdrant_url,
    sanitize_qdrant_api_key,
    sanitize_secret_text,
    is_placeholder_qdrant_url,
    QdrantCloudVectorStore,
    ResilientVectorStore,
    get_vector_store,
)
from llm_service import get_secret


def test_validate_plain_https_url():
    url = "https://bc55f8d5-f950-414a-9251-97c65f673774.us-east-1-1.aws.cloud.qdrant.io"
    assert validate_and_sanitize_qdrant_url(url) == url

    # Port included
    url_port = "https://bc55f8d5-f950-414a-9251-97c65f673774.us-east-1-1.aws.cloud.qdrant.io:6333"
    assert validate_and_sanitize_qdrant_url(url_port) == url_port

    # In-memory test token
    assert validate_and_sanitize_qdrant_url(":memory:") == ":memory:"


def test_validate_markdown_link_url():
    # Markdown link with URL as text
    md_1 = "[https://bc55f8d5.aws.cloud.qdrant.io](https://bc55f8d5.aws.cloud.qdrant.io)"
    assert validate_and_sanitize_qdrant_url(md_1) == "https://bc55f8d5.aws.cloud.qdrant.io"

    # Markdown link with custom text
    md_2 = "[My Qdrant Cluster](https://bc55f8d5.aws.cloud.qdrant.io/)"
    assert validate_and_sanitize_qdrant_url(md_2) == "https://bc55f8d5.aws.cloud.qdrant.io"

    # Bracketed URL
    md_3 = "[https://bc55f8d5.aws.cloud.qdrant.io]"
    assert validate_and_sanitize_qdrant_url(md_3) == "https://bc55f8d5.aws.cloud.qdrant.io"

    # Angle brackets
    md_4 = "<https://bc55f8d5.aws.cloud.qdrant.io>"
    assert validate_and_sanitize_qdrant_url(md_4) == "https://bc55f8d5.aws.cloud.qdrant.io"


def test_validate_url_whitespace_and_quotes():
    raw_with_spaces = "   https://bc55f8d5.aws.cloud.qdrant.io   \r\n"
    assert validate_and_sanitize_qdrant_url(raw_with_spaces) == "https://bc55f8d5.aws.cloud.qdrant.io"

    quoted_double = '"https://bc55f8d5.aws.cloud.qdrant.io"'
    assert validate_and_sanitize_qdrant_url(quoted_double) == "https://bc55f8d5.aws.cloud.qdrant.io"

    quoted_single = "'https://bc55f8d5.aws.cloud.qdrant.io/'"
    assert validate_and_sanitize_qdrant_url(quoted_single) == "https://bc55f8d5.aws.cloud.qdrant.io"


def test_validate_url_rejections():
    # Non-HTTPS scheme
    with pytest.raises(ValueError, match="Invalid QDRANT_URL scheme"):
        validate_and_sanitize_qdrant_url("ftp://qdrant.example.com")

    # Missing host
    with pytest.raises(ValueError, match="Missing host or domain name"):
        validate_and_sanitize_qdrant_url("https://")

    # Malformed text
    with pytest.raises(ValueError, match="Invalid QDRANT_URL"):
        validate_and_sanitize_qdrant_url("not_a_valid_url")


def test_sanitize_api_key_whitespace_and_newlines():
    # Normal key
    assert sanitize_qdrant_api_key("th0-valid-api-key") == "th0-valid-api-key"

    # Surrounding whitespace and carriage returns
    assert sanitize_qdrant_api_key("  th0-valid-api-key  \r\n") == "th0-valid-api-key"

    # Quoted key
    assert sanitize_qdrant_api_key('"th0-valid-api-key"') == "th0-valid-api-key"

    # Key with accidental newline in middle
    assert sanitize_qdrant_api_key("th0-valid-\napi-key") == "th0-valid-api-key"

    # Empty and None
    assert sanitize_qdrant_api_key(None) is None
    assert sanitize_qdrant_api_key("") is None
    assert sanitize_qdrant_api_key("   \r\n") is None

    # Non-ASCII rejection
    with pytest.raises(ValueError, match="non-ASCII"):
        sanitize_qdrant_api_key("key_with_unicode_\u20ac")


def test_sanitize_secret_text_leak_prevention():
    secret_key = "th0-super-secret-token-xyz"
    err = f"Failed to connect using api_key={secret_key} to remote host."
    sanitized = sanitize_secret_text(err, secret_key)
    assert secret_key not in sanitized
    assert "***REDACTED***" in sanitized

    # LocalProtocolError illegal header text
    protocol_err = f"Illegal header value b'{secret_key}\\r\\n'"
    sanitized_proto = sanitize_secret_text(protocol_err, secret_key)
    assert secret_key not in sanitized_proto
    assert "whitespace or newline in secret" in sanitized_proto


def test_qdrant_vector_store_safe_initialization_and_whitespace():
    # Store should automatically strip whitespace and quotes during __init__
    store = QdrantCloudVectorStore(
        url="  https://bc55f8d5.aws.cloud.qdrant.io/  \r\n",
        api_key='  "th0-my-test-key"\n  ',
        collection_name="  test_collection  "
    )
    assert store.url == "https://bc55f8d5.aws.cloud.qdrant.io"
    assert store.api_key == "th0-my-test-key"
    assert store.collection_name == "test_collection"


def test_qdrant_get_stats_on_connection_error_returns_clean_dict():
    # An unreachable URL should return error dict rather than raising unhandled crash
    store = QdrantCloudVectorStore(
        url="https://invalid-nonexistent-cluster-12345.aws.cloud.qdrant.io",
        api_key="th0-dummy-key"
    )

    stats = store.get_stats()
    assert stats["provider"] == "Qdrant Cloud (Connection Error)"
    assert stats["total_chunks"] == 0
    assert "error" in stats
    # Crucial security check: api_key must NOT be exposed in stats["error"]
    assert "th0-dummy-key" not in stats["error"]


def test_qdrant_ensure_collection_sanitizes_protocol_error():
    store = QdrantCloudVectorStore(
        url="https://bc55f8d5.aws.cloud.qdrant.io",
        api_key="th0-secret-key"
    )

    # Mock client.get_collections() to simulate ResponseHandlingException
    from qdrant_client.http.exceptions import ResponseHandlingException
    fake_client = MagicMock()
    fake_client.get_collections.side_effect = ResponseHandlingException("Illegal header value b'th0-secret-key'")
    store._client = fake_client

    with pytest.raises(ConnectionError) as exc_info:
        store._ensure_collection()

    err_text = str(exc_info.value)
    assert "th0-secret-key" not in err_text
    assert "HTTP protocol/header formatting error" in err_text


def test_get_secret_whitespace_and_case_insensitive(monkeypatch):
    monkeypatch.setenv("TEST_QDRANT_URL", '  "https://my-cluster.qdrant.io" \r\n ')
    val = get_secret("TEST_QDRANT_URL")
    assert val == "https://my-cluster.qdrant.io"

    # Case insensitive
    val_lower = get_secret("test_qdrant_url")
    assert val_lower == "https://my-cluster.qdrant.io"


def test_is_placeholder_qdrant_url():
    assert is_placeholder_qdrant_url("https://your-cluster-id.region.aws.cloud.qdrant.io:6333") is True
    assert is_placeholder_qdrant_url("https://your-cloud-qdrant.io") is True
    assert is_placeholder_qdrant_url("") is True
    assert is_placeholder_qdrant_url(None) is True
    assert is_placeholder_qdrant_url("https://bc55f8d5-f950-414a-9251-97c65f673774.us-east-1-1.aws.cloud.qdrant.io") is False


def test_get_vector_store_placeholder_fallback_to_chroma(monkeypatch):
    monkeypatch.setenv("VECTOR_PROVIDER", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "https://your-cluster-id.region.aws.cloud.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "your-real-qdrant-api-key-here")

    store = get_vector_store()
    stats = store.get_stats()
    assert "ChromaDB" in stats["provider"]
    assert "error" not in stats


def test_resilient_vector_store_unreachable_fallback(monkeypatch):
    monkeypatch.setenv("VECTOR_PROVIDER", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "https://unreachable-cluster-test-xyz.aws.cloud.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "some-key")

    store = get_vector_store()
    stats = store.get_stats()
    assert stats["fallback_active"] is True
    assert "ChromaDB" in stats["provider"]
    assert "error" not in stats


