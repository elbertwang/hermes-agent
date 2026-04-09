"""Tests for Google Vertex AI client builder in anthropic_adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestBuildVertexClient:
    """Tests for build_vertex_client()."""

    def test_raises_when_sdk_missing(self):
        """Should raise ImportError when anthropic SDK is not installed."""
        with patch("agent.anthropic_adapter._anthropic_sdk", None):
            from agent.anthropic_adapter import build_vertex_client

            with pytest.raises(ImportError, match="anthropic"):
                build_vertex_client("my-project", "us-east5")

    def test_raises_when_vertex_extra_missing(self):
        """Should raise ImportError when anthropic[vertex] extra is not installed."""
        mock_sdk = MagicMock(spec=[])  # No AnthropicVertex attribute
        with patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk):
            from agent.anthropic_adapter import build_vertex_client

            with pytest.raises(ImportError, match="vertex"):
                build_vertex_client("my-project", "us-east5")

    def test_creates_client_with_correct_params(self):
        """Should create AnthropicVertex with project_id, region, timeout, and betas."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.AnthropicVertex.return_value = mock_client

        with patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk):
            from agent.anthropic_adapter import build_vertex_client

            result = build_vertex_client("my-project", "us-central1")

        assert result is mock_client
        mock_sdk.AnthropicVertex.assert_called_once()
        call_kwargs = mock_sdk.AnthropicVertex.call_args[1]
        assert call_kwargs["project_id"] == "my-project"
        assert call_kwargs["region"] == "us-central1"
        assert "timeout" in call_kwargs
        assert "default_headers" in call_kwargs
        assert "anthropic-beta" in call_kwargs["default_headers"]

    def test_default_region(self):
        """Should default to us-east5 when region not specified."""
        mock_sdk = MagicMock()
        with patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk):
            from agent.anthropic_adapter import build_vertex_client

            build_vertex_client("my-project")

        call_kwargs = mock_sdk.AnthropicVertex.call_args[1]
        assert call_kwargs["region"] == "us-east5"

    def test_no_oauth_betas(self):
        """Should NOT include OAuth-only betas (those are for Anthropic direct)."""
        mock_sdk = MagicMock()
        with patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk):
            from agent.anthropic_adapter import build_vertex_client, _OAUTH_ONLY_BETAS

            build_vertex_client("my-project", "us-east5")

        call_kwargs = mock_sdk.AnthropicVertex.call_args[1]
        beta_header = call_kwargs["default_headers"]["anthropic-beta"]
        for oauth_beta in _OAUTH_ONLY_BETAS:
            assert oauth_beta not in beta_header

    def test_no_api_key_param(self):
        """Vertex client should not have api_key or auth_token params."""
        mock_sdk = MagicMock()
        with patch("agent.anthropic_adapter._anthropic_sdk", mock_sdk):
            from agent.anthropic_adapter import build_vertex_client

            build_vertex_client("my-project", "us-east5")

        call_kwargs = mock_sdk.AnthropicVertex.call_args[1]
        assert "api_key" not in call_kwargs
        assert "auth_token" not in call_kwargs
