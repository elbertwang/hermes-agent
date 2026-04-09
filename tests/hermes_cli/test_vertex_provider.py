"""Tests for Vertex AI provider registration and resolution."""

import os
from unittest.mock import patch

import pytest


class TestVertexProviderRegistry:
    """Test that vertex is properly registered in PROVIDER_REGISTRY."""

    def test_vertex_in_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "vertex" in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["vertex"]
        assert pconfig.name == "Google Vertex AI (Claude)"
        assert pconfig.auth_type == "gcloud_adc"
        assert "ANTHROPIC_VERTEX_PROJECT_ID" in pconfig.api_key_env_vars


class TestVertexProviderAliases:
    """Test provider alias resolution for vertex."""

    def test_vertex_ai_alias(self):
        from hermes_cli.auth import resolve_provider

        with patch.dict(os.environ, {"ANTHROPIC_VERTEX_PROJECT_ID": "test-project"}):
            result = resolve_provider("vertex-ai")
        assert result == "vertex"

    def test_google_vertex_alias(self):
        from hermes_cli.auth import resolve_provider

        with patch.dict(os.environ, {"ANTHROPIC_VERTEX_PROJECT_ID": "test-project"}):
            result = resolve_provider("google-vertex")
        assert result == "vertex"

    def test_vertex_direct(self):
        from hermes_cli.auth import resolve_provider

        result = resolve_provider("vertex")
        assert result == "vertex"


class TestVertexAutoDetection:
    """Test auto-detection of Vertex AI via CLAUDE_CODE_USE_VERTEX env var."""

    def test_auto_detects_vertex(self):
        from hermes_cli.auth import resolve_provider

        env = {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-project",
        }
        with patch.dict(os.environ, env, clear=False):
            result = resolve_provider("auto")
        assert result == "vertex"

    def test_no_detection_without_env(self):
        """Without CLAUDE_CODE_USE_VERTEX, should not auto-detect vertex."""
        from hermes_cli.auth import resolve_provider

        env_clear = {
            "CLAUDE_CODE_USE_VERTEX": "",
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-project",
            "OPENROUTER_API_KEY": "sk-or-test",
        }
        with patch.dict(os.environ, env_clear, clear=False):
            result = resolve_provider("auto")
        assert result != "vertex"

    def test_no_detection_without_project_id(self):
        """With CLAUDE_CODE_USE_VERTEX=1 but no project ID, should not auto-detect."""
        from hermes_cli.auth import resolve_provider

        env = {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "",
            "OPENROUTER_API_KEY": "sk-or-test",
        }
        with patch.dict(os.environ, env, clear=False):
            result = resolve_provider("auto")
        assert result != "vertex"


class TestVertexRuntimeResolution:
    """Test runtime provider resolution for vertex."""

    def test_resolve_vertex_runtime(self):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        env = {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-gcp-project",
            "CLOUD_ML_REGION": "europe-west1",
        }
        with patch.dict(os.environ, env, clear=False):
            runtime = resolve_runtime_provider(requested="vertex")

        assert runtime["provider"] == "vertex"
        assert runtime["api_mode"] == "anthropic_messages"
        assert runtime["project_id"] == "my-gcp-project"
        assert runtime["region"] == "europe-west1"
        assert runtime["source"] == "gcloud_adc"
        assert runtime["api_key"] == ""
        assert runtime["base_url"] == ""

    def test_default_region(self):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        env = {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_PROJECT_ID": "my-project",
        }
        # Remove CLOUD_ML_REGION if present
        clean_env = {k: v for k, v in os.environ.items() if k != "CLOUD_ML_REGION"}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            runtime = resolve_runtime_provider(requested="vertex")

        assert runtime["region"] == "us-east5"

    def test_missing_project_id_raises(self):
        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider

        env = {"ANTHROPIC_VERTEX_PROJECT_ID": ""}
        clean_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_VERTEX_PROJECT_ID"}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(AuthError, match="ANTHROPIC_VERTEX_PROJECT_ID"):
                resolve_runtime_provider(requested="vertex")
