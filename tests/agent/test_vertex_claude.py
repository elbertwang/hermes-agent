"""vertex-claude: Claude on Google Vertex AI through ``anthropic.AnthropicVertex`` (ADC auth).

Every client build — startup, /model switch, fallback swap, primary restore, per-request rebuild,
auxiliary — must land on AnthropicVertex with the resolved project/region, and the ``google-adc``
sentinel must never reach the generic Anthropic builder (it would be sent as an API key → 401)."""

from unittest.mock import MagicMock, patch

import pytest

import agent.anthropic_adapter as aa
from agent.context_compressor import ContextCompressor
from agent.vertex_claude_adapter import (
    ADC_SENTINEL, DEFAULT_REGION, VertexClaudeConfigError, resolve_project_region, vertex_base_url,
)
from run_agent import AIAgent

_VERTEX_ENV = ("VERTEX_PROJECT", "ANTHROPIC_VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT",
               "GOOGLE_CLOUD_PROJECT_ID", "VERTEX_LOCATION", "CLOUD_ML_REGION", "GOOGLE_CLOUD_LOCATION",
               "VERTEX_REGION")


@pytest.fixture
def vertex_env(monkeypatch):
    for name in _VERTEX_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "demo-proj")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")


def _agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "claude-opus-4.8", "anthropic"
    agent.base_url, agent.api_key, agent.api_mode = "https://api.anthropic.com", "sk-ant", "anthropic_messages"
    agent.client, agent._anthropic_client, agent._client_kwargs = None, MagicMock(), {}
    agent.quiet_mode, agent._config_context_length, agent._primary_runtime = True, None, {}
    agent.context_compressor = ContextCompressor(
        model=agent.model, threshold_percent=0.5, base_url=agent.base_url, api_key="sk-ant",
        provider="anthropic", quiet_mode=True, config_context_length=None,
    )
    return agent


def _assert_bound(agent):
    assert type(agent._anthropic_client).__name__ == "AnthropicVertex"
    assert agent._anthropic_client.project_id == "demo-proj"
    assert agent._anthropic_client.region == "us-east5"
    assert agent.api_key == agent._anthropic_api_key == ADC_SENTINEL
    assert agent.client is None and agent._client_kwargs == {}
    assert agent._is_anthropic_oauth is False


# ── client builder ────────────────────────────────────────────────────────────────────

def test_build_client_passes_project_region_without_1m_beta():
    fake_sdk = MagicMock()
    with patch.object(aa, "_get_anthropic_sdk", return_value=fake_sdk):
        aa.build_anthropic_vertex_client("my-proj", "us-east5")
    kwargs = fake_sdk.AnthropicVertex.call_args.kwargs
    assert (kwargs["project_id"], kwargs["region"], kwargs["max_retries"]) == ("my-proj", "us-east5", 0)
    betas = kwargs["default_headers"]["anthropic-beta"]
    assert "interleaved-thinking-2025-05-14" in betas
    assert "context-1m-2025-08-07" not in betas  # Vertex rejects it with HTTP 400


def test_build_client_requires_anthropic_vertex_symbol():
    with patch.object(aa, "_get_anthropic_sdk", return_value=MagicMock(spec=[])):
        with pytest.raises(ImportError):
            aa.build_anthropic_vertex_client("p", "r")


# ── project / region resolution ───────────────────────────────────────────────────────

def test_resolve_prefers_hermes_names_over_sdk_names(vertex_env, monkeypatch):
    monkeypatch.setenv("VERTEX_PROJECT", "hermes-proj")
    monkeypatch.setenv("VERTEX_LOCATION", "global")
    assert resolve_project_region() == ("hermes-proj", "global")


def test_resolve_explicit_args_win(vertex_env):
    assert resolve_project_region("p", "europe-west1") == ("p", "europe-west1")


def test_resolve_region_defaults(vertex_env, monkeypatch):
    monkeypatch.delenv("CLOUD_ML_REGION")
    assert resolve_project_region() == ("demo-proj", DEFAULT_REGION)


def test_resolve_falls_back_to_adc_project_then_errors(vertex_env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID")
    with patch("agent.vertex_claude_adapter._adc_default_project", return_value="adc-proj"):
        assert resolve_project_region()[0] == "adc-proj"
    with patch("agent.vertex_claude_adapter._adc_default_project", return_value=""):
        with pytest.raises(VertexClaudeConfigError):
            resolve_project_region()


def test_vertex_base_url_global_host():
    assert vertex_base_url("global") == "https://aiplatform.googleapis.com"
    assert vertex_base_url("us-east5") == "https://us-east5-aiplatform.googleapis.com"


# ── provider registration / runtime ───────────────────────────────────────────────────

def test_provider_registered_with_anthropic_wire():
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
    from hermes_cli.providers import determine_api_mode
    assert PROVIDER_REGISTRY["vertex-claude"].auth_type == "google_adc"
    assert resolve_provider("claude-vertex") == "vertex-claude"
    assert determine_api_mode("vertex-claude") == "anthropic_messages"
    # Upstream's Gemini ``vertex`` provider is untouched.
    assert resolve_provider("vertex") == "vertex"


def test_runtime_resolution(vertex_env):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    rt = resolve_runtime_provider(requested="vertex-claude")
    assert (rt["provider"], rt["api_mode"], rt["api_key"]) == ("vertex-claude", "anthropic_messages", ADC_SENTINEL)
    assert (rt["vertex_project"], rt["vertex_location"]) == ("demo-proj", "us-east5")
    assert rt["base_url"] == "https://us-east5-aiplatform.googleapis.com"


def test_runtime_missing_project_is_auth_error(vertex_env, monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID")
    with patch("agent.vertex_claude_adapter._adc_default_project", return_value=""):
        with pytest.raises(AuthError):
            resolve_runtime_provider(requested="vertex-claude")


# ── every agent-side client build lands on AnthropicVertex ────────────────────────────

def test_init_binds_anthropic_vertex(vertex_env):
    from agent.agent_init import _init_anthropic_client
    agent = _agent()
    agent.provider = "vertex-claude"
    _init_anthropic_client(agent, ADC_SENTINEL, "https://us-east5-aiplatform.googleapis.com", None)
    _assert_bound(agent)


@patch("agent.model_metadata.get_model_context_length", return_value=1_000_000)
def test_switch_model_binds_anthropic_vertex(_ctx, vertex_env):
    agent = _agent()
    agent.switch_model("claude-opus-5-5", "vertex-claude", api_key=ADC_SENTINEL,
                       base_url="https://us-east5-aiplatform.googleapis.com", api_mode="anthropic_messages")
    _assert_bound(agent)


def test_fallback_swap_binds_anthropic_vertex(vertex_env):
    from agent.client_lifecycle import _swap_fallback_clients
    agent = _agent()
    agent.provider = "vertex-claude"
    fb_client = MagicMock(base_url="https://us-east5-aiplatform.googleapis.com", api_key=ADC_SENTINEL)
    _swap_fallback_clients(agent, fb_client, "vertex-claude", "claude-opus-5-5", fb_client.base_url,
                           "anthropic_messages")
    _assert_bound(agent)


def test_request_client_rebuild_uses_vertex(vertex_env):
    agent = _agent()
    agent.provider, agent._vertex_project, agent._vertex_region = "vertex-claude", "demo-proj", "us-east5"
    key = agent._request_anthropic_client_key()
    assert key == ("vertex-claude", "demo-proj", "us-east5")
    client = agent._build_anthropic_client_for_key(key)
    assert type(client).__name__ == "AnthropicVertex" and client.region == "us-east5"


def test_auxiliary_client_uses_vertex(vertex_env):
    from agent.auxiliary_client import resolve_provider_client
    client, model = resolve_provider_client("vertex-claude", "claude-haiku-4-5@20251001")
    assert model == "claude-haiku-4-5@20251001"
    assert type(client._real_client).__name__ == "AnthropicVertex"
