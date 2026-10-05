"""Claude on Google Vertex AI via the Anthropic SDK's ``AnthropicVertex`` client.

Separate from ``agent/vertex_adapter.py`` (the ``vertex`` provider: Gemini over the
OpenAI-compatible endpoint). Claude on Vertex speaks the native Messages wire, so it rides
``anthropic_messages`` for full feature parity (prompt caching, adaptive thinking) — the same
shape as Bedrock + ``AnthropicBedrock``.

Auth is Application Default Credentials; the SDK mints and refreshes the bearer itself, so
``google-adc`` below is a sentinel, never a credential. Project / region resolve from env
(first set wins, SDK-native names included so a Claude Code setup works unchanged) and fall
back to the ADC default project.
"""

import logging
from typing import Optional, Tuple

from agent.secret_scope import get_secret_str

logger = logging.getLogger(__name__)

PROVIDER_ID = "vertex-claude"
ADC_SENTINEL = "google-adc"
DEFAULT_REGION = "us-east5"

_PROJECT_ENV_VARS = (
    "VERTEX_PROJECT", "ANTHROPIC_VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT",
    "GOOGLE_CLOUD_PROJECT_ID",
)
_REGION_ENV_VARS = ("VERTEX_LOCATION", "CLOUD_ML_REGION", "GOOGLE_CLOUD_LOCATION", "VERTEX_REGION")
_CLOUD_PLATFORM_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexClaudeConfigError(RuntimeError):
    """No GCP project could be resolved for Claude on Vertex."""


def _first_env(names: Tuple[str, ...]) -> str:
    for name in names:
        value = get_secret_str(name).strip()
        if value:
            return value
    return ""


def _adc_default_project() -> str:
    """ADC's default project (gcloud user creds, SA JSON, GCE metadata), or "" on any failure."""
    from agent.vertex_adapter import _ensure_google_auth
    if not _ensure_google_auth():
        return ""
    import google.auth
    try:
        _, project = google.auth.default(scopes=_CLOUD_PLATFORM_SCOPES)
        return str(project or "").strip()
    except (google.auth.exceptions.GoogleAuthError, OSError) as exc:  # caller raises the actionable error
        logger.debug("Vertex Claude: ADC default-project lookup failed: %s", exc)
        return ""


def resolve_project_region(project: Optional[str] = None, region: Optional[str] = None) -> Tuple[str, str]:
    """``(project, region)`` from explicit args → env vars → ADC default project / DEFAULT_REGION."""
    project = (project or "").strip() or _first_env(_PROJECT_ENV_VARS) or _adc_default_project()
    if not project:
        raise VertexClaudeConfigError(
            "Vertex AI project is not set for vertex-claude. Export VERTEX_PROJECT (or "
            "ANTHROPIC_VERTEX_PROJECT_ID) in ~/.hermes/.env, or run `gcloud config set project <id>`.")
    return project, (region or "").strip() or _first_env(_REGION_ENV_VARS) or DEFAULT_REGION


def vertex_base_url(region: str) -> str:
    """Display/cache-key base URL; ``AnthropicVertex`` builds its own request URLs."""
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return f"https://{host}"


def has_adc_credentials() -> bool:
    """Whether the ADC chain resolves (no token minted, no network for file-based creds)."""
    from agent.vertex_adapter import _ensure_google_auth
    if not _ensure_google_auth():
        return False
    import google.auth
    try:
        google.auth.default(scopes=_CLOUD_PLATFORM_SCOPES)
        return True
    except (google.auth.exceptions.GoogleAuthError, OSError) as exc:
        logger.debug("Vertex Claude: ADC availability check failed: %s", exc)
        return False


def google_adc_auth_status(target: str) -> dict:
    """``hermes auth status`` row for ``auth_type="google_adc"`` providers."""
    return {"logged_in": has_adc_credentials(), "provider": target}


def build_aux_client(provider: str, model: Optional[str]):
    """``(AnthropicAuxiliaryClient, model)`` over AnthropicVertex, or ``(None, None)`` when the SDK
    or project is missing so the auxiliary chain moves on to its next rung."""
    from agent.anthropic_adapter import build_anthropic_vertex_client
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient, _get_aux_model_for_provider, _normalize_resolved_model,
    )
    try:
        project, region = resolve_project_region()
        real_client = build_anthropic_vertex_client(project, region)
    except (ImportError, VertexClaudeConfigError) as exc:
        logger.warning("resolve_provider_client: cannot create vertex-claude client: %s", exc)
        return None, None
    default_model = _get_aux_model_for_provider(provider) or "claude-haiku-4-5@20251001"
    final_model = _normalize_resolved_model(model or default_model, provider) or default_model
    client = AnthropicAuxiliaryClient(real_client, final_model, api_key=ADC_SENTINEL, base_url=vertex_base_url(region))
    logger.debug("resolve_provider_client: vertex-claude (%s, %s)", final_model, region)
    return client, final_model


def bind_vertex_claude_runtime(agent) -> None:
    """Point *agent* at ``AnthropicVertex``. Startup and every later rebuild (/model switch,
    fallback swap, primary restore) share this so project/region never lag the active route and
    the ``google-adc`` sentinel never reaches the generic Anthropic/OpenAI client builders."""
    from agent.anthropic_adapter import build_anthropic_vertex_client
    project, region = resolve_project_region(getattr(agent, "_vertex_project", None),
                                             getattr(agent, "_vertex_region", None))
    agent._vertex_project, agent._vertex_region = project, region
    agent.client = None
    agent._client_kwargs = {}
    agent.api_key = agent._anthropic_api_key = ADC_SENTINEL
    agent._anthropic_base_url = vertex_base_url(region)
    agent._is_anthropic_oauth = False
    agent._anthropic_client = build_anthropic_vertex_client(project, region)
