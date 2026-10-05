"""Claude on Google Vertex AI.

Native Messages wire through ``anthropic.AnthropicVertex`` (``agent/vertex_claude_adapter.py``),
not the ``vertex`` provider's OpenAI-compatible Gemini endpoint. Auth is Application Default
Credentials refreshed inside the SDK; ``auth_type="google_adc"`` keeps it off the api_key path.
Vertex model ids use ``@``-versioned names for older models (``claude-haiku-4-5@20251001``).
"""

from providers import register_provider
from providers.base import ProviderProfile

vertex_claude = ProviderProfile(
    name="vertex-claude", aliases=("anthropic-vertex", "vertex-anthropic", "claude-vertex"),
    api_mode="anthropic_messages", display_name="Google Vertex AI (Claude)",
    description="Claude on Google Vertex AI (ADC via gcloud / service account, GCP billing)",
    env_vars=(),  # ADC — no static key env var
    base_url="",  # AnthropicVertex builds its own endpoint from project + region
    auth_type="google_adc", supports_health_check=False, supports_vision=True,
    supports_model_listing=False,  # no /models route; curated list below
    default_aux_model="claude-haiku-4-5@20251001",
    fallback_models=(
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-4-8",
        "claude-haiku-4-5@20251001",
    ),
)

register_provider(vertex_claude)
