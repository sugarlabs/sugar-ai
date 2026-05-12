"""app/backends - Model backend abstraction layer for sugar-ai.

Public API:
    BackendRouter    - routes requests to configured backends
    SugarAIConfig    - loads configuration from sugar_ai.yaml
    Message, Role    - conversation data structures
    GenerationConfig - inference parameters
    BackendResponse  - standardised response type
    BackendError     - base exception

Usage:
    from app.backends import BackendRouter, SugarAIConfig

    config = SugarAIConfig.load()
    router = BackendRouter.from_config(config)
    response = router.ask("What is photosynthesis?")
    print(response.content)
"""

from app.backends.base import (
    BackendCapabilities,
    BackendError,
    BackendQuotaError,
    BackendResponse,
    BackendTimeoutError,
    BackendUnavailableError,
    GenerationConfig,
    Message,
    ModelBackend,
    Role,
)
from app.backends.config import SugarAIConfig
from app.backends.router import BackendRegistry, BackendRouter

__all__ = [
    "BackendCapabilities",
    "BackendError",
    "BackendQuotaError",
    "BackendRegistry",
    "BackendResponse",
    "BackendRouter",
    "BackendTimeoutError",
    "BackendUnavailableError",
    "GenerationConfig",
    "Message",
    "ModelBackend",
    "Role",
    "SugarAIConfig",
]