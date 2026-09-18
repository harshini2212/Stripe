"""Runtime configuration, resolved from environment with sane offline defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, only needed to read a local .env
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "generated"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

OFFLINE_BACKEND = "offline-heuristic"

# Model backends the eval leaderboard knows how to compare.
CLAUDE_BACKENDS = ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5")


@dataclass(frozen=True)
class Config:
    seed: int = 7
    anthropic_api_key: str | None = None
    default_backend: str = "claude-opus-4-8"
    effort: str = "high"
    data_dir: Path = DATA_DIR
    artifact_dir: Path = ARTIFACT_DIR
    leaderboard_backends: tuple[str, ...] = field(default_factory=lambda: CLAUDE_BACKENDS)

    @property
    def has_live_models(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def effective_default_backend(self) -> str:
        """The backend agents use by default — falls back to offline w/o a key."""
        if self.default_backend != OFFLINE_BACKEND and not self.has_live_models:
            return OFFLINE_BACKEND
        return self.default_backend


def load_config() -> Config:
    key = os.getenv("ANTHROPIC_API_KEY") or None
    default_backend = os.getenv("COMPTROLLER_DEFAULT_BACKEND", "claude-opus-4-8").strip()
    cfg = Config(
        seed=int(os.getenv("COMPTROLLER_SEED", "7")),
        anthropic_api_key=key,
        default_backend=default_backend or "claude-opus-4-8",
        effort=os.getenv("COMPTROLLER_EFFORT", "high").strip() or "high",
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    return cfg


CONFIG = load_config()
