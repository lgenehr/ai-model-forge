"""
Global configuration settings using pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QualitySettings(BaseSettings):
    """Quality filter settings."""

    min_tokens: int = 50
    max_tokens: int = 4096
    min_quality_score: float = 0.6
    similarity_threshold: float = 0.85
    language_confidence: float = 0.95
    max_perplexity: float = 1000.0
    max_special_char_ratio: float = 0.1
    max_word_repetition: float = 0.3


class RateLimitSettings(BaseSettings):
    """Rate limiting settings for APIs."""

    news_api_per_day: int = 100
    youtube_per_day: int = 10000
    reddit_per_minute: int = 60
    semantic_scholar_per_5min: int = 100
    default_per_minute: int = 30


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Directories
    base_dir: Path = Field(default=Path("."))
    output_dir: Path = Field(default=Path("./outputs"))
    checkpoint_dir: Path = Field(default=Path("./checkpoints"))
    logs_dir: Path = Field(default=Path("./logs"))

    # API Keys
    news_api_key: str | None = Field(default=None, alias="NEWS_API_KEY")
    youtube_api_key: str | None = Field(default=None, alias="YOUTUBE_API_KEY")
    reddit_client_id: str | None = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_access_token: str | None = Field(default=None, alias="REDDIT_ACCESS_TOKEN")
    reddit_user_agent: str = Field(
        default="DatasetGenerator/1.0", alias="REDDIT_USER_AGENT"
    )
    wikipedia_user_agent: str = Field(
        default="DatasetGenerator/1.0 (+https://github.com/ai-model-forge/dataset-generator)",
        alias="WIKIPEDIA_USER_AGENT",
    )
    semantic_scholar_key: str | None = Field(
        default=None, alias="SEMANTIC_SCHOLAR_KEY"
    )
    gnews_api_key: str | None = Field(default=None, alias="GNEWS_API_KEY")

    # Collection settings
    default_language: str = "pt_br"
    max_workers: int = 4
    checkpoint_interval: int = 100
    request_timeout: int = 30
    max_retries: int = 3

    # Quality settings
    quality: QualitySettings = Field(default_factory=QualitySettings)

    # Rate limiting
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)

    # Date range for collection
    date_start: str = "2024-01-01"
    date_end: str = "2026-12-31"

    # Target sizes
    target_min_samples: int = 50_000
    target_recommended_samples: int = 200_000
    target_optimal_samples: int = 500_000

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [
            self.output_dir,
            self.output_dir / "raw",
            self.output_dir / "processed",
            self.output_dir / "final",
            self.checkpoint_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_yaml_config(file_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(file_path) as f:
        return yaml.safe_load(f)


def load_sources_config(base_dir: Path | None = None) -> dict[str, Any]:
    """Load sources configuration."""
    if base_dir is None:
        base_dir = Path(__file__).parent
    return load_yaml_config(base_dir / "sources.yaml")


def load_topics_config(base_dir: Path | None = None) -> dict[str, Any]:
    """Load topics configuration."""
    if base_dir is None:
        base_dir = Path(__file__).parent
    return load_yaml_config(base_dir / "topics.yaml")
