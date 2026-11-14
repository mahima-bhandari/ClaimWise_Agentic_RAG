
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Central configuration for ClaimWise."""

    # Groq configuration
    groq_api_key: str = Field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-120b"

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # Application configuration
    app_env: str = "development"
    log_level: str = "INFO"

    # Local storage
    data_dir: str = "data"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def absolute_data_dir(self) -> Path:
        """Return the absolute path to ClaimWise data storage."""
        path = Path(self.data_dir)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them."""
    return Settings()


settings = get_settings()
