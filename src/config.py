"""Configuration management for experiments."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration for LLM experiments."""

    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Default settings
    temperature: float = 1.0
    max_tokens: int = 1024

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.google_api_key is None:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")

    def validate(self) -> None:
        """Validate that required API keys are present."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing)}. "
                "Please set them in .env file or environment variables."
            )


def get_config() -> Config:
    """Get default configuration."""
    config = Config()
    config.validate()
    return config
