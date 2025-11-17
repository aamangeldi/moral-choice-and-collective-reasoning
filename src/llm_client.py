"""Unified client for calling different LLM APIs."""

from typing import Any, Dict, Optional

import anthropic
import google.generativeai as genai
import openai

from src.config import Config


class LLMClient:
    """Unified interface for calling Claude, GPT, and Gemini."""

    def __init__(self, config: Config):
        """Initialize API clients."""
        self.config = config

        # Initialize clients
        self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        genai.configure(api_key=config.google_api_key)

    def call(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call an LLM with a prompt and return the response.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet", "gpt-4", "gemini-2.0")
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (uses config default if not specified)
            max_tokens: Maximum tokens in response (uses config default if not specified)

        Returns:
            The model's text response
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if "claude" in model.lower():
            return self._call_claude(model, prompt, system_prompt, temperature, max_tokens)
        elif "gpt" in model.lower():
            return self._call_openai(model, prompt, system_prompt, temperature, max_tokens)
        elif "gemini" in model.lower():
            return self._call_gemini(model, prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _call_claude(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Anthropic Claude API."""
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.anthropic_client.messages.create(**kwargs)
        return response.content[0].text

    def _call_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call OpenAI GPT API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _call_gemini(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Google Gemini API."""
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

        # Gemini doesn't have a separate system prompt in the same way
        # We'll prepend it to the user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = gemini_model.generate_content(full_prompt)
        return response.text
