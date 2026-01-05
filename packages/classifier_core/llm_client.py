"""LLM client for fallback classification."""

import os
from abc import ABC, abstractmethod

import anthropic
from openai import OpenAI

from .config import LABEL2ID


# System prompt for classification
CLASSIFICATION_PROMPT = """Eres un clasificador de noticias económicas y financieras en español.

Debes clasificar el texto en UNA de estas categorías:
- Macroeconomia: Noticias sobre inflación, PIB, tasas de interés, política monetaria, indicadores económicos
- Alianzas: Fusiones, adquisiciones, partnerships, acuerdos comerciales entre empresas
- Innovacion: Nuevas tecnologías, transformación digital, startups, fintech, productos innovadores
- Regulaciones: Leyes, normativas, supervisión, compliance, decisiones de reguladores
- Sostenibilidad: ESG, medio ambiente, responsabilidad social, cambio climático, energías renovables
- Otra: Noticias que no encajan claramente en las categorías anteriores
- Reputacion: Imagen corporativa, crisis de marca, premios, rankings de empresas

Responde SOLO con el nombre exacto de la categoría, sin explicación adicional."""


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def classify(self, text: str) -> dict:
        """Classify text using LLM.

        Args:
            text: Text to classify

        Returns:
            Dict with 'label', 'confidence', and 'cost_usd'
        """
        pass


class ClaudeClient(LLMClient):
    """Anthropic Claude client for classification."""

    # Pricing per 1M tokens (as of 2024)
    INPUT_COST_PER_1M = 3.00  # Claude 3 Haiku
    OUTPUT_COST_PER_1M = 15.00

    def __init__(self, api_key: str | None = None, model: str = "claude-3-haiku-20240307"):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model to use (default: claude-3-haiku for cost efficiency)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def classify(self, text: str) -> dict:
        """Classify text using Claude."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=50,
            system=CLASSIFICATION_PROMPT,
            messages=[
                {"role": "user", "content": f"Clasifica este texto:\n\n{text[:2000]}"}
            ],
        )

        # Extract response
        response_text = message.content[0].text.strip()

        # Validate label
        label = self._parse_label(response_text)

        # Calculate cost
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost_usd = (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M +
            (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        )

        return {
            "label": label,
            "confidence": 0.95,  # LLM doesn't provide confidence, assume high
            "cost_usd": round(cost_usd, 6),
            "tokens_used": input_tokens + output_tokens,
        }

    def _parse_label(self, response: str) -> str:
        """Parse and validate label from LLM response."""
        # Clean response
        response = response.strip()

        # Check exact match
        if response in LABEL2ID:
            return response

        # Check case-insensitive match
        for label in LABEL2ID.keys():
            if response.lower() == label.lower():
                return label

        # Check if response contains a valid label
        for label in LABEL2ID.keys():
            if label.lower() in response.lower():
                return label

        # Default to "Otra" if no match
        return "Otra"


class OpenAIClient(LLMClient):
    """OpenAI-compatible client for classification (OpenAI, Groq, etc.)."""

    # Pricing per 1M tokens (GPT-4o-mini default, Groq is free/cheap)
    INPUT_COST_PER_1M = 0.15
    OUTPUT_COST_PER_1M = 0.60

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        """Initialize OpenAI-compatible client.

        Args:
            api_key: API key (uses OPENAI_API_KEY or GROQ_API_KEY env var if not provided)
            model: Model to use
            base_url: Base URL for API (e.g., https://api.groq.com/openai/v1 for Groq)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found (OPENAI_API_KEY or GROQ_API_KEY)")

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model

    def classify(self, text: str) -> dict:
        """Classify text using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=50,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"Clasifica este texto:\n\n{text[:2000]}"},
            ],
        )

        # Extract response
        response_text = response.choices[0].message.content.strip()

        # Validate label
        label = self._parse_label(response_text)

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_usd = (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M +
            (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        )

        return {
            "label": label,
            "confidence": 0.95,
            "cost_usd": round(cost_usd, 6),
            "tokens_used": input_tokens + output_tokens,
        }

    def _parse_label(self, response: str) -> str:
        """Parse and validate label from LLM response."""
        response = response.strip()

        if response in LABEL2ID:
            return response

        for label in LABEL2ID.keys():
            if response.lower() == label.lower():
                return label

        for label in LABEL2ID.keys():
            if label.lower() in response.lower():
                return label

        return "Otra"


class GroqClient(OpenAIClient):
    """Groq client for classification (uses OpenAI-compatible API)."""

    # Groq pricing is very low/free for most models
    INPUT_COST_PER_1M = 0.05
    OUTPUT_COST_PER_1M = 0.10

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ):
        """Initialize Groq client.

        Args:
            api_key: Groq API key (uses GROQ_API_KEY env var if not provided)
            model: Model to use (default: llama-3.3-70b-versatile)
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
        )


def get_llm_client(provider: str = "anthropic", **kwargs) -> LLMClient:
    """Factory function to get LLM client.

    Args:
        provider: "anthropic", "openai", or "groq"
        **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    if provider == "anthropic":
        return ClaudeClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "groq":
        return GroqClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'groq'")
