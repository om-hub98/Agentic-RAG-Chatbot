import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


def get_chat_model(
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    provider: Optional[str] = None,
) -> BaseChatModel:
    """
    Factory function to initialize a LangChain Chat Model.
    Defaults to OpenRouter if OPENROUTER_API_KEY is present in environment.
    """
    # Auto-detect provider if not explicitly given
    if not provider:
        if OPENROUTER_API_KEY:
            provider = "openrouter"
        elif GEMINI_API_KEY:
            provider = "gemini"
        else:
            provider = "openrouter"

    if provider.lower() == "openrouter":
        target_model = model_name or OPENROUTER_MODEL
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or .env file.")
        print(f"[LLM] Initializing OpenRouter Chat Model: {target_model}")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model=target_model,
            temperature=temperature,
        )

    elif provider.lower() in ["gemini", "google"]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = model_name or GEMINI_MODEL
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        print(f"[LLM] Initializing Gemini Chat Model: {target_model}")
        return ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY,
            model=target_model,
            temperature=temperature,
        )

    elif provider.lower() == "openai":
        target_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
