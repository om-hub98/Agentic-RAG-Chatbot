from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.config import EMBEDDING_MODEL


def get_embedding_model(
    model_name: Optional[str] = None,
    provider: str = "fastembed"
) -> Embeddings:
    """
    Factory function to initialize and return an embedding model.
    Default: FastEmbed with BAAI/bge-small-en-v1.5 (fast, local, no API quota needed).
    """
    selected_model = model_name or EMBEDDING_MODEL

    if provider.lower() == "fastembed":
        print(f"[Embedding] Initializing FastEmbed model: {selected_model}")
        return FastEmbedEmbeddings(model_name=selected_model)
    elif provider.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=selected_model)
    elif provider.lower() in ["gemini", "google"]:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=selected_model)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
