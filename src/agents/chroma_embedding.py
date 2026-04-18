"""Chroma embedding function selection: local by default, OpenAI optional."""
import os
from typing import Any

from chromadb.utils import embedding_functions


def get_chroma_embedding_function() -> Any:
    """
    Return embeddings for ChromaDB collections.

    Default: Chroma's ``DefaultEmbeddingFunction`` (local ONNX MiniLM, no API key).
    Set ``CORTEX_USE_OPENAI_EMBEDDINGS=1`` and ``OPENAI_API_KEY`` to use OpenAI instead.

    If you switch embedding providers, delete ``long_term_memory_store`` and
    ``long_term_memory_store_single_agent`` so collections are recreated with
    matching vector dimensions.
    """
    use_openai = os.getenv("CORTEX_USE_OPENAI_EMBEDDINGS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    key = os.getenv("OPENAI_API_KEY")
    if use_openai and key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=key,
            model_name=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )

    # Local embeddings — no cloud API (works with closed OpenAI accounts)
    return embedding_functions.DefaultEmbeddingFunction()
