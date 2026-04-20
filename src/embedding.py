"""
Embedding Model Abstraction
===========================
Provides a unified interface for both local (SentenceTransformer) and
remote (OpenRouter API) embedding models.
"""

from typing import Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config import get_settings, logger

settings = get_settings()


class EmbeddingModel:
    """
    Unified interface for embedding models.

    Supports both local (SentenceTransformer) and remote (OpenRouter API)
    embedding models based on configuration.
    """

    def __init__(self, remote: bool = False):
        """
        Initialize embedding model.

        Args:
            remote: If True, use OpenRouter API for embeddings.
                   If False, use local SentenceTransformer model.
        """
        self._remote = remote
        self._local_model: Optional[SentenceTransformer] = None
        self._remote_client: Optional[OpenAI] = None

    def _get_local_model(self) -> SentenceTransformer:
        """Get or create local SentenceTransformer model."""
        if self._local_model is None:
            model_name = settings.glossary.embedding_model
            logger.info(
                f"Loading local embedding model '{model_name}' "
                "(first run downloads ~2GB)..."
            )
            self._local_model = SentenceTransformer(model_name)
            logger.info(f"Local embedding model '{model_name}' is loaded.")
        return self._local_model

    def _get_remote_client(self) -> OpenAI:
        """Get or create OpenAI client for OpenRouter API."""
        if self._remote_client is None:
            if not settings.llm.api_key:
                raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
            self._remote_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.llm.api_key,
            )
        return self._remote_client

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Text to embed
            normalize_embeddings: Whether to normalize the embeddings

        Returns:
            numpy array containing the embedding vector
        """
        if self._remote:
            return self._encode_remote(text, normalize_embeddings)
        else:
            return self._encode_local(text, normalize_embeddings)

    def _encode_local(self, text: str, normalize_embeddings: bool) -> np.ndarray:
        """Encode text using local SentenceTransformer model."""
        model = self._get_local_model()
        result = model.encode(text, normalize_embeddings=normalize_embeddings)
        # Convert tensor to numpy array if needed
        if hasattr(result, "numpy"):
            return result.numpy()
        return np.asarray(result)

    def _encode_remote(self, text: str, normalize_embeddings: bool) -> np.ndarray:
        """Encode text using OpenRouter API."""
        client = self._get_remote_client()
        model_name = (
            settings.glossary.remote_embedding_model
            or settings.glossary.embedding_model
        )

        try:
            response = client.embeddings.create(
                model=model_name,
                input=[text],
                encoding_format="float",
            )
            embedding = response.data[0].embedding
            result = np.array(embedding, dtype=np.float32)

            if normalize_embeddings:
                norm = np.linalg.norm(result)
                if norm > 0:
                    result = result / norm

            return result
        except Exception as e:
            raise RuntimeError(f"Remote embedding failed: {str(e)}") from e

    def encode_batch(
        self, texts: list[str], normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to embed
            normalize_embeddings: Whether to normalize the embeddings

        Returns:
            2D numpy array of shape (n_texts, embedding_dim)
        """
        if self._remote:
            return self._encode_batch_remote(texts, normalize_embeddings)
        else:
            return self._encode_batch_local(texts, normalize_embeddings)

    def _encode_batch_local(
        self, texts: list[str], normalize_embeddings: bool
    ) -> np.ndarray:
        """Encode multiple texts using local SentenceTransformer model."""
        model = self._get_local_model()
        result = model.encode(texts, normalize_embeddings=normalize_embeddings)
        # Convert tensor to numpy array if needed
        if hasattr(result, "numpy"):
            return result.numpy()
        return np.asarray(result)

    def _encode_batch_remote(
        self, texts: list[str], normalize_embeddings: bool
    ) -> np.ndarray:
        """Encode multiple texts using OpenRouter API."""
        client = self._get_remote_client()
        model_name = (
            settings.glossary.remote_embedding_model
            or settings.glossary.embedding_model
        )

        try:
            response = client.embeddings.create(
                model=model_name,
                input=texts,
                encoding_format="float",
            )
            embeddings = [data.embedding for data in response.data]
            result = np.array(embeddings, dtype=np.float32)

            if normalize_embeddings:
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
                result = result / norms

            return result
        except Exception as e:
            raise RuntimeError(f"Remote embedding failed: {str(e)}") from e
