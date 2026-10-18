from ollama import embed

from src.core.logger import logger

from src.core.settings import EMBED_MODEL



def _extract_embedding_from_response(response):
    """
    Normalizes different embedding response shapes to return the primary embedding vector.

    Accepts:
      - dict-like responses with 'embeddings' or 'data'
      - objects exposing .embeddings or .data where items may have .embedding
      - objects with to_dict()

    Returns:
      list[float]
    Raises:
      ValueError on unexpected format
    """
    # Convert to plain dict if possible
    if hasattr(response, "to_dict") and callable(response.to_dict):
        try:
            response = response.to_dict()
        except Exception:
            # ignore and try other access patterns
            pass

    embeddings = None

    # Dict-like handling
    if isinstance(response, dict):
        if "embeddings" in response:
            embeddings = response.get("embeddings")
        elif "data" in response:
            data = response.get("data")
            if isinstance(data, (list, tuple)) and len(data) > 0:
                first = data[0]
                if isinstance(first, dict):
                    embeddings = first.get("embedding") or first.get("embeddings")
                    # If embedding is present as a list, wrap it so downstream treats it uniformly
                    if embeddings is not None and isinstance(embeddings[0] if isinstance(embeddings, (list, tuple)) and len(embeddings)>0 else None, (int, float)):
                        embeddings = [embeddings] if isinstance(embeddings, (list, tuple)) and not isinstance(embeddings[0], (list, tuple)) else embeddings
                else:
                    # data itself may be a list of numeric values or list-of-lists
                    embeddings = data
        else:
            embeddings = None

    # Object-like handling
    elif hasattr(response, "embeddings"):
        embeddings = response.embeddings
    elif hasattr(response, "embedding"):
        embeddings = response.embedding
    elif hasattr(response, "data"):
        data = response.data
        if isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            if hasattr(first, "embedding"):
                embeddings = [first.embedding]
            elif isinstance(first, dict):
                embeddings = first.get("embedding") or first.get("embeddings")
                if embeddings is not None and isinstance(embeddings[0] if isinstance(embeddings, (list, tuple)) and len(embeddings)>0 else None, (int, float)):
                    embeddings = [embeddings] if isinstance(embeddings, (list, tuple)) and not isinstance(embeddings[0], (list, tuple)) else embeddings
            else:
                embeddings = data
    else:
        raise ValueError(f"Invalid embedding response format: expected dict or object with .data/.embeddings/.embedding, got {type(response).__name__}")

    if embeddings is None:
        raise ValueError(f"Invalid embedding response format: embedding not found (response type {type(response).__name__})")

    # If embeddings is list-of-lists, take the first vector
    if isinstance(embeddings, (list, tuple)) and len(embeddings) > 0 and isinstance(embeddings[0], (list, tuple)):
        emb = embeddings[0]
    # If embeddings itself is a flat numeric list, accept it
    elif isinstance(embeddings, (list, tuple)) and all(isinstance(x, (int, float)) for x in embeddings):
        emb = list(embeddings)
    else:
        raise ValueError(f"Invalid embedding: unexpected embeddings shape (type={type(embeddings).__name__})")

    if not all(isinstance(v, (int, float)) for v in emb):
        raise ValueError("Invalid embedding: contains non-numeric values")

    return list(emb)


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for a single text chunk.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector

    Raises:
        ValueError: If the embedding response has an unexpected format
        Exception: If embedding generation fails
    """

    try:
        response = embed(
            model=EMBED_MODEL,
            input=text,
        )

        embedding = _extract_embedding_from_response(response)
        return embedding

    except ValueError as e:
        # Expected validation errors - log and re-raise with context
        # include response type for easier debugging
        resp_type = type(response).__name__ if 'response' in locals() else 'N/A'
        logger.error("Invalid embedding response format: %s (response_type=%s)", str(e), resp_type)
        raise
    except Exception as e:
        # Unexpected errors (network, service down, etc.)
        logger.error(
            "Embedding generation failed (error_type=%s)",
            type(e).__name__,
        )
        raise
