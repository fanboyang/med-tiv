"""Explicit infrastructure failures shared by tools and their HTTP server."""

class RetrievalServiceError(RuntimeError):
    """Retrieval transport/protocol failed; this is not a model action/result."""
