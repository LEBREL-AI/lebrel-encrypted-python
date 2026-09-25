"""Encrypted Lebrel API client. Conversation dictionaries use OpenAI's schema."""

from .client import (
    APIError,
    Completion,
    CompletionStream,
    EncryptionError,
    Lebrel,
    LebrelError,
    MODEL_ID,
    StreamError,
    TransportError,
)
from .proof import Check, Manifest, Receipt

__version__ = "0.2.0"
__all__ = ["Lebrel", "MODEL_ID", "Completion", "CompletionStream", "Receipt", "Manifest", "Check", "LebrelError", "EncryptionError", "TransportError", "APIError", "StreamError"]
