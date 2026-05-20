"""
Character embedding: encodes the character's distilled essence into a
dense vector for similarity search and retrieval.
"""
import json
import numpy as np
from openai import AsyncOpenAI

from models import CharacterProfile
from config import settings


def _build_embedding_text(character: CharacterProfile) -> str:
    """Construct a rich text representation of the character for embedding."""
    parts = [
        f"Name: {character.name}",
        f"Description: {character.description}",
    ]

    if character.background:
        parts.append(f"Background: {character.background}")

    if character.traits:
        traits_text = "; ".join(
            f"{t.name} ({t.intensity:.1f}): {t.description}"
            for t in character.traits
        )
        parts.append(f"Traits: {traits_text}")

    if character.speech_pattern:
        sp = character.speech_pattern
        parts.append(
            f"Speech: {sp.vocabulary_style}. {sp.sentence_structure}. Tone: {sp.tone}."
        )
        if sp.quirks:
            parts.append(f"Quirks: {'; '.join(sp.quirks)}")

    if character.system_prompt:
        # Truncate system prompt to avoid token limits
        parts.append(f"Character essence: {character.system_prompt[:800]}")

    return "\n".join(parts)


async def embed_character(character: CharacterProfile) -> tuple[list[float], dict]:
    """
    Returns (embedding_vector, metadata).
    metadata includes model, dimensions, and the text used for embedding.
    """
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    text = _build_embedding_text(character)

    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
        encoding_format="float",
    )

    embedding = response.data[0].embedding
    metadata = {
        "model": settings.embedding_model,
        "dimensions": len(embedding),
        "source_text_preview": text[:200],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }

    return embedding, metadata


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
