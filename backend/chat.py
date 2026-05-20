"""
Chat handler: manages conversation sessions with distilled characters.
Supports streaming and context window management.
"""
from typing import AsyncGenerator
from openai import AsyncOpenAI

from models import CharacterProfile, ChatSession, ChatMessage
from config import settings


def _build_messages(character: CharacterProfile, session: ChatSession) -> list[dict]:
    """Build the OpenAI messages list with system prompt and conversation history."""
    system_content = character.system_prompt or (
        f"You are {character.name}. {character.description}\n"
        f"Background: {character.background}\n"
        "Stay in character at all times."
    )

    # Inject trait summary into system prompt if available
    if character.traits:
        trait_lines = "\n".join(
            f"- {t.name}: {t.description}" for t in character.traits
        )
        system_content += f"\n\nCore traits:\n{trait_lines}"

    if character.speech_pattern:
        sp = character.speech_pattern
        system_content += (
            f"\n\nSpeech style: {sp.vocabulary_style} "
            f"{sp.sentence_structure} Tone: {sp.tone}."
        )
        if sp.quirks:
            system_content += f" Quirks: {', '.join(sp.quirks)}."

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Keep only the last N turns to manage context
    recent = session.messages[-settings.max_context_turns * 2:]
    for msg in recent:
        if msg.role in ("user", "assistant"):
            messages.append({"role": msg.role, "content": msg.content})

    return messages


async def get_reply(
    character: CharacterProfile,
    session: ChatSession,
    user_message: str,
) -> tuple[str, dict]:
    """
    Returns (reply_text, usage_dict).
    Appends both the user message and reply to session.messages.
    """
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    session.messages.append(ChatMessage(role="user", content=user_message))
    messages = _build_messages(character, session)

    response = await client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.85,
        max_tokens=1024,
    )

    reply = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    session.messages.append(ChatMessage(role="assistant", content=reply))
    return reply, usage


async def stream_reply(
    character: CharacterProfile,
    session: ChatSession,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """
    Yields text chunks for SSE streaming.
    Uses standard stream=True for broad API compatibility (OpenAI / SiliconFlow / etc.).
    Appends messages to session after stream completes.
    """
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    session.messages.append(ChatMessage(role="user", content=user_message))
    messages = _build_messages(character, session)

    full_reply: list[str] = []
    stream = await client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.85,
        max_tokens=1024,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_reply.append(delta)
            yield delta

    session.messages.append(ChatMessage(role="assistant", content="".join(full_reply)))
