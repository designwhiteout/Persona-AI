from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import Optional

from config import settings
from models import (
    CreateCharacterRequest,
    DistillRequest,
    ChatRequest,
    ChatResponse,
    DistillResponse,
    EmbeddingResponse,
    SimilarityResult,
    CharacterProfile,
    ChatSession,
)
import storage
import distiller
import embedder
import chat


app = FastAPI(
    title="Persona AI",
    description="Character distillation and roleplay API powered by OpenAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Characters ─────────────────────────────────────────────────────────────────

@app.post("/characters", response_model=CharacterProfile, status_code=201)
async def create_character(body: CreateCharacterRequest):
    character = CharacterProfile(
        name=body.name,
        description=body.description,
        background=body.background,
        sample_texts=body.sample_texts,
    )
    storage.save_character(character)
    return character


@app.get("/characters", response_model=list[CharacterProfile])
async def list_characters():
    return storage.list_characters()


@app.get("/characters/{character_id}", response_model=CharacterProfile)
async def get_character(character_id: str):
    character = storage.load_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    return character


@app.patch("/characters/{character_id}", response_model=CharacterProfile)
async def update_character(character_id: str, body: CreateCharacterRequest):
    character = storage.load_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    character.name = body.name
    character.description = body.description
    character.background = body.background
    if body.sample_texts:
        character.sample_texts = body.sample_texts
    storage.save_character(character)
    return character


@app.delete("/characters/{character_id}", status_code=204)
async def delete_character(character_id: str):
    if not storage.delete_character(character_id):
        raise HTTPException(status_code=404, detail="Character not found")


# ── Distillation ───────────────────────────────────────────────────────────────

@app.post("/characters/{character_id}/distill", response_model=DistillResponse)
async def distill_character(character_id: str, body: DistillRequest):
    character = storage.load_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")

    if body.sample_texts:
        character.sample_texts.extend(body.sample_texts)

    traits, speech_pattern, system_prompt = await distiller.distill_character(
        character, additional_context=body.additional_context
    )

    character.traits = traits
    character.speech_pattern = speech_pattern
    character.system_prompt = system_prompt

    # Auto-generate embedding after distillation
    embedding_vec, metadata = await embedder.embed_character(character)
    character.embedding = embedding_vec
    character.embedding_metadata = metadata

    storage.save_character(character)

    return DistillResponse(
        character_id=character.id,
        traits=traits,
        speech_pattern=speech_pattern,
        system_prompt=system_prompt,
        embedding_dimensions=len(embedding_vec),
    )


@app.post("/characters/import-text", response_model=CharacterProfile, status_code=201)
async def import_from_text(
    name: str = Form(...),
    description: str = Form(""),
    text: str = Form(...),
    auto_distill: bool = Form(True),
):
    """Create a character directly from a large text block and optionally distill immediately."""
    character = CharacterProfile(
        name=name,
        description=description or f"Character extracted from provided text",
        sample_texts=[text],
    )
    storage.save_character(character)

    if auto_distill:
        traits, speech_pattern, system_prompt = await distiller.distill_character(character)
        character.traits = traits
        character.speech_pattern = speech_pattern
        character.system_prompt = system_prompt
        embedding_vec, metadata = await embedder.embed_character(character)
        character.embedding = embedding_vec
        character.embedding_metadata = metadata
        storage.save_character(character)

    return character


@app.post("/characters/import-file", response_model=CharacterProfile, status_code=201)
async def import_from_file(
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    auto_distill: bool = Form(True),
):
    """Create a character from an uploaded text file."""
    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    character = CharacterProfile(
        name=name,
        description=description or f"Character extracted from {file.filename}",
        sample_texts=[text],
    )
    storage.save_character(character)

    if auto_distill:
        traits, speech_pattern, system_prompt = await distiller.distill_character(character)
        character.traits = traits
        character.speech_pattern = speech_pattern
        character.system_prompt = system_prompt
        embedding_vec, metadata = await embedder.embed_character(character)
        character.embedding = embedding_vec
        character.embedding_metadata = metadata
        storage.save_character(character)

    return character


# ── Embeddings ─────────────────────────────────────────────────────────────────

@app.post("/characters/{character_id}/embed", response_model=EmbeddingResponse)
async def embed_character(character_id: str):
    character = storage.load_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")

    embedding_vec, metadata = await embedder.embed_character(character)
    character.embedding = embedding_vec
    character.embedding_metadata = metadata
    storage.save_character(character)

    return EmbeddingResponse(
        character_id=character.id,
        embedding=embedding_vec,
        dimensions=len(embedding_vec),
        model=metadata["model"],
        metadata=metadata,
    )


@app.get("/characters/{character_id}/embedding", response_model=EmbeddingResponse)
async def get_embedding(character_id: str):
    character = storage.load_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    if not character.embedding:
        raise HTTPException(status_code=404, detail="No embedding found — call /embed first")

    return EmbeddingResponse(
        character_id=character.id,
        embedding=character.embedding,
        dimensions=len(character.embedding),
        model=character.embedding_metadata.get("model", settings.embedding_model),
        metadata=character.embedding_metadata,
    )


@app.get("/characters/{character_id}/similar", response_model=list[SimilarityResult])
async def find_similar(character_id: str, top_k: int = 5):
    """Find characters most similar to the given character using embedding cosine similarity."""
    target = storage.load_character(character_id)
    if not target:
        raise HTTPException(status_code=404, detail="Character not found")
    if not target.embedding:
        raise HTTPException(status_code=400, detail="Target character has no embedding")

    all_chars = storage.list_characters()
    results = []
    for char in all_chars:
        if char.id == character_id or not char.embedding:
            continue
        sim = embedder.cosine_similarity(target.embedding, char.embedding)
        results.append(SimilarityResult(character_id=char.id, name=char.name, similarity=sim))

    results.sort(key=lambda r: r.similarity, reverse=True)
    return results[:top_k]


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post("/characters/{character_id}/chat", response_model=ChatResponse)
async def chat_with_character(character_id: str, body: ChatRequest):
    character = storage.load_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    if not character.system_prompt:
        raise HTTPException(
            status_code=400,
            detail="Character has not been distilled yet — call /distill first",
        )

    if body.stream:
        # Return SSE stream
        session = _get_or_create_session(character_id, body.session_id)

        async def event_generator():
            async for chunk in chat.stream_reply(character, session, body.message):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            storage.save_session(session)
            yield f"data: {json.dumps({'done': True, 'session_id': session.id})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    session = _get_or_create_session(character_id, body.session_id)
    reply, usage = await chat.get_reply(character, session, body.message)
    storage.save_session(session)

    return ChatResponse(
        session_id=session.id,
        character_id=character_id,
        reply=reply,
        usage=usage,
    )


@app.get("/characters/{character_id}/sessions")
async def list_sessions(character_id: str):
    if not storage.load_character(character_id):
        raise HTTPException(status_code=404, detail="Character not found")
    return storage.list_sessions(character_id)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/sessions/{session_id}", status_code=204)
async def clear_session(session_id: str):
    from pathlib import Path
    path = Path(f"{settings.data_dir}/sessions/{session_id}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    path.unlink()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models": {
        "chat": settings.chat_model,
        "embedding": settings.embedding_model,
        "distill": settings.distill_model,
    }}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_or_create_session(character_id: str, session_id: Optional[str]) -> ChatSession:
    if session_id:
        session = storage.load_session(session_id)
        if session:
            return session
    return ChatSession(character_id=character_id)
