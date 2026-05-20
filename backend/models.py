from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class CharacterTrait(BaseModel):
    name: str
    description: str
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)


class SpeechPattern(BaseModel):
    vocabulary_style: str
    sentence_structure: str
    tone: str
    quirks: list[str] = []


class CharacterProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    background: str = ""
    traits: list[CharacterTrait] = []
    speech_pattern: Optional[SpeechPattern] = None
    sample_texts: list[str] = []
    system_prompt: str = ""
    embedding: Optional[list[float]] = None
    embedding_metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    character_id: str
    messages: list[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- Request/Response models ---

class CreateCharacterRequest(BaseModel):
    name: str
    description: str
    background: str = ""
    sample_texts: list[str] = []


class DistillRequest(BaseModel):
    sample_texts: list[str] = []
    additional_context: str = ""


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    character_id: str
    reply: str
    usage: dict = {}


class DistillResponse(BaseModel):
    character_id: str
    traits: list[CharacterTrait]
    speech_pattern: SpeechPattern
    system_prompt: str
    embedding_dimensions: int


class EmbeddingResponse(BaseModel):
    character_id: str
    embedding: list[float]
    dimensions: int
    model: str
    metadata: dict


class SimilarityResult(BaseModel):
    character_id: str
    name: str
    similarity: float
