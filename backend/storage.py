import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from models import CharacterProfile, ChatSession
from config import settings


def _ensure_dirs():
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{settings.data_dir}/characters").mkdir(exist_ok=True)
    Path(f"{settings.data_dir}/sessions").mkdir(exist_ok=True)


def _char_path(character_id: str) -> Path:
    return Path(f"{settings.data_dir}/characters/{character_id}.json")


def _session_path(session_id: str) -> Path:
    return Path(f"{settings.data_dir}/sessions/{session_id}.json")


def save_character(character: CharacterProfile) -> None:
    _ensure_dirs()
    character.updated_at = datetime.utcnow()
    with open(_char_path(character.id), "w", encoding="utf-8") as f:
        json.dump(character.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


def load_character(character_id: str) -> Optional[CharacterProfile]:
    path = _char_path(character_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return CharacterProfile(**json.load(f))


def list_characters() -> list[CharacterProfile]:
    _ensure_dirs()
    chars = []
    for p in Path(f"{settings.data_dir}/characters").glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                chars.append(CharacterProfile(**json.load(f)))
        except Exception:
            pass
    return sorted(chars, key=lambda c: c.created_at, reverse=True)


def delete_character(character_id: str) -> bool:
    path = _char_path(character_id)
    if path.exists():
        path.unlink()
        return True
    return False


def save_session(session: ChatSession) -> None:
    _ensure_dirs()
    with open(_session_path(session.id), "w", encoding="utf-8") as f:
        json.dump(session.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


def load_session(session_id: str) -> Optional[ChatSession]:
    path = _session_path(session_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return ChatSession(**json.load(f))


def list_sessions(character_id: str) -> list[ChatSession]:
    _ensure_dirs()
    sessions = []
    for p in Path(f"{settings.data_dir}/sessions").glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                s = ChatSession(**json.load(f))
                if s.character_id == character_id:
                    sessions.append(s)
        except Exception:
            pass
    return sorted(sessions, key=lambda s: s.created_at, reverse=True)
