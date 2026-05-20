"""
SQLite-backed persistence layer.
Public API is identical to the former JSON-file implementation so main.py
needs only one small change (clear_session now calls delete_session).

Schema: two tables, complex fields stored as JSON text.
Migration: on first init, any existing JSON files are imported automatically.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

from models import CharacterProfile, ChatSession
from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT NOT NULL,
    background     TEXT DEFAULT '',
    sample_texts   TEXT DEFAULT '[]',
    traits         TEXT DEFAULT '[]',
    speech_pattern TEXT,
    system_prompt  TEXT DEFAULT '',
    embedding      TEXT,
    embedding_meta TEXT DEFAULT '{}',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    messages     TEXT DEFAULT '[]',
    created_at   TEXT NOT NULL
);
"""

_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[str] = None


def _get_conn() -> sqlite3.Connection:
    global _conn, _conn_path
    db_path = str(Path(settings.data_dir) / "persona_ai.db")
    if _conn is None or _conn_path != db_path:
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(_SCHEMA)
        _conn.commit()
        _conn_path = db_path
    return _conn


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _char_to_row(c: CharacterProfile) -> tuple:
    d = c.model_dump(mode="json")
    return (
        d["id"], d["name"], d["description"], d["background"],
        json.dumps(d["sample_texts"], ensure_ascii=False),
        json.dumps(d["traits"], ensure_ascii=False),
        json.dumps(d["speech_pattern"], ensure_ascii=False) if d["speech_pattern"] else None,
        d["system_prompt"],
        json.dumps(d["embedding"]) if d["embedding"] else None,
        json.dumps(d["embedding_metadata"], ensure_ascii=False),
        d["created_at"], d["updated_at"],
    )


def _row_to_char(row: sqlite3.Row) -> CharacterProfile:
    d = dict(row)
    d["sample_texts"] = json.loads(d["sample_texts"] or "[]")
    d["traits"] = json.loads(d["traits"] or "[]")
    d["speech_pattern"] = json.loads(d["speech_pattern"]) if d["speech_pattern"] else None
    d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
    d["embedding_metadata"] = json.loads(d.pop("embedding_meta") or "{}")
    return CharacterProfile(**d)


def _row_to_session(row: sqlite3.Row) -> ChatSession:
    d = dict(row)
    d["messages"] = json.loads(d["messages"] or "[]")
    return ChatSession(**d)


# ── Characters ────────────────────────────────────────────────────────────────

def save_character(character: CharacterProfile) -> None:
    character.updated_at = datetime.utcnow()
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO characters
           (id, name, description, background, sample_texts, traits,
            speech_pattern, system_prompt, embedding, embedding_meta,
            created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        _char_to_row(character),
    )
    conn.commit()


def load_character(character_id: str) -> Optional[CharacterProfile]:
    row = _get_conn().execute(
        "SELECT * FROM characters WHERE id = ?", (character_id,)
    ).fetchone()
    return _row_to_char(row) if row else None


def list_characters() -> list[CharacterProfile]:
    rows = _get_conn().execute(
        "SELECT * FROM characters ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_char(r) for r in rows]


def delete_character(character_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))
    conn.commit()
    return cur.rowcount > 0


# ── Sessions ──────────────────────────────────────────────────────────────────

def save_session(session: ChatSession) -> None:
    d = session.model_dump(mode="json")
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO sessions (id, character_id, messages, created_at)
           VALUES (?,?,?,?)""",
        (d["id"], d["character_id"],
         json.dumps(d["messages"], ensure_ascii=False), d["created_at"]),
    )
    conn.commit()


def load_session(session_id: str) -> Optional[ChatSession]:
    row = _get_conn().execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return _row_to_session(row) if row else None


def list_sessions(character_id: str) -> list[ChatSession]:
    rows = _get_conn().execute(
        "SELECT * FROM sessions WHERE character_id = ? ORDER BY created_at DESC",
        (character_id,),
    ).fetchall()
    return [_row_to_session(r) for r in rows]


def delete_session(session_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return cur.rowcount > 0


# ── Migration from legacy JSON files ──────────────────────────────────────────

def _migrate_from_json() -> int:
    """
    One-time, non-destructive import of old JSON file storage.
    Runs silently on startup; skips records already present in the DB.
    Returns number of characters migrated.
    """
    old_chars = Path(settings.data_dir) / "characters"
    old_sessions = Path(settings.data_dir) / "sessions"
    migrated = 0

    if old_chars.is_dir():
        for p in old_chars.glob("*.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    char = CharacterProfile(**json.load(f))
                if not load_character(char.id):
                    save_character(char)
                    migrated += 1
            except Exception:
                pass

    if old_sessions.is_dir():
        for p in old_sessions.glob("*.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    sess = ChatSession(**json.load(f))
                if not load_session(sess.id):
                    save_session(sess)
            except Exception:
                pass

    return migrated


# Initialise DB and migrate on import
_get_conn()
_migrate_from_json()
