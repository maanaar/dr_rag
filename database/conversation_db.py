# database/conversation_db.py

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


# Get the dr_rag directory (parent of database directory)
BASE_DIR = Path(__file__).parent.parent
DB_FILE = BASE_DIR / "data" / "conversations.db"


def init_database():
    """Initialize the SQLite database with conversation summaries table"""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            summary TEXT NOT NULL,
            last_bot_response TEXT,
            speciality TEXT,
            message_count INTEGER NOT NULL,
            conversation_data TEXT,
            metadata TEXT
        )
    """)
    
    # Add speciality column if it doesn't exist (migration for existing databases)
    try:
        cursor.execute("ALTER TABLE conversation_summaries ADD COLUMN speciality TEXT")
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Add user_id column if it doesn't exist (migration for existing databases)
    try:
        cursor.execute("ALTER TABLE conversation_summaries ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Add last_bot_response column if it doesn't exist (migration for existing databases)
    try:
        cursor.execute("ALTER TABLE conversation_summaries ADD COLUMN last_bot_response TEXT")
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Create index on user_id for faster lookups (for Meta Messenger user filtering)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id 
        ON conversation_summaries(user_id)
    """)
    
    # Create index on session_id for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id 
        ON conversation_summaries(session_id)
    """)
    
    # Create index on created_at for date-based queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at 
        ON conversation_summaries(created_at)
    """)
    
    conn.commit()
    conn.close()


def save_conversation_summary(session_id: str, summary: str, 
                              conversation_history: List[Dict[str, str]],
                              metadata: Optional[Dict[str, Any]] = None,
                              last_bot_response: Optional[str] = None,
                              speciality: Optional[str] = None) -> int:
    """
    Save a conversation summary to the database.
    
    Args:
        session_id: Unique identifier for the conversation session
        summary: The summarized conversation text
        conversation_history: Full conversation history
        metadata: Optional metadata (e.g., user info, topics, etc.)
        last_bot_response: The last response from the bot/assistant
        speciality: The first speciality chosen by the LLM in a search
    
    Returns:
        The ID of the inserted record
    """
    init_database()  # Ensure database exists
    
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    # Extract user_id from Meta Messenger metadata
    # Meta Messenger typically provides user ID in various formats
    user_id = None
    if metadata:
        # Try different possible keys from Meta Messenger
        user_id = (metadata.get("user_id") or 
                  metadata.get("userId") or 
                  metadata.get("user") or
                  metadata.get("sender_id") or
                  metadata.get("senderId") or
                  metadata.get("sender") or
                  metadata.get("psid") or  # Page-Scoped ID from Meta
                  metadata.get("page_scoped_id") or
                  metadata.get("messenger_user_id"))
    
    # Extract last bot response from conversation history if not provided
    if not last_bot_response and conversation_history:
        # Find the last assistant message
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_bot_response = msg.get("content", "")
                break
    
    # Extract speciality from metadata if not provided
    if not speciality and metadata:
        speciality = metadata.get("first_speciality") or metadata.get("speciality")
    
    # Convert conversation history and metadata to JSON strings
    conversation_json = json.dumps(conversation_history, ensure_ascii=False)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    
    cursor.execute("""
        INSERT INTO conversation_summaries 
        (user_id, session_id, summary, last_bot_response, speciality, message_count, conversation_data, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        session_id,
        summary,
        last_bot_response,
        speciality,
        len(conversation_history),
        conversation_json,
        metadata_json
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id


def get_conversation_summaries(limit: int = 10, 
                               session_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve conversation summaries from the database.
    
    Args:
        limit: Maximum number of summaries to retrieve
        session_id: Optional filter by session_id
        user_id: Optional filter by user_id (from Meta Messenger)
    
    Returns:
        List of summary dictionaries
    """
    init_database()
    
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()
    
    # Build query based on filters
    query = "SELECT * FROM conversation_summaries WHERE 1=1"
    params = []
    
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert rows to dictionaries
    summaries = []
    for row in rows:
        summary_dict = {
            "id": row["id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "summary": row["summary"],
            "last_bot_response": row.get("last_bot_response", ""),
            "speciality": row.get("speciality", ""),
            "message_count": row["message_count"],
            "conversation_data": json.loads(row["conversation_data"]) if row["conversation_data"] else [],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
        summaries.append(summary_dict)
    
    return summaries


def get_conversation_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific conversation summary by ID"""
    init_database()
    
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM conversation_summaries
        WHERE id = ?
    """, (record_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "session_id": row["session_id"],
        "created_at": row["created_at"],
        "summary": row["summary"],
        "last_bot_response": row.get("last_bot_response", ""),
        "speciality": row.get("speciality", ""),
        "message_count": row["message_count"],
        "conversation_data": json.loads(row["conversation_data"]) if row["conversation_data"] else [],
        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
    }


def delete_conversation(record_id: int) -> bool:
    """Delete a conversation summary by ID. Returns True if deleted, False if not found."""
    init_database()
    
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM conversation_summaries
        WHERE id = ?
    """, (record_id,))
    
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return deleted

