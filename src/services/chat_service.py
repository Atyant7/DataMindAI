"""
Chat persistence service for DataMind AI.

Persists conversations and messages to the database scoped to project and user.
Enforces multi-user isolation across all chat operations.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.logger import get_logger
from src.database.models import ChatRecord, MessageRecord, utc_now
from src.database.session import get_db

logger = get_logger(__name__)


class ChatService:
    """Manages chat sessions and persistent message histories."""

    @staticmethod
    def get_or_create_default_chat(user_id: str, project_id: str) -> dict[str, Any]:
        """
        Get the most recent chat for the user's project, or create one if none exists.
        Enforces user_id and project_id scoping.
        """
        with get_db() as db:
            chat = (
                db.query(ChatRecord)
                .filter(
                    ChatRecord.user_id == user_id,
                    ChatRecord.project_id == project_id,
                )
                .order_by(ChatRecord.updated_at.desc())
                .first()
            )

            if not chat:
                chat = ChatRecord(
                    user_id=user_id,
                    project_id=project_id,
                    title="Main Conversation",
                )
                db.add(chat)
                db.flush()

            return chat.to_dict()

    @staticmethod
    def get_chat(user_id: str, chat_id: str) -> dict[str, Any] | None:
        """Retrieve chat record verifying user ownership."""
        with get_db() as db:
            chat = (
                db.query(ChatRecord)
                .filter(ChatRecord.id == chat_id, ChatRecord.user_id == user_id)
                .first()
            )
            if chat:
                return chat.to_dict()
        return None

    @staticmethod
    def list_project_chats(user_id: str, project_id: str) -> list[dict[str, Any]]:
        """List all chats in a project scoped to user."""
        with get_db() as db:
            chats = (
                db.query(ChatRecord)
                .filter(
                    ChatRecord.user_id == user_id,
                    ChatRecord.project_id == project_id,
                )
                .order_by(ChatRecord.updated_at.desc())
                .all()
            )

            return [c.to_dict() for c in chats]

    @staticmethod
    def create_chat(user_id: str, project_id: str, title: str = "New Chat") -> dict[str, Any]:
        """Create a new chat conversation scoped to user and project."""
        clean_title = title.strip() or "New Chat"
        with get_db() as db:
            chat = ChatRecord(
                user_id=user_id,
                project_id=project_id,
                title=clean_title,
            )
            db.add(chat)
            db.flush()
            return chat.to_dict()

    @staticmethod
    def add_message(
        chat_id: str,
        role: str,
        content: str,
        operation: str | None = None,
        evidence: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist a message to the database with optional user ownership verification.
        """
        evidence_json = json.dumps(evidence, default=str) if evidence else None

        with get_db() as db:
            chat = db.query(ChatRecord).filter(ChatRecord.id == chat_id).first()
            if not chat:
                raise ValueError(f"Chat session '{chat_id}' not found.")

            if user_id and chat.user_id != user_id:
                raise PermissionError("User does not have access to this chat session.")

            msg = MessageRecord(
                chat_id=chat_id,
                role=role,
                content=content,
                operation=operation,
                evidence_json=evidence_json,
            )
            db.add(msg)

            # Update parent chat's updated_at timestamp
            chat.updated_at = utc_now()
            db.flush()
            return msg.to_dict()

    @staticmethod
    def get_chat_messages(chat_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
        """
        Retrieve all messages in a chat in chronological order.
        If user_id is provided, verifies user owns the chat.
        """
        with get_db() as db:
            query = db.query(MessageRecord).filter(MessageRecord.chat_id == chat_id)

            if user_id:
                query = query.join(ChatRecord).filter(ChatRecord.user_id == user_id)

            messages = query.order_by(MessageRecord.created_at.asc()).all()

            result = []
            for m in messages:
                d = m.to_dict()
                if d.get("evidence_json"):
                    try:
                        d["evidence"] = json.loads(d["evidence_json"])
                    except Exception:
                        d["evidence"] = {}
                else:
                    d["evidence"] = {}
                result.append(d)
            return result

    @staticmethod
    def clear_chat_messages(chat_id: str, user_id: str | None = None) -> bool:
        """
        Delete all messages from a chat session verifying user ownership if supplied.
        """
        with get_db() as db:
            if user_id:
                chat = (
                    db.query(ChatRecord)
                    .filter(ChatRecord.id == chat_id, ChatRecord.user_id == user_id)
                    .first()
                )
                if not chat:
                    return False

            db.query(MessageRecord).filter(MessageRecord.chat_id == chat_id).delete()
            return True

    @staticmethod
    def delete_chat(user_id: str, chat_id: str) -> bool:
        """Delete a chat and its messages verifying ownership."""
        with get_db() as db:
            chat = (
                db.query(ChatRecord)
                .filter(ChatRecord.id == chat_id, ChatRecord.user_id == user_id)
                .first()
            )
            if not chat:
                return False

            db.delete(chat)
            return True
