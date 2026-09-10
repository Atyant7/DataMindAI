"""
Notification service for DataMind AI.

Manages lightweight user notifications and action feedback.
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger
from src.database.models import NotificationRecord
from src.database.session import get_db

logger = get_logger(__name__)


class NotificationService:
    """Service to create, read, and dismiss user notifications."""

    @staticmethod
    def create_notification(
        user_id: str,
        title: str,
        message: str,
        level: str = "info",
    ) -> dict[str, Any]:
        """Create a new notification for the given user."""
        if level not in ("success", "info", "warning", "error"):
            level = "info"

        with get_db() as db:
            notif = NotificationRecord(
                user_id=user_id,
                title=title,
                message=message,
                level=level,
            )
            db.add(notif)
            db.flush()
            return notif.to_dict()

    @staticmethod
    def get_unread_notifications(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Fetch unread notifications for a user."""
        with get_db() as db:
            records = (
                db.query(NotificationRecord)
                .filter(
                    NotificationRecord.user_id == user_id,
                    NotificationRecord.is_read == False,  # noqa: E712
                )
                .order_by(NotificationRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in records]

    @staticmethod
    def get_user_notifications(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch all notifications for a user."""
        with get_db() as db:
            records = (
                db.query(NotificationRecord)
                .filter(NotificationRecord.user_id == user_id)
                .order_by(NotificationRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in records]

    @staticmethod
    def mark_as_read(user_id: str, notification_id: str) -> bool:
        """Mark a single notification as read."""
        with get_db() as db:
            notif = (
                db.query(NotificationRecord)
                .filter(
                    NotificationRecord.id == notification_id,
                    NotificationRecord.user_id == user_id,
                )
                .first()
            )
            if notif:
                notif.is_read = True
                return True
        return False

    @staticmethod
    def mark_all_as_read(user_id: str) -> bool:
        """Mark all unread notifications for a user as read."""
        with get_db() as db:
            db.query(NotificationRecord).filter(
                NotificationRecord.user_id == user_id,
                NotificationRecord.is_read == False,  # noqa: E712
            ).update({"is_read": True})
            return True
