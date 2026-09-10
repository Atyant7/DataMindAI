"""
Activity tracking audit log service for DataMind AI.

Records and retrieves chronological user actions such as:
- Project creation / rename / archive / delete
- Dataset uploads
- Model training events
- Prediction requests
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger
from src.database.models import ActivityRecord
from src.database.session import get_db

logger = get_logger(__name__)


class ActivityService:
    """Service to record and query user workspace activities."""

    @staticmethod
    def log_activity(
        user_id: str,
        action: str,
        title: str,
        description: str | None = None,
        project_id: str | None = None,
        icon: str = "⚡",
    ) -> dict[str, Any]:
        """
        Record a new user activity into the audit log.
        """
        with get_db() as db:
            activity = ActivityRecord(
                user_id=user_id,
                project_id=project_id,
                action=action,
                title=title,
                description=description,
                icon=icon,
            )
            db.add(activity)
            db.flush()
            result = activity.to_dict()

        logger.debug("Logged activity: %s for user %s", action, user_id)
        return result

    @staticmethod
    def get_recent_activities(
        user_id: str,
        limit: int = 10,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent activities belonging strictly to the authenticated user.
        """
        with get_db() as db:
            query = db.query(ActivityRecord).filter(ActivityRecord.user_id == user_id)
            if project_id:
                query = query.filter(ActivityRecord.project_id == project_id)

            records = query.order_by(ActivityRecord.created_at.desc()).limit(limit).all()
            return [r.to_dict() for r in records]

    @staticmethod
    def clear_user_activities(user_id: str) -> bool:
        """Clear all activities for a user."""
        with get_db() as db:
            db.query(ActivityRecord).filter(ActivityRecord.user_id == user_id).delete()
            return True
