"""
Tests for user activity logging and notification services.
"""

import unittest
import uuid

from src.database.init_db import init_db
from src.services.activity_service import ActivityService
from src.services.auth_service import AuthService
from src.services.notification_service import NotificationService


class TestActivityAndNotifications(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        suffix = uuid.uuid4().hex[:6]
        self.user = AuthService.register_user(
            f"act_user_{suffix}",
            f"act_{suffix}@example.com",
            "password123",
        )

    def test_activity_logging_and_retrieval(self):
        user_id = self.user["id"]

        # Log activities
        a1 = ActivityService.log_activity(
            user_id=user_id,
            action="model_trained",
            title="Trained Random Forest",
            description="Target: churn",
            icon="🧠",
        )
        self.assertIsNotNone(a1["id"])

        a2 = ActivityService.log_activity(
            user_id=user_id,
            action="dataset_uploaded",
            title="Uploaded dataset test.csv",
            icon="📊",
        )
        self.assertIsNotNone(a2["id"])

        # Retrieve
        recent = ActivityService.get_recent_activities(user_id, limit=5)
        self.assertGreaterEqual(len(recent), 2)
        self.assertEqual(recent[0]["action"], "dataset_uploaded")
        self.assertEqual(recent[1]["action"], "model_trained")

    def test_notification_lifecycle(self):
        user_id = self.user["id"]

        n1 = NotificationService.create_notification(
            user_id=user_id,
            title="Model Ready",
            message="AutoML finished evaluating models.",
            level="success",
        )
        self.assertIsNotNone(n1["id"])

        n2 = NotificationService.create_notification(
            user_id=user_id,
            title="Warning",
            message="Missing values detected.",
            level="warning",
        )
        self.assertIsNotNone(n2["id"])

        # Fetch unread
        unread = NotificationService.get_unread_notifications(user_id)
        self.assertEqual(len(unread), 2)

        # Mark single as read
        self.assertTrue(NotificationService.mark_as_read(user_id, n1["id"]))
        unread_after = NotificationService.get_unread_notifications(user_id)
        self.assertEqual(len(unread_after), 1)
        self.assertEqual(unread_after[0]["id"], n2["id"])

        # Mark all as read
        self.assertTrue(NotificationService.mark_all_as_read(user_id))
        unread_final = NotificationService.get_unread_notifications(user_id)
        self.assertEqual(len(unread_final), 0)
