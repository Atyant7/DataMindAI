"""
Tests for account settings, password changes, profile updates, data export, and account deletion.
"""

import unittest
import uuid

from src.database.init_db import init_db
from src.services.auth_service import AuthService
from src.services.project_service import ProjectService


class TestAccountSettings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        suffix = uuid.uuid4().hex[:6]
        self.user = AuthService.register_user(
            f"acc_user_{suffix}",
            f"acc_{suffix}@example.com",
            "InitialPassword123",
        )

    def test_change_password(self):
        user_id = self.user["id"]

        # Wrong current password fails
        with self.assertRaises(ValueError):
            AuthService.change_password(user_id, "WrongPass", "NewPassword123")

        # Short new password fails
        with self.assertRaises(ValueError):
            AuthService.change_password(user_id, "InitialPassword123", "123")

        # Success
        self.assertTrue(AuthService.change_password(user_id, "InitialPassword123", "NewSecurePassword123"))

        # Verify old password no longer works
        self.assertIsNone(AuthService.authenticate(self.user["username"], "InitialPassword123"))

        # Verify new password works
        auth = AuthService.authenticate(self.user["username"], "NewSecurePassword123")
        self.assertIsNotNone(auth)
        self.assertEqual(auth["id"], user_id)

    def test_update_profile(self):
        user_id = self.user["id"]
        new_name = f"renamed_{uuid.uuid4().hex[:4]}"
        new_mail = f"new_{uuid.uuid4().hex[:4]}@example.com"

        updated = AuthService.update_profile(user_id, username=new_name, email=new_mail)
        self.assertEqual(updated["username"], new_name)
        self.assertEqual(updated["email"], new_mail)

        # Invalid username fails
        with self.assertRaises(ValueError):
            AuthService.update_profile(user_id, username="ab")

        # Invalid email fails
        with self.assertRaises(ValueError):
            AuthService.update_profile(user_id, email="invalid-email")

    def test_export_user_data(self):
        user_id = self.user["id"]
        # Create a sample project to export
        ProjectService.create_project(user_id, "Exported Project", "Description for export")

        export_data = AuthService.export_user_data(user_id)
        self.assertIn("user", export_data)
        self.assertIn("projects", export_data)
        self.assertIn("datasets", export_data)
        self.assertIn("chats", export_data)
        self.assertIn("models", export_data)
        self.assertIn("predictions", export_data)
        self.assertIn("activities", export_data)

        # Ensure NO password hash is exposed
        self.assertNotIn("password_hash", export_data["user"])
        self.assertEqual(len(export_data["projects"]), 1)
        self.assertEqual(export_data["projects"][0]["name"], "Exported Project")

    def test_delete_account(self):
        user_id = self.user["id"]
        ProjectService.create_project(user_id, "Project to be purged")

        # Wrong password fails
        with self.assertRaises(ValueError):
            AuthService.delete_account(user_id, "WrongPass")

        # Success
        self.assertTrue(AuthService.delete_account(user_id, "InitialPassword123"))

        # User no longer exists
        self.assertIsNone(AuthService.get_user_by_id(user_id))
        self.assertIsNone(AuthService.authenticate(self.user["username"], "InitialPassword123"))
