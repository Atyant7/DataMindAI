"""
Tests for authentication, password hashing, and user preferences.
"""

import unittest
import uuid

from src.database.init_db import init_db
from src.services.auth_service import (
    AuthService,
    hash_password,
    verify_password,
)


class TestAuthService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_password_hashing_and_verification(self):
        password = "SecurePassword123!"
        hashed = hash_password(password)

        self.assertTrue(hashed.startswith("pbkdf2_sha256$"))
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))
        self.assertFalse(verify_password("", hashed))

    def test_user_registration_and_authentication(self):
        unique_suffix = uuid.uuid4().hex[:8]
        username = f"testuser_{unique_suffix}"
        email = f"test_{unique_suffix}@example.com"
        password = "MyPassword123"

        # Register
        user = AuthService.register_user(
            username=username,
            email=email,
            password=password,
            theme_preference="light",
        )
        self.assertIsNotNone(user["id"])
        self.assertEqual(user["username"], username)
        self.assertEqual(user["theme_preference"], "light")

        # Authenticate with username
        auth_user = AuthService.authenticate(username, password)
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user["id"], user["id"])

        # Authenticate with email
        auth_email = AuthService.authenticate(email, password)
        self.assertIsNotNone(auth_email)
        self.assertEqual(auth_email["id"], user["id"])

        # Invalid credentials
        self.assertIsNone(AuthService.authenticate(username, "BadPassword"))

    def test_duplicate_registration_rejected(self):
        unique_suffix = uuid.uuid4().hex[:8]
        username = f"dupuser_{unique_suffix}"
        email = f"dup_{unique_suffix}@example.com"

        AuthService.register_user(username=username, email=email, password="password123")

        with self.assertRaises(ValueError):
            AuthService.register_user(username=username, email=f"other_{unique_suffix}@example.com", password="password123")

        with self.assertRaises(ValueError):
            AuthService.register_user(username=f"other_{unique_suffix}", email=email, password="password123")

    def test_theme_preference_update(self):
        unique_suffix = uuid.uuid4().hex[:8]
        user = AuthService.register_user(
            username=f"theme_{unique_suffix}",
            email=f"theme_{unique_suffix}@example.com",
            password="password123",
            theme_preference="dark",
        )
        self.assertEqual(user["theme_preference"], "dark")

        updated = AuthService.update_theme(user["id"], "light")
        self.assertTrue(updated)

        fetched = AuthService.get_user_by_id(user["id"])
        self.assertEqual(fetched["theme_preference"], "light")
