"""
Authentication service for DataMind AI.

Provides:
- Cryptographically secure password hashing (PBKDF2-HMAC-SHA256 + salt)
- User registration and validation
- Credential authentication
- Theme preference persistence
- User isolation
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any

from src.core.logger import get_logger
from src.database.models import User
from src.database.session import get_db

logger = get_logger(__name__)

ITERATIONS = 100_000
SALT_BYTES = 16


def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2-HMAC-SHA256 with a unique random salt.
    Format: pbkdf2_sha256$iterations$salt_hex$hash_hex
    """
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")

    salt = os.urandom(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ITERATIONS,
    )
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its stored PBKDF2 hash using constant-time comparison.
    """
    if not plain_password or not hashed_password:
        return False

    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False

        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_dk = bytes.fromhex(parts[3])

        computed_dk = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(computed_dk, expected_dk)
    except Exception as exc:
        logger.warning("Password verification failed: %s", exc)
        return False


class AuthService:
    """Handles user identity, registration, login, and preferences."""

    @staticmethod
    def register_user(
        username: str,
        email: str,
        password: str,
        theme_preference: str = "dark",
    ) -> dict[str, Any]:
        """
        Register a new user with validation.
        """
        username = username.strip()
        email = email.strip().lower()

        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", username):
            raise ValueError("Username can only contain alphanumeric characters, hyphens, and underscores.")

        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
            raise ValueError("Please provide a valid email address.")

        if not password or len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")

        password_hash = hash_password(password)

        with get_db() as db:
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()

            if existing_user:
                if existing_user.username == username:
                    raise ValueError(f"Username '{username}' is already taken.")
                else:
                    raise ValueError(f"Email '{email}' is already registered.")

            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                theme_preference=theme_preference,
            )
            db.add(new_user)
            db.flush()
            user_data = new_user.to_dict()

        logger.info("User registered successfully: %s (%s)", username, user_data["id"])
        return user_data

    @staticmethod
    def authenticate(username_or_email: str, password: str) -> dict[str, Any] | None:
        """
        Authenticate user by username or email.
        """
        identifier = username_or_email.strip()
        if not identifier or not password:
            return None

        with get_db() as db:
            user = db.query(User).filter(
                (User.username == identifier) | (User.email == identifier.lower())
            ).first()

            if user and verify_password(password, user.password_hash):
                logger.info("User authenticated: %s (%s)", user.username, user.id)
                return user.to_dict()

        logger.warning("Failed authentication attempt for: %s", identifier)
        return None

    @staticmethod
    def get_user_by_id(user_id: str) -> dict[str, Any] | None:
        """Retrieve user details by user ID."""
        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                return user.to_dict()
        return None

    @staticmethod
    def update_theme(user_id: str, theme: str) -> bool:
        """Update and persist theme preference ('light' or 'dark')."""
        if theme not in ("light", "dark"):
            theme = "dark"

        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.theme_preference = theme
                logger.info("Updated theme preference to '%s' for user %s", theme, user_id)
                return True
        return False

    @staticmethod
    def change_password(user_id: str, current_password: str, new_password: str) -> bool:
        """
        Change user password with strict validation and hashing.
        """
        if not new_password or len(new_password) < 6:
            raise ValueError("New password must be at least 6 characters.")

        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found.")

            if not verify_password(current_password, user.password_hash):
                raise ValueError("Incorrect current password.")

            user.password_hash = hash_password(new_password)
            logger.info("Password changed successfully for user %s", user_id)
            return True

    @staticmethod
    def update_profile(
        user_id: str,
        username: str | None = None,
        email: str | None = None,
    ) -> dict[str, Any]:
        """
        Update username and/or email address with validation and conflict detection.
        """
        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found.")

            if username is not None:
                clean_username = username.strip()
                if not clean_username or len(clean_username) < 3:
                    raise ValueError("Username must be at least 3 characters.")
                if not re.match(r"^[a-zA-Z0-9_\-]+$", clean_username):
                    raise ValueError("Username can only contain alphanumeric characters, hyphens, and underscores.")

                existing = db.query(User).filter(User.username == clean_username, User.id != user_id).first()
                if existing:
                    raise ValueError(f"Username '{clean_username}' is already taken.")
                user.username = clean_username

            if email is not None:
                clean_email = email.strip().lower()
                if not re.match(r"^[^@]+@[^@]+\.[^@]+$", clean_email):
                    raise ValueError("Please provide a valid email address.")

                existing = db.query(User).filter(User.email == clean_email, User.id != user_id).first()
                if existing:
                    raise ValueError(f"Email '{clean_email}' is already registered.")
                user.email = clean_email

            db.flush()
            return user.to_dict()

    @staticmethod
    def export_user_data(user_id: str) -> dict[str, Any]:
        """
        Export all non-sensitive user data including projects, datasets metadata,
        chats, models, predictions, and activity audit logs as sanitized JSON.
        """
        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found.")

            export = {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "theme_preference": user.theme_preference,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                },
                "projects": [p.to_dict() for p in user.projects],
                "datasets": [d.to_dict() for d in user.datasets],
                "chats": [
                    {
                        **c.to_dict(),
                        "messages": [m.to_dict() for m in c.messages],
                    }
                    for c in user.chats
                ],
                "models": [m.to_dict() for m in user.models],
                "predictions": [pr.to_dict() for pr in user.predictions],
                "activities": [a.to_dict() for a in user.activities],
            }
            return export

    @staticmethod
    def delete_account(user_id: str, password: str) -> bool:
        """
        Permanently delete user account, stored files, and cascaded database records.
        """
        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found.")

            if not verify_password(password, user.password_hash):
                raise ValueError("Incorrect password.")

            # Attempt deletion of user's physical files
            from src.services.storage_service import LocalStorageService
            storage = LocalStorageService()
            for ds in user.datasets:
                try:
                    storage.delete_file(ds.file_path)
                except Exception:
                    pass

            db.delete(user)
            logger.info("Permanently deleted user account %s", user_id)
            return True
