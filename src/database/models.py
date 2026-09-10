"""
SQLAlchemy ORM models for DataMind AI.

Tables:
- User: Authentication, credentials, user theme
- Project: Workspaces scoped to user
- DatasetRecord: Uploaded datasets metadata and storage paths
- ChatRecord: Conversations scoped to project and user
- MessageRecord: Chat messages with roles and evidence
- ModelRecord: Model Registry records with versioning and metadata
- PredictionRecord: Historical predictions with inputs and outputs
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a UUID4 string."""
    return str(uuid.uuid4())


def utc_now() -> datetime.datetime:
    """Return timezone-naive UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class User(Base):
    """User account model for authentication and data isolation."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    theme_preference = Column(String(20), default="dark", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    datasets = relationship("DatasetRecord", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("ChatRecord", back_populates="user", cascade="all, delete-orphan")
    models = relationship("ModelRecord", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="user", cascade="all, delete-orphan")
    activities = relationship("ActivityRecord", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("NotificationRecord", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "theme_preference": self.theme_preference,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Project(Base):
    """Project workspace model."""

    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    is_archived = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    user = relationship("User", back_populates="projects")
    datasets = relationship("DatasetRecord", back_populates="project", cascade="all, delete-orphan")
    chats = relationship("ChatRecord", back_populates="project", cascade="all, delete-orphan")
    models = relationship("ModelRecord", back_populates="project", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="project", cascade="all, delete-orphan")
    activities = relationship("ActivityRecord", back_populates="project", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "is_archived": self.is_archived,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatasetRecord(Base):
    """Uploaded dataset metadata and storage locator."""

    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    rows = Column(Integer, default=0, nullable=False)
    columns = Column(Integer, default=0, nullable=False)
    signature = Column(String(100), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="datasets")
    project = relationship("Project", back_populates="datasets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "name": self.name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "rows": self.rows,
            "columns": self.columns,
            "signature": self.signature,
            "metadata_json": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatRecord(Base):
    """Chat session model associated with a project."""

    __tablename__ = "chats"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), default="New Chat", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    user = relationship("User", back_populates="chats")
    project = relationship("Project", back_populates="chats")
    messages = relationship("MessageRecord", back_populates="chat", cascade="all, delete-orphan", order_by="MessageRecord.created_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MessageRecord(Base):
    """Message within a chat session."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    operation = Column(String(100), nullable=True)
    evidence_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    chat = relationship("ChatRecord", back_populates="messages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "role": self.role,
            "content": self.content,
            "operation": self.operation,
            "evidence_json": self.evidence_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelRecord(Base):
    """Model Registry record."""

    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(String(36), nullable=True)
    model_name = Column(String(100), nullable=False)
    display_name = Column(String(200), nullable=False)
    model_type = Column(String(50), default="tabular", nullable=False)  # "tabular", "image", "geospatial"
    framework = Column(String(50), default="scikit-learn", nullable=False)
    task = Column(String(50), nullable=False)  # "classification" or "regression"
    target_column = Column(String(100), nullable=False)
    feature_names_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    training_config_json = Column(Text, nullable=True)
    artifact_path = Column(String(500), nullable=False)
    metadata_path = Column(String(500), nullable=False)
    version = Column(Integer, default=1, nullable=False)
    is_best = Column(Boolean, default=False, nullable=False)
    is_image = Column(Boolean, default=False, nullable=False)
    is_geospatial = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="models")
    project = relationship("Project", back_populates="models")
    predictions = relationship("PredictionRecord", back_populates="model", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "framework": self.framework,
            "task": self.task,
            "target_column": self.target_column,
            "feature_names_json": self.feature_names_json,
            "metrics_json": self.metrics_json,
            "training_config_json": self.training_config_json,
            "artifact_path": self.artifact_path,
            "metadata_path": self.metadata_path,
            "version": self.version,
            "is_best": self.is_best,
            "is_image": self.is_image,
            "is_geospatial": self.is_geospatial,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PredictionRecord(Base):
    """Historical record of predictions."""

    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id = Column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    input_data_json = Column(Text, nullable=False)
    prediction_result = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=True)
    probabilities_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="predictions")
    project = relationship("Project", back_populates="predictions")
    model = relationship("ModelRecord", back_populates="predictions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "model_id": self.model_id,
            "input_data_json": self.input_data_json,
            "prediction_result": self.prediction_result,
            "confidence": self.confidence,
            "probabilities_json": self.probabilities_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ActivityRecord(Base):
    """Activity tracking audit log for user actions."""

    __tablename__ = "activities"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)
    action = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    icon = Column(String(50), default="⚡", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="activities")
    project = relationship("Project", back_populates="activities")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "action": self.action,
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class NotificationRecord(Base):
    """System and action notification model."""

    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    level = Column(String(20), default="info", nullable=False)  # success, info, warning, error
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="notifications")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "level": self.level,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
