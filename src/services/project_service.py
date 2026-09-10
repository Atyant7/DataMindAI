"""
Project management service for DataMind AI.

Manages user projects with strict multi-user isolation, metadata enrichment,
search, filtering, sorting, pagination, archiving, and duplication.
"""

from __future__ import annotations

import json
import math
from typing import Any

from sqlalchemy import func

from src.core.logger import get_logger
from src.database.models import (
    ChatRecord,
    DatasetRecord,
    ModelRecord,
    PredictionRecord,
    Project,
)
from src.database.session import get_db
from src.services.activity_service import ActivityService

logger = get_logger(__name__)


class ProjectService:
    """Handles project lifecycle and user-scoped workspaces."""

    @staticmethod
    def create_project(
        user_id: str,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new project for the given user.
        Also creates an initial default chat and logs activity.
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name cannot be empty.")

        with get_db() as db:
            project = Project(
                user_id=user_id,
                name=clean_name,
                description=description.strip() if description else None,
                is_archived=False,
            )
            db.add(project)
            db.flush()

            # Create default chat for this project
            default_chat = ChatRecord(
                user_id=user_id,
                project_id=project.id,
                title="General Analysis",
            )
            db.add(default_chat)
            db.flush()

            result = project.to_dict()

        logger.info("Created project '%s' (id: %s) for user %s", clean_name, result["id"], user_id)

        try:
            ActivityService.log_activity(
                user_id=user_id,
                project_id=result["id"],
                action="project_created",
                title=f"Created project '{clean_name}'",
                description=description,
                icon="📁",
            )
        except Exception as exc:
            logger.warning("Could not log activity for project creation: %s", exc)

        return result

    @staticmethod
    def get_project(user_id: str, project_id: str) -> dict[str, Any] | None:
        """
        Retrieve a project only if it belongs to the authenticated user.
        Enforces user isolation.
        """
        with get_db() as db:
            project = (
                db.query(Project)
                .filter(
                    Project.id == project_id,
                    Project.user_id == user_id,
                )
                .first()
            )
            if project:
                return project.to_dict()

        return None

    @staticmethod
    def get_user_dashboard_stats(user_id: str) -> dict[str, int]:
        """
        Compute high-performance database aggregated statistics for the user's dashboard.
        """
        with get_db() as db:
            proj_count = (
                db.query(func.count(Project.id))
                .filter(Project.user_id == user_id, Project.is_archived == False)  # noqa: E712
                .scalar()
                or 0
            )
            dataset_count = (
                db.query(func.count(DatasetRecord.id))
                .filter(DatasetRecord.user_id == user_id)
                .scalar()
                or 0
            )
            model_count = (
                db.query(func.count(ModelRecord.id))
                .filter(ModelRecord.user_id == user_id)
                .scalar()
                or 0
            )
            pred_count = (
                db.query(func.count(PredictionRecord.id))
                .filter(PredictionRecord.user_id == user_id)
                .scalar()
                or 0
            )

            return {
                "projects": int(proj_count),
                "datasets": int(dataset_count),
                "models": int(model_count),
                "predictions": int(pred_count),
            }

    @staticmethod
    def get_project_summary(user_id: str, project_id: str) -> dict[str, Any] | None:
        """
        Retrieve a comprehensive summary for a project card, including
        dataset details, best model, problem task, and performance metric.
        """
        with get_db() as db:
            project = (
                db.query(Project)
                .filter(Project.id == project_id, Project.user_id == user_id)
                .first()
            )
            if not project:
                return None

            summary = project.to_dict()

            # 1. Dataset information
            latest_dataset = (
                db.query(DatasetRecord)
                .filter(DatasetRecord.project_id == project_id, DatasetRecord.user_id == user_id)
                .order_by(DatasetRecord.created_at.desc())
                .first()
            )
            if latest_dataset:
                summary["dataset_name"] = latest_dataset.name
                summary["dataset_rows"] = latest_dataset.rows
                summary["dataset_columns"] = latest_dataset.columns
            else:
                summary["dataset_name"] = None
                summary["dataset_rows"] = 0
                summary["dataset_columns"] = 0

            # 2. Models information & Best Model
            models = (
                db.query(ModelRecord)
                .filter(ModelRecord.project_id == project_id, ModelRecord.user_id == user_id)
                .order_by(ModelRecord.is_best.desc(), ModelRecord.created_at.desc())
                .all()
            )
            summary["model_count"] = len(models)

            if models:
                best = models[0]
                summary["best_model"] = best.display_name
                summary["task"] = best.task
                summary["target_column"] = best.target_column
                summary["is_image"] = best.is_image
                summary["is_geospatial"] = best.is_geospatial

                # Parse metric
                score_str = None
                if best.metrics_json:
                    try:
                        metrics = json.loads(best.metrics_json)
                        if metrics:
                            metric_name, metric_val = next(iter(metrics.items()))
                            score_str = f"{metric_name.upper()}: {metric_val:.4f}" if isinstance(metric_val, (int, float)) else f"{metric_name.upper()}: {metric_val}"
                    except Exception:
                        pass
                summary["best_metric"] = score_str
            else:
                summary["best_model"] = None
                summary["task"] = None
                summary["target_column"] = None
                summary["best_metric"] = None
                summary["is_image"] = False
                summary["is_geospatial"] = False

            # 3. Prediction count
            pred_count = (
                db.query(func.count(PredictionRecord.id))
                .filter(PredictionRecord.project_id == project_id, PredictionRecord.user_id == user_id)
                .scalar()
                or 0
            )
            summary["prediction_count"] = int(pred_count)

            return summary

    @staticmethod
    def list_user_projects(
        user_id: str,
        include_archived: bool = False,
        archived_only: bool = False,
        search: str | None = None,
        task_filter: str | None = None,
        sort_by: str = "updated_at",
        page: int | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        List user projects with search, task filtering, sorting, and optional pagination.
        """
        with get_db() as db:
            query = db.query(Project).filter(Project.user_id == user_id)

            if archived_only:
                query = query.filter(Project.is_archived == True)  # noqa: E712
            elif not include_archived:
                query = query.filter(Project.is_archived == False)  # noqa: E712

            if search and search.strip():
                clean_term = f"%{search.strip().lower()}%"
                query = query.filter(func.lower(Project.name).like(clean_term))

            # Task filter (classification, regression, vision, geospatial)
            if task_filter and task_filter.lower() not in ("all", "all tasks"):
                filter_val = task_filter.lower()
                if filter_val in ("computer vision", "vision", "image"):
                    query = query.join(ModelRecord).filter(ModelRecord.is_image == True)  # noqa: E712
                elif filter_val in ("geospatial", "geo"):
                    query = query.join(ModelRecord).filter(ModelRecord.is_geospatial == True)  # noqa: E712
                else:
                    query = query.join(ModelRecord).filter(func.lower(ModelRecord.task) == filter_val)

            # Sorting
            if sort_by == "created_at":
                query = query.order_by(Project.created_at.desc())
            elif sort_by in ("name", "name_asc"):
                query = query.order_by(Project.name.asc())
            elif sort_by == "name_desc":
                query = query.order_by(Project.name.desc())
            else:
                query = query.order_by(Project.updated_at.desc())

            # Pagination if requested
            if page is not None and page_size is not None and page_size > 0:
                total = query.count()
                total_pages = max(1, math.ceil(total / page_size))
                current_page = max(1, min(page, total_pages))
                offset = (current_page - 1) * page_size
                projects = query.offset(offset).limit(page_size).all()

                project_summaries = []
                for p in projects:
                    summary = ProjectService.get_project_summary(user_id, p.id)
                    if summary:
                        project_summaries.append(summary)

                return {
                    "projects": project_summaries,
                    "total": total,
                    "page": current_page,
                    "pages": total_pages,
                    "page_size": page_size,
                }

            projects = query.all()
            project_summaries = []
            for p in projects:
                summary = ProjectService.get_project_summary(user_id, p.id)
                if summary:
                    project_summaries.append(summary)

            return project_summaries

    @staticmethod
    def update_project(
        user_id: str,
        project_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any] | None:
        """Update project details with ownership check."""
        with get_db() as db:
            project = (
                db.query(Project)
                .filter(Project.id == project_id, Project.user_id == user_id)
                .first()
            )
            if not project:
                return None

            if name is not None:
                clean_name = name.strip()
                if not clean_name:
                    raise ValueError("Project name cannot be empty.")
                project.name = clean_name

            if description is not None:
                project.description = description.strip()

            db.flush()
            result = project.to_dict()

        try:
            ActivityService.log_activity(
                user_id=user_id,
                project_id=project_id,
                action="project_renamed",
                title=f"Renamed project to '{result['name']}'",
                icon="✏️",
            )
        except Exception:
            pass

        return result

    @staticmethod
    def archive_project(
        user_id: str,
        project_id: str,
        archive: bool = True,
    ) -> bool:
        """Archive or unarchive a project."""
        with get_db() as db:
            project = (
                db.query(Project)
                .filter(Project.id == project_id, Project.user_id == user_id)
                .first()
            )
            if not project:
                return False

            project.is_archived = archive
            db.flush()
            proj_name = project.name

        action_name = "project_archived" if archive else "project_restored"
        title = f"Archived project '{proj_name}'" if archive else f"Restored project '{proj_name}'"
        icon = "📦" if archive else "♻️"

        try:
            ActivityService.log_activity(
                user_id=user_id,
                project_id=project_id,
                action=action_name,
                title=title,
                icon=icon,
            )
        except Exception:
            pass

        return True

    @staticmethod
    def duplicate_project(
        user_id: str,
        project_id: str,
        new_name: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Duplicate an existing project and its dataset reference.
        """
        with get_db() as db:
            original = (
                db.query(Project)
                .filter(Project.id == project_id, Project.user_id == user_id)
                .first()
            )
            if not original:
                return None

            dup_name = new_name.strip() if new_name else f"Copy of {original.name}"
            new_proj = Project(
                user_id=user_id,
                name=dup_name,
                description=original.description,
                is_archived=False,
            )
            db.add(new_proj)
            db.flush()

            # Create default chat
            chat = ChatRecord(
                user_id=user_id,
                project_id=new_proj.id,
                title="General Analysis",
            )
            db.add(chat)

            # Duplicate dataset reference if present
            orig_ds = (
                db.query(DatasetRecord)
                .filter(DatasetRecord.project_id == original.id)
                .order_by(DatasetRecord.created_at.desc())
                .first()
            )
            if orig_ds:
                dup_ds = DatasetRecord(
                    user_id=user_id,
                    project_id=new_proj.id,
                    name=orig_ds.name,
                    file_path=orig_ds.file_path,
                    file_type=orig_ds.file_type,
                    rows=orig_ds.rows,
                    columns=orig_ds.columns,
                    signature=orig_ds.signature,
                    metadata_json=orig_ds.metadata_json,
                )
                db.add(dup_ds)

            db.flush()
            result = new_proj.to_dict()

        try:
            ActivityService.log_activity(
                user_id=user_id,
                project_id=result["id"],
                action="project_duplicated",
                title=f"Duplicated project as '{dup_name}'",
                icon="📋",
            )
        except Exception:
            pass

        return result

    @staticmethod
    def delete_project(user_id: str, project_id: str) -> bool:
        """
        Delete a project and all its associated database records with ownership check.
        """
        with get_db() as db:
            project = (
                db.query(Project)
                .filter(Project.id == project_id, Project.user_id == user_id)
                .first()
            )
            if not project:
                return False

            proj_name = project.name

            # Explicit cleanup of child chat records to guarantee no orphaned messages
            chats = db.query(ChatRecord).filter(ChatRecord.project_id == project_id, ChatRecord.user_id == user_id).all()
            for c in chats:
                db.query(MessageRecord).filter(MessageRecord.chat_id == c.id).delete()
                db.delete(c)

            db.delete(project)
            logger.info("Deleted project %s for user %s", project_id, user_id)

        try:
            ActivityService.log_activity(
                user_id=user_id,
                project_id=None,
                action="project_deleted",
                title=f"Deleted project '{proj_name}'",
                icon="🗑",
            )
        except Exception:
            pass

        return True
