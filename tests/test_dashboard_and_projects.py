"""
Tests for user dashboard statistics, project search, filtering, sorting,
pagination, archiving, duplication, and multi-user isolation.
"""

import unittest
import uuid

from src.database.init_db import init_db
from src.services.activity_service import ActivityService
from src.services.auth_service import AuthService
from src.services.model_registry_service import ModelRegistryService
from src.services.project_service import ProjectService


class TestDashboardAndProjects(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

        s1 = uuid.uuid4().hex[:6]
        s2 = uuid.uuid4().hex[:6]
        cls.user_a = AuthService.register_user(f"dash_a_{s1}", f"dash_a_{s1}@test.com", "password123")
        cls.user_b = AuthService.register_user(f"dash_b_{s2}", f"dash_b_{s2}@test.com", "password123")

    def test_dashboard_stats_and_creation(self):
        user_id = self.user_a["id"]

        # Initial stats
        stats_before = ProjectService.get_user_dashboard_stats(user_id)

        # Create project
        proj = ProjectService.create_project(user_id, "Cardio Health", "Predicting cardiac events")
        self.assertIsNotNone(proj["id"])

        # Check stats increment
        stats_after = ProjectService.get_user_dashboard_stats(user_id)
        self.assertEqual(stats_after["projects"], stats_before["projects"] + 1)

        # Verify activity was logged
        activities = ActivityService.get_recent_activities(user_id, limit=5)
        self.assertTrue(any(a["action"] == "project_created" and "Cardio Health" in a["title"] for a in activities))

    def test_project_archiving_and_restoring(self):
        user_id = self.user_a["id"]
        proj = ProjectService.create_project(user_id, "Temporary Analysis")

        # Active listing contains it
        active_projects = ProjectService.list_user_projects(user_id, include_archived=False)
        self.assertTrue(any(p["id"] == proj["id"] for p in active_projects))

        # Archive project
        self.assertTrue(ProjectService.archive_project(user_id, proj["id"], archive=True))

        # Active listing NO LONGER contains it
        active_after = ProjectService.list_user_projects(user_id, include_archived=False)
        self.assertFalse(any(p["id"] == proj["id"] for p in active_after))

        # Archived listing DOES contain it
        archived_list = ProjectService.list_user_projects(user_id, archived_only=True)
        self.assertTrue(any(p["id"] == proj["id"] for p in archived_list))

        # Restore project
        self.assertTrue(ProjectService.archive_project(user_id, proj["id"], archive=False))
        restored_list = ProjectService.list_user_projects(user_id, include_archived=False)
        self.assertTrue(any(p["id"] == proj["id"] for p in restored_list))

    def test_project_duplication(self):
        user_id = self.user_a["id"]
        original = ProjectService.create_project(user_id, "Original Master", "Original project description")

        duplicate = ProjectService.duplicate_project(user_id, original["id"], "Cloned Project")
        self.assertIsNotNone(duplicate)
        self.assertNotEqual(duplicate["id"], original["id"])
        self.assertEqual(duplicate["name"], "Cloned Project")

        # Both projects exist in user's list
        all_projs = ProjectService.list_user_projects(user_id)
        names = [p["name"] for p in all_projs]
        self.assertIn("Original Master", names)
        self.assertIn("Cloned Project", names)

    def test_project_search_sort_and_pagination(self):
        user_id = self.user_a["id"]

        # Create distinct projects
        ProjectService.create_project(user_id, "Alpha Project")
        ProjectService.create_project(user_id, "Beta Project")
        ProjectService.create_project(user_id, "Gamma Project")

        # Search
        search_res = ProjectService.list_user_projects(user_id, search="Alpha")
        self.assertTrue(any(p["name"] == "Alpha Project" for p in search_res))
        self.assertFalse(any(p["name"] == "Beta Project" for p in search_res))

        # Sort Name A-Z
        sorted_asc = ProjectService.list_user_projects(user_id, sort_by="name_asc")
        names_asc = [p["name"] for p in sorted_asc if p["name"] in ("Alpha Project", "Beta Project", "Gamma Project")]
        self.assertEqual(names_asc, ["Alpha Project", "Beta Project", "Gamma Project"])

        # Pagination
        paginated = ProjectService.list_user_projects(user_id, page=1, page_size=2)
        self.assertIn("projects", paginated)
        self.assertIn("total", paginated)
        self.assertIn("pages", paginated)
        self.assertEqual(len(paginated["projects"]), 2)
        self.assertGreaterEqual(paginated["pages"], 2)

    def test_multi_user_isolation(self):
        # User A creates Project A
        proj_a = ProjectService.create_project(self.user_a["id"], "User A Secret Workspace")

        # User B creates Project B
        proj_b = ProjectService.create_project(self.user_b["id"], "User B Secret Workspace")

        # Verify User A only sees Project A
        projs_a = ProjectService.list_user_projects(self.user_a["id"])
        ids_a = [p["id"] for p in projs_a]
        self.assertIn(proj_a["id"], ids_a)
        self.assertNotIn(proj_b["id"], ids_a)

        # Verify User B only sees Project B
        projs_b = ProjectService.list_user_projects(self.user_b["id"])
        ids_b = [p["id"] for p in projs_b]
        self.assertIn(proj_b["id"], ids_b)
        self.assertNotIn(proj_a["id"], ids_b)

        # User B cannot access User A's summary or delete it
        self.assertIsNone(ProjectService.get_project_summary(self.user_b["id"], proj_a["id"]))
        self.assertFalse(ProjectService.delete_project(self.user_b["id"], proj_a["id"]))
