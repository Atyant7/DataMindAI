"""
Tests for storage service and security protections.
"""

import tempfile
import unittest
from pathlib import Path

from src.services.storage_service import LocalStorageService, sanitize_filename


class TestStorageService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = LocalStorageService(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("../../etc/passwd"), "passwd")
        self.assertEqual(sanitize_filename("my data file (1).csv"), "my_data_file__1_.csv")
        self.assertEqual(sanitize_filename(".hidden_file.txt"), "hidden_file.txt")

    def test_save_and_retrieve_file(self):
        content = b"col1,col2\n1,2\n3,4"
        rel_path = self.storage.save_file(content, "test_folder", "test.csv")

        full_path = self.storage.get_file_path(rel_path)
        self.assertTrue(full_path.exists())
        self.assertEqual(full_path.read_bytes(), content)

        # Delete
        self.assertTrue(self.storage.delete_file(rel_path))
        self.assertFalse(full_path.exists())

    def test_path_traversal_prevention(self):
        with self.assertRaises(ValueError):
            self.storage.get_file_path("../../../etc/passwd")
