from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from loguru import logger


class TestBase(unittest.TestCase):
    """Base TestCase that provides a class-level temporary directory.

    The directory is created once for the test class and removed after all
    tests in the class have run.

    Attributes:
        temp_dir: Directory available to all test methods in the class.

    Example:
        ```python
        class TestMyLogic(TempDirTestCase):
            def test_write_file(self):
                file_path = self.temp_dir / "example.txt"
                file_path.write_text("hello")
                self.assertEqual(file_path.read_text(), "hello")
        ```
    """

    temp_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary directory for the whole test class."""
        super().setUpClass()
        logger.info(f"Creating temp directory for {cls.__name__}…")
        temp_dir_str = tempfile.mkdtemp(prefix=f"{cls.__name__}_")
        cls.temp_dir = Path(temp_dir_str)

        # Ensure cleanup even if tearDownClass isn't reached for some reason
        cls.addClassCleanup(cls._cleanup_temp_dir)

    @classmethod
    def _cleanup_temp_dir(cls) -> None:
        """Internal helper for cleaning up the temp directory."""
        logger.info(f"Removing temp directory for {cls.__name__}…")
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        """Trigger cleanup when the test class is done."""
        try:
            cls._cleanup_temp_dir()
        finally:
            super().tearDownClass()

    def setUp(self) -> None:
        """Prepare per-test state.

        Called before every test method.
        """
        super().setUp()
        logger.debug(f"Starting test: {self.id()}")
        # Example: you could create a per-test subfolder here if you like
        self.test_subdir = self.temp_dir / self._safe_test_method_name()
        self.test_subdir.mkdir(exist_ok=True)
        self.addCleanup(self._cleanup_test_subdir)

    def tearDown(self) -> None:
        """Clean up per-test state."""
        logger.debug(f"Finished test: {self.id()}")
        super().tearDown()

    def _cleanup_test_subdir(self) -> None:
        """Internal helper to cleanup per-test subdir."""
        shutil.rmtree(self.test_subdir, ignore_errors=True)

    def _safe_test_method_name(self) -> str:
        """Return a filesystem-safe version of the test method name."""
        return self._testMethodName.replace("/", "_").replace("\\", "_")
