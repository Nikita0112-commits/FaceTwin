"""Unit tests for FaceTwin core utilities."""

import json
from pathlib import Path
import unittest

from facetwin.core.config import ConfigManager, load_yaml_config
from facetwin.core.exceptions import ConfigurationError, FaceTwinError
from facetwin.core.logging_config import setup_logger


class TestFaceTwinCore(unittest.TestCase):
    """Test suite for core configuration, exceptions, and logging."""

    def test_exception_hierarchy(self):
        """Verify that custom exceptions inherit properly from FaceTwinError."""
        err = ConfigurationError("Sample config issue")
        self.assertIsInstance(err, FaceTwinError)
        self.assertEqual(str(err), "Sample config issue")

    def test_setup_logger(self):
        """Test logger creation with console and file output."""
        log_file = Path("experiments/results/test_run.log")
        logger = setup_logger(name="test_logger", level="DEBUG", log_file=str(log_file))

        logger.info("Test informational message")
        self.assertTrue(log_file.exists())
        content = log_file.read_text(encoding="utf-8")
        self.assertIn("Test informational message", content)

        # Cleanup test log file
        if log_file.exists():
            log_file.unlink()

    def test_load_yaml_config_missing_file(self):
        """Verify that loading non-existent config raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            load_yaml_config("non_existent_file.yaml")

    def test_load_json_config(self):
        """Verify that load_yaml_config supports JSON parsing natively."""
        temp_json = Path("experiments/results/temp_test.json")
        temp_json.parent.mkdir(parents=True, exist_ok=True)
        sample_data = {"project": {"name": "TestTwin", "version": "0.1.0"}}
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        loaded = load_yaml_config(temp_json)
        self.assertEqual(loaded["project"]["name"], "TestTwin")

        if temp_json.exists():
            temp_json.unlink()


if __name__ == "__main__":
    unittest.main()
