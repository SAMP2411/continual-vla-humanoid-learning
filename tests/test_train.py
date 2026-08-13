import tempfile
import unittest
from pathlib import Path

from lifelong_vla.train import make_run_directory, validate_config


class TrainConfigurationTest(unittest.TestCase):
    def test_rejects_missing_configuration(self):
        with self.assertRaises(ValueError):
            validate_config({"seed": 1})

    def test_run_directories_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_run_directory(directory)
            second = make_run_directory(directory)
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).is_dir())


if __name__ == "__main__":
    unittest.main()
