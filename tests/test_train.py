import tempfile
import unittest
from pathlib import Path

from lifelong_vla.runs import assert_config_matches, create_run_directory, mark_complete
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

    def test_config_mismatch_is_rejected_and_completion_is_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            run = create_run_directory(directory, {"seed": 1})
            assert_config_matches(run, {"seed": 1})
            with self.assertRaises(ValueError):
                assert_config_matches(run, {"seed": 2})
            mark_complete(run)
            self.assertTrue((run / "COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
