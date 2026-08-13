import unittest
from unittest.mock import patch

from scripts.system_report import nvidia_smi_gpus


class SystemReportTest(unittest.TestCase):
    @patch("scripts.system_report.shutil.which", return_value=None)
    def test_no_nvidia_smi_is_valid(self, _which):
        self.assertEqual(nvidia_smi_gpus(), [])


if __name__ == "__main__":
    unittest.main()
