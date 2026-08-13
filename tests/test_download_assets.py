import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_assets import validate_manifest, verify


class DownloadAssetsTest(unittest.TestCase):
    def test_incomplete_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_manifest({"files": [{"name": "incomplete"}]})

    def test_checksum_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"checkpoint"
            (root / "weights.bin").write_bytes(payload)
            manifest = {
                "files": [{
                    "name": "fixture", "url": "https://example.invalid/weights.bin",
                    "path": "weights.bin", "sha256": hashlib.sha256(payload).hexdigest(),
                    "revision": "immutable-revision", "license": "Apache-2.0",
                }]
            }
            verify(manifest, root)


if __name__ == "__main__":
    unittest.main()
