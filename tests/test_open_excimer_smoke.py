import os
import unittest
from pathlib import Path

DEFAULT_DATASET_DIR = Path(
    "OSLUV Data/OSLUV Experiments/Open Excimer"
)


class TestOpenExcimerDatasetSmoke(unittest.TestCase):
    def test_dataset_files_present(self):
        dataset_dir = Path(os.environ.get("OPEN_EXCIMER_DIR", DEFAULT_DATASET_DIR))
        if not dataset_dir.exists():
            self.skipTest(f"Dataset directory not found: {dataset_dir}")

        sw3_files = sorted(dataset_dir.glob("*.sw3"))
        csv_files = sorted(dataset_dir.glob("*.csv"))

        self.assertGreaterEqual(len(sw3_files), 2)
        self.assertGreaterEqual(len(csv_files), 1)


if __name__ == "__main__":
    unittest.main()
