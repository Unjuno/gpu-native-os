"""Run v03 differential checks in an isolated process."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class V03Tests(unittest.TestCase):
    def test_differential(self):
        out = subprocess.check_output(
            [sys.executable, str(ROOT / "experiments/v03/experiment.py"), "--test"],
            text=True, timeout=120,
        )
        data = json.loads(out)
        for bits in ("32", "64"):
            self.assertEqual(data[bits]["cases"], 203)
            self.assertEqual(data[bits]["three_backend_comparisons"], 609)
            self.assertEqual(data[bits]["invalid_images_rejected"], 3)


if __name__ == "__main__":
    unittest.main()
