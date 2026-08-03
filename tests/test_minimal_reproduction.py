from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MinimalReproductionTest(unittest.TestCase):
    def test_demo_reports_zero_rigid_ape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "result"
            subprocess.run(
                [sys.executable, "script/vio_eval.py", "demo", "--output-dir", str(output_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["metrics_backend"], "internal")
            self.assertEqual(metrics["matched_samples"], 5)
            self.assertLess(metrics["metrics"]["ape_translation_se3"]["rmse"], 1e-9)
            self.assertLess(metrics["metrics"]["ape_rotation_se3"]["rmse"], 1e-9)


if __name__ == "__main__":
    unittest.main()
