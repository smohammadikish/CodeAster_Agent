"""CLI tests for workspace creation and default input handling."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
RUNTIME_ROOT = REPO_ROOT / "workspace" / "test_runtime"


class MainCliTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        temp_dir = RUNTIME_ROOT / uuid.uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def test_run_creates_workspace_with_result_file(self) -> None:
        temp_dir = self.make_temp_dir()
        workspace_root = REPO_ROOT / "workspace"
        before = {path for path in workspace_root.iterdir() if path.is_dir()} if workspace_root.exists() else set()
        completed = subprocess.run(
            [
                sys.executable,
                str(APP_ROOT / "main.py"),
                str(APP_ROOT / "examples" / "simple_beam_midpoint_load.json"),
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        after = {path for path in workspace_root.iterdir() if path.is_dir()}
        new_runs = sorted(after - before, key=lambda path: path.name)
        self.assertEqual(len(new_runs), 1)
        latest = new_runs[0]

        self.addCleanup(lambda: shutil.rmtree(latest, ignore_errors=True))

        self.assertTrue((latest / "simple_beam_midpoint_load.json").is_file())
        self.assertTrue((latest / "result.json").is_file())

        result = json.loads((latest / "result.json").read_text(encoding="utf-8"))
        stdout_result = json.loads(completed.stdout)

        self.assertEqual(result["beam"]["length"], 6.0)
        self.assertEqual(result["beam"]["length"], stdout_result["beam"]["length"])
        self.assertIn("Saved run workspace:", completed.stderr)

    def test_run_uses_root_input_folder_by_default(self) -> None:
        workspace_root = REPO_ROOT / "workspace"
        before = {path for path in workspace_root.iterdir() if path.is_dir()} if workspace_root.exists() else set()
        completed = subprocess.run(
            [
                sys.executable,
                str(APP_ROOT / "main.py"),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        result = json.loads(completed.stdout)
        self.assertEqual(result["case_type"], "cantilever_tip_load")
        self.assertTrue(any(support["type"] == "fixed" for support in result["supports"]))
        self.assertIn("Saved run workspace:", completed.stderr)
        after = {path for path in workspace_root.iterdir() if path.is_dir()}
        for run_dir in after - before:
            self.addCleanup(lambda path=run_dir: shutil.rmtree(path, ignore_errors=True))

    def test_default_report_is_saved_into_run_workspace(self) -> None:
        workspace_root = REPO_ROOT / "workspace"
        before = {path for path in workspace_root.iterdir() if path.is_dir()} if workspace_root.exists() else set()
        completed = subprocess.run(
            [
                sys.executable,
                str(APP_ROOT / "main.py"),
                str(APP_ROOT / "examples" / "simple_beam_midpoint_load.json"),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("Saved run workspace:", completed.stderr)
        after = {path for path in workspace_root.iterdir() if path.is_dir()}
        new_runs = sorted(after - before, key=lambda path: path.name)
        self.assertEqual(len(new_runs), 1)
        latest = new_runs[0]
        self.addCleanup(lambda: shutil.rmtree(latest, ignore_errors=True))
        self.assertTrue((latest / "simply_supported_report.txt").is_file())


if __name__ == "__main__":
    unittest.main()
