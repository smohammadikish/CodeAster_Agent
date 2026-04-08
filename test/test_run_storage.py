import shutil
import sqlite3
import unittest
import uuid
from pathlib import Path

from storage import (DB_PATH, ensure_run_dirs, finalize_run, init_db,
                     record_artifact, record_tool_event, register_run,
                     resolve_workspace_file, utc_now_iso)
from tools.comm_generator import generate_comm_tool
from tools.export_generator import generate_export_tool
from tools.geo_generator import generate_geo_tool


class RunStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = f"test_{uuid.uuid4().hex[:8]}"
        self.paths = ensure_run_dirs(self.run_id)

    def tearDown(self) -> None:
        if self.paths.root.exists():
            shutil.rmtree(self.paths.root)

    def test_run_layout_and_input_outputs_paths(self) -> None:
        result = generate_geo_tool.invoke(
            {
                "length_mm": 1000.0,
                "n_elems": 5,
                "geo_name": "model.geo",
                "run_id": self.run_id,
            }
        )
        self.assertIn("Generated", result)

        med_path = resolve_workspace_file("model.med", "inputs", self.run_id)
        med_path.write_text("dummy med", encoding="utf-8")

        comm_result = generate_comm_tool.invoke(
            {
                "med_name": "model.med",
                "comm_name": "model.comm",
                "E": 210e9,
                "NU": 0.3,
                "RHO": 7850.0,
                "HY_mm": 120.0,
                "HZ_mm": 100.0,
                "FY_N": -1500.0,
                "run_id": self.run_id,
            }
        )
        self.assertIn("Generated", comm_result)

        export_result = generate_export_tool.invoke(
            {
                "job_name": "beam_job",
                "comm_name": "model.comm",
                "med_name": "model.med",
                "export_name": "model.export",
                "run_id": self.run_id,
            }
        )
        self.assertIn("Export file generated", export_result)

        self.assertTrue((self.paths.inputs / "model.geo").exists())
        self.assertTrue((self.paths.inputs / "model.med").exists())
        self.assertTrue((self.paths.inputs / "model.comm").exists())
        self.assertTrue((self.paths.inputs / "model.export").exists())

        export_text = (self.paths.inputs / "model.export").read_text(encoding="utf-8")
        self.assertIn(str((self.paths.inputs / "model.med").resolve()), export_text)
        self.assertIn(str((self.paths.outputs / "tab_reac.txt").resolve()), export_text)
        self.assertIn(str((self.paths.outputs / "beam_job.resu").resolve()), export_text)

    def test_database_tracks_runs_artifacts_and_tool_events(self) -> None:
        init_db()
        register_run(self.run_id, {"length_m": 3.0}, str(self.paths.root.resolve()))

        artifact_path = resolve_workspace_file("model.geo", "inputs", self.run_id)
        artifact_path.write_text("geo", encoding="utf-8")
        record_artifact(self.run_id, "geo", "inputs", str(artifact_path))
        record_tool_event(
            self.run_id,
            "generate_geo",
            "completed",
            utc_now_iso(),
            utc_now_iso(),
            "ok",
            {"geo_name": "model.geo"},
        )
        finalize_run(self.run_id, "completed")

        with sqlite3.connect(DB_PATH) as conn:
            run_row = conn.execute(
                "SELECT status, run_dir FROM runs WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
            artifact_row = conn.execute(
                "SELECT artifact_type, stage, exists_flag FROM artifacts WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
            tool_row = conn.execute(
                "SELECT tool_name, status FROM tool_events WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()

        self.assertEqual(run_row[0], "completed")
        self.assertEqual(run_row[1], str(self.paths.root.resolve()))
        self.assertEqual(artifact_row, ("geo", "inputs", 1))
        self.assertEqual(tool_row, ("generate_geo", "completed"))


if __name__ == "__main__":
    unittest.main()
