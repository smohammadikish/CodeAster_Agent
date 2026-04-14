from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("workspace")
RUNS_ROOT = WORKSPACE_ROOT / "runs"
WORKSPACE_LOGS_ROOT = WORKSPACE_ROOT / "logs"
DATA_ROOT = Path("data")
DB_PATH = DATA_ROOT / "simulations.db"


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    inputs: Path
    outputs: Path
    viz: Path
    logs: Path
    meta: Path


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_base_dirs() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def create_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{prefix}_{suffix}"


def get_run_paths(run_id: str) -> RunPaths:
    ensure_base_dirs()
    root = RUNS_ROOT / run_id
    return RunPaths(
        run_id=run_id,
        root=root,
        inputs=root / "inputs",
        outputs=root / "outputs",
        viz=root / "viz",
        logs=root / "logs",
        meta=root / "meta",
    )


def get_registered_run_paths(run_id: str) -> RunPaths | None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT run_dir FROM runs WHERE run_id = ? ORDER BY rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()

    if not row or not row[0]:
        return None

    root = Path(row[0])
    return RunPaths(
        run_id=run_id,
        root=root,
        inputs=root / "inputs",
        outputs=root / "outputs",
        viz=root / "viz",
        logs=root / "logs",
        meta=root / "meta",
    )


def ensure_run_dirs(run_id: str) -> RunPaths:
    paths = get_run_paths(run_id)
    for folder in (
        paths.root,
        paths.inputs,
        paths.outputs,
        paths.viz,
        paths.logs,
        paths.meta,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_workspace_file(filename: str, folder: str, run_id: str | None = None) -> Path:
    if run_id:
        paths = ensure_run_dirs(run_id)
        directory = getattr(paths, folder)
    else:
        ensure_base_dirs()
        directory = WORKSPACE_ROOT
        directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def init_db() -> Path:
    ensure_base_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                params_json TEXT,
                run_dir TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                path TEXT NOT NULL,
                exists_flag INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                message TEXT,
                details_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            """
        )
        conn.commit()
    return DB_PATH


def register_run(run_id: str, params: dict[str, Any], run_dir: str) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO runs(run_id, status, created_at, params_json, run_dir)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, "running", utc_now_iso(), json.dumps(params), run_dir),
        )
        conn.commit()


def finalize_run(run_id: str, status: str, error_message: str | None = None) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, completed_at = ?, error_message = ?
            WHERE run_id = ?
            """,
            (status, utc_now_iso(), error_message, run_id),
        )
        conn.commit()


def record_artifact(run_id: str, artifact_type: str, stage: str, path: str) -> None:
    init_db()
    artifact_path = Path(path)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO artifacts(run_id, artifact_type, stage, path, exists_flag, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                artifact_type,
                stage,
                str(artifact_path),
                int(artifact_path.exists()),
                utc_now_iso(),
            ),
        )
        conn.commit()


def record_tool_event(
    run_id: str,
    tool_name: str,
    status: str,
    started_at: str,
    completed_at: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO tool_events(
                run_id, tool_name, status, started_at, completed_at, message, details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                tool_name,
                status,
                started_at,
                completed_at,
                message,
                json.dumps(details or {}),
            ),
        )
        conn.commit()
