"""Unified workspace management for integrated beam analysis.

This module creates and manages the run-based workspace structure:
  workspace/YYYYMMDD_HHMMSS_runlabel/
    ├── inputs/
    │   ├── input.json or input_snapshot/
    │   └── metadata.json
    ├── solvers/
    │   ├── automated_manual/
    │   │   ├── result.json
    │   │   └── report.txt (if available)
    │   └── opensees/
    │       ├── results.json
    │       ├── summary_results.json
    │       ├── report.txt
    │       └── run_metadata.json
    └── comparison/
        └── comparison_report.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from shared_input import InputSource, copy_input_snapshot


@dataclass
class RunMetadata:
    """Metadata about a complete analysis run."""

    run_id: str  # YYYYMMDD_HHMMSS_runlabel
    timestamp: str  # ISO format timestamp
    input_source: dict  # source_type, root_path, etc.
    input_label: str  # Label derived from input file/folder name


def create_embedded_verification_workspace(
    input_source: InputSource,
    verification_root: Path | str,
    input_label: str,
) -> Path:
    """Create a verification folder directly inside an existing Codester run."""
    verification_root = Path(verification_root)
    inputs_dir = verification_root / "inputs"
    manual_dir = verification_root / "manual"
    opensees_dir = verification_root / "opensees"
    comparison_dir = verification_root / "comparison"

    for folder in (
        verification_root,
        inputs_dir,
        manual_dir,
        opensees_dir,
        comparison_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    copy_input_snapshot(input_source, inputs_dir)

    metadata = RunMetadata(
        run_id=verification_root.name,
        timestamp=datetime.now().isoformat(),
        input_source={
            "source_type": input_source.source_type,
            "root_path": str(input_source.root_path),
        },
        input_label=input_label,
    )
    (verification_root / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2),
        encoding="utf-8",
    )
    return verification_root


def create_run_workspace(
    input_source: InputSource,
    input_label: str,
    workspace_root: Path | str = "workspace",
) -> Path:
    """Create a new run workspace with the specified structure.

    Args:
        input_source: InputSource object with type and files.
        input_label: Label for this run (e.g., "cantilever_tip_load").
        workspace_root: Root folder for all workspaces. Default: workspace.

    Returns:
        Path to the created run workspace.
    """
    workspace_root = Path(workspace_root)
    timestamp = datetime.now()
    run_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{timestamp.microsecond // 1000:03d}_{input_label}"

    run_dir = workspace_root / run_id
    inputs_dir = run_dir / "inputs"
    solvers_dir = run_dir / "solvers"
    comparison_dir = run_dir / "comparison"

    # Create all required directories
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (solvers_dir / "automated_manual").mkdir(parents=True, exist_ok=True)
    (solvers_dir / "opensees").mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Copy input files
    copy_input_snapshot(input_source, inputs_dir)

    # Write metadata
    metadata = RunMetadata(
        run_id=run_id,
        timestamp=timestamp.isoformat(),
        input_source={
            "source_type": input_source.source_type,
            "root_path": str(input_source.root_path),
        },
        input_label=input_label,
    )
    (inputs_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2),
        encoding="utf-8",
    )

    return run_dir


def get_solver_output_dir(run_dir: Path, solver: str) -> Path:
    """Get the output directory for a specific solver within a run.

    Args:
        run_dir: Path to the run workspace.
        solver: Solver name: "automated_manual" or "opensees".

    Returns:
        Path to the solver's output directory.

    Raises:
        ValueError: If solver name is not recognized.
    """
    if solver not in {"automated_manual", "manual", "opensees"}:
        raise ValueError(f"Unknown solver: {solver}")
    if (run_dir / "solvers").exists():
        if solver == "manual":
            solver = "automated_manual"
        return run_dir / "solvers" / solver
    if solver == "automated_manual":
        solver = "manual"
    return run_dir / solver


def get_comparison_dir(run_dir: Path) -> Path:
    """Get the comparison output directory within a run."""
    return run_dir / "comparison"


def save_solver_result(
    run_dir: Path,
    solver: str,
    result_data: dict[str, Any],
    output_filename: str = "result.json",
) -> Path:
    """Save analysis result from a solver.

    Args:
        run_dir: Path to the run workspace.
        solver: Solver name: "automated_manual" or "opensees".
        result_data: Analysis result as dictionary.
        output_filename: Name of the output file. Default: result.json.

    Returns:
        Path to the saved file.
    """
    output_dir = get_solver_output_dir(run_dir, solver)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_filename
    output_path.write_text(
        json.dumps(result_data, indent=2),
        encoding="utf-8",
    )

    return output_path


def save_report(
    run_dir: Path,
    solver: str,
    report_text: str,
    report_filename: str = "report.txt",
) -> Path:
    """Save text report from a solver.

    Args:
        run_dir: Path to the run workspace.
        solver: Solver name: "automated_manual" or "opensees".
        report_text: Report content as string.
        report_filename: Name of the report file. Default: report.txt.

    Returns:
        Path to the saved report.
    """
    output_dir = get_solver_output_dir(run_dir, solver)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / report_filename
    report_path.write_text(report_text, encoding="utf-8")

    return report_path


def save_comparison_report(
    run_dir: Path,
    comparison_text: str,
    report_filename: str = "comparison_report.txt",
) -> Path:
    """Save comparison report between solvers.

    Args:
        run_dir: Path to the run workspace.
        comparison_text: Comparison report content as string.
        report_filename: Name of the report file. Default: comparison_report.txt.

    Returns:
        Path to the saved comparison report.
    """
    comparison_dir = get_comparison_dir(run_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    report_path = comparison_dir / report_filename
    report_path.write_text(comparison_text, encoding="utf-8")

    return report_path


def list_runs(workspace_root: Path | str = "workspace") -> list[Path]:
    """List all run directories in the workspace, sorted by timestamp.

    Args:
        workspace_root: Root workspace folder.

    Returns:
        List of run directories, most recent first.
    """
    workspace_root = Path(workspace_root)
    if not workspace_root.exists():
        return []

    runs = sorted(
        [d for d in workspace_root.iterdir() if d.is_dir()],
        reverse=True,
        key=lambda p: p.name,
    )
    return runs


def get_latest_run(workspace_root: Path | str = "workspace") -> Path | None:
    """Get the most recent run directory.

    Args:
        workspace_root: Root workspace folder.

    Returns:
        Path to the most recent run, or None if no runs exist.
    """
    runs = list_runs(workspace_root)
    return runs[0] if runs else None
