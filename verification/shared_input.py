"""Unified input handling for both beam analysis applications.

This module provides a common interface for reading inputs from JSON files
or Code_Aster input sets (model.comm, model.export, beam.geo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputSource:
    """Metadata about where the input came from."""

    source_type: str  # "json" or "code_aster"
    root_path: Path  # Path to the JSON file or folder with Code_Aster files
    snapshot_files: tuple[Path, ...] = ()  # Files to copy for traceability


CODE_ASTER_REQUIRED_EXACT = ("model.comm", "model.export")
CODE_ASTER_GEOMETRY_NAMES = ("beam.geo", "model.geo")


def resolve_input_source_path(path: Path | str) -> Path:
    """Resolve and validate the input source path.

    Args:
        path: Path to a JSON file, Code_Aster file, or folder containing
              Code_Aster files (model.comm, model.export, beam.geo).

    Returns:
        Path object pointing to the resolved source.

    Raises:
        ValueError: If the path is not a recognized input source.
    """
    resolved = Path(path).resolve()

    if resolved.is_dir():
        return resolved

    if resolved.suffix.lower() in {".json", ".comm", ".geo", ".export", ".med"}:
        return resolved

    raise ValueError(
        "Input must be a JSON file, a Code_Aster companion file, or a directory "
        "containing model.comm, model.export, and beam.geo."
    )


def detect_input_type(path: Path) -> tuple[str, Path]:
    """Detect the input type (JSON or Code_Aster) and return the root path.

    Args:
        path: Resolved path to input source.

    Returns:
        Tuple of (source_type, root_path) where:
        - source_type: "json" or "code_aster"
        - root_path: Path to use for reading the input

    Raises:
        ValueError: If neither JSON nor Code_Aster files are found.
    """
    if path.is_file() and path.suffix.lower() == ".json":
        return "json", path

    if path.is_file() and path.suffix.lower() in {".comm", ".geo", ".export", ".med"}:
        # Code_Aster companion file - use parent directory
        return "code_aster", path.parent

    if path.is_dir():
        # Check if it contains Code_Aster files
        required_files = set(CODE_ASTER_REQUIRED_EXACT)
        existing_files = {f.name for f in path.iterdir() if f.is_file()}
        if required_files.issubset(existing_files) and any(
            name in existing_files for name in CODE_ASTER_GEOMETRY_NAMES
        ):
            return "code_aster", path

        snapshot_dir = path / "input_snapshot"
        if snapshot_dir.is_dir():
            snapshot_files = {f.name for f in snapshot_dir.iterdir() if f.is_file()}
            if required_files.issubset(snapshot_files) and any(
                name in snapshot_files for name in CODE_ASTER_GEOMETRY_NAMES
            ):
                return "code_aster", snapshot_dir

        # Check if it contains JSON files
        json_files = list(path.glob("*.json"))
        if json_files:
            # Return the first JSON file found
            return "json", json_files[0]

        if snapshot_dir.is_dir():
            snapshot_json_files = list(snapshot_dir.glob("*.json"))
            if snapshot_json_files:
                return "json", snapshot_json_files[0]

        raise ValueError(
            f"Directory '{path}' does not contain valid Code_Aster files "
            "(model.comm, model.export, beam.geo) or JSON files."
        )

    raise ValueError(f"Cannot determine input type for '{path}'.")


def load_json_content(json_path: Path) -> dict[str, Any]:
    """Load and parse a JSON input file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    return json.loads(json_path.read_text(encoding="utf-8"))


def validate_code_aster_files(folder: Path) -> dict[str, Path]:
    """Validate that a folder contains all required Code_Aster files.

    Args:
        folder: Path to folder containing Code_Aster input set.

    Returns:
        Dictionary mapping file names to their paths.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    required_files = ["model.comm", "model.export"]
    file_paths = {}

    for file_name in required_files:
        file_path = folder / file_name
        if not file_path.exists():
            missing = [f for f in required_files if not (folder / f).exists()]
            raise FileNotFoundError(
                f"Code_Aster folder is missing required files: {', '.join(missing)}"
            )
        file_paths[file_name] = file_path

    geometry_path = next(
        (folder / name for name in CODE_ASTER_GEOMETRY_NAMES if (folder / name).exists()),
        None,
    )
    if geometry_path is None:
        raise FileNotFoundError(
            "Code_Aster folder is missing required geometry file: beam.geo or model.geo"
        )
    file_paths[geometry_path.name] = geometry_path

    return file_paths


def get_input_source_info(path: Path | str) -> InputSource:
    """Get comprehensive metadata about an input source.

    Args:
        path: Path to JSON file or Code_Aster folder.

    Returns:
        InputSource object with type, root path, and snapshot files.

    Raises:
        ValueError: If the input source is invalid.
    """
    resolved = resolve_input_source_path(path)
    source_type, root_path = detect_input_type(resolved)

    if source_type == "json":
        snapshot_files = (root_path,)
    else:  # code_aster
        file_paths = validate_code_aster_files(root_path)
        snapshot_files = tuple(sorted(file_paths.values()))

    return InputSource(
        source_type=source_type,
        root_path=root_path,
        snapshot_files=snapshot_files,
    )


def copy_input_snapshot(input_source: InputSource, destination: Path) -> None:
    """Copy input files to a destination for traceability.

    Args:
        input_source: InputSource object with snapshot files to copy.
        destination: Folder where files should be copied. Will be created if needed.
    """
    destination.mkdir(parents=True, exist_ok=True)

    if len(input_source.snapshot_files) == 1:
        source_file = input_source.snapshot_files[0]
        if source_file.suffix.lower() == ".json":
            # For single JSON file, copy with recognizable name
            (destination / "input.json").write_text(
                source_file.read_text(encoding="utf-8")
            )
            return

    # For Code_Aster files or multiple files, copy to subdirectory
    snapshot_dir = destination / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source_file in input_source.snapshot_files:
        import shutil

        shutil.copy2(source_file, snapshot_dir / source_file.name)
