from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from ver_opensees.models import BeamInput, InputSource
from ver_opensees.results import write_results_json
from ver_opensees.tcl_writer import write_model_tcl


# Run the full OpenSees workflow for one input case.
def run_analysis(beam_input: BeamInput, input_source: InputSource) -> Path:
    # Find the OpenSees executable before creating any files.
    executable = shutil.which(beam_input.opensees_executable)
    if executable is None:
        raise FileNotFoundError(
            f"Could not find OpenSees executable '{beam_input.opensees_executable}' in PATH."
        )

    # Create a self-contained workspace for this analysis case.
    run_dir = beam_input.output_root / beam_input.analysis_name
    inputs_dir = run_dir / "inputs"
    results_dir = run_dir / "results"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the original input and run details for traceability.
    _write_input_snapshot(inputs_dir, input_source)
    (run_dir / "run_metadata.json").write_text(
        _metadata_json(beam_input, executable, input_source, run_dir, results_dir),
        encoding="utf-8",
    )

    # Generate the Tcl model file used by OpenSees.
    tcl_path = results_dir / "model.tcl"
    write_model_tcl(beam_input, tcl_path)

    # Execute OpenSees inside the results folder.
    process = subprocess.run(
        [executable, tcl_path.name],
        cwd=results_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    # Save the console output from OpenSees.
    (results_dir / "stdout.txt").write_text(process.stdout, encoding="utf-8")
    (results_dir / "stderr.txt").write_text(process.stderr, encoding="utf-8")

    # Stop and report the problem if OpenSees failed.
    if process.returncode != 0:
        raise RuntimeError(
            f"OpenSees analysis failed with exit code {process.returncode}. "
            f"See {results_dir / 'stderr.txt'} and {results_dir / 'stdout.txt'}."
        )

    # Parse the OpenSees output files into JSON and a text report.
    write_results_json(beam_input, results_dir, results_dir / "results.json")
    return run_dir


# Convert run information into JSON text for the metadata file.
def _metadata_json(
    beam_input: BeamInput,
    executable: str,
    input_source: InputSource,
    run_dir: Path,
    results_dir: Path,
) -> str:
    import json

    # Convert the dataclass to a serializable dictionary.
    payload = asdict(beam_input)
    payload["output_root"] = str(beam_input.output_root)
    payload["run_directory"] = str(run_dir)
    payload["inputs_directory"] = str(run_dir / "inputs")
    payload["results_directory"] = str(results_dir)
    payload["resolved_opensees_executable"] = executable
    payload["input_source"] = {
        "source_type": input_source.source_type,
        "root_path": str(input_source.root_path),
        "snapshot_files": [str(path) for path in input_source.snapshot_files],
    }
    return json.dumps(payload, indent=2)


def _write_input_snapshot(inputs_dir: Path, input_source: InputSource) -> None:
    if input_source.source_type == "json" and len(input_source.snapshot_files) == 1:
        snapshot_path = inputs_dir / "input_snapshot.json"
        snapshot_path.write_text(
            input_source.snapshot_files[0].read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return

    snapshot_dir = inputs_dir / "input_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    for source_path in input_source.snapshot_files:
        shutil.copy2(source_path, snapshot_dir / source_path.name)
