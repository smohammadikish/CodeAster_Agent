# Beam Analysis Verification System

This repository integrates two beam-analysis applications into one root workflow:

1. `Automated_manual_calculations` for statically determinate beam verification
2. `opensees` for FEM-based analysis with OpenSees

The root app reads one shared input source, runs both solvers against it, and stores outputs in a unified workspace.

## Quick Start

Run the integrated workflow:

```bash
# Analyze a JSON input file
python main.py input/cantilever_beam.json

# Analyze a Code_Aster input folder
python main.py input/

# Use a custom workspace root
python main.py input/ --workspace-root my_results/

# List previous runs
python main.py --list-runs
```

## Supported Inputs

Both solvers can read:

### 1. Canonical JSON Input

Supported cases:
- `cantilever_tip_load`
- `simply_supported_midspan_point_load`

Example:

```json
{
  "model_id": "my_beam",
  "case_type": "cantilever_tip_load",
  "beam": {
    "length": 6.0,
    "num_elements": 20
  },
  "material": {
    "E": 30000.0,
    "nu": 0.3,
    "rho": 0.0025
  },
  "section": {
    "A": 1.0,
    "Iz": 0.0833,
    "Iy": 0.0833
  },
  "load": {
    "type": "point_force",
    "magnitude": 10.0,
    "direction": "y_negative"
  }
}
```

### 2. Code_Aster Input Set

A folder containing:
- `model.comm`
- `model.export`
- `beam.geo`

Example:

```text
input/
|-- model.comm
|-- model.export
`-- beam.geo
```

The integrated workflow treats that 3-file set as the shared input source.

## Workspace Layout

Integrated runs are stored under the root `workspace/` folder:

```text
workspace/
`-- 20260412_120530_input/
    |-- inputs/
    |   |-- metadata.json
    |   `-- input_snapshot/
    |       |-- beam.geo
    |       |-- model.comm
    |       `-- model.export
    |-- solvers/
    |   |-- automated_manual/
    |   |   |-- result.json
    |   |   `-- report.txt
    |   `-- opensees/
    |       |-- results.json
    |       `-- report.txt
    `-- comparison/
        `-- comparison_report.txt
```

Notes:
- `inputs/metadata.json` stores run metadata and the original input source path.
- For JSON inputs, `inputs/input.json` is stored instead of `input_snapshot/`.
- The integrated OpenSees output currently stores the canonical summary used by the orchestrator.

## Result Format

### Automated Manual Result

`solvers/automated_manual/result.json` contains canonical output with:
- `supports` and per-support `reactions`
- `extrema`
- `response_points`
- `stations`
- `summary_text`
- `warnings`

### OpenSees Result

`solvers/opensees/results.json` contains the canonical OpenSees summary used by the integration layer with:
- `supports` and per-support `reactions`
- `extrema`
- `response_points`
- `stations`
- `summary_text`
- optional `warnings`

### Comparison Report

`comparison/comparison_report.txt` compares:
- support reactions, matched by support position
- controlling absolute axial, shear, and bending values with their `x` locations
- controlling absolute displacement and rotation values with their `x` locations
- aggregate agreement statistics

Agreement buckets:
- excellent: `< 1%`
- acceptable: `1-5%`
- poor: `> 5%`

## Root Modules

- `main.py`: root CLI entry point
- `orchestrator.py`: runs both solvers and stores results
- `shared_input.py`: shared input discovery and snapshot logic
- `workspace_manager.py`: run-folder creation and result persistence
- `comparison_reporter.py`: solver-to-solver comparison

## Running Solvers Independently

You can still run each app on its own:

```bash
# Automated manual solver
cd Automated_manual_calculations
python main.py

# OpenSees solver
cd opensees
python main.py
```

The root workflow adds:
- one shared input entry point
- one root workspace
- one comparison report

## Dependencies

Root project:
- Python `>= 3.12`
- `uv` optional for lockfile management
- one repository-wide `pyproject.toml` at the repo root

OpenSees solver:
- OpenSees executable available in `PATH`, or
- `OPENSEES_EXECUTABLE` environment variable set, preferably in the repository-root `.env`, or
- `opensees_executable` provided in JSON input

## Environment Setup

If OpenSees is not in `PATH`, create a repository-root `.env` file:

```dotenv
OPENSEES_EXECUTABLE=C:\path\to\OpenSees.exe
```

You can also set it directly in the shell:

```bash
# Windows
set OPENSEES_EXECUTABLE=C:\path\to\OpenSees.exe

# Linux / macOS
export OPENSEES_EXECUTABLE=/path/to/OpenSees
```

## Current Limitations

- The integrated workflow completes even if OpenSees is unavailable, but then only the manual solver result is written and comparison is skipped.
- The root integration currently stores canonical OpenSees summary output, not every raw OpenSees artifact from the standalone app workspace.
- The integrated OpenSees step currently runs from a temporary workspace. On some Windows setups this can fail with `[WinError 5] Access is denied` inside the temp directory.

## Units

- For Code_Aster folder inputs, `beam.geo` geometry coordinates are interpreted in metres.
- In the sample Code_Aster files, rectangular section dimensions in `AFFE_CARA_ELEM` are interpreted in millimetres and converted to SI before analysis.
- Comparison reports use absolute values for envelope and displacement checks and also report the `x` location where each controlling value occurs.

## Development

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` does not pin any third-party Python packages because this repository currently uses only the Python standard library. OpenSees is still required separately as an external executable.

Create the virtual environment with:

```bash
uv venv
uv sync
```

If `uv` cannot write to the default user cache on your machine, use a local cache:

```bash
# Windows PowerShell
$env:UV_CACHE_DIR=".uv-cache"
uv venv
uv sync
```

Create or refresh the lockfile with:

```bash
uv lock
```

If `uv` cannot write to the default user cache on your machine, use a local cache:

```bash
UV_CACHE_DIR=.uv-cache uv lock
```
