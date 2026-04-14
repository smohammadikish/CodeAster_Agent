# OpenSees Component

This folder contains the OpenSees-based solver used by the root verification workflow.

What remains here:
- `src/ver_opensees/`: OpenSees input parsing, Tcl generation, run orchestration, and result parsing
- `template_report/`: OpenSees report templates
- `schema/`: component-level schemas

Repository-wide defaults now live at the repo root:
- shared input folder: [`../input`](../input)
- shared workspace: [`../workspace`](../workspace)
- shared dependency file: [`../pyproject.toml`](../pyproject.toml)

Standalone usage:

```bash
python main.py
python main.py ../input
python main.py ../input/cantilever_beam.json
```

Note:
- OpenSees must still be installed and available through `PATH` or `OPENSEES_EXECUTABLE`.
- The preferred place for `OPENSEES_EXECUTABLE` is the repository-root `.env`.
- Standalone runs write an OpenSees workspace under the configured output root with `inputs/`, `results/`, and `run_metadata.json`.
