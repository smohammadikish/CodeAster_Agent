# Automated Manual Calculations

This folder contains the hand-calculation beam solver used by the root verification workflow.

What remains here:
- `beam_analysis.py`: solver logic
- `report_renderer.py`: text or docx report rendering
- `template_report/`: built-in report templates
- `schema/`: component-level schemas
- `examples/`: JSON examples used by component tests
- `tests/`: solver and CLI tests

Repository-wide defaults now live at the repo root:
- shared input folder: [`../input`](../input)
- shared workspace: [`../workspace`](../workspace)
- shared dependency file: [`../pyproject.toml`](../pyproject.toml)

Standalone usage:

```bash
python main.py
python main.py ../input
python main.py examples/simple_beam_midpoint_load.json
```

Notes:
- For Code_Aster folder inputs, geometry is read from `beam.geo` in metres.
- Rectangular section dimensions from `AFFE_CARA_ELEM` are normalized from millimetres to SI before area, inertia, and displacement are computed.
