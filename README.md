# Code_Aster Beam Agent

This project is a LangChain + OpenAI beam-simulation assistant that drives a deterministic pipeline built on `Gmsh` and `Code_Aster`.

The agent collects beam parameters in chat, asks for confirmation, and then runs:

- geometry generation
- meshing
- `.comm` generation
- `.export` generation
- Code_Aster solve
- result visualization

All user-facing geometry inputs must be given in meters. The internal tools work in millimeters and the pipeline converts units automatically.

## Requirements

- Windows with working `Gmsh` and `Code_Aster`
- Python `3.12+`
- An OpenAI API key

## Environment

Create a `.env` file with values like:

```env
GMSH_BIN=C:\path\to\gmsh.exe
AS_RUN_BIN=C:\path\to\as_run.bat
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

Optional:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

## Install

If you already have a virtual environment:

```powershell
.\.venv\Scripts\activate
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

Or with plain pip:

```powershell
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run (CLI)

Interactive chat mode:

```powershell
python main_agent.py --mode chat
```

One-shot mode:

```powershell
python main_agent.py --mode agent --input "simulate a 4m beam with 25 elements and 2000 N load"
```

## Run Frontend

Frontend entry points:

- `frontend/gui_streamlit.py` (recommended)
- `frontend/gui_gradio.py`

From repo root:

```powershell
cd C:\Users\Michele\training
.\.venv\Scripts\activate
python -m pip install streamlit gradio
```

### Streamlit (Recommended)

```powershell
streamlit run frontend\gui_streamlit.py
```

Open:

```text
http://localhost:8501
```

### Gradio

```powershell
python frontend\gui_gradio.py
```

Open:

```text
http://localhost:7860
```

Both frontends reuse the same backend agent and pipeline.

What the frontend provides:

- Reuses the same `main_agent.py` backend agent as the CLI flow.
- Shows the current workflow phase, parameter sheet, run overview, execution timeline, generated plots, and artifact previews.
- Browses prior runs from SQLite and the local `workspace/runs/<run_id>` folders.
- Exposes run-scoped verification and a side-by-side compare area for image artifacts.
- Accepts stop commands such as `stop`, `cancel`, `exit`, and `done`.

## Recommended Chat Flow

Use `--mode chat` for the best experience. A good test conversation is:

```text
You: simulate a 4 m beam with 25 elements and 2000 N tip load
Agent: asks for missing material and section values or suggests using the catalog

You: use C35/45 concrete and IPE200
Agent: resolves the material and section from the catalog, summarizes E/NU/RHO and A/IY/IZ/JX/AY/AZ, and asks for confirmation

You: yes
Agent: runs the full simulation
```

Another valid full input set:

```text
simulate a 4 m beam with 25 elements, E=30e9 Pa, NU=0.2, RHO=2400 kg/m3, HY=0.20 m, HZ=0.10 m, and 2000 N load
```

The chat agent can also help with catalog browsing, for example:

```text
list available steel materials
list available steel sections
use S355 steel
use RHS 150x100x5
use a circular section with diameter 80 mm
```

Important:

- Always include explicit units for geometry.
- Inputs like `hy 20, hz 10` are now rejected as ambiguous.
- The agent only runs the simulation after explicit confirmation.

## Catalogs

Materials and sections are loaded from `library/` through a manifest-driven
catalog layer.

- Material references are stored with each family. Steel, aluminum, concrete,
  and wood materials are tagged with their Eurocode source.
- Concrete supports a formula fallback when a requested grade is not present in
  the CSV table. The fallback uses:
  `fcm = fck + 8 MPa` and `Ecm = 22 * (fcm / 10)^0.3 GPa`.
- Simple rectangular and circular sections can be generated without a CSV row.
  They return `A`, `Iy`, `Iz`, `IT`, `Avy`, and `Avz` in solver-facing units.
- For generated solid rectangles, `IT` uses the Saint-Venant approximation
  supplied in the project requirements and `Avy = Avz = 5/6 * A`.
- For generated solid circles, `IT = pi*d^4/32` and `Avy = Avz = 0.9 * A`.

## Outputs

Generated files are written under `workspace/`:

- `model.geo`
- `model.med`
- `model.comm`
- `model.export`
- `beam_job.mess`
- `beam_job.resu`
- `beam_job_def.png`
- `beam_job_rot.png`
- `shear_diagram.png`
- `moment_diagram.png`
- `mfz_contour.png`
- `freebody.png`
- `hybrid_pipeline_agentic.log`

## Notes

- The chat agent exposes catalog lookup tools for materials and sections plus the guarded high-level simulation tool.
- The LLM may use catalog tools to understand user requests and answer list/lookup questions.
- Final analysis input resolution and simulation execution still remain deterministic in Python code.
- Beam section transfer to Code_Aster uses `A`, `IY`, `IZ`, `JX`, `AY`, and `AZ`.
- Code_Aster result-file naming is aligned with the export `nomjob`.
- The MFZ contour reports beam length in meters correctly.
- The deterministic pipeline now passes the applied load correctly as `FY_N`.

## Troubleshooting

- `Connection error`
  Check `OPENAI_API_KEY`, internet access, firewall, and proxy settings.

- `Code_Aster binary not found`
  Check `AS_RUN_BIN` in `.env`.

- `mesh file 'workspace\\model.med' not found`
  This should no longer happen during normal chat flow. If it does, the environment is likely running old code.

- Import errors around LangChain
  Reinstall dependencies from `requirements.txt`.

- Geometry rejected as ambiguous
  Rewrite dimensions with units, for example `HY=0.20 m` and `HZ=100 mm`.
