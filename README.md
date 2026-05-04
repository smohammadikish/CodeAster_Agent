# Code_Aster Beam Agent

This project is a LangChain + OpenAI beam-simulation assistant that runs a deterministic `Gmsh` + `Code_Aster` pipeline.

## Scope (Current Version)

- No human-in-the-loop confirmation gate
- No decision-support recommendations flow
- No frontend UI (CLI only)
- Visualization output is limited to:
  - `shear_diagram.png`
  - `moment_diagram.png`

## Requirements

- Windows with working `Gmsh` and `Code_Aster`
- Python `3.12+`
- OpenAI API key

## Environment

Create `.env` with:

```env
GMSH_BIN=C:\path\to\gmsh.exe
AS_RUN_BIN=C:\path\to\as_run.bat
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

## Install

```powershell
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run (CLI)

Run with a one-shot input string:

```powershell
python .\main_agent.py --mode agent --input "cantilever beam, length 4 m, 25 elements, E 210e9 Pa, NU 0.3, RHO 7850 kg/m3, HY 200 mm, HZ 100 mm, load 2000 N"
```

Alternative explicit scenario format:

```powershell
python .\main_agent.py --mode agent --input "cantilever_tip_load, length 4 m, n_elems 25, E 210e9 Pa, NU 0.3, RHO 7850 kg/m3, HY 200 mm, HZ 100 mm, FY -2000 N"
```

Simply supported beam with a midpoint load:

```powershell
python .\main_agent.py --mode agent --input "simply supported beam, length 4 m, 24 elements, E 210e9 Pa, NU 0.3, RHO 7850 kg/m3, HY 200 mm, HZ 100 mm, load 2000 N"
```

Alternative explicit scenario format:

```powershell
python .\main_agent.py --mode agent --input "simply_supported_midpoint_load, length 4 m, n_elems 24, E 210e9 Pa, NU 0.3, RHO 7850 kg/m3, HY 200 mm, HZ 100 mm, FY -2000 N"
```

Important for simply supported beams:
- Use an even number of elements so the mesh has a midpoint node.
- The point load is applied at midspan.

Show input help:

```powershell
python .\main_agent.py --mode agent
```

Input text example:

```text
cantilever beam, length 4 m, 25 elements, E 210e9 Pa, NU 0.3, RHO 7850 kg/m3, HY 0.2 m, HZ 0.1 m, load 2000 N
```

Important:
- Do not paste plain input text directly into PowerShell as a separate line.
- PowerShell will treat it as a shell command and fail with `CommandNotFoundException`.
- Always pass the request through `--input "..."`.

## Outputs

Run artifacts are written under `workspace/runs/<run_id>/` including:

- inputs: `model.geo`, `model.med`, `model.comm`, `model.export`
- solver outputs: `beam_job.mess`, `beam_job.resu`, `tab_reac.txt`, `tab_disp.txt`, `tab_sie.txt`
- visualization: `shear_diagram.png`, `moment_diagram.png`
- metadata/log: `run.json`, `hybrid_pipeline_agentic.log`

## Notes

- No catalog recommendation/lookup flow in this version.
- Final execution stays deterministic in Python code.
- Geometry values without explicit units (e.g. `hy 20`) are rejected as ambiguous.
