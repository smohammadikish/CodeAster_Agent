# 🧠 Code_Aster LLM Agent — Intelligent FEM Automation

This project combines **LangChain**, **Code_Aster**, and **Gmsh**  
to build an intelligent agent capable of performing **structural analysis workflows** automatically.

---

## 🚀 Overview

The system can operate in **two modes**:

1. **🤖 Agent Mode** — When a local or online LLM (like Ollama or OpenAI) is available  
   → The agent interprets natural language commands and automatically executes all analysis steps.

2. **⚙️ Manual Mode** — When no LLM is available (offline mode)  
   → The system executes all tools in order, step by step, ensuring full transparency.

---

## 🧩 Architecture

Each step of the FEM workflow is implemented as an independent **LangChain Tool**:

| Step | Tool | Description |
|------|------|-------------|
| 1️⃣ | `generate_geo_tool` | Generates the `.geo` file (geometry for Gmsh). |
| 2️⃣ | `mesh_with_gmsh_tool` | Calls Gmsh to mesh the geometry and export `.med`. |
| 3️⃣ | `generate_comm_tool` | Builds the `.comm` file for Code_Aster. |
| 4️⃣ | `generate_export_tool` | Creates the `.export` file linking all inputs. |
| 5️⃣ | `run_code_aster_tool` | Executes Code_Aster analysis (`as_run.bat --run ...`). |
| 6️⃣ | `summarize_results_tool` | Extracts textual summaries from results (`.mess`, `.resu`). |
| 7️⃣ | `visualize_resu_tool` | Creates visual outputs (`.png`, `.csv`, plots). |

All files are stored in a dedicated working directory:
```
workspace/
 ├─ model.geo
 ├─ model.med
 ├─ model.comm
 ├─ job.export
 ├─ result.mess
 ├─ result.resu
 ├─ beam_job.png
 └─ pipeline.log
```

---

## 🧱 Main Components

| File | Purpose |
|------|----------|
| `main_agent.py` | Unified entry point — detects if an LLM is available and runs either Agent or Manual mode. |
| `agent_pipeline_demo.py` | Simplified demo for learning LangChain Agent interaction. |
| `tools/` | Folder containing all modular tools for geometry, meshing, analysis, and visualization. |
| `env_loader.py` | Loads environment paths for Gmsh and Code_Aster executables. |

---

## ⚙️ How It Works

### 1️⃣ Start the pipeline

Default (auto-detect mode):

```bash
uv run python main_agent.py
```

The program checks whether a local LLM (Ollama) is available.

- ✅ If **available** → Runs **Agent Mode**
- ⚙️ If **not available** → Falls back to **Manual Mode**

---

### 2️⃣ Force a specific mode

You can explicitly choose which mode to run:

```bash
uv run python main_agent.py --mode agent
uv run python main_agent.py --mode manual
```

---

### 3️⃣ Example command (Agent Mode)

```text
🧱 You: Simulate a 3-meter steel beam with 15 elements and visualize the results.
🤖 Agent: Geometry created → Mesh done → Analysis completed → Results plotted.
```

---

## 🧠 Technical Details

- **LLM Interface**  
  - Default model: `ChatOllama(model="mistral")`
  - Fallback: `FakeListChatModel` (offline mock model)
- **Logging**  
  - All process logs are written to `workspace/pipeline.log`
- **Dependencies**
  - `langchain==0.2.14`
  - `langchain-community==0.2.12`
  - `langchain-ollama==0.1.1`
  - `numpy==1.26.4`
  - `matplotlib`, `meshio`, `gmsh` CLI, `Code_Aster 14+`

---

## 🧩 Environment Configuration

Add paths for executables in a `.env` file at the project root:

```env
GMSH_BIN=/usr/local/bin/gmsh
AS_RUN_BIN=C:/Program Files/CodeAster14/bin/as_run.bat
DATA_DIR=workspace
```

---

## 📦 Typical Workflow

1. User describes the analysis in natural language  
   (e.g., "Analyze a cantilever beam of 3 m with a tip load of 1000 N")
2. Agent or pipeline executes:
   - Generates geometry (.geo)
   - Meshes geometry (.med)
   - Prepares .comm and .export files
   - Runs Code_Aster
   - Summarizes and visualizes results
3. Results (plots + summaries) appear in `workspace/`

---

## 🧰 Development Notes

- Works both on **macOS** and **Windows**
- Designed for **Code_Aster v14** and **MED format 4.0.0**
- Can be integrated later with **LangGraph** for advanced flow control

---

## 🔮 Future Roadmap

| Phase | Feature |
|-------|----------|
| 1️⃣ | Multi-geometry support (beam, plate, solid) |
| 2️⃣ | Dynamic, modal, and nonlinear analysis |
| 3️⃣ | Real-time data and time-history input |
| 4️⃣ | Smart reasoning: AI suggests model refinements |
| 5️⃣ | GUI / Web dashboard for visualization |
| 6️⃣ | Integration with sensor data for SHM applications |

---

## 🧠 Authors
 **Saeideh Mohammadikish** 
 **Amir Yarmohamadi**
Structural Engineer & AI Developer  
Politecnico di Torino  