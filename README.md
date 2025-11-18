# 🤖 Code_Aster LLM Agent  
### *AI-Native Automation for Finite Element Analysis*  
**Authors: Saeideh Mohammadikish / Amir Yarmohamadi — 2025**

A modular, agent-driven system that uses Large Language Models to fully automate **Code_Aster** Finite Element Analysis workflows — from natural-language input to complete FEM results, including geometry, mesh, solver execution, and visualization.

---

# 🌟 Overview

This project demonstrates an **AI-assisted structural analysis pipeline** capable of transforming a plain-English description of a structural system into:

- Gmsh geometry (`.geo`)
- Mesh (`.med`)
- Code_Aster input deck (`.comm`)
- Code_Aster `.export` file
- Automated solver execution
- Extraction of reactions, stresses, deformations
- Visualization of results

It is powered by a **tool-based LLM agent** (OpenAI API or local Ollama) that decides which operation to perform at each stage of the analysis pipeline.

---

# 🧠 Key Idea  
> **“Natural language in → Full FEM analysis out.”**

This repository shows how **AI can serve as a control layer** on top of established simulation tools, reducing the manual work of pre-processing and post-processing.

---

# 🧩 Architecture

```
User description
        │
        ▼
LLM Agent (OpenAI / Ollama)
        │ selects appropriate tool
        ▼
 ┌───────────────────────────────┐
 │          Pipeline Tools        │
 │ 1. Parameter Extractor         │
 │ 2. Geometry Generator (.geo)   │
 │ 3. Mesh Builder (.med)         │
 │ 4. Comm Generator (.comm)      │
 │ 5. Export Generator (.export)  │
 │ 6. Code_Aster Runner           │
 │ 7. Visualization Tool          │
 │ 8. Free-Body Diagram Tool      │
 └───────────────────────────────┘
        │
        ▼
FEM Results + Plots + Reports
```

Each component is fully modular and can be replaced or extended.

---

# 🔧 Features

### 🗣 Natural-Language Structural Modeling  
Provide dimensions, loads, materials, and constraints in plain English.

### 🧠 Agent-Based Automation  
The LLM determines which tool to run next: geometry, mesh, comm generation, solver, visualization, etc.

### 📐 Automated Geometry & Meshing (Gmsh)  
Generates `.geo` files and produces `.med` meshes.

### 🛠 Automatic Code_Aster Input Generation  
Creates all required solver input files without templates.

### 📊 Result Extraction & Visualization  
- Stress plots  
- Displacement plots  
- Reaction forces  
- Free-body diagrams  

### 🧱 Modular Design  
Every step lives in `tools/` as an independent component.

---

# 📂 Folder Structure

```
codeaster_agent/
│
├── agents/
│   ├── codeaster_agent.py
│   └── (...)
│
├── tools/
│   ├── geo_generator.py
│   ├── mesher_gmsh.py
│   ├── comm_generator.py
│   ├── export_generator.py
│   ├── run_aster_tool.py
│   ├── visualize_tool.py
│   ├── freebody_tool.py
│
├── config/
│   ├── paths.yaml
│
├── workspace/
│   ├── generated .geo / .comm / .export / .med files
│   ├── logs
│
├── report/
│   ├── plots, screenshots, outputs
│
└── main_agent.py
```

---

# 🚀 Running the Agent

## 1️⃣ Install Environment

```bash
uv venv
uv pip install -r requirements.txt
```

## 2️⃣ Export Required Paths

```
GMSH_BIN=/path/to/gmsh
AS_RUN_BIN=/path/to/as_run
OLLAMA_URL=http://localhost:11434
```

## 3️⃣ Launch Interactive Mode

```bash
python -m main_agent --mode chat
```

---

# 🎮 Example Usage

```
🟢 INTERACTIVE CHAT MODE

🤖 Describe your beam:
👤 6.85 m concrete beam, 20 elements, 0.30×0.20 m, tip load 1500 N

🤖 What is the beam material?
👤 C25/30

→ Geometry generated  
→ Mesh created  
→ .comm file ready  
→ Code_Aster running  
→ Plots saved  
→ Reactions extracted  
```

---

# 📸 Example Outputs

- Free-body diagram 
- Beam geometry  
- Finite element mesh  
- Von Mises contour  
- Displacement plot  
- Reaction table   

---

# 🔭 Roadmap

- [ ] Support for multiple cross-section shapes  
- [ ] 2D and 3D frame structures  
- [ ] Modal, torsional, and buckling analysis  
- [ ] Automatic PDF report generator  
- [ ] Web-based visualization dashboard  

---

# 👤 Author
**Saeideh Mohammadikish**  
**Amir Yarmohamadi**  

---

